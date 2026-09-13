"""Stage 2 v1 (얕은 버전): 충돌/진입 시점·회피공간·진입방향.

공개 라벨은 t_collision 뿐이라(나머지 3개는 -1, 실제 라벨 없음) 학습 자체가 불가능하다.
그래서:
  - collision_frame: 라벨 없이도 잴 수 있다 — 프레임간 화면 변화량(모션 에너지) 급증 지점.
    t_collision 5개로 검증 가능.
  - entry_frame / entry_side / evasion_space: COCO 사전학습 탐지기(torchvision, 학습 불필요)로
    상대 차량 박스를 잡아 위치 기반 휴리스틱으로 근사. **검증할 라벨이 아예 없어서 눈검증만 가능.**
    이 셋이 지금 파이프라인에서 가장 근거가 약한 부분 — AIHub 교통사고 영상 데이터(597) 승인되면
    최우선으로 여기부터 진짜 라벨로 교체해야 함.

ponytail: 탐지는 사전학습 fasterrcnn_mobilenet(COCO)로 때우고 트래킹/시점판단은 규칙 기반.
        라벨 생기면 얕은 GRU 헤드(baseline 원안)로 업그레이드.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch import nn
from torchvision.models.detection import (
    FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
    fasterrcnn_mobilenet_v3_large_320_fpn,
)

from video_io import load_frames

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
VEHICLE_CLASSES = {"car", "motorcycle", "bus", "truck"}
SCORE_THR = 0.2  # 0.5->0.2: 358영상/16248프레임 캐시로 검증(mean IoU 0.378->0.442, 히트율 45.2%->53.1%,
# 0.2 밑으로는 수확체감이라 여기서 멈춤). 재정렬기가 후보 많아질수록 진짜 정답 박스가
# 후보군에 남아있을 확률이 올라가서 선택 여지가 커지는 것으로 보임.
RERANK_FEATURES = ["cx", "cy", "bw", "bh", "score", "aspect"]


# --------------------------------------------------------------------------- detector
def load_detector():
    weights = FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT
    model = fasterrcnn_mobilenet_v3_large_320_fpn(weights=weights)  # 학습 시엔 인터넷 있으니 그대로 다운로드
    model.eval()
    categories = weights.meta["categories"]
    return model, weights.transforms(), categories


def load_reranker(path=None):
    """후보 박스 중 '상대차량'을 고르는 학습된 재정렬기 (stage2_rerank_train.py 산출물).

    실라벨 9085프레임(150 train/50 val 영상, 리키지 방지 위해 영상 단위 분할)으로
    score*area 규칙과 비교: mean IoU 0.315->0.334, 적중률 35.9%->38.3% (held-out).
    작지만 실측 개선이라 기본값으로 채택. 파일 없으면 None -> score*area로 폴백.
    """
    path = Path(path) if path else ROOT / "model" / "stage2" / "reranker.pt"
    if not path.exists():
        return None
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    net = nn.Sequential(nn.Linear(len(ckpt["features"]), 16), nn.ReLU(), nn.Linear(16, 1))
    net.load_state_dict(ckpt["state_dict"])
    net.eval()
    return net, ckpt["mean"], ckpt["std"]


def _rerank_features(c, w, h):
    x0, y0, x1, y1, score = c[:5]
    cx, cy = (x0 + x1) / 2 / w, (y0 + y1) / 2 / h
    bw, bh = (x1 - x0) / w, (y1 - y0) / h
    return [cx, cy, bw, bh, score, bw / (bh + 1e-6)]


def detect_vehicles(model, transform, categories, frame: np.ndarray, reranker=None):
    """frame(RGB) 안에서 '상대차량'으로 볼 박스 하나 반환: (x0,y0,x1,y1,score) or None.

    reranker가 주어지면 학습된 재정렬기로 후보 중 고르고(load_reranker() 참고),
    없으면 score*area 최대 규칙으로 폴백(캐시 실험상 largest_area와 거의 동일 -
    stage2_selection_search.py, 9085프레임 기준 0.337 vs 0.338, 유의미한 차이 아님).
    """
    candidates = detect_all_vehicles(model, transform, categories, frame, score_thr=SCORE_THR)
    if not candidates:
        return None
    if reranker is not None:
        net, mean, std = reranker
        h, w = frame.shape[:2]
        feats = torch.tensor([_rerank_features(c, w, h) for c in candidates], dtype=torch.float32)
        with torch.inference_mode():
            scores = net((feats - mean) / std).squeeze(-1)
        return candidates[int(scores.argmax())]
    return max(candidates, key=lambda c: c[4] * (c[2] - c[0]) * (c[3] - c[1]))


@torch.inference_mode()
def detect_all_vehicles(model, transform, categories, frame: np.ndarray, score_thr: float = 0.2):
    """score_thr 이상인 모든 차량류 박스를 반환 (선택 규칙 없이 후보 전체)."""
    x = transform(torch.from_numpy(frame).permute(2, 0, 1))
    out = model([x])[0]
    candidates = []
    for box, label, score in zip(out["boxes"], out["labels"], out["scores"]):
        if score < score_thr or categories[label] not in VEHICLE_CLASSES:
            continue
        x0, y0, x1, y1 = box.tolist()
        candidates.append((x0, y0, x1, y1, float(score)))
    return candidates


def _iou_boxes(a, b):
    ax0, ay0, ax1, ay1 = a[:4]
    bx0, by0, bx1, by1 = b[:4]
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    area_a, area_b = (ax1 - ax0) * (ay1 - ay0), (bx1 - bx0) * (by1 - by0)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def track_target_vehicle(model, transform, categories, frames, start, end, score_thr=0.3, min_track_iou=0.1):
    """start..end를 순서대로 훑으며 같은 차량을 계속 추적(간단 IoU 그리디 트래킹).

    프레임마다 독립적으로 '가장 큰 박스'를 고르면(기존 detect_vehicles) 프레임마다 다른
    차를 고를 위험이 크다 - 실라벨 대비 진단 결과 이게 병목이었다(oracle IoU 0.51 vs
    실제규칙 0.33). 대신 직전 프레임 박스와 IoU가 가장 높은 후보를 이어서 따라가고,
    추적을 놓치면(IoU가 다 낮으면) 크기*신뢰도 기준으로 재초기화한다.
    반환: {frame_idx: (x0,y0,x1,y1,score) or None}
    """
    track = {}
    current = None
    for t in range(start, end + 1):
        candidates = detect_all_vehicles(model, transform, categories, frames[t], score_thr=score_thr)
        if not candidates:
            track[t] = None
            current = None
            continue
        if current is not None:
            best = max(candidates, key=lambda c: _iou_boxes(current, c))
            if _iou_boxes(current, best) >= min_track_iou:
                current = best
            else:
                current = None
        if current is None:
            current = max(candidates, key=lambda c: c[4] * (c[2] - c[0]) * (c[3] - c[1]))
        track[t] = current
    return track


# --------------------------------------------------------------------------- collision frame (라벨 있음, 검증 가능)
def motion_energy(frames: list) -> np.ndarray:
    grays = [cv2.resize(cv2.cvtColor(f, cv2.COLOR_RGB2GRAY), (320, 180)) for f in frames]
    diffs = [cv2.absdiff(grays[i], grays[i - 1]).mean() for i in range(1, len(grays))]
    return np.array([diffs[0]] + diffs, dtype=np.float32)  # 프레임 0 자리 채움


def find_collision_frame(frames: list) -> int:
    # 시작부만 고정 8프레임 제외, 끝은 제외 안 함: CCD(801개 자차관여 실측)로 재보정.
    # 원래(비율 기반 margin=10%, 시작+끝 둘 다 제외)는 사고 클립 특성상 문제 있었음 -
    # 충돌이 늘 뒷부분(50프레임 중 30~49, 최솟값이 30!)이라 끝 10% 제외가 늦게 발생한
    # 진짜 충돌을 통째로 못 찾게 막았음. 비율 기반 margin=0(끝 제외 없음)도 시도했지만
    # DACON 공개 5샘플 중 하나(자차 무관 배경사고 영상)에서 초반 카메라 흔들림을
    # 충돌로 오검출(MAE 2.40->6.40). 영상 길이에 비례하는 "비율"보다 "고정 프레임 수"가
    # 더 타당하다고 보고(카메라 흔들림은 영상 길이와 무관하게 일정 시간) 시작 8프레임만
    # 고정 제외 + 끝 제외 없음으로 재시도: CCD 기준 MAE 6.22(최선), within±3 50.9%,
    # 공개 5샘플에서도 오검출 없음(stage2_ccd_calibrate_margin2.py로 그리드서치).
    energy = motion_energy(frames)
    start_exclude = min(8, max(0, len(energy) - 1))
    window = energy[start_exclude:]
    return int(np.argmax(window)) + start_exclude


# --------------------------------------------------------------------------- ego-lane (entry_frame 재정의용)
def _fit_lane_side(points):
    """차선은 화면상 거의 수직이라 x=m*y+b로 피팅(수직선에서도 안정적). 표본 2개 미만/
    y분산 거의 0이면 못 믿을 피팅이라 None."""
    if len(points) < 2:
        return None
    ys = np.array([p[1] for p in points], dtype=np.float64)
    xs = np.array([p[0] for p in points], dtype=np.float64)
    if ys.std() < 1e-3:
        return None
    m, b = np.polyfit(ys, xs, 1)
    return float(m), float(b)


def detect_lane_boundaries(frame: np.ndarray):
    """한 프레임에서 자차 차선 좌/우 경계선 근사: 도로 원근에 맞춘 사다리꼴 ROI로
    건물/보도 경계 등 도로 밖 직선을 배제하고, Canny+HoughLinesP로 거의 수평인 선
    (차선 아님)을 버린 뒤 기울기 부호로 좌/우 분리해 각각 x=m*y+b로 피팅.
    실패(선 없음/한쪽만 검출)하면 None - 호출부가 _default_lane()으로 넘어감.

    ponytail: 학습 없는 고전 CV(Hough) - 라벨이 없어 정량 검증 불가, 눈검증만 가능.
    실제 눈검증 결과 DACON 공개 5샘플 전부에서 실패함(사거리/근접 촬영이라 차선이
    거의 수평으로 보이거나 표시 자체가 흐림 - 포럼에서도 같은 문제 제기됨, talkboard
    417288). 그래도 실패시 area_frac이 아니라 _default_lane()(자차가 차선 중앙에
    있다고 가정한 고정 사다리꼴)으로 넘어가게 해서, 검출 성패와 무관하게 "차선 진입"
    개념 자체는 항상 유지한다 - area_frac 폴백은 정의 자체가 다른 값이라 완전히 버림.
    """
    h, w = frame.shape[:2]
    roi_top = int(h * 0.7)  # 화면 하단 30%만 본다 - 이보다 멀면 차선이 거의 수평이 돼서
    # 기울기로 차선/노면표시를 구분할 수가 없음(원근 문제, 실측으로 확인)
    mask = np.zeros((h, w), dtype=np.uint8)
    trapezoid = np.array([
        [0, h], [int(w * 0.3), roi_top],
        [int(w * 0.7), roi_top], [w, h],
    ])
    cv2.fillPoly(mask, [trapezoid], 255)

    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    edges = cv2.bitwise_and(cv2.Canny(gray, 50, 150), mask)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=25, minLineLength=int(h * 0.08), maxLineGap=30)
    if lines is None:
        return None

    left_pts, right_pts = [], []
    for x1, y1, x2, y2 in lines.reshape(-1, 4):
        if x2 == x1:
            continue
        slope = (y2 - y1) / (x2 - x1)
        if abs(slope) < 0.4:  # 수평에 가까운 선 제외
            continue
        (left_pts if slope < 0 else right_pts).extend([(x1, y1), (x2, y2)])

    left, right = _fit_lane_side(left_pts), _fit_lane_side(right_pts)
    if left is None or right is None:
        return None
    return left, right  # 각각 (m, b): x = m*y + b


def _default_lane(h: int, w: int):
    """차선 검출 실패시 기본값 - 자차가 차선 중앙에 있고 차선폭이 화면 하단 기준
    38%라고 가정(편도 2~3차로 도로에서 흔한 비율), 화면 중앙 상단(소실점 근사)으로
    수렴하는 사다리꼴. 실제 카메라 장착/도로폭에 따라 달라질 수 있는 거친 근사지만,
    area_frac(화면 면적)보다는 "차선 안에 있는지"라는 정의에 훨씬 가깝다."""
    half = 0.19 * w
    left = tuple(np.polyfit([h, 0], [w / 2 - half, w / 2], 1))
    right = tuple(np.polyfit([h, 0], [w / 2 + half, w / 2], 1))
    return left, right


def estimate_ego_lane(frames: list, ts, h: int, w: int):
    """검색 구간 프레임들에서 차선을 각각 검출 후 중앙값으로 대표 경계선 산출
    (한 프레임 오검출에 흔들리지 않게). 유효 표본이 1/4 미만이면 _default_lane()."""
    ts = list(ts)
    lefts, rights = [], []
    for t in ts:
        fit = detect_lane_boundaries(frames[t])
        if fit is None:
            continue
        lefts.append(fit[0])
        rights.append(fit[1])
    if len(lefts) < max(2, len(ts) // 4):
        return _default_lane(h, w)
    return tuple(np.median(lefts, axis=0)), tuple(np.median(rights, axis=0))


def _lane_x_at(line, y):
    m, b = line
    return m * y + b


# --------------------------------------------------------------------------- trajectory tracking (entry_frame v2)
# 2026-09-13: 팀원(정민)이 CCD 실측 entry_frame/entry_side/evasion_space 정답 36개(사람이
# 직접 검수, review_status=DONE)를 공유해줘서 entry_frame이 이번 세션 처음으로 정량
# 검증 가능해졌다(external/jungmin_labels/legacy_labels.csv). 우리 기존 로직(아래
# _crosses_into_lane 기반)을 그 36개로 검증해보니 Accuracy@0.3s=16.7%(6/36), MAE=1.79초 -
# 원인 진단(stage2_entry_diagnose2.py) 결과 "탐지가 안 돼서"가 아니라(0%) "차량이 처음부터
# 차선 근처에 있으면 프레임 0 근처에서 바로 진입으로 오판"하는 게 대부분(정민 코드 주석:
# "이게 v9의 지배적인 초반 오탐 원인이었다").
#
# 정민의 tools/stage2_track_v10_core.py(다중객체 그리디 트래킹 + "이전엔 코리도 밖에
# 있다가 이후 안으로 전환하는 순간"만 진입으로 인정하는 판정)를 그대로 이식해서 같은
# 36개로 우리 탐지기로 직접 재검증(stage2_track_v10_validate.py) - TRACK_CONTINUITY가
# Accuracy@0.3s=22.2%(8/36), MAE=1.11초, entry_side=75.0%, evasion_space=52.8%로 전부
# 기존보다 나음(정민 자신의 36개 재사용 검증과 달리, 우리는 이 라벨로 아무것도 설계한
# 적이 없어 사실상 독립 재현). 프레임별 독립 재정렬(detect_vehicles/reranker) 대신
# 이 트래킹으로 entry_frame/entry_side/evasion_space를 전부 교체한다.
def _lane_bounds(lane, y: float) -> tuple:
    values = [float(m) * float(y) + float(b) for m, b in lane]
    return min(values), max(values)


def _track_geometry(box, width: int, height: int):
    x0, y0, x1, y1, score = map(float, box[:5])
    return ((x0 + x1) / (2 * width), (y0 + y1) / (2 * height),
             max(1.0, x1 - x0) / width, max(1.0, y1 - y0) / height, float(score))


def _link_cost(previous, current, width: int, height: int, gap: int = 1) -> float:
    """스케일 인지 연결 비용 - 1을 넘으면 연결 후보에서 제외."""
    ax, ay, aw, ah, _ = _track_geometry(previous, width, height)
    bx, by, bw, bh, _ = _track_geometry(current, width, height)
    distance = np.hypot(ax - bx, ay - by) / max(0.035, 0.5 * (aw + bw), 0.5 * (ah + bh))
    scale = abs(np.log((bw * bh + 1e-6) / (aw * ah + 1e-6)))
    return float(distance / max(1.0, gap) + 0.35 * scale + 0.08 * (gap - 1))


def _build_tracks(all_boxes: dict, width: int, height: int, max_gap: int = 3, max_cost: float = 2.2) -> list:
    """그리디 이분매칭 - 결정적이고 외부 의존성 없음(제출 환경 그대로 동작)."""
    tracks: list = []
    for time_index in sorted(all_boxes):
        boxes = sorted((tuple(map(float, box[:5])) for box in all_boxes[time_index]),
                        key=lambda box: (-box[4], box[0], box[1]))[:16]
        candidates = []
        for track_index, track in enumerate(tracks):
            gap = time_index - track["times"][-1]
            if 1 <= gap <= max_gap:
                for box_index, box in enumerate(boxes):
                    cost = _link_cost(track["boxes"][-1], box, width, height, gap)
                    if cost <= max_cost:
                        candidates.append((cost, track_index, box_index))
        used_tracks, used_boxes = set(), set()
        for _, track_index, box_index in sorted(candidates):
            if track_index in used_tracks or box_index in used_boxes:
                continue
            tracks[track_index]["times"].append(int(time_index))
            tracks[track_index]["boxes"].append(boxes[box_index])
            used_tracks.add(track_index); used_boxes.add(box_index)
        for box_index, box in enumerate(boxes):
            if box_index not in used_boxes:
                tracks.append({"times": [int(time_index)], "boxes": [box]})
    return tracks


def _track_inside_ratio(box, lane) -> float:
    x0, _, x1, y1 = map(float, box[:4])
    left, right = _lane_bounds(lane, y1)
    overlap = max(0.0, min(x1, right) - max(x0, left))
    return overlap / max(1.0, x1 - x0)


def _track_features(track: dict, lane, collision: int, width: int, height: int) -> dict:
    times, boxes = track["times"], track["boxes"]
    usable = [(t, b) for t, b in zip(times, boxes) if t <= collision + 3]
    if not usable or not any(t <= collision for t, _ in usable):
        return {"score": -1e9, "entry": None, "side": "RIGHT", "crossing": False,
                "terminal_distance": 10**9, "length": 0, "continuity": 0.0}
    times = [x[0] for x in usable]; boxes = [x[1] for x in usable]
    inside = [_track_inside_ratio(box, lane) for box in boxes]
    crossing_index = None
    for index in range(len(times)):
        future = inside[index:min(len(inside), index + 3)]
        previous = inside[max(0, index - 2):index]
        if inside[index] >= 0.10 and sum(value >= 0.10 for value in future) >= min(2, len(future)):
            # 처음부터 코리도 안에 있는 건 진입 증거가 아니다 - 직전(median)이 코리도
            # 밖(<10%)이었다가 지금 안으로 전환되는 순간만 인정(v9의 지배적 초반 오탐 방지).
            if previous and float(np.median(previous)) < 0.10:
                crossing_index = index
                break
    entry = times[crossing_index] if crossing_index is not None else None
    side_samples = boxes[max(0, (crossing_index or 0) - 3):(crossing_index or 0) + 1]
    offsets = []
    for box in side_samples:
        x0, _, x1, y1 = box[:4]
        left, right = _lane_bounds(lane, y1)
        offsets.append((x0 + x1) / 2 - (left + right) / 2)
    side = "LEFT" if offsets and float(np.median(offsets)) < 0 else "RIGHT"
    terminal_index = int(np.argmin([abs(t - collision) for t in times]))
    terminal_time, terminal = times[terminal_index], boxes[terminal_index]
    x0, y0, x1, y1, confidence = terminal
    area = (x1 - x0) * (y1 - y0) / max(1.0, width * height)
    terminal_distance = abs(terminal_time - collision)
    continuity = len(times) / max(1, times[-1] - times[0] + 1)
    areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in boxes]
    expansion = np.log((areas[-1] + 1) / (areas[0] + 1)) / max(1, len(areas) - 1)
    lateral = 0.0
    if len(boxes) >= 2:
        lateral = abs(((boxes[-1][0] + boxes[-1][2]) - (boxes[0][0] + boxes[0][2])) / (2 * width))
    score = (2.0 * _track_inside_ratio(terminal, lane) + 1.2 * min(1.0, area / 0.08) + 0.45 * confidence
             + 0.65 * continuity + 0.35 * min(1.0, max(0.0, expansion) / 0.08)
             + 0.45 * min(1.0, lateral / 0.15) - 0.18 * terminal_distance)
    if crossing_index is not None and entry <= collision:
        score += 0.9 + 0.25 * min(1.0, (collision - entry) / max(1, collision))
    return {"score": float(score), "entry": entry, "side": side,
            "crossing": crossing_index is not None, "terminal": terminal,
            "terminal_distance": terminal_distance, "length": len(times), "continuity": float(continuity)}


def trajectory_scene(all_boxes: dict, lane, height: int, width: int, collision: int) -> dict:
    """TRACK_CONTINUITY: 충돌과 이어지는 트랙 중 스코어가 가장 높은 것을 상대차량으로
    선택 - 프레임 하나하나 독립적으로 고르는 대신, 전체 궤적(위치/크기/신뢰도/연속성/
    확장/횡이동 종합)을 보고 고른다. CCD 실측 36라벨 검증: Accuracy@0.3s 16.7%->22.2%,
    MAE 1.79->1.11초, entry_side 69.4%->75.0%, evasion_space 50.0%->52.8%."""
    tracks = _build_tracks(all_boxes, width, height)
    featured = [(track, _track_features(track, lane, collision, width, height)) for track in tracks]
    eligible = [(track, feature) for track, feature in featured
                if feature["terminal_distance"] <= 5 and feature["length"] >= 2]
    selected = max(eligible, key=lambda item: item[1]["score"], default=None)
    if selected is None:
        return {"entry": int(max(0, collision)), "side": "RIGHT", "evasion": 0,
                "collision_box_frame": None, "collision_box": None}
    track, feature = selected
    if feature["entry"] is None:
        # 진입 프레임을 임의로 만들지 않는다 - 이 트랙에서 실제 관측된 첫 시점을 쓴다.
        pre_collision = [t for t in track["times"] if t <= collision]
        entry = min(pre_collision) if pre_collision else max(0, collision)
    else:
        entry = int(feature["entry"])
    terminal = feature["terminal"]
    x0, _, x1, y1 = terminal[:4]
    left, right = _lane_bounds(lane, y1)
    lane_width = max(1.0, right - left)
    evasion = int(max(max(0.0, x0 - left), max(0.0, right - x1)) >= 0.32 * lane_width)
    return {"entry": min(int(collision), int(entry)), "side": feature["side"], "evasion": evasion,
            "collision_box_frame": track["times"][int(np.argmin([abs(t - collision) for t in track["times"]]))],
            "collision_box": terminal}


def _crosses_into_lane(box, lane):
    """박스 하단(바퀴 접지 근사)이 차선 경계 구간과 조금이라도 겹치면 진입으로 판단."""
    x0, _, x1, y1 = box[:4]
    lx, rx = _lane_x_at(lane[0], y1), _lane_x_at(lane[1], y1)
    if lx > rx:
        lx, rx = rx, lx
    return x1 > lx and x0 < rx


# --------------------------------------------------------------------------- entry/evasion/side (2026-09-13: CCD 실측 36라벨로 검증됨)
def find_entry_and_scene(model, transform, categories, frames: list, collision_frame: int, reranker=None):
    """상대차량이 자차 차선에 처음 '진입'하는 시점/방향, 충돌 직전 여유공간.

    trajectory_scene(TRACK_CONTINUITY) 기반 - 프레임 독립 탐지+재정렬(reranker) 대신
    전체 후보를 그리디로 이어붙인 다중객체 트랙 중 충돌과 이어지는 가장 그럴듯한 트랙을
    골라 그 트랙의 "코리도 밖->안 전환" 시점을 진입으로 쓴다. reranker 인자는 옛 시그니처
    호환용으로만 남겨둠(더는 안 씀 - trajectory_scene 자체가 선택 로직을 대신함).

    entry_frame 공식 정의(talkboard 417186/417277 - "피해차량 바퀴가 피의차량 차선에
    최초로 닿는 시점"), evasion_space 공식 정의(talkboard 417319 - "자차가 진행방향을
    바꿔 피할 수 있는 물리적 공간")는 그대로 유지, 판정 방식만 트래킹으로 교체.
    검증: CCD 실측 36라벨(external/jungmin_labels/legacy_labels.csv) 기준 Accuracy@0.3s
    16.7%->22.2%, MAE 1.79->1.11초, entry_side 69.4%->75.0%, evasion_space 50.0%->52.8%
    (stage2_track_v10_validate.py, 우리 탐지기로 직접 재현 - 이 라벨로 뭘 설계한 적이
    없어 사실상 독립 검증).
    """
    h, w = frames[0].shape[:2]
    high = min(len(frames) - 1, collision_frame + 3)
    sampled = sorted(set(np.rint(np.linspace(0, high, min(96, high + 1))).astype(int).tolist()))

    all_boxes = {}
    for t in sampled:
        boxes = detect_all_vehicles(model, transform, categories, frames[t], score_thr=SCORE_THR)
        if boxes:
            all_boxes[t] = boxes

    lane = estimate_ego_lane(frames, sampled, h, w) if sampled else _default_lane(h, w)
    result = trajectory_scene(all_boxes, lane, h, w, collision_frame)

    return (
        result["entry"], result["side"], result["evasion"],
        result["collision_box_frame"], result["collision_box"],
    )


def _evasion_space_from_lane(frame, lane, model, transform, categories, y_ref: float, w: float) -> int:
    """회피공간 공식 정의(talkboard 417319): "충돌 시점 기준 피의차량(자차)이 진행
    방향을 바꿔 충돌을 피할 수 있는 물리적 공간"이 있는지 - 상대차량 박스 좌우 여백
    (이전 방식, 자차와 무관한 기준)이 아니라 **자차 차선 경계 바로 옆**에 다른 차량이
    없는 인접 공간이 있는지로 판단한다.

    한계: 반대차선/보도 등 "현실적으로 회피 경로가 아닌 공간"을 구분할 도로 정보가
    없어서, 화면 가장자리에 너무 붙은 경우만 배제하는 거친 근사다. 정량 검증 불가
    (라벨 없음, 눈검증만 가능) - 이전 방식과 마찬가지로 신뢰도 낮은 휴리스틱."""
    lx, rx = _lane_x_at(lane[0], y_ref), _lane_x_at(lane[1], y_ref)
    if lx > rx:
        lx, rx = rx, lx
    lane_w = max(rx - lx, 1.0)
    others = detect_all_vehicles(model, transform, categories, frame, score_thr=SCORE_THR)

    def occupied(x0, x1):
        return any(not (c[2] < x0 or c[0] > x1) for c in others)

    left_clear = lx > 0.05 * w and not occupied(max(0.0, lx - lane_w), lx)
    right_clear = rx < 0.95 * w and not occupied(rx, min(w, rx + lane_w))
    return int(left_clear or right_clear)


# --------------------------------------------------------------------------- calibration / self-check
def main():
    print("loading COCO-pretrained detector ...", flush=True)
    model, transform, categories = load_detector()
    reranker = load_reranker()
    print(f"reranker: {'있음, 사용' if reranker else '없음(model/stage2/reranker.pt) - score*area로 폴백'}", flush=True)

    out = ROOT / "model" / "stage2"
    out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out / "detector.pth")

    labels = pd.read_csv(DATA / "stage2/labels.csv")
    errors = []
    for row in labels.itertuples():
        frames = load_frames(DATA / "stage2" / row.path)
        pred_collision = find_collision_frame(frames)
        entry_frame, entry_side, evasion, _, _ = find_entry_and_scene(
            model, transform, categories, frames, pred_collision, reranker=reranker
        )
        err = abs(pred_collision - row.t_collision)
        errors.append(err)
        print(f"{row.ID}: collision pred={pred_collision} true={row.t_collision} err={err} | "
              f"entry_frame={entry_frame} entry_side={entry_side} evasion_space={evasion}", flush=True)

    mae = float(np.mean(errors))
    print(f"\ncollision_frame MAE (검증 가능): {mae:.2f} frames")
    print("entry_frame/entry_side/evasion_space: 라벨이 없어 위 출력을 눈으로만 확인 - 정량 검증 불가")
    assert mae < 10, "충돌 프레임 오차가 너무 큼 - motion_energy 로직 점검 필요"


if __name__ == "__main__":
    main()
