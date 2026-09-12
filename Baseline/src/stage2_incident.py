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
    energy = motion_energy(frames)
    # 시작/끝 근처 튐(카메라 켜짐/영상 끝)은 충돌일 가능성이 낮으니 제외
    margin = max(1, len(energy) // 10)
    window = energy[margin:-margin] if len(energy) > 2 * margin else energy
    return int(np.argmax(window)) + margin


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


def _crosses_into_lane(box, lane):
    """박스 하단(바퀴 접지 근사)이 차선 경계 구간과 조금이라도 겹치면 진입으로 판단."""
    x0, _, x1, y1 = box[:4]
    lx, rx = _lane_x_at(lane[0], y1), _lane_x_at(lane[1], y1)
    if lx > rx:
        lx, rx = rx, lx
    return x1 > lx and x0 < rx


# --------------------------------------------------------------------------- entry/evasion/side (라벨 없음, 휴리스틱)
def find_entry_and_scene(model, transform, categories, frames: list, collision_frame: int, reranker=None):
    """상대차량이 자차 차선에 처음 '진입'하는 시점/방향, 충돌 직전 여유공간.

    entry_frame: 대회 공식 정의("피해차량 바퀴가 피의차량 차선에 최초로 닿는 시점", talkboard
    417186/417277)에 맞춰 화면 면적(area_frac) 프록시를 완전히 버리고 차선 추정 기반으로
    교체. Hough 검출이 성공하면 그 값을, 실패하면(DACON 공개 5샘플 전부 실패 - 사거리/
    근접촬영이 많아 차선이 거의 수평이거나 표시가 흐림, talkboard 417288 참고)
    _default_lane()(자차 중앙, 차선폭 화면폭의 38% 가정)을 써서 "차선 진입" 개념 자체는
    검출 성패와 무관하게 유지한다(area_frac은 정의 자체가 다른 값이라 폴백으로도 안 씀).
    entry_side/evasion_space는 이번 변경에서 안 건드림(레버 하나씩 바꿔야 다음 제출에서
    뭐가 통했는지 구분 가능).

    검색 구간을 collision_frame-30에서 -90으로 넓히고(entry가 area 기준보다 훨씬 이른
    프레임에서 걸릴 수 있어서), 못 찾으면 collision_frame이 아니라 window_lo로 폴백
    (talkboard 417277: "영상 시작 이전에 이미 진입했다면 첫 프레임을 제출" - window_lo가
    0이면 정확히 이 규칙과 일치).
    """
    h, w = frames[0].shape[:2]
    window_lo = max(0, collision_frame - 90)
    window_hi = min(len(frames) - 1, collision_frame + 5)  # 충돌 직후 몇 프레임도 봐야 근접탐지 fallback 가능

    detections = {}
    for t in range(window_lo, window_hi + 1):
        det = detect_vehicles(model, transform, categories, frames[t], reranker=reranker)
        if det is not None:
            detections[t] = det

    lane = estimate_ego_lane(frames, detections.keys(), h, w) if detections else _default_lane(h, w)

    entry_frame, entry_side = None, None
    for t in sorted(t for t in detections if t <= collision_frame):
        det = detections[t]
        x0, _, x1, _, _ = det
        if _crosses_into_lane(det, lane):
            entry_frame = t
            entry_side = "LEFT" if (x0 + x1) / 2 < w / 2 else "RIGHT"
            break

    if entry_frame is None:  # 못 찾으면 검색 구간 시작점으로 폴백(0이면 "시작 전 진입" 규칙과 일치)
        entry_frame, entry_side = window_lo, "RIGHT"

    # 충돌 순간은 모션블러로 탐지가 자주 빠진다 — 가장 가까운 프레임의 박스로 대체(0으로 뭉개지 않게)
    box_at_collision = None
    if detections:
        nearest_t = min(detections, key=lambda t: abs(t - collision_frame))
        box_at_collision = detections[nearest_t]

    evasion_space = 0
    collision_box_frame, collision_box = None, None
    if box_at_collision is not None:
        collision_box_frame, collision_box = nearest_t, box_at_collision
        x0, y0, x1, y1, score = box_at_collision
        free_left, free_right = x0, w - x1
        evasion_space = int(max(free_left, free_right) > 0.15 * w)

    return entry_frame, entry_side, evasion_space, collision_box_frame, collision_box


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
