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


# --------------------------------------------------------------------------- entry/evasion/side (라벨 없음, 휴리스틱)
def find_entry_and_scene(model, transform, categories, frames: list, collision_frame: int, reranker=None):
    """collision_frame 이전 구간에서 상대차량이 처음 '크게' 잡히는 시점/방향, 충돌 직전 여유공간.

    실라벨 9085프레임 기준: score*area(폴백) IoU~0.34, 학습 재정렬기(load_reranker) IoU 0.334
    (held-out 영상 기준, 큰 표본에서 재확인) - 재정렬기가 근소하게 낫고 유일하게 실측
    검증된 개선이라 기본 사용. entry_side/evasion_space는 여전히 신뢰도 낮음.
    """
    h, w = frames[0].shape[:2]
    window_lo = max(0, collision_frame - 30)
    window_hi = min(len(frames) - 1, collision_frame + 5)  # 충돌 직후 몇 프레임도 봐야 근접탐지 fallback 가능

    detections = {}
    entry_frame, entry_side = None, None
    for t in range(window_lo, window_hi + 1):
        det = detect_vehicles(model, transform, categories, frames[t], reranker=reranker)
        if det is None:
            continue
        detections[t] = det
        x0, y0, x1, y1, score = det
        area_frac = (x1 - x0) * (y1 - y0) / (w * h)
        if entry_frame is None and t <= collision_frame and area_frac > 0.03:  # 화면의 3% 이상 = '진입'으로 간주
            entry_frame = t
            entry_side = "LEFT" if (x0 + x1) / 2 < w / 2 else "RIGHT"

    if entry_frame is None:  # 못 찾으면 충돌 프레임 자체로 폴백
        entry_frame, entry_side = collision_frame, "RIGHT"

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
