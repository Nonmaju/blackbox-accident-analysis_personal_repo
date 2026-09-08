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
from torchvision.models.detection import (
    FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
    fasterrcnn_mobilenet_v3_large_320_fpn,
)

from video_io import load_frames

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
VEHICLE_CLASSES = {"car", "motorcycle", "bus", "truck"}
SCORE_THR = 0.5


# --------------------------------------------------------------------------- detector
def load_detector():
    weights = FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT
    model = fasterrcnn_mobilenet_v3_large_320_fpn(weights=weights)
    model.eval()
    categories = weights.meta["categories"]
    return model, weights.transforms(), categories


@torch.inference_mode()
def detect_vehicles(model, transform, categories, frame: np.ndarray):
    """frame(RGB) 안의 차량류 박스 중 가장 큰 것 1개 반환: (x0,y0,x1,y1,score) or None."""
    x = transform(torch.from_numpy(frame).permute(2, 0, 1))
    out = model([x])[0]
    best = None
    for box, label, score in zip(out["boxes"], out["labels"], out["scores"]):
        if score < SCORE_THR or categories[label] not in VEHICLE_CLASSES:
            continue
        x0, y0, x1, y1 = box.tolist()
        area = (x1 - x0) * (y1 - y0)
        if best is None or area > best[-1]:
            best = (x0, y0, x1, y1, float(score), area)
    return best[:5] if best else None


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
def find_entry_and_scene(model, transform, categories, frames: list, collision_frame: int):
    """collision_frame 이전 구간에서 상대차량이 처음 '크게' 잡히는 시점/방향, 충돌 직전 여유공간."""
    h, w = frames[0].shape[:2]
    entry_frame, entry_side, box_at_collision = None, None, None
    for t in range(max(0, collision_frame - 30), collision_frame + 1):
        det = detect_vehicles(model, transform, categories, frames[t])
        if det is None:
            continue
        x0, y0, x1, y1, score = det
        area_frac = (x1 - x0) * (y1 - y0) / (w * h)
        if entry_frame is None and area_frac > 0.03:  # 화면의 3% 이상 = '진입'으로 간주
            entry_frame = t
            entry_side = "LEFT" if (x0 + x1) / 2 < w / 2 else "RIGHT"
        if t == collision_frame:
            box_at_collision = (x0, y0, x1, y1)

    if entry_frame is None:  # 못 찾으면 충돌 프레임 자체로 폴백
        entry_frame, entry_side = collision_frame, "RIGHT"

    evasion_space = 0
    if box_at_collision is not None:
        x0, y0, x1, y1 = box_at_collision
        free_left, free_right = x0, w - x1
        evasion_space = int(max(free_left, free_right) > 0.15 * w)

    return entry_frame, entry_side, evasion_space


# --------------------------------------------------------------------------- calibration / self-check
def main():
    print("loading COCO-pretrained detector ...", flush=True)
    model, transform, categories = load_detector()

    out = ROOT / "model" / "stage2"
    out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out / "detector.pth")

    labels = pd.read_csv(DATA / "stage2/labels.csv")
    errors = []
    for row in labels.itertuples():
        frames = load_frames(DATA / "stage2" / row.path)
        pred_collision = find_collision_frame(frames)
        entry_frame, entry_side, evasion = find_entry_and_scene(model, transform, categories, frames, pred_collision)
        err = abs(pred_collision - row.t_collision)
        errors.append(err)
        print(f"{row.ID}: collision pred={pred_collision} true={row.t_collision} err={err} | "
              f"entry_frame={entry_frame} entry_side={entry_side} evasion_space={evasion}", flush=True)

    mae = float(np.mean(errors))
    print(f"\ncollision_frame MAE (검증 가능): {mae:.2f} frames")
    print("entry_frame/entry_side/evasion_space: 라벨이 없어 위 출력을 눈으로만 확인 - 정량 검증 불가")
    assert mae < 10, "충돌 프레임 오차가 너무 큼 — motion_energy 로직 점검 필요"


if __name__ == "__main__":
    main()
