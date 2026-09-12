"""팀원(정민) 제출 코드(submit-3.zip) Stage3 분석에서 얻은 개선안 두 가지를
공개 라벨 50개(기존 CAN 보정 임계값 그대로)에 적용했을 때 회귀가 없는지 확인.

1. 텍스처/화질 게이트: Laplacian variance가 너무 낮은(거의 단색/블러) 프레임은
   optical flow를 못 믿고 CONSTANT/STRAIGHT로 강제.
2. 평균(박스) 스무딩 대신 중앙값(median) 슬라이딩 윈도우 - 이상치 스파이크에 강함.

기존 임계값(model/stage3/best.pt: stopped=0.1, accel_eps=0.01, steer=0.65)은 그대로
쓰고 두 개선만 켰다 껐다 하며 공개 50라벨 정확도 비교 - 물리량을 안 바꾸는(노이즈만
억제하는) 변경이라 재보정 없이도 안전한지 확인하는 목적.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

from video_io import load_frames

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FLOW_SIZE = (160, 90)
QUALITY_THR = 2.0


def compute_flow_series_q(frames):
    """compute_flow_series + 프레임별 quality(Laplacian variance)."""
    small = [cv2.resize(cv2.cvtColor(f, cv2.COLOR_RGB2GRAY), FLOW_SIZE) for f in frames]
    n = len(small) // 2
    w, h = FLOW_SIZE
    road = slice(int(h * 0.55), h)
    horizon = slice(int(h * 0.25), int(h * 0.55))
    speed = np.zeros(n, dtype=np.float32)
    steer = np.zeros(n, dtype=np.float32)
    quality = np.zeros(n, dtype=np.float32)
    for t in range(n):
        i0 = min(2 * t, len(small) - 3)
        i1 = i0 + 2
        flow = cv2.calcOpticalFlowFarneback(small[i0], small[i1], None, 0.5, 2, 15, 3, 5, 1.2, 0)
        mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
        speed[t] = float(np.median(mag[road]))
        steer[t] = float(np.median(flow[horizon, :, 0]))
        quality[t] = float(cv2.Laplacian(small[i1], cv2.CV_32F).var())
    return speed, steer, quality


def _smooth_mean(x, k=3):
    if len(x) < 2 * k + 1:
        return x
    return np.convolve(x, np.ones(2 * k + 1) / (2 * k + 1), mode="same")


def _smooth_median(x, k=3):
    if len(x) < 2 * k + 1:
        return x
    width = 2 * k + 1
    padded = np.pad(x, (k, k), mode="edge")
    return np.median(np.lib.stride_tricks.sliding_window_view(padded, width), axis=-1)


def classify(speed, steer, quality, stopped_thr, accel_eps, steer_thr, use_median, use_gate):
    smooth = _smooth_median if use_median else _smooth_mean
    speed_s = smooth(speed)
    n = len(speed_s)
    accel_out, steer_out = [], []
    for t in range(n):
        if speed_s[t] < stopped_thr:
            accel_out.append("STOPPED")
        else:
            lo, hi = max(0, t - 3), min(n, t + 4)
            slope = speed_s[hi - 1] - speed_s[lo] if hi - 1 > lo else 0.0
            accel_out.append("ACCELERATING" if slope > accel_eps else "DECELERATING" if slope < -accel_eps else "CONSTANT")
        s = steer[t]
        steer_out.append("LEFT" if s > steer_thr else "RIGHT" if s < -steer_thr else "STRAIGHT")
        if use_gate and quality[t] < QUALITY_THR:
            accel_out[-1], steer_out[-1] = "CONSTANT", "STRAIGHT"
    return accel_out, steer_out


def main():
    checkpoint = torch.load(ROOT / "model/stage3/best.pt", map_location="cpu", weights_only=False)
    stopped_thr, accel_eps, steer_thr = checkpoint["stopped_thr"], checkpoint["accel_eps"], checkpoint["steer_thr"]
    print(f"기존 임계값: stopped={stopped_thr} accel_eps={accel_eps} steer={steer_thr}")

    labels = pd.read_csv(DATA / "stage3/labels.csv")
    cache = {}
    for vid_id, group in labels.groupby("ID"):
        frames = load_frames(DATA / "stage3/videos" / f"{vid_id}.mp4")
        cache[vid_id] = (compute_flow_series_q(frames), group)

    for use_median, use_gate, name in [
        (False, False, "기존(평균 스무딩, 게이트 없음)"),
        (True, False, "중앙값 스무딩만"),
        (False, True, "화질 게이트만"),
        (True, True, "중앙값+화질 게이트 둘 다"),
    ]:
        correct = total = 0
        for vid_id, ((speed, steer, quality), group) in cache.items():
            accel_pred, steer_pred = classify(speed, steer, quality, stopped_thr, accel_eps, steer_thr, use_median, use_gate)
            for row in group.itertuples():
                idx = min(row.sample_index, len(accel_pred) - 1)
                total += 2
                correct += accel_pred[idx] == row.accel_label
                correct += steer_pred[idx] == row.steer_label
        print(f"  {name}: acc={correct/total:.3f} ({correct}/{total})")


if __name__ == "__main__":
    main()
