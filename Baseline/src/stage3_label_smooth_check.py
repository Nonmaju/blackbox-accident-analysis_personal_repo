"""Stage3 가감속/조향 라벨(문자열)에 다수결(mode) 스무딩을 적용해보는 실험 - 부정 결과.

한 프레임짜리 튐(노이즈)을 없애면 좋을 것 같아서 시도. 공개 50라벨(기존 CAN 보정
임계값 그대로) 기준 정확도가 0.570 -> 0.560으로 오히려 나빠져서 미적용 - 실제
운전 상태가 프레임 단위로 자주 바뀌는 경우가 (다수결이 지우는) 노이즈보다 많다는 뜻일
수 있음. stage3_teammate_check.py와 마찬가지로 "시도했지만 효과 없었음"을 남겨두는
용도(재시도 방지).
"""
from __future__ import annotations

from collections import Counter

import pandas as pd
import torch

from stage3_behavior import DATA, ROOT, classify, compute_flow_series
from video_io import load_frames


def mode_smooth(labels: list, k: int = 2) -> list:
    n = len(labels)
    out = []
    for t in range(n):
        lo, hi = max(0, t - k), min(n, t + k + 1)
        out.append(Counter(labels[lo:hi]).most_common(1)[0][0])
    return out


def main():
    checkpoint = torch.load(ROOT / "model/stage3/best.pt", map_location="cpu", weights_only=False)
    stopped_thr, accel_eps, steer_thr = checkpoint["stopped_thr"], checkpoint["accel_eps"], checkpoint["steer_thr"]

    labels = pd.read_csv(DATA / "stage3/labels.csv")
    correct_base = correct_smooth = total = 0
    for vid_id, group in labels.groupby("ID"):
        frames = load_frames(DATA / "stage3/videos" / f"{vid_id}.mp4")
        speed, steer, quality = compute_flow_series(frames)
        accel_pred, steer_pred = classify(speed, steer, quality, stopped_thr, accel_eps, steer_thr)
        accel_sm, steer_sm = mode_smooth(accel_pred), mode_smooth(steer_pred)
        for row in group.itertuples():
            idx = min(row.sample_index, len(accel_pred) - 1)
            total += 2
            correct_base += accel_pred[idx] == row.accel_label
            correct_base += steer_pred[idx] == row.steer_label
            correct_smooth += accel_sm[idx] == row.accel_label
            correct_smooth += steer_sm[idx] == row.steer_label

    print(f"base(현재)              acc={correct_base/total:.3f}")
    print(f"label-mode-smooth(k=2) acc={correct_smooth/total:.3f}  <- 더 나쁨, 미채택")


if __name__ == "__main__":
    main()
