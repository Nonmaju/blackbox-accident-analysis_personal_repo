"""entry_frame이 프레임 0 근처에서 너무 자주(허위) crossing으로 잡히는 문제 진단 결과
(stage2_entry_diagnose2.py: 탐지 0건은 0%, crossing 자체는 88.9%에서 찾음 - 즉 '못 찾아서'가
아니라 '너무 이른 프레임에서 과도하게 걸려서' 문제)에 대한 첫 번째 저위험 수정 후보:
_crosses_into_lane의 "조금이라도 겹치면 인정"을 "차량 폭의 N% 이상 겹쳐야 인정"으로
강화. CCD 실측 36라벨로 바로 국소 재검증(DACON 재제출 없이) - UFLD 때 교훈 반영.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from stage2_incident import (
    _default_lane,
    _lane_x_at,
    detect_vehicles,
    estimate_ego_lane,
    find_collision_frame,
    load_detector,
    load_reranker,
)
from video_io import load_frames

ROOT = Path(__file__).resolve().parents[1]
CCD_ROOT = ROOT / "data/stage2/videos/CCD/Crash-1500"
LABELS_CSV = ROOT / "external/jungmin_labels/legacy_labels.csv"
FPS = 10.0


def crosses_frac(box, lane, min_overlap_frac: float) -> bool:
    x0, _, x1, y1 = box[:4]
    lx, rx = _lane_x_at(lane[0], y1), _lane_x_at(lane[1], y1)
    if lx > rx:
        lx, rx = rx, lx
    inter = max(0.0, min(x1, rx) - max(x0, lx))
    box_w = max(x1 - x0, 1e-6)
    return inter / box_w >= min_overlap_frac


def main():
    rows = [r for r in csv.DictReader(open(LABELS_CSV, encoding="utf-8")) if r["review_status"] == "DONE" and r["entry_frame"]]
    model, transform, categories = load_detector()
    reranker = load_reranker()

    # 탐지/차선/충돌은 min_overlap_frac과 무관하게 동일 -> 한 번만 계산해서 캐시
    cache = []
    for row in rows:
        vid = row["video_id"]
        path = CCD_ROOT / f"{vid}.mp4"
        if not path.exists():
            continue
        frames = load_frames(path)
        h, w = frames[0].shape[:2]
        collision = find_collision_frame(frames)
        window_lo, window_hi = max(0, collision - 90), min(len(frames) - 1, collision + 5)
        detections = {}
        for t in range(window_lo, window_hi + 1):
            det = detect_vehicles(model, transform, categories, frames[t], reranker=reranker)
            if det is not None:
                detections[t] = det
        lane = estimate_ego_lane(frames, detections.keys(), h, w) if detections else _default_lane(h, w)
        cache.append((vid, row, collision, window_lo, detections, lane))
        print(f"  cached {vid}", flush=True)

    for min_overlap_frac in [0.0, 0.15, 0.3, 0.5]:
        correct = 0
        errs = []
        for vid, row, collision, window_lo, detections, lane in cache:
            true_entry = int(row["entry_frame"])
            entry_idx = None
            for t in sorted(t for t in detections if t <= collision):
                if crosses_frac(detections[t], lane, min_overlap_frac):
                    entry_idx = t
                    break
            if entry_idx is None:
                entry_idx = window_lo
            err = abs(entry_idx - true_entry) / FPS
            errs.append(err)
            correct += err <= 0.3
        n = len(cache)
        print(f"min_overlap_frac={min_overlap_frac:.2f}  Accuracy@0.3s={correct}/{n}={correct/n:.3f}  MAE={np.mean(errs):.2f}s")


if __name__ == "__main__":
    main()
