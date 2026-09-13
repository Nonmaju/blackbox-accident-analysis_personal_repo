"""entry_frame이 왜 자주 0(폴백)으로 나오는지 원인 분리 - 탐지 자체가 없는 건지,
탐지는 있는데 차선crossing이 안 걸리는 건지."""
from __future__ import annotations

import csv
from pathlib import Path

from stage2_incident import (
    _crosses_into_lane,
    _default_lane,
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


def main():
    rows = [r for r in csv.DictReader(open(LABELS_CSV, encoding="utf-8")) if r["review_status"] == "DONE" and r["entry_frame"]]
    model, transform, categories = load_detector()
    reranker = load_reranker()

    no_detection = no_crossing = has_crossing = 0
    for row in rows:
        vid = row["video_id"]
        path = CCD_ROOT / f"{vid}.mp4"
        if not path.exists():
            continue
        frames = load_frames(path)
        collision = find_collision_frame(frames)
        h, w = frames[0].shape[:2]
        window_lo, window_hi = max(0, collision - 90), min(len(frames) - 1, collision + 5)

        detections = {}
        for t in range(window_lo, window_hi + 1):
            det = detect_vehicles(model, transform, categories, frames[t], reranker=reranker)
            if det is not None:
                detections[t] = det

        if not detections:
            no_detection += 1
            print(f"  {vid}: 탐지 0건 (전체 {window_hi-window_lo+1}프레임 중)")
            continue

        lane = estimate_ego_lane(frames, detections.keys(), h, w)
        crossed = any(_crosses_into_lane(detections[t], lane) for t in detections if t <= collision)
        if crossed:
            has_crossing += 1
        else:
            no_crossing += 1
            print(f"  {vid}: 탐지 {len(detections)}건 있지만 차선crossing 0건 (window {window_lo}-{window_hi}, collision={collision})")

    n = len(rows)
    print(f"\n총 {n}개 중:")
    print(f"  탐지 자체가 0건: {no_detection} ({no_detection/n:.1%})")
    print(f"  탐지는 있지만 차선crossing 없음: {no_crossing} ({no_crossing/n:.1%})")
    print(f"  crossing 찾음(0이 아닌 실제값 나옴): {has_crossing} ({has_crossing/n:.1%})")


if __name__ == "__main__":
    main()
