"""팀원(정민) v10 trajectory_scene(다중객체 그리디 트래킹 + 코리도 진입 판정)을
CCD 실측 36라벨로 직접 검증 - 정민 코드를 그대로 옮겨와서(원본 tools/stage2_track_v10_core.py,
dependency-free 순수 로직) 우리 탐지기/차선추정으로 실행. 정민이 보고한 수치(우리와
같은 36개라 circular)를 우리 환경에서 독립적으로 재현되는지 확인하는 목적.

핵심 차이 - 우리 현재 _crosses_into_lane: "조금이라도 겹치면" 즉시 진입 판정(프레임0
근처에서 과다 오탐, 실측으로 확인: stage2_entry_diagnose2.py). 이 트래킹 로직은
"이전엔 코리도 밖 있었다가(median<10%) 이후 코리도 안(>=10%)으로 전환"하는 순간만
진입으로 인정 - 우리가 진단한 '초반 오탐' 문제를 구조적으로 막는 설계.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from stage2_incident import (
    _default_lane,
    detect_all_vehicles,
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

VARIANTS = ("TRACK_CONTINUITY", "TRACK_CROSSING", "TRACK_ENSEMBLE")


# --------------------------------------------------------------------------- 정민 v10 core 그대로 이식 (dependency-free)
def lane_bounds(lane, y: float) -> Tuple[float, float]:
    values = [float(m) * float(y) + float(b) for m, b in lane]
    return min(values), max(values)


def _geometry(box, width: int, height: int):
    x0, y0, x1, y1, score = map(float, box[:5])
    return ((x0 + x1) / (2 * width), (y0 + y1) / (2 * height),
             max(1.0, x1 - x0) / width, max(1.0, y1 - y0) / height, float(score))


def link_cost(previous, current, width: int, height: int, gap: int = 1) -> float:
    ax, ay, aw, ah, _ = _geometry(previous, width, height)
    bx, by, bw, bh, _ = _geometry(current, width, height)
    distance = np.hypot(ax - bx, ay - by) / max(.035, .5 * (aw + bw), .5 * (ah + bh))
    scale = abs(np.log((bw * bh + 1e-6) / (aw * ah + 1e-6)))
    return float(distance / max(1.0, gap) + .35 * scale + .08 * (gap - 1))


def build_tracks(all_boxes: Dict[int, Sequence[Sequence[float]]], width: int, height: int,
                  max_gap: int = 3, max_cost: float = 2.2) -> List[dict]:
    tracks: List[dict] = []
    for time_index in sorted(all_boxes):
        boxes = sorted((tuple(map(float, box[:5])) for box in all_boxes[time_index]),
                        key=lambda box: (-box[4], box[0], box[1]))[:16]
        candidates = []
        for track_index, track in enumerate(tracks):
            gap = time_index - track["times"][-1]
            if 1 <= gap <= max_gap:
                for box_index, box in enumerate(boxes):
                    cost = link_cost(track["boxes"][-1], box, width, height, gap)
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


def _inside_ratio(box, lane) -> float:
    x0, _, x1, y1 = map(float, box[:4])
    left, right = lane_bounds(lane, y1)
    overlap = max(0.0, min(x1, right) - max(x0, left))
    return overlap / max(1.0, x1 - x0)


def _track_features(track: dict, lane, collision: int, width: int, height: int) -> dict:
    times, boxes = track["times"], track["boxes"]
    usable = [(t, b) for t, b in zip(times, boxes) if t <= collision + 3]
    if not usable or not any(t <= collision for t, _ in usable):
        return {"score": -1e9, "entry": None, "side": "RIGHT", "crossing": False,
                "terminal_distance": 10**9, "length": 0, "continuity": 0.0}
    times = [x[0] for x in usable]; boxes = [x[1] for x in usable]
    inside = [_inside_ratio(box, lane) for box in boxes]
    crossing_index = None
    for index in range(len(times)):
        future = inside[index:min(len(inside), index + 3)]
        previous = inside[max(0, index - 2):index]
        if inside[index] >= .10 and sum(value >= .10 for value in future) >= min(2, len(future)):
            if previous and float(np.median(previous)) < .10:
                crossing_index = index
                break
    entry = times[crossing_index] if crossing_index is not None else None
    side_samples = boxes[max(0, (crossing_index or 0) - 3):(crossing_index or 0) + 1]
    offsets = []
    for box in side_samples:
        x0, _, x1, y1 = box[:4]
        left, right = lane_bounds(lane, y1)
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
    score = (2.0 * _inside_ratio(terminal, lane) + 1.2 * min(1.0, area / .08) + .45 * confidence
             + .65 * continuity + .35 * min(1.0, max(0.0, expansion) / .08)
             + .45 * min(1.0, lateral / .15) - .18 * terminal_distance)
    if crossing_index is not None and entry <= collision:
        score += .9 + .25 * min(1.0, (collision - entry) / max(1, collision))
    return {"score": float(score), "entry": entry, "side": side,
            "crossing": crossing_index is not None, "terminal": terminal,
            "terminal_distance": terminal_distance, "length": len(times), "continuity": float(continuity)}


def trajectory_scene(all_boxes, lane, height: int, width: int, collision: int, variant: str) -> dict:
    tracks = build_tracks(all_boxes, width, height)
    featured = [(track, _track_features(track, lane, collision, width, height)) for track in tracks]
    eligible = [(track, feature) for track, feature in featured
                if feature["terminal_distance"] <= 5 and feature["length"] >= 2]
    crossing = [(track, feature) for track, feature in eligible if feature["crossing"]]
    pool = crossing if variant in {"TRACK_CROSSING", "TRACK_ENSEMBLE"} and crossing else eligible
    if variant == "TRACK_CONTINUITY":
        pool = eligible
    selected = max(pool, key=lambda item: item[1]["score"], default=None)
    if selected is None:
        return {"entry": int(max(0, collision)), "side": "RIGHT", "evasion": 0, "fallback": True}
    track, feature = selected
    if feature["entry"] is None:
        pre_collision = [t for t in track["times"] if t <= collision]
        entry = min(pre_collision) if pre_collision else max(0, collision)
    else:
        entry = int(feature["entry"])
    if variant == "TRACK_ENSEMBLE" and crossing:
        top = sorted(crossing, key=lambda item: item[1]["score"], reverse=True)[:3]
        plausible = [int(item[1]["entry"]) for item in top if item[1]["entry"] is not None]
        if len(plausible) >= 2:
            entry = int(round(float(np.median(plausible))))
            nearest = min(top, key=lambda item: abs(int(item[1]["entry"]) - entry))
            track, feature = nearest
    terminal = feature["terminal"]
    x0, _, x1, y1 = terminal[:4]
    left, right = lane_bounds(lane, y1)
    lane_width = max(1.0, right - left)
    evasion = int(max(max(0.0, x0 - left), max(0.0, right - x1)) >= .32 * lane_width)
    return {"entry": min(int(collision), int(entry)), "side": feature["side"], "evasion": evasion}


# --------------------------------------------------------------------------- 검증
def main():
    rows = [r for r in csv.DictReader(open(LABELS_CSV, encoding="utf-8")) if r["review_status"] == "DONE" and r["entry_frame"]]
    model, transform, categories = load_detector()

    cache = []
    for row in rows:
        vid = row["video_id"]
        path = CCD_ROOT / f"{vid}.mp4"
        if not path.exists():
            continue
        frames = load_frames(path)
        h, w = frames[0].shape[:2]
        collision = find_collision_frame(frames)
        high = min(len(frames) - 1, collision + 3)
        sampled = sorted(set(np.rint(np.linspace(0, high, min(96, high + 1))).astype(int).tolist()))
        all_boxes = {}
        for t in sampled:
            boxes = detect_all_vehicles(model, transform, categories, frames[t], score_thr=0.2)
            if boxes:
                all_boxes[t] = boxes
        lane = estimate_ego_lane(frames, sampled, h, w) if sampled else _default_lane(h, w)
        cache.append((vid, row, collision, h, w, all_boxes, lane))
        print(f"  cached {vid}", flush=True)

    for variant in VARIANTS:
        correct = side_correct = evasion_correct = 0
        errs = []
        for vid, row, collision, h, w, all_boxes, lane in cache:
            true_entry, true_side, true_evasion = int(row["entry_frame"]), row["entry_side"], int(row["evasion_space"])
            result = trajectory_scene(all_boxes, lane, h, w, collision, variant)
            err = abs(result["entry"] - true_entry) / FPS
            errs.append(err)
            correct += err <= 0.3
            side_correct += result["side"] == true_side
            evasion_correct += result["evasion"] == true_evasion
        n = len(cache)
        print(f"{variant:18s} entry_acc@0.3s={correct}/{n}={correct/n:.3f}  MAE={np.mean(errs):.2f}s  "
              f"side_acc={side_correct/n:.3f}  evasion_acc={evasion_correct/n:.3f}")


if __name__ == "__main__":
    main()
