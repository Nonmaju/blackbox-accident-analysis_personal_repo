"""stage2_diagnose.py의 트래킹 기반 선택 검증 - 전체 554영상 x 0..max_frame을 매 프레임
탐지하면 몇 시간이 걸려서(영상당 평균 150프레임, 전부 FasterRCNN 통과) 대표성 있는
부분집합(라벨 프레임 주변 ±15프레임)만 빠르게 확인. 같은 부분집합에서 현재규칙/
재정렬기/트래킹을 나란히 비교해 공정하게 비교한다."""
from __future__ import annotations

import random

import numpy as np
import torch
from torch import nn

from stage2_aihub_eval import SAMPLE, collect_b_labels, iou
from stage2_incident import (
    detect_all_vehicles,
    detect_vehicles,
    load_detector,
    load_reranker,
    track_target_vehicle,
)
from video_io import load_frames


def main():
    labels = collect_b_labels()
    videos = {p.stem: p for p in (SAMPLE / "videos").glob("*.mp4")}
    model, transform, categories = load_detector()
    reranker = load_reranker()

    names = sorted(n for n in labels if n in videos)
    rng = random.Random(0)
    rng.shuffle(names)
    names = names[:20]  # 대표 부분집합(속도 확보 - 진행상황 출력하며 실행)

    largest_ious, rerank_ious, track_ious = [], [], []
    n_frames = 0
    for vi, name in enumerate(names):
        print(f"[{vi+1}/{len(names)}] {name}", flush=True)
        frame_boxes = labels[name]
        frames = load_frames(videos[name])
        frame_boxes = [(fn, gt) for fn, gt in frame_boxes if fn < len(frames)]
        if not frame_boxes:
            continue

        # 현재규칙(largest) + 재정렬기: 라벨 프레임만 독립적으로
        for frame_no, gt_box in frame_boxes:
            n_frames += 1
            candidates = detect_all_vehicles(model, transform, categories, frames[frame_no], score_thr=0.2)
            if not candidates:
                largest_ious.append(0.0)
                rerank_ious.append(0.0)
                continue
            strict = [c for c in candidates if c[4] >= 0.5]
            largest = max(strict, key=lambda c: (c[2] - c[0]) * (c[3] - c[1])) if strict else max(candidates, key=lambda c: c[4])
            largest_ious.append(iou(largest[:4], gt_box))

            picked = detect_vehicles(model, transform, categories, frames[frame_no], reranker=reranker)
            rerank_ious.append(iou(picked[:4], gt_box) if picked else 0.0)

        # 트래킹: 라벨 프레임 주변 ±15프레임 구간만 순서대로 훑음(대표성 유지, 속도 확보)
        lo = max(0, min(fn for fn, _ in frame_boxes) - 10)
        hi = min(len(frames) - 1, max(fn for fn, _ in frame_boxes) + 10)
        track = track_target_vehicle(model, transform, categories, frames, lo, hi)
        for frame_no, gt_box in frame_boxes:
            box = track.get(frame_no)
            track_ious.append(iou(box[:4], gt_box) if box else 0.0)

    largest_ious, rerank_ious, track_ious = np.array(largest_ious), np.array(rerank_ious), np.array(track_ious)
    print(f"videos={len(names)} frames={n_frames}")
    print(f"현재규칙(largest)     mean IoU={largest_ious.mean():.3f}  IoU>0.5={(largest_ious>0.5).mean():.1%}")
    print(f"재정렬기(reranker.pt) mean IoU={rerank_ious.mean():.3f}  IoU>0.5={(rerank_ious>0.5).mean():.1%}")
    print(f"트래킹(track_target_vehicle) mean IoU={track_ious.mean():.3f}  IoU>0.5={(track_ious>0.5).mean():.1%}")


if __name__ == "__main__":
    main()
