"""트래킹(V2)이 기존(재정렬기, score*area) 대비 실제로 나은지 AIHub 실라벨로 비교.

각 라벨 프레임을 '충돌 시점'이라 치고 실제 파이프라인과 동일한 36프레임 윈도우
(frame_no-30 ~ frame_no+5)로 추적한 뒤, frame_no에서의 트랙 박스를 GT와 비교한다.
"""
import json
import random
import time

import numpy as np
import torch

from stage2_aihub_eval import SAMPLE, collect_b_labels, iou
from stage2_incident import detect_vehicles, load_detector, load_reranker
from stage2_tracker import track_sequence
from video_io import load_frames

N_SAMPLE = 50  # 재정렬기 검증 때도 held-out 50개로 신뢰할 만한 결과 나왔음 - 그거에 맞춤


def main():
    labels = collect_b_labels()
    videos = {p.stem: p for p in (SAMPLE / "videos").glob("*.mp4")}
    matched = sorted(labels.keys() & videos.keys())
    if len(matched) > N_SAMPLE:
        matched = random.Random(20260909).sample(matched, N_SAMPLE)
    print(f"평가 대상 비디오: {len(matched)}개(샘플링), torch 스레드: {torch.get_num_threads()}")

    model, transform, categories = load_detector()
    reranker = load_reranker()

    tracker_ious, current_ious = [], []
    t0 = time.time()
    for vi, name in enumerate(matched, 1):
        frames = load_frames(videos[name])
        print(f"[{vi}/{len(matched)}] {name} ({len(frames)}프레임, {time.time()-t0:.0f}s 경과)", flush=True)
        for frame_no, gt_box in labels[name]:
            if frame_no >= len(frames):
                continue
            lo, hi = max(0, frame_no - 30), min(len(frames) - 1, frame_no + 5)

            track = track_sequence(model, transform, categories, frames, lo, hi)
            tbox = track.get(frame_no)
            tracker_ious.append(iou(tbox[:4], gt_box) if tbox else 0.0)

            cbox = detect_vehicles(model, transform, categories, frames[frame_no], reranker=reranker)
            current_ious.append(iou(cbox[:4], gt_box) if cbox else 0.0)

    tracker_ious, current_ious = np.array(tracker_ious), np.array(current_ious)
    n = len(tracker_ious)
    print(f"\n평가 프레임: {n}개")
    print(f"현재(재정렬기/score*area) mean IoU: {current_ious.mean():.3f}  IoU>0.5: {(current_ious>0.5).mean():.1%}")
    print(f"트래킹(V2)              mean IoU: {tracker_ious.mean():.3f}  IoU>0.5: {(tracker_ious>0.5).mean():.1%}")


if __name__ == "__main__":
    main()
