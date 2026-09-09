"""Stage2 탐지 부실(mean IoU 0.33, 적중률 39%) 원인 진단.

두 가지를 분리해서 본다:
1. Recall: score_thr=0.2로 낮췄을 때 GT 근처에 후보 박스가 아예 하나도 없는 비율
   (없다 = 탐지기 자체가 못 봄, 선택 규칙 문제가 아님)
2. Oracle IoU: 후보들 중 GT와 가장 잘 맞는 것만 골랐을 때의 IoU
   (oracle이 높다 = 후보는 있는데 '가장 큰 박스' 규칙이 엉뚱한 걸 고름 -> 선택 규칙만 고치면 됨)
   (oracle도 낮다 = 탐지기가 애초에 GT랑 안 맞는 박스만 냄 -> 파인튜닝 필요)
"""
import json
from pathlib import Path

import numpy as np

from stage2_aihub_eval import SAMPLE, collect_b_labels, iou
from stage2_incident import detect_all_vehicles, load_detector, track_target_vehicle
from video_io import load_frames


def main():
    labels = collect_b_labels()
    videos = {p.stem: p for p in (SAMPLE / "videos").glob("*.mp4")}
    model, transform, categories = load_detector()

    oracle_ious, current_rule_ious, n_no_candidate = [], [], 0
    for name, frame_boxes in labels.items():
        if name not in videos:
            continue
        frames = load_frames(videos[name])
        for frame_no, gt_box in frame_boxes:
            if frame_no >= len(frames):
                continue
            candidates = detect_all_vehicles(model, transform, categories, frames[frame_no], score_thr=0.2)
            if not candidates:
                n_no_candidate += 1
                oracle_ious.append(0.0)
                current_rule_ious.append(0.0)
                continue
            ious = [iou(c[:4], gt_box) for c in candidates]
            oracle_ious.append(max(ious))
            # 현재 규칙(가장 큰 박스, score>=0.5)과 동일하게 재현
            strict = [c for c in candidates if c[4] >= 0.5]
            if strict:
                largest = max(strict, key=lambda c: (c[2] - c[0]) * (c[3] - c[1]))
                current_rule_ious.append(iou(largest[:4], gt_box))
            else:
                current_rule_ious.append(0.0)

    n = len(oracle_ious)
    oracle_ious, current_rule_ious = np.array(oracle_ious), np.array(current_rule_ious)
    print(f"평가 프레임: {n}개")
    print(f"score_thr=0.2로 낮춰도 후보 박스 자체가 없는 비율: {n_no_candidate/n:.1%}  <- 이게 높으면 탐지기 리콜 문제")
    print(f"현재 규칙(최대박스, thr=0.5) mean IoU: {current_rule_ious.mean():.3f}  IoU>0.5: {(current_rule_ious>0.5).mean():.1%}")
    print(f"oracle(후보 중 최선) mean IoU:        {oracle_ious.mean():.3f}  IoU>0.5: {(oracle_ious>0.5).mean():.1%}")

    # 트래킹 기반 선택 규칙 평가 (비디오 전체를 순서대로 훑어야 해서 별도 루프)
    print("\n--- 트래킹 기반 선택 규칙 평가 ---")
    track_ious = []
    for name, frame_boxes in labels.items():
        if name not in videos:
            continue
        frames = load_frames(videos[name])
        max_frame = max(fn for fn, _ in frame_boxes)
        if max_frame >= len(frames):
            continue
        track = track_target_vehicle(model, transform, categories, frames, 0, max_frame)
        for frame_no, gt_box in frame_boxes:
            box = track.get(frame_no)
            track_ious.append(iou(box[:4], gt_box) if box else 0.0)

    track_ious = np.array(track_ious)
    print(f"평가 프레임: {len(track_ious)}개")
    print(f"트래킹 규칙 mean IoU: {track_ious.mean():.3f}  IoU>0.5: {(track_ious>0.5).mean():.1%}")
    print(f"\n=> oracle이 현재규칙보다 많이 높으면: 선택 규칙만 고치면 됨 (트래킹/휴리스틱)")
    print(f"=> oracle도 낮으면: 탐지기 자체 개선(threshold/모델 교체/파인튜닝) 필요")


if __name__ == "__main__":
    main()
