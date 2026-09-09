"""AIHub 597(교통사고 영상 데이터) 실라벨로 Stage2 탐지기를 검증.

주의: 이 데이터셋엔 t_collision/entry_frame 같은 '사고 시점' 라벨이 없다(과실비율/객체
식별 위주). sequence_frame_number+bbox는 있지만 A/B가 같은 프레임에 같이 잡히는 경우가
거의 없어서 collision_frame 검증에는 못 쓴다. 대신 이걸로 확실히 검증되는 것:
"우리 탐지기가 피해차량(B) 박스를 실제로 찾아내는가?" (S2_001~005 전부 RIGHT로 나온
문제의 그 detect_vehicles() 함수를 실라벨 IoU로 채점).
"""
from pathlib import Path

import cv2
import numpy as np

from stage2_incident import detect_vehicles, load_detector, load_reranker
from video_io import load_frames

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "data" / "_aihub_extract" / "stage2_sample"


def xywh_to_xyxy(b):
    x, y, w, h = b
    return x, y, x + w, y + h


def iou(box_a, box_b):
    ax0, ay0, ax1, ay1 = box_a
    bx0, by0, bx1, by1 = box_b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    area_a = (ax1 - ax0) * (ay1 - ay0)
    area_b = (bx1 - bx0) * (by1 - by0)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def collect_b_labels():
    """{video_file_name: [(frame_no, bbox_xyxy), ...]} — objectB(피해차량)만."""
    import json

    out = {}
    for jf in SAMPLE.glob("img_labels/*/*.json"):
        d = json.loads(jf.read_text(encoding="utf-8"))
        for obj in d["objects"]:
            if obj.get("isObjectB"):
                out.setdefault(d["video_file_name"], []).append(
                    (d["sequence_frame_number"], xywh_to_xyxy(obj["bbox"]))
                )
    return out


def main():
    labels = collect_b_labels()
    videos = {p.stem: p for p in (SAMPLE / "videos").glob("*.mp4")}
    print(f"라벨 있는 비디오: {len(labels)}개 / 실제 보유 비디오: {len(videos)}개 / 교집합: {len(labels.keys() & videos.keys())}개")

    model, transform, categories = load_detector()
    reranker = load_reranker()
    print(f"reranker: {'있음, 사용' if reranker else '없음 - score*area 폴백'}")

    ious, hits = [], 0
    n_frames = 0
    for name, frame_boxes in labels.items():
        if name not in videos:
            continue
        frames = load_frames(videos[name])
        for frame_no, gt_box in frame_boxes:
            if frame_no >= len(frames):
                continue
            n_frames += 1
            det = detect_vehicles(model, transform, categories, frames[frame_no], reranker=reranker)
            if det is None:
                ious.append(0.0)
                continue
            x0, y0, x1, y1, score = det
            score_iou = iou((x0, y0, x1, y1), gt_box)
            ious.append(score_iou)
            hits += score_iou > 0.5

    ious = np.array(ious)
    print(f"평가한 (비디오,프레임) 쌍: {n_frames}개")
    print(f"mean IoU: {ious.mean():.3f}  median IoU: {np.median(ious):.3f}")
    print(f"IoU>0.5 비율(대충 맞음): {hits}/{n_frames} = {hits/n_frames:.1%}")
    print(f"탐지 자체를 못한 비율(박스 없음): {(ious == 0).mean():.1%}  # 0 IoU엔 '박스는 찾았는데 틀린 차' 도 섞여있음")


if __name__ == "__main__":
    main()
