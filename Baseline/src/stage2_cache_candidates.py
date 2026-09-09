"""라벨된 (비디오,프레임)마다 후보 박스 전체 + GT를 한 번만 계산해서 캐시.

선택 규칙을 여러 개 실험할 때마다 탐지기를 다시 돌리면 느리다 - 여기서 한 번 뽑아
JSON으로 저장해두면 stage2_selection_search.py에서 순간적으로 이것저것 시도 가능.
"""
import json
from pathlib import Path

from stage2_aihub_eval import SAMPLE, collect_b_labels
from stage2_incident import detect_all_vehicles, load_detector
from video_io import load_frames

OUT = Path(__file__).resolve().parents[1] / "data" / "_aihub_extract" / "stage2_candidates_cache.json"


def main():
    labels = collect_b_labels()
    videos = {p.stem: p for p in (SAMPLE / "videos").glob("*.mp4")}
    model, transform, categories = load_detector()

    rows = []
    for name, frame_boxes in labels.items():
        if name not in videos:
            continue
        frames = load_frames(videos[name])
        h, w = frames[0].shape[:2]
        for frame_no, gt_box in frame_boxes:
            if frame_no >= len(frames):
                continue
            candidates = detect_all_vehicles(model, transform, categories, frames[frame_no], score_thr=0.2)
            rows.append({"video": name, "frame": frame_no, "w": w, "h": h, "gt": gt_box, "candidates": candidates})
        print(f"{name}: {len(frame_boxes)}프레임 처리", flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rows), encoding="utf-8")
    print(f"\n캐시 저장: {len(rows)}개 (비디오,프레임) 쌍 -> {OUT}")


if __name__ == "__main__":
    main()
