"""Stage2 탐지 결과를 눈으로 확인하기 위한 스크립트.

entry_frame/collision_frame 시점의 프레임에 탐지 박스를 그려서 PNG로 저장한다.
entry_side가 전부 RIGHT로 나온 게 실제인지 탐지 버그인지 이걸로 확인한다.
"""
from pathlib import Path

import cv2
import pandas as pd

from stage2_incident import DATA, ROOT, find_collision_frame, find_entry_and_scene, load_detector, load_reranker
from video_io import load_frames


def draw_box(frame, box, label, color=(255, 0, 0)):
    img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR).copy()
    if box is not None:
        x0, y0, x1, y1, *_ = box
        cv2.rectangle(img, (int(x0), int(y0)), (int(x1), int(y1)), color, 3)
    cv2.putText(img, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
    return img


def imwrite_unicode(path: Path, img):
    # cv2.imwrite는 Windows에서 비ASCII 경로를 조용히 무시한다 — imencode + 파일쓰기로 우회
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise RuntimeError(f"encode failed: {path}")
    path.write_bytes(buf.tobytes())


def main():
    model, transform, categories = load_detector()
    reranker = load_reranker()
    out_dir = ROOT / "output" / "stage2_viz"
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = pd.read_csv(DATA / "stage2/labels.csv")
    saved = []
    for row in labels.itertuples():
        frames = load_frames(DATA / "stage2" / row.path)
        collision = find_collision_frame(frames)
        entry_frame, entry_side, evasion, box_frame, box = find_entry_and_scene(
            model, transform, categories, frames, collision, reranker=reranker
        )

        from stage2_incident import detect_vehicles
        entry_det = detect_vehicles(model, transform, categories, frames[entry_frame], reranker=reranker)

        p1 = out_dir / f"{row.ID}_entry_{entry_side}.png"
        imwrite_unicode(p1, draw_box(frames[entry_frame], entry_det, f"entry frame={entry_frame} side={entry_side}"))
        # evasion_space 판단에 실제로 쓰인 박스를 그린다(충돌 프레임에서 탐지 실패 시 가장 가까운 프레임으로 대체됨)
        box_label = f"collision frame={collision} evasion={evasion}"
        if box_frame is not None and box_frame != collision:
            box_label += f" (box from frame={box_frame}, no detection at collision)"
        p2 = out_dir / f"{row.ID}_collision_evasion{evasion}.png"
        imwrite_unicode(p2, draw_box(frames[box_frame if box_frame is not None else collision], box, box_label, (0, 0, 255)))
        saved += [p1, p2]
        print(f"{row.ID}: saved {p1.name}, {p2.name}", flush=True)

    print(f"\n{len(saved)}장 저장 완료 -> {out_dir}")


if __name__ == "__main__":
    main()
