"""Stage2 entry_frame/entry_side/evasion_space를 처음으로 실측 검증한다.

팀원(정민)이 CCD 실측 영상(우리도 이미 다운로드해 갖고 있는 Crash-1500) 36개에 대해
사람이 직접 확인한 entry_frame/entry_side/evasion_space 정답(HIGH/MEDIUM 신뢰도)을
공유해줬다 - 지금까지 이 세 값은 "공개 라벨이 아예 없어 정량 검증 불가, 눈검증만
가능"이었는데, 처음으로 실측 정답이 생긴 것.

여기서는 우리 현재 프로덕션 코드(stage2_incident.py, Hough 기반)를 이 36개에 그대로
돌려서 정량 점수를 낸다 - 뭘 바꾸기 전에 우리가 지금 어디 서 있는지부터 아는 게 목적
(UFLD/트래킹을 검증 없이 반영했다가 실제 점수 회귀를 겪은 교훈).

라벨 출처: assets/v13_seed/legacy_labels.csv (DONE 상태만, review_status=DONE),
사람이 실제로 눈으로 검수한 36개 - AI 초안(ai_draft.json)은 "GO 근거 아님"이라고
명시돼 있어 여기선 안 씀.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from stage2_incident import find_collision_frame, find_entry_and_scene, load_detector, load_reranker
from video_io import load_frames

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
CCD_ROOT = DATA / "stage2/videos/CCD/Crash-1500"
LABELS_CSV = ROOT / "external" / "jungmin_labels" / "legacy_labels.csv"
FPS = 10.0  # CCD 명시 source_fps


def load_labels():
    rows = list(csv.DictReader(open(LABELS_CSV, encoding="utf-8")))
    return [r for r in rows if r["review_status"] == "DONE" and r["entry_frame"]]


def main():
    labels = load_labels()
    print(f"실측 라벨(사람 검수 DONE): {len(labels)}개", flush=True)

    model, transform, categories = load_detector()
    reranker = load_reranker()
    print(f"reranker: {'있음' if reranker else '없음(score*area 폴백)'}", flush=True)

    entry_correct = collision_correct = side_correct = evasion_correct = 0
    entry_errs, collision_errs = [], []
    n = 0
    for row in labels:
        vid = row["video_id"]
        path = CCD_ROOT / f"{vid}.mp4"
        if not path.exists():
            print(f"  {vid}: 영상 없음, 건너뜀")
            continue
        n += 1
        frames = load_frames(path)
        true_entry = int(row["entry_frame"])
        true_side = row["entry_side"]
        true_evasion = int(row["evasion_space"])

        pred_collision = find_collision_frame(frames)
        pred_entry, pred_side, pred_evasion, _, _ = find_entry_and_scene(
            model, transform, categories, frames, pred_collision, reranker=reranker
        )

        entry_err = abs(pred_entry - true_entry) / FPS  # seconds
        entry_errs.append(entry_err)
        entry_correct += entry_err <= 0.3
        side_correct += pred_side == true_side
        evasion_correct += pred_evasion == true_evasion

        print(
            f"  {vid}: entry pred={pred_entry:2d} true={true_entry:2d} err={entry_err:.1f}s "
            f"{'OK' if entry_err<=0.3 else 'X '} | side pred={pred_side:5s} true={true_side:5s} "
            f"{'OK' if pred_side==true_side else 'X '} | evasion pred={pred_evasion} true={true_evasion} "
            f"{'OK' if pred_evasion==true_evasion else 'X '}",
            flush=True,
        )

    print(f"\n=== 실측 {n}개 결과 (우리 현재 프로덕션 코드, Hough 기반) ===")
    print(f"entry_frame Accuracy@0.3s: {entry_correct}/{n} = {entry_correct/n:.3f}")
    print(f"entry_frame MAE: {np.mean(entry_errs):.2f}초")
    print(f"entry_side accuracy: {side_correct}/{n} = {side_correct/n:.3f}")
    print(f"evasion_space accuracy: {evasion_correct}/{n} = {evasion_correct/n:.3f}")
    print("\n참고 - 정민 development_evaluation.json 기준:")
    print("  PARENT:     entry_acc=0.167 entry_mae=1.90s side_f1=0.692")
    print("  ENTRY_ONLY: entry_acc=0.250 entry_mae=1.01s side_f1=0.692 (같은 36개 재사용 - circular 가능성 명시됨)")


if __name__ == "__main__":
    main()
