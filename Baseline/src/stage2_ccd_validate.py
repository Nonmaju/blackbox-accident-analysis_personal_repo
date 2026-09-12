"""Stage2 collision_frame 휴리스틱(motion_energy 급증 지점) 검증 - CCD(Crash-1500) 실측.

CCD Crash-1500.txt: 영상당 50프레임(10fps, 5초) 이진 배열(사고 발생 여부) + 자차 관여
여부(Yes/No). DACON 비공개 평가데이터는 "블랙박스 차량이 충돌에 직접 관여"(talkboard
417269)라고 확인했으므로 Yes만 필터링 - 실제 대회 조건과 같은 서브셋으로 검증한다.

정답 프레임 = 이진 배열이 0->1로 바뀌는 첫 인덱스(사고 시작 시점). 우리 collision_frame은
"실제 접촉 시점"이라 사고 시작(위험 개시)보다 조금 늦게 잡힐 수 있음 - 완전히 같은 정의는
아니지만 실측 대량 검증으로는 가장 근접한 자료.
"""
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np

from stage2_incident import find_collision_frame
from video_io import load_frames

ROOT = Path(__file__).resolve().parents[1] / "data" / "stage2" / "videos" / "CCD"


def parse_labels():
    rows = []
    for line in (ROOT / "Crash-1500.txt").read_text().splitlines():
        parts = line.split(",", 1)
        vid = parts[0]
        rest = parts[1]
        # [배열] 뒤에 콤마로 구분된 나머지 필드
        arr_end = rest.rindex("]") + 1
        arr = ast.literal_eval(rest[:arr_end])
        tail = rest[arr_end + 1 :].split(",")
        ego_involved = tail[-1].strip()
        rows.append((vid, arr, ego_involved))
    return rows


def onset_frame(arr):
    for i, v in enumerate(arr):
        if v == 1:
            return i
    return None


def main():
    rows = parse_labels()
    ego_rows = [(vid, arr) for vid, arr, ego in rows if ego == "Yes"]
    print(f"전체 {len(rows)}개 중 자차 관여(Yes) {len(ego_rows)}개로 검증")

    errors = []
    missing = 0
    for vid, arr in ego_rows:
        path = ROOT / "Crash-1500" / f"{vid}.mp4"
        if not path.exists():
            missing += 1
            continue
        true_frame = onset_frame(arr)
        if true_frame is None:
            continue
        frames = load_frames(path)
        pred_frame = find_collision_frame(frames)
        errors.append(abs(pred_frame - true_frame))

    errors = np.array(errors)
    print(f"영상 없음: {missing}개, 검증 완료: {len(errors)}개")
    print(f"MAE: {errors.mean():.2f}프레임 (10fps, 0.3초 tolerance = 3프레임)")
    print(f"median: {np.median(errors):.1f}  within±3(0.3s): {(errors<=3).mean()*100:.1f}%  "
          f"within±1: {(errors<=1).mean()*100:.1f}%  max: {errors.max():.0f}")


if __name__ == "__main__":
    main()
