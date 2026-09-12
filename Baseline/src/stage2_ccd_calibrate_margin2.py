"""collision_frame margin 2차 재보정 - 시작부만 고정 프레임수로 제외, 끝은 제외 안 함.

1차 재보정(margin_frac 그리드)에서 margin=0이 CCD(801개)에선 가장 좋았지만, DACON 공개
5샘플(S2_002, 자차와 무관한 배경 사고 촬영 영상)에서 프레임 4에 스퓨리어스 피크(초반
카메라 흔들림 등)를 잡아 MAE가 2.40->6.40으로 나빠짐. CCD는 50프레임 고정 클립이라
margin_frac(비율)이 이 특정 영상 길이에서만 맞고, 더 긴 실제 영상엔 그대로 비례하지
않을 수 있음 - "시작부 고정 프레임 수"로 다시 접근(끝은 제외 안 함, CCD가 증명한 대로
사고가 항상 뒷부분이라)."""
from __future__ import annotations

import numpy as np

from stage2_ccd_validate import ROOT, parse_labels, onset_frame
from stage2_incident import motion_energy
from video_io import load_frames


def main():
    rows = parse_labels()
    ego_rows = [(vid, arr) for vid, arr, ego in rows if ego == "Yes"]

    energies, true_frames = {}, {}
    for vid, arr in ego_rows:
        path = ROOT / "Crash-1500" / f"{vid}.mp4"
        if not path.exists():
            continue
        t = onset_frame(arr)
        if t is None:
            continue
        frames = load_frames(path)
        energies[vid] = motion_energy(frames)
        true_frames[vid] = t

    print(f"videos used: {len(energies)}")
    for start_exclude in [0, 1, 2, 3, 5, 8]:
        errors = []
        for vid, energy in energies.items():
            window = energy[start_exclude:] if start_exclude < len(energy) else energy
            pred = int(np.argmax(window)) + start_exclude
            errors.append(abs(pred - true_frames[vid]))
        errors = np.array(errors)
        print(f"start_exclude={start_exclude}  MAE={errors.mean():.2f}  within±3={100*(errors<=3).mean():.1f}%")


if __name__ == "__main__":
    main()
