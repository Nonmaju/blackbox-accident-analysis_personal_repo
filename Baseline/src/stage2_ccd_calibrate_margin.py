"""collision_frame의 margin(시작/끝 제외 구간) 재보정 - CCD 801개 자차관여 영상 기준.

stage2_ccd_validate.py로 검증한 결과 MAE 7.53프레임(0.3초 tolerance 내 49.1%)로 예상보다
안 좋았음 - 원인 분석: CCD는 "사고 임박" 클립이라 실제 충돌이 항상 뒷부분(30~49/50프레임,
최솟값이 30!)에서 발생하는데, find_collision_frame의 margin=10%가 끝에서 5프레임을
통째로 검색 범위 밖으로 제외해서 45~49프레임에서 발생하는 진짜 충돌(전체의 상당수)을
원천적으로 못 찾음. DACON 실제 영상도 "사고 영상"이라 비슷한 구조일 가능성이 높음.
"""
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np

from stage2_incident import motion_energy
from stage2_ccd_validate import ROOT, parse_labels, onset_frame
from video_io import load_frames


def find_collision_frame_v2(frames: list, margin_frac: float) -> int:
    energy = motion_energy(frames)
    margin = max(1, int(len(energy) * margin_frac))
    if margin * 2 >= len(energy):
        return int(np.argmax(energy))
    window = energy[margin:-margin] if margin > 0 else energy
    return int(np.argmax(window)) + margin


def main():
    rows = parse_labels()
    ego_rows = [(vid, arr) for vid, arr, ego in rows if ego == "Yes"]

    energies = {}
    true_frames = {}
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

    best = None
    for margin_frac in [0.0, 0.02, 0.05, 0.1, 0.15, 0.2]:
        errors = []
        for vid, energy in energies.items():
            margin = max(1, int(len(energy) * margin_frac))
            if margin * 2 >= len(energy):
                pred = int(np.argmax(energy))
            else:
                window = energy[margin:-margin] if margin > 0 else energy
                pred = int(np.argmax(window)) + margin
            errors.append(abs(pred - true_frames[vid]))
        errors = np.array(errors)
        mae = errors.mean()
        within3 = (errors <= 3).mean() * 100
        print(f"margin_frac={margin_frac:.2f}  MAE={mae:.2f}  within±3={within3:.1f}%")
        if best is None or within3 > best[1]:
            best = (margin_frac, within3, mae)

    print(f"\nbest margin_frac={best[0]} within±3={best[1]:.1f}% MAE={best[2]:.2f}")


if __name__ == "__main__":
    main()
