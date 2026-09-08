"""공용 비디오 프레임 로딩 (학습 스크립트 전용 — 제출 inference.py에는 각자 인라인한다)."""
from pathlib import Path

import cv2
import numpy as np


def load_frames(path: Path) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise ValueError(f"cannot decode: {path}")
    return frames


def sample_frames(frames: list[np.ndarray], n: int = 8) -> list[np.ndarray]:
    idx = np.linspace(0, len(frames) - 1, min(n, len(frames))).round().astype(int)
    return [frames[i] for i in idx]
