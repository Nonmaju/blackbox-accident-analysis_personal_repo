"""Stage 1 v1 (얕은 버전): 재녹화(recapture) 판별.

MViT를 라벨 10개로 학습시키면 그대로 외운다 — 대신 포렌식에서 쓰는 재녹화 특징
(모아레/플리커/블록아티팩트/블러/테두리)을 손으로 뽑아 로지스틱 회귀 하나로 분류한다.
라벨이 부족한 문제를 학습 데이터를 늘리는 방향(직접 재녹화를 합성)으로 우회한다.

ponytail: 딥러닝 대신 5개 핸드크래프트 피처 + 로지스틱 회귀. 실제 재녹화 샘플이
쌓이면(AIHub든 직접수집이든) 이 피처들을 입력으로 쓰는 얕은 MLP나 GBM으로 업그레이드.

이 파일은 학습 전용이다 — 실제 제출 zip에 들어가는 inference.py는 노트북 셀 기반으로
따로 빌드되므로, extract_features()는 나중에 그 셀에 그대로 복사해 넣는다(동일 로직 유지 필수).
"""
from __future__ import annotations

import random
from pathlib import Path

import cv2
import numpy as np
import torch

from video_io import load_frames, sample_frames

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FEATURE_DIM = 5


# --------------------------------------------------------------------------- recapture simulation (재녹화 합성)
def _moire_overlay(frame: np.ndarray, rng: random.Random) -> np.ndarray:
    h, w = frame.shape[:2]
    freq = rng.uniform(0.15, 0.4)  # 픽셀 그리드 주기 시뮬레이션
    phase = rng.uniform(0, np.pi)
    yy, xx = np.mgrid[0:h, 0:w]
    grid = 0.5 + 0.5 * np.sin(freq * (xx + yy) + phase)
    strength = rng.uniform(6, 18)
    out = frame.astype(np.float32) + (grid[..., None] - 0.5) * strength
    return np.clip(out, 0, 255).astype(np.uint8)


def _flicker_stack(frames: list[np.ndarray], rng: random.Random) -> list[np.ndarray]:
    # 촬영 화면 주사율과 카메라 프레임레이트가 안 맞을 때 생기는 밝기 요동
    period = rng.uniform(3.5, 9.0)
    amp = rng.uniform(0.06, 0.16)
    return [
        np.clip(f.astype(np.float32) * (1 + amp * np.sin(2 * np.pi * i / period)), 0, 255).astype(np.uint8)
        for i, f in enumerate(frames)
    ]


def _double_compress(frame: np.ndarray, rng: random.Random) -> np.ndarray:
    quality = rng.randint(25, 55)
    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _add_border(frame: np.ndarray, rng: random.Random) -> np.ndarray:
    h, w = frame.shape[:2]
    b = int(min(h, w) * rng.uniform(0.02, 0.06))
    out = frame.copy()
    out[:b] = out[-b:] = out[:, :b] = out[:, -b:] = 0
    return out


def simulate_rerecording(frames: list[np.ndarray], seed: int) -> list[np.ndarray]:
    rng = random.Random(seed)
    frames = _flicker_stack(frames, rng)
    frames = [_moire_overlay(f, rng) for f in frames]
    frames = [_double_compress(f, rng) for f in frames]
    frames = [_add_border(f, rng) for f in frames]
    return frames


def benign_augment(frames: list[np.ndarray], seed: int) -> list[np.ndarray]:
    """재녹화가 아닌 일반적인 변형(밝기/약한 압축) — ORIGINAL 클래스 다양성용."""
    rng = random.Random(seed)
    gain = rng.uniform(0.85, 1.15)
    quality = rng.randint(75, 95)
    out = []
    for f in frames:
        bright = np.clip(f.astype(np.float32) * gain, 0, 255).astype(np.uint8)
        bgr = cv2.cvtColor(bright, cv2.COLOR_RGB2BGR)
        ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
        bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        out.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    return out


# --------------------------------------------------------------------------- handcrafted features
def _fft_high_freq_ratio(gray: np.ndarray) -> float:
    f = np.fft.fftshift(np.fft.fft2(gray.astype(np.float32)))
    mag = np.abs(f)
    h, w = gray.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    radius = min(h, w) * 0.15
    high = mag[r > radius].sum()
    total = mag.sum() + 1e-6
    return float(high / total)


def _blockiness(gray: np.ndarray) -> float:
    # 8x8 JPEG 블록 경계에서의 밝기 불연속량 (이중압축일수록 커짐)
    g = gray.astype(np.float32)
    h, w = g.shape
    h, w = h - h % 8, w - w % 8
    g = g[:h, :w]
    boundary = np.abs(np.diff(g[:, 7:w:8], axis=1)).mean() if w > 8 else 0.0
    boundary += np.abs(np.diff(g[7:h:8, :], axis=0)).mean() if h > 8 else 0.0
    interior = np.abs(np.diff(g, axis=1)).mean() + np.abs(np.diff(g, axis=0)).mean()
    return float(boundary / (interior + 1e-6))


def extract_features(frames: list[np.ndarray]) -> np.ndarray:
    """5개 핸드크래프트 피처. frames는 RGB uint8 리스트(임의 개수)."""
    frames = sample_frames(frames, 8)
    grays = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in frames]
    resized = [cv2.resize(g, (256, 256)) for g in grays]

    fft_ratio = float(np.mean([_fft_high_freq_ratio(g) for g in resized]))
    brightness = np.array([g.mean() for g in grays], dtype=np.float32)
    flicker_std = float(brightness.std() / (brightness.mean() + 1e-6))
    border = float(np.mean([np.concatenate([g[:4].ravel(), g[-4:].ravel(), g[:, :4].ravel(), g[:, -4:].ravel()]).mean() for g in grays]))
    blur = float(np.mean([cv2.Laplacian(g, cv2.CV_64F).var() for g in resized]))
    block = float(np.mean([_blockiness(g) for g in resized]))

    return np.array([fft_ratio, flicker_std, border, blur, block], dtype=np.float32)


# --------------------------------------------------------------------------- dataset + training
def _video_paths() -> tuple[list[Path], list[Path]]:
    """(originals, real_rerecorded) — stage2/3 원본 영상도 ORIGINAL로 재활용."""
    originals = sorted((DATA / "stage1/original").glob("*.mp4"))
    originals += sorted((DATA / "stage2/videos").glob("*.mp4"))
    originals += sorted((DATA / "stage3/videos").glob("*.mp4"))
    rerecorded = sorted((DATA / "stage1/rerecorded").glob("*.mp4"))
    return originals, rerecorded


def build_dataset(variants_per_video: int = 3) -> tuple[np.ndarray, np.ndarray]:
    originals, real_rerecorded = _video_paths()
    X, y = [], []

    for path in originals:
        print(f"  {path}", flush=True)
        frames = sample_frames(load_frames(path), 8)  # 증강 전에 먼저 8프레임으로 줄여서 비용 절감
        X.append(extract_features(frames)); y.append(0)
        for v in range(variants_per_video):
            X.append(extract_features(benign_augment(frames, seed=hash((path.name, v)) & 0xFFFF))); y.append(0)
            X.append(extract_features(simulate_rerecording(frames, seed=hash((path.name, v, "r")) & 0xFFFF))); y.append(1)

    for path in real_rerecorded:
        print(f"  {path}", flush=True)
        X.append(extract_features(load_frames(path))); y.append(1)

    return np.stack(X), np.array(y, dtype=np.float32)


def train_logreg(X: np.ndarray, y: np.ndarray, epochs: int = 300, lr: float = 0.1):
    mean, std = X.mean(0), X.std(0) + 1e-6
    Xn = torch.tensor((X - mean) / std, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.float32)

    weight = torch.zeros(X.shape[1], requires_grad=True)
    bias = torch.zeros(1, requires_grad=True)
    opt = torch.optim.Adam([weight, bias], lr=lr)
    for _ in range(epochs):
        logit = Xn @ weight + bias
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logit, yt)
        opt.zero_grad(); loss.backward(); opt.step()

    with torch.no_grad():
        acc = float(((Xn @ weight + bias > 0).float() == yt).float().mean())
    return {"weight": weight.detach(), "bias": bias.detach(), "feat_mean": mean, "feat_std": std}, acc


def main():
    print("building synthetic + real dataset ...")
    X, y = build_dataset()
    print(f"dataset: {len(y)} samples ({int(y.sum())} rerecorded / {int((1 - y).sum())} original)")

    checkpoint, acc = train_logreg(X, y)
    print(f"train accuracy: {acc:.3f}")
    assert acc > 0.85, "학습 정확도가 너무 낮음 - 피처/증강 로직을 점검할 것"

    out = ROOT / "model" / "stage1"
    out.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, out / "best.pt")
    print(f"saved -> {out / 'best.pt'}")


if __name__ == "__main__":
    main()
