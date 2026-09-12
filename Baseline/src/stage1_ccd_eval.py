"""Stage1 재녹화 핸드크래프트 피처(stage1_recapture.py) - CCD 실측 원본으로 다양성 증강 후,
DACON 실제 5쌍을 완전히 held-out으로 검증.

기존 stage1_recapture.build_dataset()은 원본 풀이 stage2/3 공개샘플 10개뿐이라 합성
재녹화 augmentation이 다양성 부족으로 과적합했을 가능성 있음(LOO 1~3/10 실패 원인 후보).
CCD Normal(실제 블랙박스 원본 3000개, 재녹화 아님 확인됨)을 원본 풀에 추가해 재검증.

핵심: DACON 실제 5쌍(ORIGINAL/RERECORDED)은 학습에서 완전히 제외하고 최종 테스트에만 사용
(기존 build_dataset은 이 5쌍도 학습에 섞어써서 LOO 없이는 일반화 확인이 안 됐음).
"""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np

from stage1_recapture import benign_augment, extract_features, simulate_rerecording, train_logreg
from video_io import load_frames, sample_frames

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
CCD_NORMAL = DATA / "stage2/videos/CCD/Normal"


def build_train(n_ccd: int = 200, variants_per_video: int = 3, seed: int = 0):
    """DACON 5쌍은 전혀 안 씀 - stage2/3 공개샘플 10개 + CCD 실측 원본 n_ccd개만 사용."""
    originals = sorted((DATA / "stage2/videos").glob("*.mp4"))
    originals += sorted((DATA / "stage3/videos").glob("*.mp4"))
    rng = random.Random(seed)
    ccd_all = sorted(CCD_NORMAL.glob("*.mp4"))
    originals += rng.sample(ccd_all, min(n_ccd, len(ccd_all)))

    X, y = [], []
    for i, path in enumerate(originals):
        print(f"  [{i+1}/{len(originals)}] {path.name}", flush=True)
        frames = sample_frames(load_frames(path), 8)
        X.append(extract_features(frames)); y.append(0)
        for v in range(variants_per_video):
            X.append(extract_features(benign_augment(frames, seed=hash((path.name, v)) & 0xFFFF))); y.append(0)
            X.append(extract_features(simulate_rerecording(frames, seed=hash((path.name, v, "r")) & 0xFFFF))); y.append(1)
    return np.stack(X), np.array(y, dtype=np.float32)


def eval_on_dacon_pairs(checkpoint):
    import csv
    rows = list(csv.DictReader(open(DATA / "stage1/labels.csv", encoding="utf-8")))
    mean, std, w, b = checkpoint["feat_mean"], checkpoint["feat_std"], checkpoint["weight"], checkpoint["bias"]
    correct = 0
    for r in rows:
        frames = load_frames(DATA / "stage1" / r["path"])
        feat = extract_features(frames)
        logit = float((((feat - mean) / std) @ w.numpy()).item() + b.item())
        pred = "RERECORDED" if logit > 0 else "ORIGINAL"
        ok = pred == r["label"]
        correct += ok
        print(f"  {r['ID']:10s} true={r['label']:11s} pred={pred:11s} {'OK' if ok else 'X'}")
    print(f"held-out DACON 5쌍 accuracy: {correct}/{len(rows)}")
    return correct, len(rows)


def main():
    print("합성 학습셋 구성 중 (stage2/3 샘플 + CCD Normal 200개) ...")
    X, y = build_train()
    print(f"dataset: {len(y)} samples ({int(y.sum())} rerecorded / {int((1 - y).sum())} original)")

    checkpoint, acc = train_logreg(X, y)
    print(f"synthetic train accuracy: {acc:.3f}")

    print("\nDACON 실제 5쌍(held-out)으로 검증:")
    eval_on_dacon_pairs(checkpoint)


if __name__ == "__main__":
    main()
