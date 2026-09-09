"""휴리스틱 재정렬로 안 되니(score*area가 최선, mean IoU 0.35) 학습형 재정렬기 시도.

COCO 탐지기(고정)가 뱉는 후보 박스들 중 '어떤 게 사고 상대차량(B)인가'를 고르는 얕은
분류기를 실라벨로 학습한다. 탐지기 자체를 파인튜닝하는 게 아니라 이미 나온 후보들
사이에서 고르는 재정렬기라 훨씬 가볍다 - oracle(후보 중 최선)이 0.51/63.4%였으니
목표는 거기 근접하는 것.

비디오 단위로 train/val을 나눠서 같은 영상이 양쪽에 섞이지 않게 한다(리키지 방지).
"""
import json
import random

import numpy as np
import torch
from torch import nn

from stage2_aihub_eval import iou
from stage2_cache_candidates import OUT

FEATURES = ["cx", "cy", "bw", "bh", "score", "aspect"]


def featurize(c, w, h):
    x0, y0, x1, y1, score = c
    cx, cy = (x0 + x1) / 2 / w, (y0 + y1) / 2 / h
    bw, bh = (x1 - x0) / w, (y1 - y0) / h
    aspect = bw / (bh + 1e-6)
    return [cx, cy, bw, bh, score, aspect]


def build_examples(rows):
    X, y, groups = [], [], []  # groups = row index, 같은 프레임의 후보들을 나중에 묶어서 평가
    for gi, r in enumerate(rows):
        for c in r["candidates"]:
            X.append(featurize(c, r["w"], r["h"]))
            y.append(1.0 if iou(c[:4], r["gt"]) > 0.5 else 0.0)
            groups.append(gi)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32), groups


def train_mlp(X, y, epochs=400, lr=0.01):
    mean, std = X.mean(0), X.std(0) + 1e-6
    Xn = torch.tensor((X - mean) / std)
    yt = torch.tensor(y)
    model = nn.Sequential(nn.Linear(X.shape[1], 16), nn.ReLU(), nn.Linear(16, 1))
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    pos_weight = torch.tensor([(y == 0).sum() / max(1, (y == 1).sum())])  # 양성이 훨씬 적음
    for _ in range(epochs):
        logit = model(Xn).squeeze(-1)
        loss = nn.functional.binary_cross_entropy_with_logits(logit, yt, pos_weight=pos_weight)
        opt.zero_grad(); loss.backward(); opt.step()
    return model, mean, std


def evaluate(rows, score_fn):
    ious = []
    for r in rows:
        cands = [c for c in r["candidates"] if c[4] >= 0.2]
        if not cands:
            ious.append(0.0)
            continue
        best = max(cands, key=lambda c: score_fn(c, r["w"], r["h"]))
        ious.append(iou(best[:4], r["gt"]))
    n = len(ious)
    return sum(ious) / n, sum(i > 0.5 for i in ious) / n


def main():
    rows = json.loads(OUT.read_text(encoding="utf-8"))
    videos = sorted({r["video"] for r in rows})
    random.Random(20260909).shuffle(videos)
    n_val = max(1, len(videos) // 4)
    val_videos, train_videos = set(videos[:n_val]), set(videos[n_val:])
    train_rows = [r for r in rows if r["video"] in train_videos]
    val_rows = [r for r in rows if r["video"] in val_videos]
    print(f"train {len(train_videos)}개 영상/{len(train_rows)}프레임, val {len(val_videos)}개 영상/{len(val_rows)}프레임")

    X, y, _ = build_examples(train_rows)
    model, mean, std = train_mlp(X, y)

    def learned_score(c, w, h):
        feat = torch.tensor([featurize(c, w, h)], dtype=torch.float32)
        feat = (feat - mean) / std
        with torch.no_grad():
            return float(model(feat))

    baseline_mean, baseline_hit = evaluate(val_rows, lambda c, w, h: c[4] * (c[2] - c[0]) * (c[3] - c[1]))
    learned_mean, learned_hit = evaluate(val_rows, learned_score)
    print(f"\n[val, {len(val_videos)}개 영상 held-out]")
    print(f"score*area (기존): mean IoU {baseline_mean:.3f}  IoU>0.5 {baseline_hit:.1%}")
    print(f"학습 재정렬기:      mean IoU {learned_mean:.3f}  IoU>0.5 {learned_hit:.1%}")

    if learned_mean > baseline_mean:
        out = OUT.parents[2] / "model" / "stage2" / "reranker.pt"
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": model.state_dict(), "mean": mean, "std": std, "features": FEATURES}, out)
        print(f"\n개선 확인됨 -> 저장: {out}")
    else:
        print("\n개선 안 됨 - 저장 안 함")


if __name__ == "__main__":
    main()
