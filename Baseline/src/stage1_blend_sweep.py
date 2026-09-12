"""Stage1 bpp/MViT 블렌드 가중치 스윕 - LOO(leave-one-out) 검증.

현재 프로덕션(0.75 bpp / 0.25 MViT)은 감으로 잡은 값이었음(공격적 시도, 미검증).
실제 제출 2회 연속 0.589로 안정 재현됐지만, 이 가중치 자체가 최적인지는 확인 안 됨.

방법: bpp_thr/scale 산출(calibrate_bpp_threshold)과 MViT 파인튜닝(fit_stage1과 동일한
1 epoch, 9샘플)을 fold마다 다시 계산하는 완전 LOO - bpp 단독 검증(9~10/10) 때와 같은
방법론으로 블렌드 가중치별 정확도를 비교한다.

주의: 10샘플짜리 로컬 LOO가 실제 DACON 채점과 거의 무관하다는 게 이번 시즌 반복
확인된 구조적 한계(Stage3 스무딩 사례, 팀 이슈 #10) - 이 스윕 결과도 "참고용 힌트"
이지 확정적 근거는 아님. 그래도 지금 갖고 있는 유일한 근거이므로 참고해서 다음
제출값을 정한다.
"""
from __future__ import annotations

import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch import nn
from torchvision.models.video import mvit_v2_s

from stage1_bpp import calibrate_bpp_threshold, bpp_prob

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "stage1"
SIZE = 224
MEAN = torch.tensor([0.45, 0.45, 0.45])[:, None, None, None]
STD = torch.tensor([0.225, 0.225, 0.225])[:, None, None, None]
DEVICE = torch.device("cpu")  # 샘플 9개짜리 1 epoch라 CPU로 충분히 빠름


def _load_frames(path):
    cap = cv2.VideoCapture(str(path))
    out = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        out.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    return out


def _crop_tensor(rgb, size=SIZE):
    h, w = rgb.shape[:2]
    scale = size / min(h, w)
    nh, nw = max(size, round(h * scale)), max(size, round(w * scale))
    rgb = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)
    y, x = (nh - size) // 2, (nw - size) // 2
    return torch.from_numpy(rgb[y : y + size, x : x + size].copy()).permute(2, 0, 1).float() / 255


def _clip(path, n=16):
    frames = _load_frames(path)
    idx = np.linspace(0, len(frames) - 1, n).round().astype(int)
    return torch.stack([_crop_tensor(frames[int(i)]) for i in idx], 1)


class Stage1MViT(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = mvit_v2_s(weights=None)
        self.net.head[1] = nn.Linear(self.net.head[1].in_features, 2)

    def forward(self, x):
        return self.net(x)


def _train_mvit(train_rows):
    model = Stage1MViT().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), 1e-4)
    model.train()
    for r in train_rows.sample(frac=1, random_state=20260825).itertuples():
        x = _clip(DATA / r.path)
        x = (x - MEAN) / STD
        y = torch.tensor([0 if r.label == "ORIGINAL" else 1], device=DEVICE)
        loss = nn.functional.cross_entropy(model(x[None].to(DEVICE)), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model


@torch.inference_mode()
def _mvit_prob(model, path):
    x = _clip(path)
    x = ((x - MEAN) / STD)[None]
    logits = model(x)
    return float(torch.softmax(logits, dim=1)[0, 1])  # P(RERECORDED)


def main():
    df = pd.read_csv(DATA / "labels.csv")
    weights = np.round(np.linspace(0.0, 1.0, 11), 2)  # 0.0, 0.1, ..., 1.0
    correct = {w: 0 for w in weights}
    margins = {w: [] for w in weights}  # |prob-0.5| 평균 - 확신도 참고용

    for i, held in enumerate(df.itertuples()):
        train_rows = df.drop(index=held.Index)
        thr, scale = calibrate_bpp_threshold(train_rows, DATA)
        model = _train_mvit(train_rows)

        bp = bpp_prob(DATA / held.path, thr, scale)
        mp = _mvit_prob(model, DATA / held.path)
        y_true = 1 if held.label == "RERECORDED" else 0

        for w in weights:
            blended = w * bp + (1 - w) * mp
            pred = 1 if blended >= 0.5 else 0
            correct[w] += int(pred == y_true)
            margins[w].append(abs(blended - 0.5))
        print(f"[{i+1}/10] {held.ID} label={held.label:11s} bpp_prob={bp:.3f} mvit_prob={mp:.3f}")

    print("\nweight(bpp)  accuracy  avg_margin")
    for w in weights:
        print(f"  {w:.2f}       {correct[w]}/10     {np.mean(margins[w]):.3f}")


if __name__ == "__main__":
    main()
