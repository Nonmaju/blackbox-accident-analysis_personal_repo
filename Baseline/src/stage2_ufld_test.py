"""Ultra-Fast-Lane-Detection(TuSimple, ResNet18, MIT license) 도입 가능성 검증.

우리 Hough 기반 _detect_lane_boundaries가 DACON 공개 5샘플 전부 실패하는 문제를
사전학습된 딥러닝 차선검출로 대체할 수 있는지 확인. 반환 형식(m,b 직선 두 개)은
기존 stage2_incident.py의 _fit_lane_side/_estimate_ego_lane과 그대로 호환되게 맞춤 -
성공하면 _detect_lane_boundaries 자리만 이걸로 바꾸면 나머지 파이프라인은 안 건드림.

가중치 출처: https://github.com/cfzd/Ultra-Fast-Lane-Detection (MIT) TuSimple res18,
external/ufld/tusimple_18.pth (245MB, gdown으로 받음).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision.models import resnet18
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
CKPT = ROOT / "external/ufld/tusimple_18.pth"
DATA = ROOT / "data"

ROW_ANCHOR = [64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124, 128, 132,
              136, 140, 144, 148, 152, 156, 160, 164, 168, 172, 176, 180, 184, 188, 192, 196,
              200, 204, 208, 212, 216, 220, 224, 228, 232, 236, 240, 244, 248, 252, 256, 260,
              264, 268, 272, 276, 280, 284]
GRIDING_NUM = 100
CLS_NUM_PER_LANE = 56
NUM_LANES = 4

_TRANSFORM = transforms.Compose([
    transforms.Resize((288, 800)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])


class _UFLDNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = resnet18(weights=None)
        self.pool = nn.Conv2d(512, 8, 1)
        self.cls = nn.Sequential(nn.Linear(1800, 2048), nn.ReLU(), nn.Linear(2048, 22624))

    def forward(self, x):
        m = self.model
        x = m.conv1(x); x = m.bn1(x); x = m.relu(x); x = m.maxpool(x)
        x = m.layer1(x); x = m.layer2(x); x = m.layer3(x); x = m.layer4(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.cls(x)
        return x.view(-1, GRIDING_NUM + 1, CLS_NUM_PER_LANE, NUM_LANES)


def load_ufld():
    net = _UFLDNet()
    state = torch.load(CKPT, map_location="cpu", weights_only=False)["model"]
    net.load_state_dict(state, strict=False)
    net.eval()
    return net


def _fit_lane_side(points):
    if len(points) < 2:
        return None
    ys = np.array([p[1] for p in points], dtype=np.float64)
    xs = np.array([p[0] for p in points], dtype=np.float64)
    if ys.std() < 1e-3:
        return None
    m, b = np.polyfit(ys, xs, 1)
    return float(m), float(b)


@torch.inference_mode()
def detect_lane_boundaries_ufld(frame: np.ndarray, net) -> tuple | None:
    """frame: RGB uint8 HWC. 반환: ((m,b) left, (m,b) right) 또는 실패시 None."""
    h, w = frame.shape[:2]
    x = _TRANSFORM(Image.fromarray(frame))
    out = net(x[None])[0].numpy()  # (101, 56, 4)
    out = out[:, ::-1, :]
    exp = np.exp(out[:-1] - out[:-1].max(axis=0, keepdims=True))
    prob = exp / exp.sum(axis=0, keepdims=True)
    idx = (np.arange(GRIDING_NUM) + 1).reshape(-1, 1, 1)
    loc = (prob * idx).sum(axis=0)  # (56, 4)
    argmax_full = out.argmax(axis=0)
    loc[argmax_full == GRIDING_NUM] = 0
    col_sample_w = 799.0 / (GRIDING_NUM - 1)

    lanes = []
    for lane_i in range(NUM_LANES):
        if np.sum(loc[:, lane_i] != 0) <= 2:
            continue
        pts = []
        for k in range(CLS_NUM_PER_LANE):
            if loc[k, lane_i] > 0:
                x_px = loc[k, lane_i] * col_sample_w * w / 800 - 1
                y_px = h * (ROW_ANCHOR[CLS_NUM_PER_LANE - 1 - k] / 288) - 1
                pts.append((x_px, y_px))
        if len(pts) >= 2:
            lanes.append(pts)
    if len(lanes) < 2:
        return None

    centers = [float(np.mean([p[0] for p in pts])) for pts in lanes]
    order = np.argsort(centers)
    lanes_sorted = [lanes[i] for i in order]
    centers_sorted = [centers[i] for i in order]

    left_i = None
    for i in range(len(centers_sorted) - 1):
        if centers_sorted[i] <= w / 2 <= centers_sorted[i + 1]:
            left_i = i
            break
    if left_i is None:
        dists = [abs(c - w / 2) for c in centers_sorted]
        a, b = sorted(np.argsort(dists)[:2].tolist())
        left_pts, right_pts = lanes_sorted[a], lanes_sorted[b]
    else:
        left_pts, right_pts = lanes_sorted[left_i], lanes_sorted[left_i + 1]

    left, right = _fit_lane_side(left_pts), _fit_lane_side(right_pts)
    if left is None or right is None:
        return None
    return left, right


def main():
    from video_io import load_frames

    net = load_ufld()
    labels_csv = DATA / "stage2/labels.csv"
    import csv
    ids = [r["ID"] for r in csv.DictReader(open(labels_csv, encoding="utf-8"))] if labels_csv.exists() else None

    videos = sorted((DATA / "stage2/videos").glob("*.mp4"))
    for path in videos:
        frames = load_frames(path)
        h, w = frames[0].shape[:2]
        n_frames_with_lane = 0
        for t in range(0, len(frames), max(1, len(frames) // 10)):
            result = detect_lane_boundaries_ufld(frames[t], net)
            if result is not None:
                n_frames_with_lane += 1
        print(f"{path.name}: 10프레임 샘플 중 차선 검출 성공 {n_frames_with_lane}/10")


if __name__ == "__main__":
    main()
