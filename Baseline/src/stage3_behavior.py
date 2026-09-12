"""Stage 3 v1 (얕은 버전): 0.1초 단위 가감속/조향 범주.

라벨이 6초 간격으로만 있어서(공개 예제) 딥러닝으로 학습할 데이터가 사실상 없다.
대신 이 문제는 라벨이 필요 없다 — optical flow만으로 전방 이동량(가감속)과
좌우 편향(조향)을 직접 잴 수 있다. 순수 휴리스틱 + 공개 라벨 50개로 임계값만 보정.

ponytail: 학습 없는 optical-flow 휴리스틱. AIHub CAN데이터(실측 가감속/조향) 승인되면
그걸로 임계값을 다시 보정하거나, 필요시 이 피처를 입력으로 쓰는 얕은 분류기로 교체.

이 파일도 학습/보정 전용 — inference.py에는 compute_flow_series+classify를 그대로 복사한다.
"""
from __future__ import annotations

import itertools
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

from video_io import load_frames

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
ACCEL = ["ACCELERATING", "DECELERATING", "CONSTANT", "STOPPED"]
STEER = ["LEFT", "STRAIGHT", "RIGHT"]
FLOW_SIZE = (160, 90)  # (w, h) — 다운스케일해서 Farneback 속도 확보


# --------------------------------------------------------------------------- optical flow features
QUALITY_THR = 2.0  # Laplacian variance - 이보다 낮으면 거의 단색/블러라 flow를 못 믿음(팀원 정민 submit-3 참고)


def compute_flow_series(frames: list) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """0.1초(=2프레임) 간격의 (전방속도 proxy, 좌우편향 proxy, 화질 quality) 시계열을 반환."""
    small = [cv2.resize(cv2.cvtColor(f, cv2.COLOR_RGB2GRAY), FLOW_SIZE) for f in frames]
    n = len(small) // 2
    w, h = FLOW_SIZE
    road = slice(int(h * 0.55), h)          # 하단: 도로면, 전진속도에 민감
    horizon = slice(int(h * 0.25), int(h * 0.55))  # 중단: 소실점 부근, 조향에 민감

    speed = np.zeros(n, dtype=np.float32)
    steer = np.zeros(n, dtype=np.float32)
    quality = np.zeros(n, dtype=np.float32)
    for t in range(n):
        i0 = min(2 * t, len(small) - 3)
        i1 = i0 + 2
        flow = cv2.calcOpticalFlowFarneback(small[i0], small[i1], None, 0.5, 2, 15, 3, 5, 1.2, 0)
        mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
        speed[t] = float(np.median(mag[road]))
        steer[t] = float(np.median(flow[horizon, :, 0]))
        quality[t] = float(cv2.Laplacian(small[i1], cv2.CV_32F).var())
    return speed, steer, quality


def _smooth(x: np.ndarray, k: int = 3) -> np.ndarray:
    # 평균(박스) 대신 중앙값 슬라이딩 윈도우 - 이상치 스파이크에 강함(팀원 정민 submit-3
    # 참고, 공개 50라벨 acc 0.550->0.570로 확인된 개선, 물리량은 안 바꿔서 재보정 불필요).
    if len(x) < 2 * k + 1:
        return x
    width = 2 * k + 1
    padded = np.pad(x, (k, k), mode="edge")
    return np.median(np.lib.stride_tricks.sliding_window_view(padded, width), axis=-1)


def classify(speed: np.ndarray, steer: np.ndarray, quality: np.ndarray, stopped_thr: float, accel_eps: float, steer_thr: float):
    # steer 스무딩을 추가했다가(LOVO 0.670->0.710, 공개 50샘플 기준) 실제 제출 점수가
    # 0.521->0.48->0.47로 계속 떨어져서 원복. Macro-F1 공식(0.7*accel+0.3*steer, STOPPED
    # 제외)으로 다시 그리드서치해도 스무딩 여부와 무관하게 같은 임계값이 최적이라, 스무딩
    # 자체의 문제라기보다 "공개 5비디오 로컬 검증이 실제 숨은 평가셋과 거의 무관하다"는
    # 구조적 한계로 보임(팀장님도 반대 방향으로 동일 현상: 로컬 나빠졌는데 실제 Stage2는
    # 올랐음 - 팀 이슈 #10 참고). 실측 검증된 유일한 상태(0.521)로 되돌림.
    speed_s = _smooth(speed)
    n = len(speed_s)
    accel_out, steer_out = [], []
    for t in range(n):
        if speed_s[t] < stopped_thr:
            accel_out.append("STOPPED")
        else:
            lo, hi = max(0, t - 3), min(n, t + 4)
            slope = speed_s[hi - 1] - speed_s[lo] if hi - 1 > lo else 0.0
            if slope > accel_eps:
                accel_out.append("ACCELERATING")
            elif slope < -accel_eps:
                accel_out.append("DECELERATING")
            else:
                accel_out.append("CONSTANT")
        s = steer[t]
        # 부호 주의: 카메라가 좌회전하면 정지된 배경은 화면에서 오른쪽으로 흐른다(flow_x 양수).
        # AIHub 실측 자이로(angZAve) + 실제 프레임 확인(좌회전 차선에서 회전)으로 검증된 부호.
        steer_out.append("LEFT" if s > steer_thr else "RIGHT" if s < -steer_thr else "STRAIGHT")
        # 텍스처 게이트: 거의 단색/블러 프레임은 flow를 못 믿음(팀원 정민 submit-3 참고).
        if quality[t] < QUALITY_THR:
            accel_out[-1], steer_out[-1] = "CONSTANT", "STRAIGHT"
    return accel_out, steer_out


# --------------------------------------------------------------------------- threshold calibration
def calibrate() -> dict:
    labels = pd.read_csv(DATA / "stage3/labels.csv")
    cache = {}
    for vid_id, group in labels.groupby("ID"):
        frames = load_frames(DATA / "stage3/videos" / f"{vid_id}.mp4")
        cache[vid_id] = (compute_flow_series(frames), group)

    best = None
    grid = itertools.product(
        np.linspace(0.1, 1.5, 6),   # stopped_thr - 그리드 넓힌 버전이 LOVO에서도 더 나빴음(과적합), 원복
        np.linspace(0.02, 0.3, 6),  # accel_eps
        np.linspace(0.1, 1.0, 6),   # steer_thr
    )
    for stopped_thr, accel_eps, steer_thr in grid:
        correct, total = 0, 0
        for vid_id, ((speed, steer, quality), group) in cache.items():
            accel_pred, steer_pred = classify(speed, steer, quality, stopped_thr, accel_eps, steer_thr)
            for row in group.itertuples():
                idx = min(row.sample_index, len(accel_pred) - 1)
                total += 2
                correct += accel_pred[idx] == row.accel_label
                correct += steer_pred[idx] == row.steer_label
        acc = correct / total
        if best is None or acc > best[0]:
            best = (acc, stopped_thr, accel_eps, steer_thr)

    acc, stopped_thr, accel_eps, steer_thr = best
    print(f"calibrated acc on 50 sparse labels: {acc:.3f}  "
          f"(stopped_thr={stopped_thr:.3f}, accel_eps={accel_eps:.3f}, steer_thr={steer_thr:.3f})")
    assert acc > 0.5, "휴리스틱이 라벨과 거의 무관 - flow ROI/부호를 재점검할 것"
    return {"stopped_thr": stopped_thr, "accel_eps": accel_eps, "steer_thr": steer_thr}


def main():
    # CAN 후보 테스트 적용 - 공개 50샘플 그리드서치(calibrate()) 대신 AIHub 실측 CAN
    # (LOVO 0.687, stage3_can_candidate.py) 값을 그대로 저장. 이전 확정값(real 0.521)은
    # stage3_can_candidate.CURRENT_PRODUCTION 참고, 되돌리려면 calibrate()를 다시 호출.
    from stage3_can_candidate import CAN_CANDIDATE as params
    out = ROOT / "model" / "stage3"
    out.mkdir(parents=True, exist_ok=True)
    torch.save(params, out / "best.pt")
    print(f"CAN 후보 적용: {params} -> {out / 'best.pt'}")


if __name__ == "__main__":
    main()
