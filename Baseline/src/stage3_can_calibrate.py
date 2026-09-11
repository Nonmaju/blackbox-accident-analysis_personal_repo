"""AIHub 실측 CAN(10개 영상)으로 Stage3 임계값을 재보정 - 공개 50샘플보다 훨씬 큰 표본.

접근:
  1. 실측 gps_vss(속도)/angZAve(자이로 z축 각속도)로 accel/steer '실측 기반' 라벨을 만든다.
     절대 단위(km/h, deg/s)를 확신할 수 없어서, 공개 50라벨의 클래스 비율(대략
     accel: CONST 60%/ACC 18%/DEC 16%/STOP 6%, steer: STRAIGHT 78%/LEFT 12%/RIGHT 10%)에
     맞춘 분위수(quantile) 임계값을 쓴다 - 절대 단위 추측 대신 상대적 분포로 라벨링.
  2. 우리 optical flow 프록시(stream_flow_series)를 같은 영상에서 계산.
  3. 영상 단위 leave-one-video-out으로 flow 임계값(stopped_thr/accel_eps/steer_thr)을
     그리드서치 - 표본이 수천 프레임이라 그리드를 넓혀도 과적합 위험이 훨씬 낮다.

주의: 실측 CAN 기반 '라벨'도 우리가 정한 분위수 규칙으로 만든 것이라 DACON의 진짜
숨은 기준과 정확히 같으리라는 보장은 없다(팀 문서의 A2 가설과 동일한 위험). 그래도
공개 50개보다는 훨씬 크고 실측 물리량 기반이라, 이 결과가 기존 50샘플 보정과
방향이 일치하는지가 핵심 확인 포인트.
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "data" / "_aihub_extract" / "stage3_sample"
CACHE = ROOT / "data" / "_aihub_extract" / "stage3_can_cache.npz"
FLOW_SIZE = (160, 90)


def stream_flow_series(video_path):
    w, h = FLOW_SIZE
    road = slice(int(h * 0.55), h)
    horizon = slice(int(h * 0.25), int(h * 0.55))
    cap = cv2.VideoCapture(str(video_path))
    speed, steer = [], []
    ok, prev = cap.read()
    if not ok:
        cap.release()
        return np.array([]), np.array([])
    prev = cv2.resize(cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY), FLOW_SIZE)
    while True:
        ok, cur = cap.read()
        if not ok:
            break
        cur = cv2.resize(cv2.cvtColor(cur, cv2.COLOR_BGR2GRAY), FLOW_SIZE)
        flow = cv2.calcOpticalFlowFarneback(prev, cur, None, 0.5, 2, 15, 3, 5, 1.2, 0)
        mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
        speed.append(float(np.median(mag[road])))
        steer.append(float(np.median(flow[horizon, :, 0])))
        prev = cur
    cap.release()
    return np.array(speed, dtype=np.float32), np.array(steer, dtype=np.float32)


def _smooth(x, k=3):
    if len(x) < 2 * k + 1:
        return x
    return np.convolve(x, np.ones(2 * k + 1) / (2 * k + 1), mode="same")


def classify(speed, steer, stopped_thr, accel_eps, steer_thr):
    speed_s = _smooth(speed)
    steer_s = _smooth(steer)
    n = len(speed_s)
    accel_out, steer_out = [], []
    for t in range(n):
        if speed_s[t] < stopped_thr:
            accel_out.append("STOPPED")
        else:
            lo, hi = max(0, t - 3), min(n, t + 4)
            slope = speed_s[hi - 1] - speed_s[lo] if hi - 1 > lo else 0.0
            accel_out.append("ACCELERATING" if slope > accel_eps else "DECELERATING" if slope < -accel_eps else "CONSTANT")
        s = steer_s[t]
        steer_out.append("LEFT" if s > steer_thr else "RIGHT" if s < -steer_thr else "STRAIGHT")
    return accel_out, steer_out


def build_can_labels(real_speed, real_yaw):
    """분위수 기반 accel/steer '실측 유래' 라벨. 공개 50샘플 클래스 비율에 근사."""
    speed_s = _smooth(real_speed, k=5)
    stopped_q = np.quantile(speed_s, 0.06)  # 공개: STOPPED ~6%
    slope = np.array([speed_s[min(i + 3, len(speed_s) - 1)] - speed_s[max(i - 3, 0)] for i in range(len(speed_s))])
    acc_hi = np.quantile(slope, 0.82)   # 상위 18% = ACCELERATING
    acc_lo = np.quantile(slope, 0.16)   # 하위 16% = DECELERATING

    accel_labels = []
    for i in range(len(speed_s)):
        if speed_s[i] <= stopped_q:
            accel_labels.append("STOPPED")
        elif slope[i] >= acc_hi:
            accel_labels.append("ACCELERATING")
        elif slope[i] <= acc_lo:
            accel_labels.append("DECELERATING")
        else:
            accel_labels.append("CONSTANT")

    yaw_s = np.clip(real_yaw, -200, 200)  # 명백한 센서 글리치(수천대) 제거
    yaw_s = _smooth(yaw_s, k=5)
    left_q = np.quantile(yaw_s, 0.88)   # 공개: LEFT ~12%
    right_q = np.quantile(yaw_s, 0.10)  # 공개: RIGHT ~10%
    steer_labels = ["LEFT" if v >= left_q else "RIGHT" if v <= right_q else "STRAIGHT" for v in yaw_s]
    return accel_labels, steer_labels


def main():
    video_dir = SAMPLE / "videos"
    can_dir = SAMPLE / "can"
    videos = {p.stem.replace("NIA_CM_SOU", "NIA_CAN_SOU"): p for p in video_dir.glob("*.MP4")}

    if CACHE.exists():
        print(f"캐시 사용: {CACHE}")
        data = np.load(CACHE, allow_pickle=True)
        cache = data["cache"].item()
    else:
        cache = {}
        for can_key, video_path in videos.items():
            can_path = can_dir / f"{can_key}.JSON"
            if not can_path.exists():
                continue
            d = json.loads(can_path.read_text(encoding="utf-8"))
            real_speed = np.array([f["aim_micom"]["gps_vss"] for f in d["frames"]], dtype=np.float32)
            real_yaw = np.array([f["aim_gsensor"]["angZAve"] for f in d["frames"]], dtype=np.float32)
            print(f"{video_path.name}: flow 계산 중...", flush=True)
            flow_speed, flow_steer = stream_flow_series(video_path)
            n = min(len(real_speed) - 1, len(flow_speed))
            cache[can_key] = {
                "real_speed": real_speed[1:1+n], "real_yaw": real_yaw[1:1+n],
                "flow_speed": flow_speed[:n], "flow_steer": flow_steer[:n],
            }
        np.savez(CACHE, cache=cache)
        print(f"캐시 저장: {CACHE}")

    # 영상별 실측 라벨 생성
    per_video = {}
    for k, v in cache.items():
        accel_labels, steer_labels = build_can_labels(v["real_speed"], v["real_yaw"])
        per_video[k] = {"flow_speed": v["flow_speed"], "flow_steer": v["flow_steer"],
                         "accel": accel_labels, "steer": steer_labels}

    all_accel = sum((p["accel"] for p in per_video.values()), [])
    all_steer = sum((p["steer"] for p in per_video.values()), [])
    print(f"\n총 프레임: {len(all_accel)}개 (영상 {len(per_video)}개)")
    for lbl in ["STOPPED", "CONSTANT", "ACCELERATING", "DECELERATING"]:
        print(f"  accel={lbl}: {all_accel.count(lbl)}개 ({all_accel.count(lbl)/len(all_accel):.1%})")
    for lbl in ["STRAIGHT", "LEFT", "RIGHT"]:
        print(f"  steer={lbl}: {all_steer.count(lbl)}개 ({all_steer.count(lbl)/len(all_steer):.1%})")

    # leave-one-video-out 그리드서치
    grid = list(itertools.product(
        np.linspace(0.1, 2.0, 8), np.linspace(0.01, 0.5, 8), np.linspace(0.05, 1.5, 8),
    ))

    def eval_acc(video_keys, stopped_thr, accel_eps, steer_thr):
        correct = total = 0
        for k in video_keys:
            p = per_video[k]
            accel_pred, steer_pred = classify(p["flow_speed"], p["flow_steer"], stopped_thr, accel_eps, steer_thr)
            for gt_a, gt_s, pa, ps in zip(p["accel"], p["steer"], accel_pred, steer_pred):
                total += 2
                correct += (gt_a == pa) + (gt_s == ps)
        return correct / total if total else 0.0

    keys = list(per_video.keys())
    held_out_accs = []
    for held_out in keys:
        train_keys = [k for k in keys if k != held_out]
        best = max(grid, key=lambda p: eval_acc(train_keys, *p))
        acc = eval_acc([held_out], *best)
        held_out_accs.append(acc)
        print(f"[LOVO held_out={held_out}] best={best} held_out_acc={acc:.3f}")
    print(f"\nLOVO 평균 정확도(실측 CAN 유래 라벨 기준): {np.mean(held_out_accs):.3f}")

    best_full = max(grid, key=lambda p: eval_acc(keys, *p))
    print(f"전체 fit 최적 임계값: stopped_thr={best_full[0]:.3f} accel_eps={best_full[1]:.3f} steer_thr={best_full[2]:.3f}")
    print(f"전체 fit 정확도: {eval_acc(keys, *best_full):.3f}")


if __name__ == "__main__":
    main()
