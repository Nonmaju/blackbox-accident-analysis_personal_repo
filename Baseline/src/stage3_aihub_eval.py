"""AIHub 71851(사고위험 환경 운전습관 데이터) 실측 CAN으로 Stage3 optical-flow 휴리스틱 검증.

주의 두 가지를 확인했다:
  - CAN의 steering_angle/yawrate 필드는 이 데이터셋에서 전부 0 (여러 파일 확인) - 이 블랙박스
    유닛은 OEM 조향각 CAN을 못 읽는 걸로 보임. 대신 자이로 각속도(aim_gsensor.angZAve, Z축
    회전율)를 조향 실측치로 쓴다 - 부호는 모르니 상관계수로 어느 쪽이든 맞는지만 확인.
  - 영상이 1920x1080/30fps/2분(=3600프레임)이라 다 메모리에 올리면 20GB+ 걸린다. 그래서
    stage3_behavior.compute_flow_series(전체 프레임 리스트를 받음)를 그대로 못 쓰고,
    스트리밍으로 프레임 하나씩만 들고 처리하는 버전을 따로 둔다.
"""
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "data" / "_aihub_extract" / "stage3_sample"
FLOW_SIZE = (160, 90)


def stream_flow_series(video_path: Path):
    """영상을 프레임 하나씩만 들고 순서대로 훑으며 (speed_proxy, steer_proxy)를 프레임마다 계산."""
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


def pearson(a, b):
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    if a.std() == 0 or b.std() == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def main():
    video_dir = SAMPLE / "videos"
    can_dir = SAMPLE / "can"
    videos = {p.stem.replace("NIA_CM_SOU", "NIA_CAN_SOU"): p for p in video_dir.glob("*.MP4")}

    speed_corrs, steer_corrs = [], []
    for can_key, video_path in videos.items():
        can_path = can_dir / f"{can_key}.JSON"
        if not can_path.exists():
            print(f"CAN 라벨 없음, 스킵: {video_path.name}")
            continue
        d = json.loads(can_path.read_text(encoding="utf-8"))
        real_speed = np.array([f["aim_micom"]["gps_vss"] for f in d["frames"]], dtype=np.float32)
        real_yawrate = np.array([f["aim_gsensor"]["angZAve"] for f in d["frames"]], dtype=np.float32)

        print(f"{video_path.name}: optical flow 계산 중 (스트리밍, 프레임 하나씩)...", flush=True)
        flow_speed, flow_steer = stream_flow_series(video_path)

        # 실측 speed(레벨)와 flow speed(순간 이동량)는 같은 물리량이어야 함 - 상관계수로 확인
        sc = pearson(real_speed[1:], flow_speed)  # flow는 frame i->i+1이라 한 칸 밀림
        tc = pearson(real_yawrate[1:], flow_steer)
        speed_corrs.append(sc)
        steer_corrs.append(tc)
        print(f"  speed corr(real gps_vss, flow speed_proxy) = {sc:.3f}")
        print(f"  steer corr(real gyro angZAve, flow steer_proxy) = {tc:.3f}")

    print(f"\n=== 평균 (영상 {len(speed_corrs)}개) ===")
    print(f"speed correlation 평균: {np.mean(speed_corrs):.3f}")
    print(f"steer correlation 평균: {np.mean(steer_corrs):.3f}  (부호는 임의 - abs로도 참고: {np.mean(np.abs(steer_corrs)):.3f})")
    print("\n0.5 이상이면 optical flow가 실제 물리량을 꽤 잘 따라간다는 뜻, 0 근처면 지금 로직이 안 맞는 것")


if __name__ == "__main__":
    main()
