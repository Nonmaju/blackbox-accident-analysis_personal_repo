"""Stage3 steer_thr 캘리브레이션 - SullyChen driving_dataset(실측 조향각, degree 단위).

AIHub CAN 데이터의 steering_angle 필드가 전 영상·전 차종에서 100% 0으로 죽어있어서
(placeholder - stage3_can_candidate.py 참고) 각속도(yaw rate)로 대체했었는데, SullyChen
driving_dataset(45,406프레임, 실제 도로 주행, 조향각 실측값 채워짐)으로 진짜 조향각 검증이
가능해짐. optical-flow steer_proxy와의 상관계수 -0.845(AIHub 각속도 -0.519보다 강함).

data/stage3/videos/driving_dataset/ (0.jpg, 1.jpg, ..., data.txt) 필요 - 구글드라이브
https://drive.google.com/file/d/0B-KJCaaF7elleG1RbzVPZWV4Tlk/view 에서 받아 압축 해제.

ponytail: stopped_thr/accel_eps는 이 데이터에 속도 정보가 없어서 손 안 댐 - steer_thr만.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1] / "data" / "stage3" / "videos" / "driving_dataset"
FLOW_SIZE = (160, 90)


def _imread_unicode(path: Path):
    # cv2.imread는 Windows 비ASCII 경로(이 프로젝트 폴더명 자체가 한글)에서 조용히
    # 실패한다 - inference.py의 동일 이름 함수와 같은 우회(np.fromfile+imdecode).
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def load_labels(root: Path = ROOT) -> dict[str, float]:
    labels = {}
    for line in (root / "data.txt").read_text().splitlines():
        name, angle = line.split()
        labels[name] = float(angle)
    return labels


def compute_flow_steer(root: Path, labels: dict[str, float]) -> tuple[np.ndarray, np.ndarray]:
    """연속 프레임 쌍마다 optical-flow steer_proxy 계산, 실측각과 나란히 반환."""
    n_total = len(labels)
    real_angle, flow_steer = [], []
    prev_gray = None
    processed = i = 0
    while processed < n_total and i <= n_total + 100:
        fname = f"{i}.jpg"
        i += 1
        if fname not in labels:
            continue
        img = _imread_unicode(root / fname)
        if img is None:
            continue
        small = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), FLOW_SIZE)
        if prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(prev_gray, small, None, 0.5, 2, 15, 3, 5, 1.2, 0)
            h = FLOW_SIZE[1]
            horizon = slice(int(h * 0.25), int(h * 0.55))
            flow_steer.append(float(np.median(flow[horizon, :, 0])))
            real_angle.append(labels[fname])
        prev_gray = small
        processed += 1
    return np.array(real_angle), np.array(flow_steer)


def calibrate(real_angle: np.ndarray, flow_steer: np.ndarray) -> dict:
    """풀링 분위수(12%/90%)로 실측 LEFT/STRAIGHT/RIGHT 정답을 만들고 steer_thr 그리드서치."""
    left_q, right_q = np.quantile(real_angle, [0.12, 0.90])

    def real_label(a):
        return "A" if a <= left_q else "B" if a >= right_q else "STRAIGHT"

    true_labels = np.array([real_label(a) for a in real_angle])

    best = None
    for steer_thr in np.linspace(0.05, 3.0, 60):
        for sign in (1, -1):
            s = sign * flow_steer
            pred = np.where(s > steer_thr, "POS", np.where(s < -steer_thr, "NEG", "STRAIGHT"))
            for neg_label, pos_label in (("A", "B"), ("B", "A")):
                mapped = np.where(pred == "POS", pos_label, np.where(pred == "NEG", neg_label, "STRAIGHT"))
                acc = float((mapped == true_labels).mean())
                if best is None or acc > best["acc"]:
                    best = {"acc": acc, "steer_thr": float(steer_thr), "sign": sign}
    return best


def main():
    labels = load_labels()
    print(f"labeled frames: {len(labels)}")
    real_angle, flow_steer = compute_flow_steer(ROOT, labels)
    corr = float(np.corrcoef(flow_steer, real_angle)[0, 1])
    print(f"pairs: {len(flow_steer)}  correlation(flow_steer, real_angle): {corr:.3f}")
    result = calibrate(real_angle, flow_steer)
    print(f"best steer_thr={result['steer_thr']:.3f} acc={result['acc']:.3f} sign={result['sign']}")
    assert abs(corr) > 0.5, "상관관계가 너무 약함 - flow ROI/부호 재점검 필요"
    assert result["acc"] > 0.7, "그리드서치 정확도가 너무 낮음 - 분위수/문턱 로직 재점검 필요"


if __name__ == "__main__":
    main()
