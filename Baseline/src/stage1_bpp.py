"""Stage1 비트레이트(픽셀당 비트수) 기반 보조 신호 - 공격적 시도.

LOO 검증: 원본 최대 bpp=0.091 / 재녹화 최소 bpp=0.107, 깨끗하게 갈라짐(9~10/10).
MViT 파인튜닝(LOO 0/10 - 페어 암기로 역방향 예측)이나 순수 콘텐츠 피처(fft/blur/border
단독, LOO 1~3/10 - 씬마다 절대값 편차가 커서 무의미)보다 훨씬 강한 로컬 신호.

주의: DACON이 공식적으로 "재녹화 클래스 예제는 재녹화 과정을 모사한 파생 예제이며
실제 다른 기기로 재촬영한 데이터가 아니다"라고 명시함 - 즉 이 비트레이트 차이는
DACON의 예제 생성 파이프라인 특성(다른 인코더/코덱 사용)일 수 있고, 실제 평가셋의
진짜 재촬영 영상에서는 안 통할 수 있음. 그래도 로컬 검증상 가장 강한 신호라 사용자
결정으로 공격적으로 채택(비중 0.75) + MViT를 보조(0.25)로 블렌드해 완전히 한쪽에만
걸지는 않음.

해상도/길이가 다른 실제 평가 영상에도 이식되게, 절대 비트레이트가 아니라 "프레임당
픽셀당 비트수(bpp)"로 정규화(해상도·길이 무관 압축난이도 지표).
"""
from pathlib import Path

import cv2


def video_bpp(path) -> float:
    """size(bytes)*8 / (frame_count * width * height) - 프레임당 픽셀당 비트수."""
    cap = cv2.VideoCapture(str(path))
    nframes = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    cap.release()
    if nframes <= 0 or w <= 0 or h <= 0:
        return 0.0
    size = Path(path).stat().st_size
    return size * 8 / (nframes * w * h)


def calibrate_bpp_threshold(labels_df, data_dir):
    """공개 라벨(ORIGINAL/RERECORDED)로 bpp 임계값+스케일을 데이터 기반으로 산출."""
    bpps = {r.ID: video_bpp(Path(data_dir) / r.path) for r in labels_df.itertuples()}
    o_vals = [bpps[r.ID] for r in labels_df.itertuples() if r.label == "ORIGINAL"]
    r_vals = [bpps[r.ID] for r in labels_df.itertuples() if r.label == "RERECORDED"]
    thr = (max(o_vals) + min(r_vals)) / 2
    scale = max((min(r_vals) - max(o_vals)) / 2, 1e-6)
    return thr, scale


def bpp_prob(path, thr: float, scale: float) -> float:
    """bpp를 시그모이드로 [0,1] 확률(재녹화일 확률)로 변환."""
    import math

    bpp = video_bpp(path)
    return 1 / (1 + math.exp(-(bpp - thr) / scale))


if __name__ == "__main__":
    # 블렌드(0.8*bpp + 0.2*MViT) 풀핏 검증 - 실제 노트북에 넣기 전 확인용
    import pandas as pd
    from pathlib import Path as _P

    DATA = _P(__file__).resolve().parents[1] / "data"
    df = pd.read_csv(DATA / "stage1/labels.csv")
    thr, scale = calibrate_bpp_threshold(df, DATA / "stage1")
    print(f"thr={thr:.5f} scale={scale:.5f}")
    for r in df.itertuples():
        p = bpp_prob(DATA / "stage1" / r.path, thr, scale)
        pred = "RERECORDED" if p >= 0.5 else "ORIGINAL"
        print(f"{r.ID} label={r.label:11s} bpp_prob={p:.3f} pred={pred} {'OK' if pred==r.label else 'X'}")
