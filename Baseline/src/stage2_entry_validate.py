"""Stage2 entry_frame 휴리스틱(area_frac>0.03) 검증 - AIHub 실측 ObjectB 궤적 기준.

기존 재정렬기 후보 캐시(stage2_candidates_cache.json, 358영상/16248프레임, B가 라벨된
모든 프레임)를 재사용 - 새로 디텍터를 돌리지 않아도 됨. 영상별로 B가 처음 라벨된
프레임(=실측 진입 프레임)과, 재정렬기가 고른 박스의 area_frac이 0.03을 처음 넘는
프레임(=현재 휴리스틱의 예측)을 비교한다.

주의: 이 캐시는 ObjectB가 "라벨된"(=화면에 나타난) 프레임만 담고 있어서, 그 이전에
아주 작게라도 감지가 가능했는지는 알 수 없음 - "실측 진입 프레임"은 라벨링 기준
최초 가시 프레임이지 물리적 최초 등장 순간은 아닐 수 있음(라벨링 정책에 따라 약간의
편차 가능). 그래도 지금 가진 것 중 가장 신뢰할 수 있는 기준이다.
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from stage2_incident import load_reranker, _rerank_features

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "_aihub_extract" / "stage2_candidates_cache.json"
AREA_THR = 0.03


def _select(candidates, w, h, reranker):
    if not candidates:
        return None
    if reranker is not None:
        net, mean, std = reranker
        feats = torch.tensor([_rerank_features(c, w, h) for c in candidates], dtype=torch.float32)
        with torch.inference_mode():
            scores = net((feats - mean) / std).squeeze(-1)
        return candidates[int(scores.argmax())]
    return max(candidates, key=lambda c: c[4] * (c[2] - c[0]) * (c[3] - c[1]))


def _iou(a, b):
    ax0, ay0, ax1, ay1 = a[:4]
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    area_a = max(0, ax1 - ax0) * max(0, ay1 - ay0)
    area_b = max(0, bx1 - bx0) * max(0, by1 - by0)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def main():
    rows = json.loads(CACHE.read_text(encoding="utf-8"))
    reranker = load_reranker()
    print(f"reranker: {'있음, 사용' if reranker else '없음 - score*area 폴백'}")

    by_video = defaultdict(list)
    for r in rows:
        by_video[r["video"]].append(r)

    lags, never, ious = [], 0, []
    for video, vrows in by_video.items():
        vrows.sort(key=lambda r: r["frame"])
        true_entry = vrows[0]["frame"]
        predicted_entry = None
        for r in vrows:
            sel = _select(r["candidates"], r["w"], r["h"], reranker)
            if sel is None:
                continue
            ious.append(_iou(sel, r["gt"]))
            x0, y0, x1, y1 = sel[:4]
            area_frac = (x1 - x0) * (y1 - y0) / (r["w"] * r["h"])
            if area_frac > AREA_THR:
                predicted_entry = r["frame"]
                break
        if predicted_entry is None:
            never += 1
        else:
            lags.append(predicted_entry - true_entry)

    lags = np.array(lags)
    print(f"\n영상 수: {len(by_video)}, 매칭 정확도(mean IoU, 선택박스 vs GT): {np.mean(ious):.3f}")
    print(f"0.03 문턱을 끝까지 못 넘은 영상: {never}/{len(by_video)} ({100*never/len(by_video):.1f}%)")
    print(f"나머지 {len(lags)}개 영상의 lag(예측진입-실측진입, 프레임 단위):")
    print(f"  median={np.median(lags):.1f}  mean={np.mean(lags):.1f}  "
          f"lag<=0: {(lags<=0).mean()*100:.1f}%  lag<=5: {(lags<=5).mean()*100:.1f}%  "
          f"lag<=15: {(lags<=15).mean()*100:.1f}%  max={lags.max():.0f}")


if __name__ == "__main__":
    main()
