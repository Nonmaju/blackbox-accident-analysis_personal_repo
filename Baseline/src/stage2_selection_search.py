"""캐시된 후보 박스 위에서 여러 '어떤 박스가 상대차량인가' 규칙을 순간적으로 비교.

stage2_cache_candidates.py를 먼저 돌려야 한다. 탐지 재실행 없이 규칙만 바꿔가며 실험.
"""
import json

from stage2_aihub_eval import iou
from stage2_cache_candidates import OUT


def load_rows():
    return json.loads(OUT.read_text(encoding="utf-8"))


def evaluate(rows, score_fn, min_score=0.5):
    ious = []
    for r in rows:
        cands = [c for c in r["candidates"] if c[4] >= min_score]
        if not cands:
            ious.append(0.0)
            continue
        best = max(cands, key=lambda c: score_fn(c, r["w"], r["h"]))
        ious.append(iou(best[:4], r["gt"]))
    n = len(ious)
    hit = sum(i > 0.5 for i in ious) / n
    mean = sum(ious) / n
    return mean, hit


def area(c):
    return (c[2] - c[0]) * (c[3] - c[1])


def center(c):
    return (c[0] + c[2]) / 2, (c[1] + c[3]) / 2


RULES = {
    "largest_area (기존)": lambda c, w, h: area(c),
    "highest_score": lambda c, w, h: c[4],
    "score*area": lambda c, w, h: c[4] * area(c),
    "closest_to_center": lambda c, w, h: -((center(c)[0] - w / 2) ** 2 + (center(c)[1] - h / 2) ** 2) ** 0.5,
    "score*area/dist_to_center": lambda c, w, h: c[4] * area(c) / (1 + ((center(c)[0] - w / 2) ** 2 + (center(c)[1] - h / 2) ** 2) ** 0.5),
    "closest_to_bottom_center": lambda c, w, h: -((center(c)[0] - w / 2) ** 2 + (c[3] - h) ** 2) ** 0.5,
    "largest_width (가로폭)": lambda c, w, h: c[2] - c[0],
}


def main():
    rows = load_rows()
    print(f"캐시된 (비디오,프레임): {len(rows)}개\n")
    results = []
    for name, fn in RULES.items():
        mean, hit = evaluate(rows, fn)
        results.append((mean, hit, name))
        print(f"{name:35s}  mean IoU {mean:.3f}   IoU>0.5 {hit:.1%}")

    results.sort(reverse=True)
    print(f"\n최고: {results[0][2]}  (mean IoU {results[0][0]:.3f})")


if __name__ == "__main__":
    main()
