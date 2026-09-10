"""Stage2 V2: 속도예측 기반 트래킹(간이 SORT) + 자차 진행경로 진입 정의.

기존(stage2_incident.find_entry_and_scene)의 문제:
  - "박스 첫 등장 = entry"는 화면 어디든 큰 차만 나오면 진입으로 침 (entry_side 실측 IoU 0.33 수준)
  - 프레임간 IoU 그리디 트래킹은 개선 없었음(첫 프레임에서 틀리면 계속 틀림, 탐지 실패시
    바로 트랙을 버리고 재초기화 - 실제로는 몇 프레임 정도는 '속도로 밀고가며 버텨야' 함)

여기서는:
  1. 속도예측 트래킹: 위치를 이전 속도로 예측해두고, 예측 위치 근처 후보만 매칭(게이팅).
     탐지가 안 잡혀도 max_coast 프레임까지는 예측값으로 트랙을 유지한다(진짜 놓치면 폐기).
  2. entry = 트랙 박스의 바닥-중심이 '자차 진행경로'(사다리꼴, 소실점으로 좁아지는 영역)에
     처음 들어온 프레임. entry_side는 그 직전(진행경로 밖에 있던 마지막) 프레임의 좌우 위치.

레인 검출 모델 없이 고정 사다리꼴로 근사한다 - README의 lane/drivable-area 모델은
데이터도 학습도 필요해서, 우선 이 근사가 실제로 오라클 IoU를 갉아먹지 않는지부터 본다.
"""
from __future__ import annotations

from pathlib import Path

from stage2_incident import detect_all_vehicles

ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------- 속도예측 트래킹
class _Track:
    def __init__(self, box):
        x0, y0, x1, y1, score = box
        self.cx, self.cy = (x0 + x1) / 2, (y0 + y1) / 2
        self.w, self.h = x1 - x0, y1 - y0
        self.vx, self.vy = 0.0, 0.0
        self.score = score
        self.misses = 0

    def predict(self):
        self.cx += self.vx
        self.cy += self.vy

    def update(self, box):
        x0, y0, x1, y1, score = box
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        if self.misses > 0:
            self.vx, self.vy = (cx - self.cx) / (self.misses + 1), (cy - self.cy) / (self.misses + 1)
        else:
            self.vx, self.vy = 0.6 * self.vx + 0.4 * (cx - self.cx), 0.6 * self.vy + 0.4 * (cy - self.cy)
        self.cx, self.cy, self.w, self.h = cx, cy, 0.7 * self.w + 0.3 * (x1 - x0), 0.7 * self.h + 0.3 * (y1 - y0)
        self.score, self.misses = score, 0

    def box(self):
        return (self.cx - self.w / 2, self.cy - self.h / 2, self.cx + self.w / 2, self.cy + self.h / 2, self.score)


def track_sequence(model, transform, categories, frames, start, end, score_thr=0.3, gate_frac=0.12, max_coast=5):
    """start..end를 훑으며 속도예측+게이팅으로 한 대를 계속 추적. 반환: {frame_idx: box5 or None}."""
    h, w = frames[0].shape[:2]
    gate = gate_frac * (w + h) / 2
    track, results = None, {}

    for t in range(start, end + 1):
        candidates = detect_all_vehicles(model, transform, categories, frames[t], score_thr=score_thr)
        if track is not None:
            track.predict()

        matched = None
        if track is not None and candidates:
            def dist(c):
                cx, cy = (c[0] + c[2]) / 2, (c[1] + c[3]) / 2
                return ((cx - track.cx) ** 2 + (cy - track.cy) ** 2) ** 0.5
            near = [c for c in candidates if dist(c) < gate]
            if near:
                matched = min(near, key=dist)

        if track is None and candidates:
            track = _Track(max(candidates, key=lambda c: c[4] * (c[2] - c[0]) * (c[3] - c[1])))
        elif matched is not None:
            track.update(matched)
        elif track is not None:
            track.misses += 1
            if track.misses > max_coast:
                track = None

        results[t] = track.box() if track is not None else None
    return results


# --------------------------------------------------------------------------- 자차 진행경로(고정 사다리꼴)
def ego_path_trapezoid(w, h):
    """소실점(화면 중앙 상단쪽)으로 좁아지는 사다리꼴. (하단 절반폭, 상단 절반폭, 상단y)."""
    return {"bottom_half_w": 0.30 * w, "top_half_w": 0.06 * w, "top_y": 0.35 * h, "cx": 0.5 * w}


def in_ego_path(bottom_center_x, bottom_center_y, w, h):
    tz = ego_path_trapezoid(w, h)
    if bottom_center_y < tz["top_y"] or bottom_center_y > h:
        return False
    frac = (h - bottom_center_y) / (h - tz["top_y"])  # 0(하단)~1(소실점)
    half_w = tz["bottom_half_w"] + frac * (tz["top_half_w"] - tz["bottom_half_w"])
    return abs(bottom_center_x - tz["cx"]) <= half_w


def find_entry_v2(model, transform, categories, frames, collision_frame):
    """트랙 진행경로 진입 프레임/방향. 반환: (entry_frame, entry_side, track_dict)."""
    h, w = frames[0].shape[:2]
    lo, hi = max(0, collision_frame - 30), min(len(frames) - 1, collision_frame + 5)
    track = track_sequence(model, transform, categories, frames, lo, hi)

    entry_frame, entry_side, prev_side = None, None, None
    for t in range(lo, hi + 1):
        box = track.get(t)
        if box is None:
            continue
        x0, y0, x1, y1, score = box
        bcx, bcy = (x0 + x1) / 2, y1
        if in_ego_path(bcx, bcy, w, h):
            entry_frame = t
            entry_side = prev_side or ("LEFT" if bcx < w / 2 else "RIGHT")
            break
        prev_side = "LEFT" if bcx < w / 2 else "RIGHT"

    if entry_frame is None:
        entry_frame, entry_side = collision_frame, "RIGHT"
    return entry_frame, entry_side, track
