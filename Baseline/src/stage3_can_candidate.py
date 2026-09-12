"""Stage3 캘리브레이션 이력. 최신 = SullyChen 실측 조향각(steer_thr) + AIHub CAN(stopped/accel).

## 1세대: AIHub 실측 CAN(10영상/35982프레임)

stage3_can_calibrate.py의 build_can_labels()가 두 가지로 망가져 있었음:
  1. 부호 반대(양수=LEFT로 잘못 가정 - 실제는 음수=LEFT, stage3_aihub_eval의 상관계수로 검증됨)
  2. 자이로(angZAve) 원시값에 물리적으로 불가능한 글리치(최대 53467, 정상은 수십)가 섞여있는데
     clip()으로 경계에 뭉쳐서 분위수 계산이 왜곡됨 - clip 대신 완전히 제외(전방채움)로 수정.

수정 후 LOVO(leave-one-video-out) 평균 정확도 0.687 (10개 영상 35982프레임 기준). **실제
제출로 검증 완료(2026-09-12): real 0.521 -> 0.53113로 개선.** stopped_thr=0.1/accel_eps=0.01은
이 세대 값 그대로 유지 - AIHub 데이터엔 속도 신호가 있어서 여기서 계속 씀.

## 2세대: SullyChen 실측 조향각(steer_thr만 교체, 2026-09-12)

steer_thr는 DACON이 실제로 "조향각(steering angle)" 신호를 쓴다고 확인(talkboard) - 그런데
AIHub CAN 데이터는 조향각 필드(steering_angle)가 있는데도 전 영상·전 차종에서 100% 0으로
죽어있어서(placeholder) 실제로는 각속도(yaw rate, angZAve)로 대체 계산한 값이었음. SullyChen
driving_dataset(45,406프레임, 실측 조향각 degree 단위, 라이선스: 연구용)으로 다시 검증:

  - optical-flow steer_proxy vs 실측 조향각 상관계수: **-0.845** (AIHub 각속도 대비 -0.519보다
    훨씬 강함 - 각속도보다 조향각이 화면 흐름과 물리적으로 더 직접 연관되는 것으로 보임)
  - 풀링 분위수(12%/90%)로 LEFT/STRAIGHT/RIGHT 정답 생성(클래스 비율 12.2%/77.8%/10.0% -
    AIHub 작업 때와 거의 동일한 비율로 재현됨)
  - steer_thr 그리드서치: 기존 0.464는 정확도 0.865, **0.650이 0.886으로 더 나음** → 교체.

stopped_thr/accel_eps는 SullyChen에 속도 정보가 없어서 그대로 둠. **아직 실제 제출로 검증
안 됨 - 다음 제출 대상.**
"""

CAN_CANDIDATE = {
    "stopped_thr": 0.1,
    "accel_eps": 0.01,
    "steer_thr": 0.65,  # SullyChen 45406프레임 검증(정확도 0.886, 이전 0.464는 0.865)
}

PREV_CAN_CANDIDATE = {  # real 0.53113로 검증된 1세대 값(steer_thr만 이전 세대) - 원복용
    "stopped_thr": 0.1,
    "accel_eps": 0.01,
    "steer_thr": 0.464,
}

CURRENT_PRODUCTION = CAN_CANDIDATE  # fit_stage3()이 저장하는 값과 동일(원복용 참고 별칭)

OLD_50SAMPLE_BASELINE = {  # 공개 50샘플 튜닝, real 0.521로 검증됐던 가장 오래된 확정값
    "stopped_thr": 0.1,
    "accel_eps": 0.3,
    "steer_thr": 0.28,
}

# 검증 수치 요약:
#   AIHub CAN 기준(35982프레임, LOVO):        0.687        -> real 0.53113 (steer_thr=0.464 포함)
#   SullyChen 조향각 기준(45406프레임, 정확도): steer_thr=0.464: 0.865 / steer_thr=0.65: 0.886
#   공개 50샘플 기준(가장 오래된 확정값):       0.710        -> real 0.521
