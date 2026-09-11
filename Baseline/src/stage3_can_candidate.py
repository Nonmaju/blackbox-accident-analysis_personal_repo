"""Stage3 대안 후보: AIHub 실측 CAN(10영상/35982프레임) 기준 캘리브레이션.

stage3_can_calibrate.py의 build_can_labels()가 두 가지로 망가져 있었음:
  1. 부호 반대(양수=LEFT로 잘못 가정 - 실제는 음수=LEFT, stage3_aihub_eval의 상관계수로 검증됨)
  2. 자이로(angZAve) 원시값에 물리적으로 불가능한 글리치(최대 53467, 정상은 수십)가 섞여있는데
     clip()으로 경계에 뭉쳐서 분위수 계산이 왜곡됨 - clip 대신 완전히 제외(전방채움)로 수정.

수정 후 LOVO(leave-one-video-out) 평균 정확도 0.687 (10개 영상 35982프레임 기준, 전 폴드
0.436~0.855로 랜덤보다 확실히 위). 현재 프로덕션(공개 50샘플 튜닝, stopped=0.1/eps=0.3/steer=0.28)의
LOVO스러운 신뢰도(0.710, 근데 5비디오짜리라 신뢰구간 넓음)와 비슷한 수준이지만 훨씬 큰/다양한
실데이터 기준이라는 게 다름.

이 후보(stopped=0.1/eps=0.01/steer=0.671)를 공개 50샘플로 교차검증하면 0.55(현재 프로덕션은
0.71) - 서로 다른 데이터 특성을 학습해서 그런 것으로 보이나, 공개 50샘플 자체가 실제 DACON
성능을 못 예측한다는 게 이미 여러 번 확인된 상태라 이 낮은 교차검증 점수가 "나쁘다"는 뜻은
아님. 실제로 나은지는 제출해봐야 앎 - 로컬 검증만으로 프로덕션 교체하지 않음.

사용법: model/stage3/best.pt를 이 파일의 CAN_CANDIDATE 값으로 바꿔서 재빌드 후 제출,
결과 비교.
"""

CAN_CANDIDATE = {
    "stopped_thr": 0.1,
    "accel_eps": 0.01,
    "steer_thr": 0.464,
}

CURRENT_PRODUCTION = {  # 공개 50샘플 튜닝, 실측 0.521 검증된 값
    "stopped_thr": 0.1,
    "accel_eps": 0.3,
    "steer_thr": 0.28,
}

# 검증 수치 요약(재현하려면 stage3_can_calibrate.py의 clean_yaw/build_labels 로직 참고):
#   CAN 기준(35982프레임, LOVO):     0.687
#   공개 50샘플 기준(현재 프로덕션):  0.710
#   CAN 후보를 공개 50샘플로 교차검증: 0.55
#   현재 프로덕션을 CAN 기준으로 교차검증: (미실행 - 필요시 stage3_can_calibrate.eval_acc 재사용)
