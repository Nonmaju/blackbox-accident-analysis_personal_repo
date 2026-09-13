# DACON 블랙박스 영상 기반 교통사고 분석

블랙박스(대시캠) 영상으로 교통사고를 탐지·분석하는 DACON 대회용 프로젝트입니다.
3단계(Stage) 파이프라인으로 구성되어 있습니다.

## 파이프라인

- **Stage 1 — 전처리/보정**: 영상 재촬영(recapture) 보정, 블렌딩 스윕, CCD 평가
- **Stage 2 — 추적/후보 선정**: 객체 추적(tracker), 사고 후보 선택·재순위화(rerank), AI Hub 데이터 검증
- **Stage 3 — 행동/신호 보정**: 운전자 행동 분석, CAN 신호 보정, SullyChen 데이터셋 기반 보정

## 사용 기술

`PyTorch` · `torchvision` · `OpenCV` · `pandas` · `numpy`

## 시작하기

```bash
pip install -r Baseline/requirements.txt
```

- 학습: `Baseline/[Baseline_Train]_3Stage_학습.ipynb`
- 추론 & 제출 파일 생성: `Baseline/[Baseline_Inference]_3Stage_추론및ZIP생성.ipynb`
