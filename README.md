# Dashcam-Based Traffic Accident Analysis (DACON)

A DACON competition project that detects and analyzes traffic accidents from dashcam footage,
built as a 3-stage pipeline.

## Pipeline

- **Stage 1 — Preprocessing/calibration**: recapture correction, blend sweep, CCD evaluation
- **Stage 2 — Tracking/candidate selection**: object tracking, incident candidate selection & reranking, AI Hub dataset validation
- **Stage 3 — Behavior/signal calibration**: driver behavior analysis, CAN signal calibration, calibration against the SullyChen dataset

## Tech Stack

`PyTorch` · `torchvision` · `OpenCV` · `pandas` · `numpy`

## Getting Started

```bash
pip install -r Baseline/requirements.txt
```

- Train: `Baseline/[Baseline_Train]_3Stage_학습.ipynb`
- Inference & generate submission zip: `Baseline/[Baseline_Inference]_3Stage_추론및ZIP생성.ipynb`
