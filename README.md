# Warden — School Shooting Detection System

Two complementary AI components publish to a shared real-time channel that
drives a unified incident dashboard:

- **Component A — Audio.** YAMNet + Dense head detects gunshots from a
  microphone or audio file. ([inference/](inference/))
- **Component B — Vision.** YOLOv11s detects firearms in a webcam or video
  file. ([vision/](vision/))

Both components publish detections to the same Ably channel
(`gunshot-detection`) which the React dashboard consumes via WebSocket. A
FastAPI backend persists incidents to SQLite (or Postgres) and serves the
frontend a short-lived Ably token so it never sees the API key.

---

## Results

### Component A — Audio (YAMNet + Dense head)

Trained on **24,144 clips** (4,621 gunshot / 19,523 not_gunshot). Test set: 3,622 samples.

| Metric | Value |
|---|---|
| Accuracy | 97.2% |
| F1 | 0.925 |
| AUC-ROC | 0.994 |
| Precision | 93.6% |
| Recall | 91.3% |
| TP / FP / FN / TN | 633 / 43 / 60 / 2886 |

> Threshold: **0.64** — selected via sweep over 0.02–0.98, see `experiments/plots/threshold_sweep/`

### Component B — Vision (YOLOv11s fine-tuned)

Evaluated on **1,737 held-out test images** (919 positives / 818 negatives, 974 GT boxes).
Threshold sweep at IoU-match ≥ 0.50, see `experiments/plots/yolo_threshold_sweep/`.

| Metric | Value |
|---|---|
| F1 (best) | 0.915 |
| Precision | 0.934 |
| Recall | 0.897 |
| PR-AUC | 0.915 |
| TP / FP / FN | 874 / 62 / 100 |

> Confidence threshold: **0.35** (best F1) | NMS IoU: **0.45** | Inference resolution: **1280**

---

## Repository Layout

```
.
├── README.md
├── requirements.txt
├── .env.example                        ← copy to .env, fill in secrets
│
├── demo_data/                          ← pre-recorded clips for offline demo
│   ├── audio_gun.wav                   ← 2 s gunshot clip (prob peaks at 1.00)
│   ├── video_gun.mp4                   ← raw CCTV footage with visible firearm
│   └── video_gun.annotated.mp4         ← same clip with bounding boxes baked in
│
├── frontend/                           ← React dashboard (TanStack Start + shadcn/ui)
│   └── src/
│       ├── hooks/useAuth.ts
│       ├── hooks/useAbly.ts
│       ├── lib/ably.ts
│       └── lib/incidentStore.ts
│
├── api/                                ← FastAPI backend
│   ├── main.py
│   ├── database.py
│   ├── models.py
│   ├── schemas.py
│   └── routes/
│       ├── auth.py
│       ├── devices.py
│       ├── incidents.py
│       ├── messages.py
│       └── ably_token.py
│
├── inference/                          ← Component A — audio live inference
│   ├── live_inference.py               ← real-time mic / file inference
│   ├── cascade.py                      ← two-stage audio → video pipeline
│   └── config.py                       ← YOLO weights path resolution
│
├── vision/                             ← Component B — vision live inference
│   └── live_inference.py               ← real-time webcam / video file inference
│
├── models/
│   ├── head_dense.py                   ← Dense MLP head (build_dense_head)
│   ├── saved_weights/
│   │   └── dense_head_best.keras       ← best audio head checkpoint (LFS)
│   └── yolo_finetuned/
│       └── best.pt                     ← fine-tuned YOLOv11s checkpoint (LFS)
│
├── pipeline/                           ← audio embedding extraction (training only)
│   ├── extract_embeddings.py
│   ├── modal_extract.py
│   └── split_dataset.py
│
├── training/
│   ├── train_head.py
│   ├── evaluate_test.py
│   └── modal_train_yolo.py             ← YOLOv11s fine-tuning on Modal A100
│
├── experiments/
│   ├── threshold_sweep.py              ← audio threshold sweep (0.02–0.98)
│   ├── yolo_threshold_sweep.py         ← YOLO confidence sweep on test set
│   └── plots/
│       ├── threshold_sweep/            ← audio: PR curve, ROC, F1 vs threshold
│       └── yolo_threshold_sweep/       ← vision: PR curve, F1 vs threshold, IoU grid
│
├── configs/
│   ├── yamnet_pipeline.yaml
│   └── experiment_template.yaml
│
├── scripts/
│   ├── dev.sh                          ← start full stack (Linux/Mac)
│   └── dev.ps1                         ← start full stack (Windows)
│
└── tests/
    ├── test_extract_embeddings.py
    └── test_yamnet_integration.py
```

---

## Prerequisites

```bash
pip install -r requirements.txt
```

Requires **Python 3.10+** and **Node.js 18+**.

Both model weights are stored in the repo via Git LFS:
- `models/saved_weights/dense_head_best.keras` — audio head
- `models/yolo_finetuned/best.pt` — fine-tuned YOLOv11s

> Make sure `git lfs` is installed before cloning so the model files download
> correctly (`git lfs install` once, then `git clone` as usual).

---

## Quick Start

### 1. Clone

```bash
git lfs install          # only needed once per machine
git clone https://github.com/MartinKostadinov04/school_shooting_detection.git
cd school_shooting_detection
pip install -r requirements.txt
```

### 2. Environment

```bash
cp .env.example .env
# Edit .env — set ABLY_API_KEY and JWT_SECRET at minimum
```

### 3. Full stack

```bash
# Linux / Mac
bash scripts/dev.sh

# Windows (PowerShell)
.\scripts\dev.ps1
```

This starts:
1. **FastAPI backend** at `http://localhost:8000` — auto-seeds SQLite with demo data
2. **React dashboard** at `http://localhost:5173` — school and police views
3. **Audio inference** — mic listener on `gunshot-detection` channel
4. **Vision inference** — webcam listener on the same channel

### Demo credentials

| Role | Email | Password |
|---|---|---|
| School Operator | school@demo.com | school123 |
| Dispatch Officer | police@demo.com | police123 |

---

## Cascade Demo (Offline, No Webcam Required)

The cascade pipeline chains both components: audio stage 1 triggers video
stage 2 for visual confirmation. Pre-recorded demo clips are in `demo_data/`.

### What it does

```
demo_data/audio_gun.wav  →  Stage 1: YAMNet + Dense head
                                prob=0.758 ≥ 0.64  →  GUNSHOT DETECTED
                                Ably: audio:detected  +  audio:snippet
                             ↓
demo_data/video_gun.mp4  →  Stage 2: YOLOv11s (conf ≥ 0.35, 4-of-6 gate)
                                conf=0.627 → 0.759, count=2
                                Ably: video:detected  +  video:segment
```

### Command

```bash
# Start the FastAPI backend first (needed for stable media URLs):
uvicorn api.main:app --reload --port 8000

# Then in a second terminal — run the cascade in REPL mode:
python -m inference.cascade --no_sahi

# At the REPL prompt, submit the audio clip:
cascade [Cafeteria]> demo_data/audio_gun.wav

# When asked for the video path, submit the video clip:
Video path for visual confirmation (Enter to skip): demo_data/video_gun.mp4
```

> `--no_sahi` skips tiled inference to avoid a known torch/numpy/cv2
> incompatibility on some Windows setups. Omit it on Linux/Mac.

### Expected output

```
  TacticalEye — Cascade Pipeline
  ══════════════════════════════════════════════════════
  Audio:    threshold=0.64  window=2.0s  hop=0.5s
  Video:    threshold=0.35  model=best.pt
  Location: Cafeteria
  Mode:     REPL (file submission)
  ══════════════════════════════════════════════════════

  cascade [Cafeteria]> demo_data/audio_gun.wav

  ▶  'audio_gun.wav'  (2.0 s · 4 chunks · window=2.0s  hop=0.5s)

  🔴  GUNSHOT DETECTED  prob=0.758  loc=Cafeteria

  Video path for visual confirmation (Enter to skip): demo_data/video_gun.mp4

  ▶  STAGE-2  'video_gun.mp4'  threshold=0.35

  🔴  GUN DETECTED  conf=0.759  count=2  loc=Cafeteria
  → Police alert WITH visual reference
```

Both alerts are published to Ably and appear live on the React dashboard.

### Live-mic + live-video mode

```bash
python -m inference.cascade --live --location "Main Entrance"
```

Runs the microphone continuously. On a gunshot detection it pauses and asks
for a video path. Press `Ctrl+C` to stop.

### Cascade CLI flags

| Flag | Default | Description |
|---|---|---|
| `--audio_threshold` | `0.64` | Min gunshot probability to fire stage 2 |
| `--video_threshold` | `0.35` | Min YOLO confidence to confirm a gun |
| `--location` | `Cafeteria` | Label sent in every Ably message |
| `--audio_model` | `models/saved_weights/dense_head_best.keras` | Audio head weights |
| `--video_model` | `models/yolo_finetuned/best.pt` | YOLO weights |
| `--no_sahi` | off | Disable tiled inference (Windows stability fix) |
| `--no_pose` | off | Disable pose-overlap FP filter |
| `--live` | off | Live-mic mode instead of REPL file mode |
| `--show` | off | Open OpenCV window during video stage |
| `--ably_key` | `$ABLY_API_KEY` | Override env var |

---

## Real-time Detection Flow

```
┌──────────────────┐                       ┌──────────────────┐
│  Audio (mic /    │                       │  Vision (webcam /│
│  WAV file)       │                       │  video file)     │
└────────┬─────────┘                       └────────┬─────────┘
         │  YAMNet → Dense head                     │  YOLOv11s
         │  prob ≥ 0.64                             │  conf ≥ 0.35
         ▼                                          ▼
   ┌─────────────────────────────────────────────────────┐
   │  Ably channel:  gunshot-detection                   │
   │    audio:detected   audio:snippet                   │
   │    video:detected   video:segment                   │
   └─────────────────────┬───────────────────────────────┘
                         │  WebSocket
                         ▼
              ┌─────────────────────┐
              │  React dashboard    │
              │  (school + police)  │
              └─────────────────────┘
```

Both components have a 5-second alert cooldown. The vision component also
uses a 4-of-6 temporal gate — a gun must appear in 4 out of 6 consecutive
frames before an alert fires, suppressing single-frame false positives.

### Ably message format

| Event name | Data |
|---|---|
| `audio:detected` | `audio:detected:{location}:{prob}` |
| `audio:snippet` | `audio:snippet:{location}:{url}` |
| `video:detected` | `video:detected:{location}:{conf}` |
| `video:segment` | `video:segment:{location}:{url}` |
| `video:negative` | `video:negative:{location}` |

### Environment variables

| Variable | Required | Description |
|---|---|---|
| `ABLY_API_KEY` | For WS alerts | Ably API key (or pass `--ably_key`) |
| `AWS_ACCESS_KEY_ID` | For S3 upload | AWS credentials |
| `AWS_SECRET_ACCESS_KEY` | For S3 upload | |
| `AWS_DEFAULT_REGION` | For S3 upload | Falls back to `--aws_region` arg |
| `JWT_SECRET` | API auth | Backend JWT signing key |
| `DATABASE_URL` | API persistence | Defaults to `sqlite:///data/tacticaleye.db` |

> **Never hardcode credentials.** All secrets are read from environment variables only.

---

## Component A — Audio

### Pipeline

```
Audio source (one of):
  A) Microphone      → sounddevice InputStream callback
  B) --demo_file     → soundfile.read → chunked in-process
          ↓
  Ring buffer (32,000 samples = last 2 s at 16 kHz)
          ↓
  YAMNet → (1024,) embedding    ← one forward pass per 0.5 s chunk
          ↓
  Dense head → gunshot probability
          ↓
  if prob >= 0.64 (with 5 s cooldown):
      • console log
      • append to inference/detections.jsonl
      • Ably WS → "audio:detected:{location}:{prob}"
      • Ably WS → "audio:snippet:{location}:{url}"
```

Latency: ≤ 0.5 s from audio to detection.

### Architecture

```
Input(1024)           ← YAMNet mean-pooled clip embedding
  → Dense(256, relu)
  → Dropout(0.3)
  → Dense(1, sigmoid) → gunshot probability in [0, 1]
```

### Live inference

```bash
python -m inference.live_inference --location "Main Entrance"

# With Ably + S3
export ABLY_API_KEY="your-key"
python -m inference.live_inference --location "Main Entrance" --s3_bucket my-bucket
```

### Audio CLI flags

| Flag | Default | Description |
|---|---|---|
| `--model_path` | `models/saved_weights/dense_head_best.keras` | Head weights |
| `--threshold` | `0.64` | Min probability to alert |
| `--device` | `None` | sounddevice input device index |
| `--location` | `unknown` | Label sent in every Ably message |
| `--channel` | `gunshot-detection` | Ably channel name |
| `--ably_key` | `$ABLY_API_KEY` | Override env var |
| `--s3_bucket` | `None` | Omit to skip S3 upload |

---

## Component B — Vision

### Pipeline

```
Video source (one of):
  A) Webcam (--source 0)        → cv2.VideoCapture device index
  B) --source path/to/video.mp4 → cv2.VideoCapture file path
          ↓
  YOLOv11s → bounding boxes + confidence scores
          ↓
  FP-reduction stack:
    1. Temporal 4-of-6 gate     ← gun must appear in 4 of 6 consecutive frames
    2. SAHI tiled inference     ← improves recall on small/distant guns
    3. Pose-overlap filter      ← suppresses detections not near a hand region
          ↓
  if gate fires (with 5 s cooldown):
      • console log
      • append to vision/detections.jsonl
      • Ably WS → "video:detected:{location}:{conf}"
      • Ably WS → "video:segment:{location}:{url}"
```

### Live inference

```bash
# Webcam (default)
python -m vision.live_inference --location "Main Entrance"

# Video file with annotated playback window
python -m vision.live_inference \
    --source demo_data/video_gun.mp4 \
    --location "Gymnasium" \
    --show
```

### Vision CLI flags

| Flag | Default | Description |
|---|---|---|
| `--model_path` | `models/yolo_finetuned/best.pt` | YOLO weights |
| `--threshold` | `0.35` | Min confidence to alert |
| `--iou` | `0.45` | NMS IoU threshold |
| `--imgsz` | `1280` | Inference resolution |
| `--source` | `0` | `0` = webcam, or path to video file |
| `--show` | `False` | Open OpenCV window with annotated feed |
| `--no_sahi` | off | Disable SAHI tiled inference |
| `--no_pose` | off | Disable pose-overlap FP filter |
| `--location` | `unknown` | Label sent in every Ably message |
| `--ably_key` | `$ABLY_API_KEY` | Override env var |
| `--s3_bucket` | `None` | Omit to skip S3 upload |

---

## Threshold Selection

### Audio

```bash
python -m experiments.threshold_sweep
```

Sweeps 0.02–0.98 on the test set and saves PR curve, ROC, F1 vs threshold,
and a metrics table to `experiments/plots/threshold_sweep/`.
Best threshold: **0.64** (F1 = 0.925).

### Vision (YOLO)

```bash
python -m experiments.yolo_threshold_sweep
```

Runs YOLO once at conf=0.001 to collect all raw detections, then sweeps
0.05–0.95 in post-processing using IoU-match ≥ 0.50. Saves four plots to
`experiments/plots/yolo_threshold_sweep/` and raw numbers to
`experiments/runs/yolo_sweep_results.json`.
Best threshold: **0.35** (F1 = 0.915, PR-AUC = 0.915).

---

## Training from Scratch

### Audio head

```bash
# 1. Extract YAMNet embeddings
python -m pipeline.extract_embeddings \
    --data_dir data/raw \
    --output_dir data/processed/embeddings

# 2. Split dataset
python -m pipeline.split_dataset \
    --embeddings_dir data/processed/embeddings \
    --output_dir data/processed/splits

# 3. Train
python -m training.train_head

# 4. Evaluate
python -m training.evaluate_test --threshold 0.64

# 5. Sweep thresholds
python -m experiments.threshold_sweep
```

### YOLOv11s (Modal cloud GPU)

```bash
# Requires a Modal account — fine-tuning runs on an A100-80GB
modal run training/modal_train_yolo.py
```

---

## Notes on YAMNet

- Requires: mono, 16 kHz, float32, values in `[-1.0, +1.0]`
- Computes its own mel spectrogram internally — do **not** pass spectrograms
- 0.96 s windows, 0.48 s hop — a 2 s clip produces 3 frames, mean-pooled to (1024,)
- YAMNet class names are resolved dynamically via `load_class_map()` — do not hardcode indices

---

## References

- Valliappan et al. (2024, IEEE Access) — YAMNet + Dense head, 94.96% accuracy on 12-class firearm ID
- Wu (DAML 2024) — YAMNet + BiLSTM, strong generalization on UrbanSound8K
- Ultralytics YOLOv11 — fine-tuned for firearm detection on curated CCTV dataset
