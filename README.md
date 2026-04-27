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

> Threshold: **0.64** (selected via full sweep, see `experiments/plots/threshold_sweep/`)

### Component B — Vision (YOLOv11s)

Fine-tuned on a curated firearm-detection dataset (see
[vision/](vision/) and `YOLO_hugging-main/`).

| Metric | Value |
|---|---|
| mAP@50 | 0.984 |
| Precision | 0.995 |
| Recall | 0.980 |

> Confidence threshold: **0.6** | NMS IoU: **0.45** | Inference resolution: **1280**

---

## Repository Layout

```
.
├── README.md
├── requirements.txt
├── .env.example                        ← copy to .env, fill in secrets
├── wav_info.py                         ← WAV file inspector utility
│
├── frontend/                           ← React dashboard (TanStack Start + shadcn/ui)
│   └── src/
│       ├── hooks/useAuth.ts            ← auth → POST /api/auth/login
│       ├── hooks/useAbly.ts            ← Ably subscriber → audio/video events
│       ├── lib/ably.ts                 ← Ably client → GET /api/ably-token
│       └── lib/incidentStore.ts        ← Zustand store, API-backed
│
├── api/                                ← FastAPI backend
│   ├── main.py                         ← app, CORS, router mount, lifespan seed
│   ├── database.py                     ← SQLAlchemy engine + session
│   ├── models.py                       ← ORM: users, devices, incidents, messages
│   ├── schemas.py                      ← Pydantic request/response models
│   └── routes/
│       ├── auth.py                     ← POST /api/auth/login, GET /api/auth/me
│       ├── devices.py                  ← GET/PATCH /api/schools/{id}/devices
│       ├── incidents.py                ← CRUD /api/incidents
│       ├── messages.py                 ← GET/POST /api/incidents/{id}/messages
│       └── ably_token.py               ← GET /api/ably-token
│
├── configs/
│   ├── yamnet_pipeline.yaml            ← audio pipeline + Modal config
│   └── experiment_template.yaml
│
├── data/
│   ├── raw/
│   │   ├── gunshot/                    ← 4,621 WAVs (label 1)
│   │   └── not_gunshot/                ← 19,523 WAVs (label 0)
│   └── processed/
│       ├── embeddings/                 ← output of extract_embeddings.py
│       └── splits/                     ← 70/15/15 train/val/test splits
│
├── pipeline/                           ← audio embedding extraction
│   ├── extract_embeddings.py           ← YAMNet extraction CLI (local)
│   ├── modal_extract.py                ← YAMNet extraction on Modal cloud GPU (T4)
│   └── split_dataset.py                ← stratified train/val/test split
│
├── models/                             ← classification heads + saved weights
│   ├── head_dense.py                   ← Dense MLP head (build_dense_head)
│   └── saved_weights/
│       └── dense_head_best.keras       ← best checkpoint (val_loss)
│
├── training/                           ← audio head training
│   ├── train_head.py                   ← training CLI with early stopping + class weights
│   └── evaluate_test.py                ← held-out test set evaluation
│
├── inference/                          ← Component A — audio live inference
│   └── live_inference.py               ← real-time mic / file inference (sliding window)
│
├── vision/                             ← Component B — vision live inference
│   ├── live_inference.py               ← real-time webcam / video file inference
│   └── inference_breakdown.md          ← component-by-component code walkthrough
│
├── experiments/
│   ├── threshold_sweep.py              ← sweep audio thresholds 0.02–0.98
│   ├── runs/                           ← per-run JSON (train + test metrics)
│   └── plots/
│       └── threshold_sweep/            ← PR curve, ROC, F1 vs threshold, metrics table
│
├── scripts/
│   ├── dev.sh                          ← start API + frontend + audio + vision (Linux/Mac)
│   └── dev.ps1                         ← same, Windows
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

Requires Python 3.10+ and Node.js 18+. The vision component additionally needs
`ultralytics` and `opencv-python` (already in `requirements.txt`) and the
fine-tuned YOLO weights at `YOLO_hugging-main/best.pt`.

---

## Full-Stack Quick Start

### One command (after setting up `.env`)

```bash
# Linux / Mac
cp .env.example .env   # fill in ABLY_API_KEY, JWT_SECRET
bash scripts/dev.sh

# Windows (PowerShell)
Copy-Item .env.example .env   # fill in secrets
.\scripts\dev.ps1
```

This starts:
1. **FastAPI backend** at `http://localhost:8000` — auto-creates SQLite DB and seeds demo data
2. **React dashboard** at `http://localhost:5173` — school and police views
3. **Audio inference** — microphone listener publishing to Ably channel `gunshot-detection`
4. **Vision inference** — webcam listener publishing to the same channel

### Manual startup (four terminals)

```bash
# 1. Backend
uvicorn api.main:app --reload --port 8000

# 2. Frontend
cd frontend && npm install && npm run dev

# 3. Audio pipeline
python -m inference.live_inference --location "Main Entrance"

# 4. Vision pipeline
python -m vision.live_inference --location "Main Entrance"
```

### Demo credentials

| Role | Email | Password |
|---|---|---|
| School Operator | school@demo.com | school123 |
| Dispatch Officer | police@demo.com | police123 |

---

## Real-time Detection Flow

```
┌──────────────────┐                       ┌──────────────────┐
│  Audio (mic /    │                       │  Vision (webcam /│
│  WAV file)       │                       │  video file)     │
└────────┬─────────┘                       └────────┬─────────┘
         │  YAMNet → Dense head                     │  YOLOv11s
         │  prob ≥ 0.64                             │  conf ≥ 0.6
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

Both components have a 5-second alert cooldown and bounded S3-upload thread
pools — sustained detections won't flood the channel or the bucket.

### Ably message format

| Event name | Data |
|---|---|
| `audio:detected` | `audio:detected:{location}` |
| `audio:snippet` | `audio:snippet:{location}:{presigned_url}` |
| `video:detected` | `video:detected:{location}` |
| `video:segment` | `video:segment:{location}:{presigned_url}` |

### Environment variables

| Variable | Required | Description |
|---|---|---|
| `ABLY_API_KEY` | For WS alerts | Ably API key (or pass `--ably_key`) |
| `AWS_ACCESS_KEY_ID` | For S3 upload | AWS credentials (standard boto3 env vars) |
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
  B) --demo_file     → librosa.load → chunked in-process
  C) --run + sender  → UDP socket (127.0.0.1:9999) ← --demo_file in T2
          ↓
  Ring buffer (32,000 samples = last 2 s)
          ↓
  YAMNet → (1024,) embedding    ← one forward pass per 0.5 s chunk
          ↓
  Dense head → gunshot probability
          ↓
  if prob >= 0.64 (with 5 s cooldown):
      • console log
      • append to inference/detections.jsonl
      • Ably WS → "audio:detected:{location}"
      • (optional) upload 2 s WAV → S3 → "audio:snippet:{location}:{url}"
```

Latency: ≤ 0.5 s from audio to detection.

### Architecture

```
Input(1024)           ← YAMNet mean-pooled clip embedding
  → Dense(256, relu)
  → Dropout(0.3)
  → Dense(1, sigmoid) → gunshot probability in [0, 1]
```

### Training

```bash
# Default hyperparameters
python -m training.train_head

# Override class weight for higher recall
python -m training.train_head --class_weight_gunshot 8.0 --threshold 0.35
```

Best weights saved to `models/saved_weights/dense_head_best.keras`.

### Live inference

```bash
# Live mic, console + JSONL log only
python -m inference.live_inference --location "building-a/entrance"

# With Ably WebSocket alerts
export ABLY_API_KEY="your-ably-key"
python -m inference.live_inference --location "building-a/entrance"

# With Ably + S3 audio snippet upload
python -m inference.live_inference \
    --location   "building-a/entrance" \
    --s3_bucket  my-bucket \
    --aws_region eu-west-1
```

### File / demo modes

```bash
# Single terminal — loads model, plays audio, runs inference
python -m inference.live_inference --demo_file path/to/shot.wav --location "Gymnasium"

# Two terminals — engine stays running, audio injected on demand
# Terminal 1:
python -m inference.live_inference --run --location "Gymnasium"
# Terminal 2:
python -m inference.live_inference --demo_file path/to/shot.wav
```

`--demo_file` auto-detects whether `--run` is listening on the UDP port. If
it is, it acts as a sender; otherwise it falls back to the single-terminal
path.

### Audio CLI flags

| Flag | Default | Description |
|---|---|---|
| `--model_path` | `models/saved_weights/dense_head_best.keras` | Head weights |
| `--threshold` | `0.64` | Min probability to alert |
| `--device` | `None` | sounddevice input device index |
| `--source` (via `--demo_file`) | mic | Audio file path |
| `--run` | `False` | Bind UDP listener instead of opening mic |
| `--port` | `9999` | UDP port for `--run` / `--demo_file` IPC |
| `--location` | `unknown` | Label sent in every Ably message |
| `--channel` | `gunshot-detection` | Ably channel name |
| `--ably_key` | `$ABLY_API_KEY` | Override env var |
| `--s3_bucket` | `None` | Omit to skip S3 upload |
| `--aws_region` | `us-east-1` | AWS region for boto3 |

---

## Component B — Vision

### Pipeline

```
Video source (one of):
  A) Webcam (--source 0)        → cv2.VideoCapture device index
  B) --source path/to/video.mp4 → cv2.VideoCapture file path
          ↓
  YOLOv11s → bounding boxes + confidence
          ↓
  if max_conf >= 0.6 (with 5 s cooldown):
      • console log
      • append to vision/detections.jsonl
      • Ably WS → "video:detected:{location}"
      • (optional) upload annotated frame → S3 → "video:segment:{location}:{url}"
```

### Live inference

```bash
# Webcam (default), no display window
python -m vision.live_inference --location "Main Entrance"

# Webcam with annotated playback window (press 'q' to stop)
python -m vision.live_inference --location "Main Entrance" --show

# Video file
python -m vision.live_inference \
    --source YOLO_hugging-main/footage_worked/footage_pistol.avi \
    --location "Gymnasium" \
    --show

# With Ably + S3
export ABLY_API_KEY="your-ably-key"
python -m vision.live_inference \
    --location  "Gymnasium" \
    --s3_bucket my-bucket
```

### Vision CLI flags

| Flag | Default | Description |
|---|---|---|
| `--model_path` | `YOLO_hugging-main/best.pt` | Path to YOLO weights |
| `--threshold` | `0.6` | Min confidence to alert |
| `--iou` | `0.45` | NMS IoU threshold |
| `--imgsz` | `1280` | Inference resolution |
| `--source` | `0` | `0` = webcam, or path to video file |
| `--show` | `False` | Open OpenCV window with annotated feed |
| `--location` | `unknown` | Label sent in every Ably message |
| `--channel` | `gunshot-detection` | Ably channel name |
| `--ably_key` | `$ABLY_API_KEY` | Override env var |
| `--s3_bucket` | `None` | Omit to skip S3 upload |
| `--aws_region` | `us-east-1` | AWS region for boto3 |

---

## Audio Pipeline (one-time setup)

If you want to retrain the audio head from scratch, run the data pipeline
first.

### Step 1 — Extract YAMNet embeddings

```bash
# Local
python -m pipeline.extract_embeddings \
    --data_dir data/raw \
    --output_dir data/processed/embeddings \
    --workers 8

# Or on Modal T4 GPU
modal run pipeline/modal_extract.py
```

### Step 2 — Split dataset

```bash
python -m pipeline.split_dataset \
    --embeddings_dir data/processed/embeddings \
    --output_dir data/processed/splits
```

Stratified 70/15/15 split, `random_state=42`.

### Step 3 — Train head

```bash
python -m training.train_head
```

### Step 4 — Evaluate on test set

```bash
python -m training.evaluate_test --threshold 0.64
```

### Step 5 — Threshold sweep

```bash
python -m experiments.threshold_sweep
```

Saves PR curve, ROC, F1 vs threshold, and a full metrics table to
`experiments/plots/threshold_sweep/`.

---

## Modal Cloud GPU

Run audio embedding extraction on a Modal T4 GPU instead of locally.

```bash
# One-time setup
pip install modal
modal token new
modal volume create gunshot-data

# Upload raw WAVs
modal volume put gunshot-data data/raw/gunshot     gunshot
modal volume put gunshot-data data/raw/not_gunshot not_gunshot

# Run extraction
modal run pipeline/modal_extract.py

# Download results
modal volume get gunshot-data embeddings data/processed/embeddings
```

---

## Data Contracts

| Stage | Reads | Writes |
|---|---|---|
| `extract_embeddings.py` | `data/raw/gunshot/`, `data/raw/not_gunshot/` | `data/processed/embeddings/` |
| `modal_extract.py` | Modal volume: `gunshot/`, `not_gunshot/` | Modal volume: `embeddings/` |
| `split_dataset.py` | `data/processed/embeddings/` | `data/processed/splits/` |
| `train_head.py` | `data/processed/splits/` | `models/saved_weights/`, `experiments/runs/` |
| `evaluate_test.py` | `data/processed/splits/`, `models/saved_weights/` | `experiments/runs/test_results.json` |
| `threshold_sweep.py` | `data/processed/splits/`, `models/saved_weights/` | `experiments/plots/threshold_sweep/` |
| `inference/live_inference.py` | `models/saved_weights/`, audio source | `inference/detections.jsonl`, Ably, S3 |
| `vision/live_inference.py` | `YOLO_hugging-main/best.pt`, video source | `vision/detections.jsonl`, Ably, S3 |

---

## Notes on YAMNet

- Requires: mono, 16 kHz, float32, values in `[-1.0, +1.0]`
- Computes its own mel spectrogram internally — do **not** pass spectrograms
- 0.96-second windows, 0.48-second hop — a 2-second clip produces 3 frames, mean-pooled to (1024,)
- The "Gunshot, gunfire" AudioSet class index is resolved dynamically at
  runtime via `resolve_gunshot_class_idx()` — do not hardcode it. The
  bundled class map places it at index 421.
- YAMNet outputs numbers, not strings — class names are resolved via `load_class_map()`

---

## Reproducibility

- All random operations use `random_state=42`
- `split_info.json` stores exact original indices
- `metadata.json` timestamps every extraction run and embeds the full 521-class map
- `configs/yamnet_pipeline.yaml` is the single source of truth for audio pipeline hyperparameters

---

## References

- Valliappan et al. (2024, IEEE Access) — YAMNet + Dense head, 94.96% accuracy on 12-class firearm ID
- Wu (DAML 2024) — YAMNet + BiLSTM, strong generalization on UrbanSound8K
- TensorFlow official YAMNet tutorial — embeddings + 2-layer Dense head for binary classification
- Ultralytics YOLOv11 — fine-tuned for firearm detection on the curated dataset in `YOLO_hugging-main/`
