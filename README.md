# Gunshot Detection — Part A: YAMNet + Dense Head

Acoustic gunshot detection for school safety. Part A (this repo) covers the
full audio classification pipeline: preprocessing, YAMNet feature extraction,
dense head training, threshold optimisation, and real-time microphone inference.
Part B (separate repo) is a YOLO-based vision pipeline triggered when a gunshot
is detected.

---

## Results

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

---

## Repository Layout

```
.
├── README.md
├── requirements.txt
├── wav_info.py                         ← WAV file inspector utility
│
├── configs/
│   ├── yamnet_pipeline.yaml            ← pipeline + Modal config (single source of truth)
│   └── experiment_template.yaml
│
├── data/
│   ├── raw/
│   │   ├── gunshot/                   ← 4,621 WAVs (label 1)
│   │   └── not_gunshot/               ← 19,523 WAVs (label 0)
│   └── processed/
│       ├── embeddings/                ← output of extract_embeddings.py
│       └── splits/                    ← 70/15/15 train/val/test splits
│
├── pipeline/
│   ├── extract_embeddings.py          ← YAMNet extraction CLI (local)
│   ├── modal_extract.py               ← YAMNet extraction on Modal cloud GPU (T4)
│   └── split_dataset.py               ← stratified train/val/test split
│
├── models/
│   ├── head_dense.py                  ← Dense MLP head (build_dense_head)
│   └── saved_weights/
│       └── dense_head_best.keras      ← best checkpoint (val_loss)
│
├── training/
│   ├── train_head.py                  ← training CLI with early stopping + class weights
│   └── evaluate_test.py               ← held-out test set evaluation
│
├── inference/
│   └── live_inference.py              ← real-time microphone inference (sliding window)
│
├── experiments/
│   ├── threshold_sweep.py             ← sweep thresholds 0.02–0.98, save plots
│   ├── runs/                          ← per-run JSON (train + test metrics)
│   └── plots/
│       └── threshold_sweep/           ← PR curve, ROC, F1 vs threshold, metrics table
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

Requires Python 3.10+.

---

## Quick Start

### Step 1 — Extract YAMNet embeddings

```bash
# Local
python -m pipeline.extract_embeddings \
    --data_dir data/raw \
    --output_dir data/processed/embeddings \
    --workers 8

# Or on Modal T4 GPU (see Modal section below)
modal run pipeline/modal_extract.py
```

Output in `data/processed/embeddings/`:

| File | Shape | Description |
|---|---|---|
| `X_embeddings.npy` | `(N, 1024)` float32 | Mean-pooled YAMNet embeddings |
| `y_labels.npy` | `(N,)` float32 | 1.0 = gunshot, 0.0 = not_gunshot |
| `zero_shot_scores.npy` | `(N,)` float32 | YAMNet class-427 score (zero-shot baseline) |
| `yamnet_top_class_indices.npy` | `(N,)` int32 | Top AudioSet class index per clip |
| `yamnet_top_class_scores.npy` | `(N,)` float32 | Confidence of top class |
| `yamnet_top_class_names.npy` | `(N,)` str | Top class name |
| `metadata.json` | — | Counts, skipped files, timestamp, full 521-class map |

### Step 2 — Split dataset

```bash
python -m pipeline.split_dataset \
    --embeddings_dir data/processed/embeddings \
    --output_dir data/processed/splits
```

Stratified 70/15/15 split, `random_state=42`.

### Step 3 — Train head

```bash
# Default hyperparameters
python -m training.train_head

# Override class weight for higher recall
python -m training.train_head --class_weight_gunshot 8.0 --threshold 0.35
```

Best weights saved to `models/saved_weights/dense_head_best.keras`.
Per-run JSON saved to `experiments/runs/`.

### Step 4 — Evaluate on test set

```bash
python -m training.evaluate_test --threshold 0.64
```

### Step 5 — Threshold sweep

```bash
python -m experiments.threshold_sweep
```

Saves 5 plots to `experiments/plots/threshold_sweep/`:
- `precision_recall_vs_threshold.png`
- `f1_vs_threshold.png`
- `precision_recall_curve.png`
- `roc_curve.png`
- `metrics_table.png` (full TP/FP/FN/TN table, best F1 highlighted)

### Step 6 — Live inference

```bash
# Minimal (console + JSONL log only)
python -m inference.live_inference --location "building-a/entrance"

# With Ably WebSocket alerts
export ABLY_API_KEY="your-ably-key"
python -m inference.live_inference \
    --location  "building-a/entrance" \
    --channel   "gunshot-detection"

# With Ably + S3 audio snippet upload
export ABLY_API_KEY="your-ably-key"
export AWS_ACCESS_KEY_ID="..."
export AWS_SECRET_ACCESS_KEY="..."
python -m inference.live_inference \
    --location   "building-a/entrance" \
    --channel    "gunshot-detection" \
    --s3_bucket  my-bucket \
    --aws_region eu-west-1
```

Press **Enter** or **Ctrl+C** to stop. Detections are appended to `inference/detections.jsonl`.

---

## Model Architecture

```
Input(1024)           ← YAMNet mean-pooled clip embedding
  → Dense(256, relu)
  → Dropout(0.3)
  → Dense(1, sigmoid) → gunshot probability in [0, 1]
```

Training config (default):
- Optimizer: Adam, lr=3e-4
- Loss: binary crossentropy
- Class weights: balanced (~4.2× for gunshot)
- Early stopping: patience=10 on val_loss
- Best run: epoch 12/100

---

## Live Inference Architecture

```
Microphone (16 kHz mono float32)
  → sounddevice callback (8,000 samples = 0.5 s chunks)
  → Ring buffer (32,000 samples = last 2 s)
  → YAMNet → (1024,) embedding          ← one forward pass per chunk
  → Dense head → gunshot probability
  → if prob >= 0.64:
      • console log (timestamp + probability + location)
      • append to inference/detections.jsonl
      • Ably WS publish → "audio:detected:{location}"
      • (if --s3_bucket) upload 2 s WAV snippet → S3 presigned URL
                         Ably WS publish → "audio:snippet:{location}:{url}"
```

Latency: ≤ 0.5 s from gunshot to detection.

### Ably message format

| Event name | Data |
|---|---|
| `audio:detected` | `audio:detected:{location}` |
| `audio:snippet` | `audio:snippet:{location}:{presigned_url}` |

### Environment variables

| Variable | Required | Description |
|---|---|---|
| `ABLY_API_KEY` | For WS alerts | Ably API key (or pass `--ably_key`) |
| `AWS_ACCESS_KEY_ID` | For S3 snippets | AWS credentials (standard boto3 env vars) |
| `AWS_SECRET_ACCESS_KEY` | For S3 snippets | |
| `AWS_DEFAULT_REGION` | For S3 snippets | Falls back to `--aws_region` arg |

> **Never hardcode credentials.** All secrets are read from environment variables only.

---

## Modal Cloud GPU

Run embedding extraction on a Modal T4 GPU instead of locally.

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

> **Never commit actual token values.** Store credentials via `modal token new` or
> environment variables — the config stores only the env var names.

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
| `live_inference.py` | `models/saved_weights/` | `inference/detections.jsonl`, Ably WS, S3 |

---

## Notes on YAMNet

- Requires: mono, 16 kHz, float32, values in `[-1.0, +1.0]`
- Computes its own mel spectrogram internally — do **not** pass spectrograms
- 0.96-second windows, 0.48-second hop — a 2-second clip produces 3 frames, mean-pooled to (1024,)
- AudioSet class index **421** = `"Gunshot, gunfire"` (confirmed from model's class map)
- YAMNet outputs numbers, not strings — class names are resolved via `load_class_map()`

---

## Reproducibility

- All random operations use `random_state=42`
- `split_info.json` stores exact original indices
- `metadata.json` timestamps every extraction run and embeds the full 521-class map
- `configs/yamnet_pipeline.yaml` is the single source of truth for pipeline hyperparameters

---

## References

- Valliappan et al. (2024, IEEE Access) — YAMNet + Dense head, 94.96% accuracy on 12-class firearm ID
- Wu (DAML 2024) — YAMNet + BiLSTM, strong generalization on UrbanSound8K
- TensorFlow official YAMNet tutorial — embeddings + 2-layer Dense head for binary classification
