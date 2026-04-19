# Gunshot Detection — Part A: YAMNet Embedding Pipeline

Acoustic gunshot detection for school safety. Part A (this repo) covers the
audio classification pipeline: preprocessing, YAMNet feature extraction, and
dataset splitting. Part B (separate repo) is a YOLO-based vision pipeline
triggered when a gunshot is detected.

---

## What This Repo Currently Contains

- **`pipeline/preprocessing.py`** — Audio normalization and length fixing
- **`pipeline/extract_embeddings.py`** — YAMNet embedding extraction (run once, cache to disk)
- **`pipeline/split_dataset.py`** — Stratified 70/15/15 train/val/test split
- **`tests/`** — Unit tests for each pipeline module
- **`configs/yamnet_pipeline.yaml`** — All pipeline hyperparameters

## What Is NOT Yet Implemented

Head training, the two-stage cascade system, threshold tuning, live inference,
and weapon-type sub-classification are all planned for future sessions.
Placeholder files with comments exist for each.

---

## Repository Layout

```
.
├── README.md
├── requirements.txt
├── .gitignore
│
├── configs/
│   ├── yamnet_pipeline.yaml       ← YAMNet pipeline config (edit paths here)
│   └── experiment_template.yaml   ← [FUTURE] head training config template
│
├── data/
│   ├── raw/
│   │   ├── gunshot/               ← Put gunshot WAVs here (read-only)
│   │   └── not_gunshot/           ← Put non-gunshot WAVs here (read-only)
│   └── processed/
│       ├── embeddings/            ← Output of extract_embeddings.py
│       └── splits/                ← Output of split_dataset.py
│
├── pipeline/
│   ├── config.py                  ← CNN experiment mel params (NOT for YAMNet)
│   ├── preprocessing.py           ← preprocess_clip(), audit_dataset()
│   ├── extract_embeddings.py      ← YAMNet embedding extraction CLI
│   └── split_dataset.py           ← Train/val/test split CLI
│
├── models/
│   ├── head_dense.py              ← [FUTURE] Dense MLP head
│   ├── head_bilstm.py             ← [FUTURE] BiLSTM head
│   └── cascade/gate.py            ← [FUTURE] Two-stage cascade gate
│
├── training/
│   └── train_head.py              ← [FUTURE] Head training script
│
├── inference/
│   └── live_inference.py          ← [FUTURE] Live microphone inference
│
├── experiments/
│   └── runs/                      ← [FUTURE] Per-run JSON/CSV metric logs
│
└── tests/
    ├── test_preprocessing.py
    ├── test_extract_embeddings.py
    └── test_split_dataset.py
```

> **Important:** `pipeline/config.py` holds mel spectrogram parameters for a
> separate CNN experiment. It is **not** used by the YAMNet pipeline. The
> YAMNet pipeline reads `configs/yamnet_pipeline.yaml` exclusively.

---

## Prerequisites

```bash
pip install -r requirements.txt
```

Requires Python 3.10+. TensorFlow is only needed for `extract_embeddings.py`;
all preprocessing works without it.

---

## Quick Start

### Step 1 — Prepare raw data

Place your WAV files in the correct directories:

```
data/raw/gunshot/        ← gunshot recordings
data/raw/not_gunshot/    ← everything else (fireworks, door slams, etc.)
```

Files may be at any sample rate and any number of channels — preprocessing
handles resampling and mono conversion automatically.

### Step 2 — Audit data quality (optional but recommended)

Run the audit before extraction to catch bad recordings early:

```python
from pipeline.preprocessing import audit_dataset

report = audit_dataset("data/raw")
print(f"Total files : {report['total_files']}")
print(f"Too short   : {report['too_short']}")
print(f"Too long    : {report['too_long']}")
print(f"Silent      : {report['nearly_silent']}")
print(f"Has NaN     : {report['has_nan']}")
print(f"Unreadable  : {report['unreadable']}")
```

### Step 3 — Extract YAMNet embeddings

```bash
python -m pipeline.extract_embeddings \
    --data_dir data/raw \
    --output_dir data/processed/embeddings
```

Downloads YAMNet from TensorFlow Hub on the first run (~17 MB, cached
afterwards). Produces:

| File | Shape | Description |
|---|---|---|
| `X_embeddings.npy` | `(N, 1024)` float32 | One embedding per clip |
| `y_labels.npy` | `(N,)` float32 | 1.0 = gunshot, 0.0 = not_gunshot |
| `metadata.json` | — | Counts, skipped files, timestamp |

To re-run and overwrite existing outputs: add `--force`.

### Step 4 — Split the dataset

```bash
python -m pipeline.split_dataset \
    --embeddings_dir data/processed/embeddings \
    --output_dir data/processed/splits
```

Produces six `.npy` arrays plus `split_info.json` with exact indices and class
distributions. Splits: **70% train / 15% val / 15% test**, stratified,
`random_state=42`.

---

## Data Contracts

Each stage reads from and writes to known paths. Nothing in between.

| Stage | Reads from | Writes to |
|---|---|---|
| `preprocessing.py` | `data/raw/**/*.wav` | (in-memory only) |
| `extract_embeddings.py` | `data/raw/gunshot/`, `data/raw/not_gunshot/` | `data/processed/embeddings/` |
| `split_dataset.py` | `data/processed/embeddings/` | `data/processed/splits/` |
| **[FUTURE]** `train_head.py` | `data/processed/splits/` | `experiments/runs/`, `models/saved_weights/` |
| **[FUTURE]** `live_inference.py` | `models/saved_weights/` | stdout / alert to Part B |

---

## Running Tests

```bash
python -m pytest tests/ -v
```

The tests for `extract_embeddings.py` mock YAMNet so no network download is
required. The tests for `preprocessing.py` and `split_dataset.py` are
fully self-contained.

---

## Pipeline Architecture

```
Raw WAV file (any SR, any channels)
        │
        ▼
preprocess_clip()
  1. librosa.load(sr=16000)     ← always resample
  2. average channels → mono
  3. cast to float32
  4. warn if < 0.5 s
  5. normalize to [-1, +1]
  6. center-pad or center-trim → exactly 32000 samples
        │
        ▼  shape: (32000,) float32
        │
YAMNet (frozen, TF Hub)
  Internal: mel spectrogram → MobileNetV1 backbone
  Output:   per-frame embeddings (num_frames, 1024)
        │
        ▼
tf.reduce_mean(axis=0)          ← mean-pool over time
        │
        ▼  shape: (1024,) float32
        │
[FUTURE] Classification Head
  Dense: (1024,) → Dense(256) → Dense(128) → Dense(1, sigmoid)
  OR
  BiLSTM: (num_frames, 1024) → BiLSTM(128) → Dense(1, sigmoid)
        │
        ▼
Binary label: 1 = gunshot, 0 = not_gunshot
```

---

## Notes on YAMNet

- YAMNet requires: mono, 16 kHz, float32, values in `[-1.0, +1.0]`
- YAMNet computes its own mel spectrogram internally — **do not** pass
  spectrograms to YAMNet
- YAMNet uses 0.96-second windows with 0.48-second hop; a 2-second clip
  produces 3 embedding frames which are mean-pooled to 1
- AudioSet class index **427** is "Gunshot, gunfire" — used by the zero-shot
  baseline in `extract_zero_shot_score()`

---

## Reproducibility

- All random operations use `random_state=42`
- `split_info.json` stores the exact original indices for each split — you
  can reconstruct the exact same splits from any copy of `X_embeddings.npy`
- `metadata.json` timestamps every extraction run
- `configs/yamnet_pipeline.yaml` records all hyperparameters

---

## Future Work

These modules are stubbed and documented but not yet implemented:

| File | Description |
|---|---|
| `models/head_dense.py` | Dense MLP head — fast, strong baseline |
| `models/head_bilstm.py` | BiLSTM head — temporal modeling of per-frame embeddings |
| `models/cascade/gate.py` | Class-427 score gate for two-stage cascade |
| `training/train_head.py` | Head training script with early stopping and metric logging |
| `inference/live_inference.py` | Real-time microphone inference, alert trigger to Part B |

See each file for detailed design notes and references.

---

## References

- Valliappan et al. (2024, IEEE Access) — YAMNet + Dense head, 94.96% accuracy on 12-class firearm ID
- Wu (DAML 2024) — YAMNet + BiLSTM, strong generalization on UrbanSound8K
- TensorFlow official YAMNet tutorial — embeddings + 2-layer Dense head for binary classification
