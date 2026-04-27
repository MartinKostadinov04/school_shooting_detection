"""
modal_train_yolo.py
===================
Fine-tune YOLOv8m on a firearm-detection dataset using Modal cloud GPU.

The dataset must be in standard YOLO format (a data.yaml pointing at
images/train, images/val and labels/train, labels/val directories).
Everything lives on a Modal persistent Volume so nothing is lost between runs.

Setup (one-time)
----------------
1. Install Modal and authenticate::

       pip install modal
       modal token new          # saves credentials to ~/.modal.toml

2. Create the persistent volume::

       modal volume create yolo-training

3. Upload your dataset (YOLO format)::

       modal volume put yolo-training data/yolo/dataset  dataset

   The volume will contain::

       /vol/dataset/
           data.yaml
           images/train/   images/val/
           labels/train/   labels/val/

Run
---
::

    # Fine-tune from YOLOv8m pretrained weights (recommended for first run)
    modal run training/modal_train_yolo.py

    # Resume from a previous checkpoint saved in the volume
    modal run training/modal_train_yolo.py --resume

    # Override epochs or GPU
    modal run training/modal_train_yolo.py --epochs 50 --gpu A10G

Download weights
----------------
::

    modal volume get yolo-training weights/best.pt  models/yolo_finetuned/best.pt

Volume layout (inside /vol)
----------------------------
::

    /vol/
      dataset/          ← YOLO dataset uploaded before running
      weights/          ← written by this script
        best.pt         ← best checkpoint (mAP@50)
        last.pt         ← final checkpoint
        results.csv     ← per-epoch metrics
        args.yaml       ← training args snapshot

Authentication
--------------
Modal reads credentials from ``~/.modal.toml`` (set by ``modal token new``)
or from the environment variables::

    MODAL_TOKEN_ID
    MODAL_TOKEN_SECRET
"""

import os
import sys
from pathlib import Path

import modal

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

APP_NAME      = "yolo-finetune"
VOLUME_NAME   = "yolo-training"
VOLUME_MOUNT  = "/vol"

DATASET_DIR   = f"{VOLUME_MOUNT}/dataset"   # uploaded before run
WEIGHTS_DIR   = f"{VOLUME_MOUNT}/weights"   # written by this script

DEFAULT_BASE_MODEL = "yolov8m.pt"   # downloaded from Ultralytics hub if not in volume
DEFAULT_EPOCHS     = 100
DEFAULT_IMGSZ      = 1280
DEFAULT_BATCH      = 8              # safe for 16 GB T4; raise to 16 for A10G
DEFAULT_GPU        = "T4"
DEFAULT_PATIENCE   = 20

# ---------------------------------------------------------------------------
# Modal app + image
# ---------------------------------------------------------------------------

app = modal.App(APP_NAME)

training_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "ultralytics>=8.3.0",
        "torch>=2.2.0",
        "torchvision>=0.17.0",
        "numpy>=1.24.0",
        "pyyaml>=6.0",
        "opencv-python-headless>=4.8.0",
        "pillow>=10.0.0",
        "sahi>=0.11.0",   # tiled inference at eval time
    )
)

data_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)


# ---------------------------------------------------------------------------
# Remote training function
# ---------------------------------------------------------------------------

@app.function(
    image=training_image,
    gpu=DEFAULT_GPU,
    volumes={VOLUME_MOUNT: data_volume},
    timeout=18000,   # 5 hours — enough for 100 epochs on T4
    memory=16384,
)
def run_training(
    base_model:  str  = DEFAULT_BASE_MODEL,
    epochs:      int  = DEFAULT_EPOCHS,
    imgsz:       int  = DEFAULT_IMGSZ,
    batch:       int  = DEFAULT_BATCH,
    patience:    int  = DEFAULT_PATIENCE,
    resume:      bool = False,
) -> dict:
    """
    Fine-tune YOLO on the dataset stored in the Modal volume.

    Parameters
    ----------
    base_model : str
        Starting weights. Either a model name (downloaded from Ultralytics hub,
        e.g. ``"yolov8m.pt"``) or a path inside the volume (e.g.
        ``"/vol/weights/last.pt"`` to resume from a previous run).
    epochs : int
        Maximum training epochs.
    imgsz : int
        Training image size (square). 1280 matches inference resolution.
    batch : int
        Batch size per GPU. T4: 8; A10G: 16.
    patience : int
        Early-stopping patience (epochs without mAP improvement).
    resume : bool
        If True, resume from ``/vol/weights/last.pt`` instead of ``base_model``.

    Returns
    -------
    dict
        Final metrics from the best checkpoint.
    """
    import shutil
    from ultralytics import YOLO

    weights_path = Path(WEIGHTS_DIR)
    weights_path.mkdir(parents=True, exist_ok=True)

    data_yaml = Path(DATASET_DIR) / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(
            f"Dataset not found at {data_yaml}. "
            "Upload with: modal volume put yolo-training data/yolo/dataset dataset"
        )

    # Pick starting weights
    if resume and (weights_path / "last.pt").exists():
        model_source = str(weights_path / "last.pt")
        print(f"[modal_train_yolo] Resuming from {model_source}")
    else:
        model_source = base_model
        print(f"[modal_train_yolo] Starting from {model_source}")

    model = YOLO(model_source)

    print(f"[modal_train_yolo] Dataset : {data_yaml}")
    print(f"[modal_train_yolo] Epochs  : {epochs}  imgsz={imgsz}  batch={batch}")

    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        patience=patience,
        device=0,                       # GPU 0
        project="/tmp/yolo_run",
        name="train",
        exist_ok=True,
        val=True,
        save=True,
        save_period=10,                 # checkpoint every 10 epochs
        # Hard-negative aware augmentation
        mosaic=1.0,
        mixup=0.1,
        copy_paste=0.1,
        # Recommended for CCTV / small objects
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        scale=0.5,
        fliplr=0.5,
    )

    run_dir = Path("/tmp/yolo_run/train")

    # Copy best.pt and last.pt to the persistent volume
    for fname in ("best.pt", "last.pt", "results.csv", "args.yaml"):
        src = run_dir / "weights" / fname if fname.endswith(".pt") else run_dir / fname
        if src.exists():
            shutil.copy(src, weights_path / fname)
            print(f"[modal_train_yolo] Saved {fname} → {weights_path / fname}")

    data_volume.commit()

    metrics = {
        "mAP50":    float(results.results_dict.get("metrics/mAP50(B)",    0)),
        "mAP50_95": float(results.results_dict.get("metrics/mAP50-95(B)", 0)),
        "precision":float(results.results_dict.get("metrics/precision(B)",0)),
        "recall":   float(results.results_dict.get("metrics/recall(B)",   0)),
    }

    print("\n[modal_train_yolo] ── Final metrics ──────────────────────────")
    for k, v in metrics.items():
        print(f"  {k:<14} {v:.4f}")
    print("[modal_train_yolo] ────────────────────────────────────────────")

    return metrics


# ---------------------------------------------------------------------------
# Local entrypoint
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main(
    base_model: str  = DEFAULT_BASE_MODEL,
    epochs:     int  = DEFAULT_EPOCHS,
    imgsz:      int  = DEFAULT_IMGSZ,
    batch:      int  = DEFAULT_BATCH,
    patience:   int  = DEFAULT_PATIENCE,
    gpu:        str  = DEFAULT_GPU,
    resume:     bool = False,
) -> None:
    """
    Submit a YOLO fine-tuning job to Modal.

    Usage::

        modal run training/modal_train_yolo.py
        modal run training/modal_train_yolo.py --epochs 50 --gpu A10G
        modal run training/modal_train_yolo.py --resume
        modal run training/modal_train_yolo.py --base_model /vol/weights/last.pt
    """
    if not Path("~/.modal.toml").expanduser().exists() \
            and not os.environ.get("MODAL_TOKEN_ID"):
        print(
            "ERROR: Modal credentials not found.\n"
            "Run `modal token new` to authenticate, or set:\n"
            "  export MODAL_TOKEN_ID=...\n"
            "  export MODAL_TOKEN_SECRET=..."
        )
        sys.exit(1)

    # Override GPU at call time by patching the function decorator dynamically
    fn = run_training.with_options(gpu=gpu)

    print(f"Submitting YOLO fine-tune job to Modal (GPU={gpu}) ...")
    print(f"  base_model={base_model}  epochs={epochs}  imgsz={imgsz}  batch={batch}")

    metrics = fn.remote(
        base_model=base_model,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        patience=patience,
        resume=resume,
    )

    print("\nTraining complete.")
    print(f"  mAP@50        : {metrics['mAP50']:.4f}")
    print(f"  mAP@50-95     : {metrics['mAP50_95']:.4f}")
    print(f"  Precision     : {metrics['precision']:.4f}")
    print(f"  Recall        : {metrics['recall']:.4f}")
    print(f"\nDownload best weights with:")
    print(f"  modal volume get {VOLUME_NAME} weights/best.pt models/yolo_finetuned/best.pt")
