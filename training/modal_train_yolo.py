"""
modal_train_yolo.py
===================
Fine-tune YOLOv8m on a firearm-detection dataset using Modal cloud GPU.

Setup (one-time)
----------------
1. Install Modal and authenticate::

       pip install modal
       modal token new

2. Create the persistent volume::

       modal volume create yolo-training

3. Upload your merged dataset (YOLO format) to the vision-train folder::

       modal volume put yolo-training <local_dataset_dir>/  vision-train/

   The volume must contain::

       /vol/vision-train/
           data.yaml
           images/train/   images/val/
           labels/train/   labels/val/

Run
---
::

    modal run training/modal_train_yolo.py
    modal run training/modal_train_yolo.py --resume
    modal run training/modal_train_yolo.py --epochs 50 --gpu A10G
    modal run training/modal_train_yolo.py --imgsz 1280 --batch 4

Download weights
----------------
::

    modal volume get yolo-training weights/best.pt  models/yolo_finetuned/best.pt
    modal volume get yolo-training weights/results.csv experiments/plots/modal_results.csv
"""

import os
import sys
from pathlib import Path

import modal

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

APP_NAME     = "yolo-finetune"
VOLUME_NAME  = "yolo-training"
VOLUME_MOUNT = "/vol"

DATASET_DIR  = f"{VOLUME_MOUNT}/vision-train"   # matches the upload folder name
WEIGHTS_DIR  = f"{VOLUME_MOUNT}/weights"

DEFAULT_BASE_MODEL = "yolov8m.pt"
DEFAULT_EPOCHS     = 100
DEFAULT_IMGSZ      = 640    # matches original training resolution; use 1280 + batch 4 on A10G
DEFAULT_BATCH      = 16     # safe for T4 16GB at imgsz=640; drop to 4 for imgsz=1280
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
    )
)

data_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fix_data_yaml(yaml_path: Path) -> Path:
    """
    Rewrite the data.yaml so all paths are absolute inside the volume.

    Roboflow exports set `path:` relative to wherever the file was exported.
    When uploaded to Modal, those relative paths break. This rewrites them
    to absolute paths anchored at the yaml file's parent directory.
    """
    import yaml

    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)

    dataset_root = str(yaml_path.parent)
    cfg["path"] = dataset_root

    # Ensure train/val/test are relative strings (ultralytics joins path + split)
    for split in ("train", "val", "test"):
        if split in cfg and cfg[split] and Path(cfg[split]).is_absolute():
            # Strip the old absolute prefix and keep only the relative part
            cfg[split] = str(Path(cfg[split]).relative_to(Path(cfg[split]).anchor))

    fixed_path = yaml_path.parent / "data_fixed.yaml"
    with open(fixed_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    print(f"[modal_train_yolo] data.yaml rewritten → {fixed_path}")
    print(f"[modal_train_yolo]   path  : {cfg['path']}")
    print(f"[modal_train_yolo]   train : {cfg.get('train')}")
    print(f"[modal_train_yolo]   val   : {cfg.get('val')}")
    print(f"[modal_train_yolo]   nc    : {cfg.get('nc')}  names: {cfg.get('names')}")
    return fixed_path


def _extract_metrics(results) -> dict:
    """
    Pull final metrics from an ultralytics Results object robustly.
    Key names vary slightly across ultralytics versions.
    """
    rd = getattr(results, "results_dict", {}) or {}

    def _get(*keys: str) -> float:
        for k in keys:
            if k in rd:
                return float(rd[k])
        return 0.0

    return {
        "mAP50":    _get("metrics/mAP50(B)",    "mAP_0.5"),
        "mAP50_95": _get("metrics/mAP50-95(B)", "mAP_0.5:0.95"),
        "precision":_get("metrics/precision(B)", "precision"),
        "recall":   _get("metrics/recall(B)",    "recall"),
    }


# ---------------------------------------------------------------------------
# Remote training function
# ---------------------------------------------------------------------------

@app.function(
    image=training_image,
    gpu=DEFAULT_GPU,
    volumes={VOLUME_MOUNT: data_volume},
    timeout=18000,   # 5 hours — enough for 100 epochs on T4
    memory=32768,    # 32 GB RAM; YOLO + dataloader workers are memory hungry
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
    Fine-tune YOLO on the dataset stored at /vol/vision-train in the Modal volume.

    Parameters
    ----------
    base_model : str
        Starting weights — Ultralytics hub name (e.g. ``"yolov8m.pt"``) or an
        absolute path inside the volume (e.g. ``"/vol/weights/last.pt"``).
    epochs : int
        Maximum training epochs (early stopping via patience).
    imgsz : int
        Training image size. 640 for T4 batch=16; 1280 for A10G batch=4.
    batch : int
        Batch size. Reduce if OOM: T4@640→16, T4@1280→4, A10G@640→32.
    patience : int
        Early-stopping patience in epochs without mAP improvement.
    resume : bool
        Resume from /vol/weights/last.pt if it exists.
    """
    import shutil
    import torch
    from ultralytics import YOLO

    print(f"[modal_train_yolo] PyTorch  : {torch.__version__}")
    print(f"[modal_train_yolo] CUDA     : {torch.cuda.is_available()}  "
          f"device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'}")

    weights_path = Path(WEIGHTS_DIR)
    weights_path.mkdir(parents=True, exist_ok=True)

    # Locate dataset
    dataset_root = Path(DATASET_DIR)
    data_yaml    = dataset_root / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(
            f"data.yaml not found at {data_yaml}.\n"
            f"Upload with:\n"
            f"  modal volume put {VOLUME_NAME} <local_dir>/  vision-train/"
        )

    # Rewrite paths in data.yaml to absolute volume paths
    data_yaml = _fix_data_yaml(data_yaml)

    # Pick starting weights
    if resume and (weights_path / "last.pt").exists():
        model_source = str(weights_path / "last.pt")
        print(f"[modal_train_yolo] Resuming from {model_source}")
    else:
        model_source = base_model
        print(f"[modal_train_yolo] Starting from  {model_source}")

    model = YOLO(model_source)

    print(f"[modal_train_yolo] Dataset  : {data_yaml}")
    print(f"[modal_train_yolo] Epochs   : {epochs}  imgsz={imgsz}  batch={batch}  patience={patience}")

    # Clean up any stale /tmp run to avoid exist_ok collision issues
    run_dir = Path("/tmp/yolo_run/train")
    if run_dir.exists():
        shutil.rmtree(run_dir, ignore_errors=True)

    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        patience=patience,
        device=0,
        project="/tmp/yolo_run",
        name="train",
        exist_ok=False,
        val=True,
        save=True,
        save_period=10,
        # Augmentation tuned for CCTV / small objects
        mosaic=1.0,
        mixup=0.1,
        # copy_paste requires segmentation masks — omitted for bbox-only datasets
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        scale=0.5,
        fliplr=0.5,
    )

    # Persist weights and logs to the volume
    weights_run_dir = run_dir / "weights"
    for fname in ("best.pt", "last.pt"):
        src = weights_run_dir / fname
        if src.exists():
            shutil.copy(src, weights_path / fname)
            print(f"[modal_train_yolo] Saved {fname} → {weights_path / fname}")

    for fname in ("results.csv", "args.yaml"):
        src = run_dir / fname
        if src.exists():
            shutil.copy(src, weights_path / fname)
            print(f"[modal_train_yolo] Saved {fname} → {weights_path / fname}")

    data_volume.commit()

    metrics = _extract_metrics(results)

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
    resume:     bool = False,
) -> None:
    """
    Submit a YOLO fine-tuning job to Modal.

    Usage::

        modal run training/modal_train_yolo.py
        modal run training/modal_train_yolo.py --epochs 50
        modal run training/modal_train_yolo.py --imgsz 1280 --batch 4
        modal run training/modal_train_yolo.py --resume
        modal run training/modal_train_yolo.py --base_model /vol/weights/last.pt

    To use a different GPU, change DEFAULT_GPU at the top of this file and
    adjust DEFAULT_BATCH accordingly before running:
        T4  (16GB):  imgsz=640  batch=16  or  imgsz=1280  batch=4
        A10G (24GB): imgsz=640  batch=32  or  imgsz=1280  batch=8
        A100 (40GB): imgsz=1280 batch=16
    """
    if not Path("~/.modal.toml").expanduser().exists() \
            and not os.environ.get("MODAL_TOKEN_ID"):
        print(
            "ERROR: Modal credentials not found.\n"
            "Run `modal token new` to authenticate."
        )
        sys.exit(1)

    print(f"Submitting YOLO fine-tune job to Modal ...")
    print(f"  base_model={base_model}  epochs={epochs}  imgsz={imgsz}  batch={batch}")
    print(f"  dataset=/vol/vision-train  weights → /vol/weights/")

    metrics = run_training.remote(
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
    print(f"\nDownload weights:")
    print(f"  modal volume get {VOLUME_NAME} weights/best.pt models/yolo_finetuned/best.pt")
    print(f"  modal volume get {VOLUME_NAME} weights/results.csv experiments/plots/modal_results.csv")
