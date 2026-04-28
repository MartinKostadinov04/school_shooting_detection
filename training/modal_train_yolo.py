"""
modal_train_yolo.py
===================
Fine-tune YOLOv8m on a firearm-detection dataset using Modal cloud GPU (A100-80GB).

Setup (one-time)
----------------
1. Install Modal and authenticate::

       pip install modal
       modal token new

2. Create the persistent volume::

       modal volume create vision-train

3. Upload your merged dataset (YOLO format)::

       modal volume put --force vision-train data/vision_yolo/images yolo/
       modal volume put --force vision-train data/vision_yolo/labels yolo/
       modal volume put --force vision-train data/vision_yolo/data.yaml yolo/

   The volume must contain::

       /vol/yolo/
           data.yaml
           images/train/   images/val/
           labels/train/   labels/val/

Run
---
**Always use ``--detach`` for full training runs.** Without it, closing your
terminal / sleeping your laptop / dropping VPN tears down the Modal app
mid-epoch (you'll see ``Stopping app - local client disconnected`` in the
logs followed by a benign DataLoader teardown traceback).

::

    # Recommended — survives client disconnects (~4-5 h on 2× A100-80GB)
    modal run --detach training/modal_train_yolo.py

    # Tail logs of a detached run
    modal app logs yolo-finetune

    # Resume after a crash / preemption / explicit stop
    modal run --detach training/modal_train_yolo.py --resume

    # Other options
    modal run --detach training/modal_train_yolo.py --epochs 50
    modal run --detach training/modal_train_yolo.py --imgsz 1280 --batch 32

Download weights
----------------
::

    modal volume get vision-train weights/best.pt  models/yolo_finetuned/best.pt
    modal volume get vision-train weights/results.csv experiments/plots/modal_results.csv
"""

import os
import sys
from pathlib import Path

import modal

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

APP_NAME     = "yolo-finetune"
VOLUME_NAME  = "vision-train"           # modal volume ls → vision-train
VOLUME_MOUNT = "/vol"

DATASET_DIR  = f"{VOLUME_MOUNT}/yolo"  # vision_yolo data uploaded to yolo/
WEIGHTS_DIR  = f"{VOLUME_MOUNT}/weights"

DEFAULT_BASE_MODEL = "yolov8m.pt"
DEFAULT_EPOCHS     = 100
DEFAULT_IMGSZ      = 1280
# Effective batch 64 across 2× A100-80GB (32 per GPU). The first attempt at
# 2× A100-40GB OOMed: a per-rank batch of 32 at imgsz=1280 with mosaic+mixup
# needs ~75 GB of VRAM (observed empirically), so 40 GB cards cannot hold it.
# Single-GPU fallback for 80GB also auto-reduces 64→32 with grad-accum (nbs=64).
DEFAULT_BATCH      = 64
DEFAULT_GPU        = "A100-80GB:2"   # 2× A100-80GB, DDP (device=[0,1])
DEFAULT_PATIENCE   = 30     # large batch = fewer steps/epoch, needs more epochs to plateau

# ---------------------------------------------------------------------------
# Modal app + image
# ---------------------------------------------------------------------------

app = modal.App(APP_NAME)

training_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(["libglib2.0-0", "libsm6", "libxext6", "ffmpeg"])
    .pip_install(
        "ultralytics>=8.3.0",
        "torch>=2.2.0",
        "torchvision>=0.17.0",
        "numpy>=1.24.0",
        "pyyaml>=6.0",
        "pillow>=10.0.0",
    )
    # ultralytics pulls in opencv-python (GUI build) as a transitive dep, which
    # requires libGL.so.1. Force-reinstall the headless build afterward so cv2
    # never tries to load any OpenGL/display libraries.
    .run_commands("pip install 'opencv-python-headless>=4.8.0' --force-reinstall --quiet")
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
    timeout=86400,   # 24 hours — 2× A100-80GB DDP completes 100 epochs in ~4-5 hours
    # Ultralytics DDP spawns one process per rank; each rank's `cache="ram"`
    # holds its own copy of the dataset. Empirically the train+val cache is
    # ~63 GB per rank on this dataset, so 2 ranks need 2 × 63 = ~126 GB just
    # for image caches, plus model copies, dataloader workers, and OS overhead.
    memory=196608,   # 192 GB RAM: ~126 GB for caches + headroom for workers / scratch
)
def run_training(
    base_model:  str  = DEFAULT_BASE_MODEL,
    epochs:      int  = DEFAULT_EPOCHS,
    imgsz:       int  = DEFAULT_IMGSZ,
    batch:       int  = DEFAULT_BATCH,
    patience:    int  = DEFAULT_PATIENCE,
    # ``resume`` defaults to True so that Modal preemption auto-restarts
    # pick up at the last checkpoint instead of clobbering the run dir.
    # When no checkpoint exists on the volume, the resume logic below
    # transparently falls through to a fresh start from ``base_model`` —
    # so the default is safe for a first-time run too.
    resume:      bool = True,
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
        Training image size. Default 1280 for A100; use 640 for T4.
    batch : int
        Total batch size across all GPUs (Ultralytics splits it evenly per rank).
        2×A100-80GB@1280 → 64 (32/GPU), 1×A100-80GB@1280 → 64 (auto-grad-accum).
        Note: A100-40GB cannot hold the per-rank activations at imgsz=1280.
    patience : int
        Early-stopping patience in epochs without mAP improvement.
    resume : bool
        Resume training from a previous run on the volume. Defaults to True
        so Modal preemption-triggered auto-restarts pick up at the last
        checkpoint instead of clobbering ``run/`` (which Ultralytics will do
        on a fresh start because ``exist_ok=True``).

        Three-tier lookup:

        1. **True resume** — if ``/vol/weights/run/weights/last.pt`` AND
           ``/vol/weights/run/args.yaml`` exist, Ultralytics restores the
           epoch counter, optimizer state, and LR schedule, picking up
           exactly where the previous run stopped.
        2. **Warm-start** — otherwise, if ``/vol/weights/last.pt`` (the
           mirrored alias) exists, weights are loaded but the trainer
           starts from epoch 0 with a fresh schedule.
        3. Else (no checkpoint anywhere) falls through to ``base_model`` —
           making the True default safe for first-time runs as well.

        Pass ``--no-resume`` on the CLI to force a fresh start over an
        existing run dir (it WILL be wiped — that is intentional).
    """
    import shutil
    import torch
    from ultralytics import YOLO

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

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
            f"  modal volume put {VOLUME_NAME} data/vision_yolo/images yolo/\n"
            f"  modal volume put {VOLUME_NAME} data/vision_yolo/labels yolo/\n"
            f"  modal volume put {VOLUME_NAME} data/vision_yolo/data.yaml yolo/\n"
        )

    # Rewrite paths in data.yaml to absolute volume paths
    data_yaml = _fix_data_yaml(data_yaml)

    # ── Checkpoint layout ──────────────────────────────────────────────────
    # Ultralytics writes during training to:
    #   /vol/weights/run/weights/last.pt   (overwritten every epoch — full optimizer state)
    #   /vol/weights/run/weights/best.pt   (overwritten when mAP improves)
    #   /vol/weights/run/weights/epoch{N}.pt   (every save_period epochs — for rollback)
    #   /vol/weights/run/args.yaml         (trainer config — REQUIRED for resume=True)
    #
    # We mirror last.pt / best.pt up to /vol/weights/{last,best}.pt as well-known
    # aliases, so any inference/eval script can find them at a stable path even
    # while training is still running or if the run was killed before the
    # post-training copy block ran.
    # ───────────────────────────────────────────────────────────────────────
    vol_run_dir       = weights_path / "run"
    vol_run_dir.mkdir(parents=True, exist_ok=True)
    run_weights_dir   = vol_run_dir / "weights"
    run_last_pt       = run_weights_dir / "last.pt"
    run_best_pt       = run_weights_dir / "best.pt"
    alias_last_pt     = weights_path / "last.pt"
    alias_best_pt     = weights_path / "best.pt"

    # ── Pick starting weights and whether to pass resume=True ──────────────
    # True resume needs `last.pt` to live next to its sibling `args.yaml` so
    # Ultralytics can rebuild epoch counter / optimizer / LR schedule. That
    # only holds for the run-dir copy; the alias copy is warm-start only.
    if resume and run_last_pt.exists() and (vol_run_dir / "args.yaml").exists():
        model_source   = str(run_last_pt)
        trainer_resume = True
        print(f"[modal_train_yolo] Resuming run state from {model_source}")
    elif resume and alias_last_pt.exists():
        model_source   = str(alias_last_pt)
        trainer_resume = False  # warm-start: trainer state lost, schedule restarts from epoch 0
        print(f"[modal_train_yolo] Warm-starting from alias {model_source}")
        print(f"[modal_train_yolo]   (no run/args.yaml found — optimizer/LR schedule will restart)")
    else:
        model_source   = base_model
        trainer_resume = False
        print(f"[modal_train_yolo] Starting fresh from {model_source}")

    model = YOLO(model_source)

    print(f"[modal_train_yolo] Dataset  : {data_yaml}")
    print(f"[modal_train_yolo] Epochs   : {epochs}  imgsz={imgsz}  batch={batch}  patience={patience}")

    # ── Alias-refresh helper ───────────────────────────────────────────────
    # Mirrors the live last.pt/best.pt up to /vol/weights/{last,best}.pt.
    # Atomic: torch.save uses temp-file + rename, so reads always see a
    # complete checkpoint. Failures are logged but never raise — checkpoint
    # mirroring must never crash training.
    def _refresh_aliases():
        for src, dst in ((run_last_pt, alias_last_pt), (run_best_pt, alias_best_pt)):
            if not src.exists():
                continue
            try:
                shutil.copy2(src, dst)
            except Exception as exc:
                print(f"[modal_train_yolo] alias refresh failed for {src.name}: {exc}")

    # ── Volume commit + alias mirroring ────────────────────────────────────
    # In DDP mode Ultralytics spawns torchrun subprocesses; callbacks registered
    # on the parent model object are NOT transferred to those worker processes.
    # The background thread runs in the parent (blocked on subprocess.wait),
    # so it covers DDP runs. The on_fit_epoch_end callback covers single-GPU
    # runs. Both refresh aliases before committing so that every commit
    # captures the latest mirrored last.pt / best.pt in one durable snapshot.
    import threading
    _commit_stop = threading.Event()

    def _bg_commit():
        while not _commit_stop.wait(timeout=300):
            _refresh_aliases()
            print("[modal_train_yolo] Background commit (aliases refreshed) …")
            data_volume.commit()

    _commit_thread = threading.Thread(target=_bg_commit, daemon=True)
    _commit_thread.start()

    def _commit_after_checkpoint(trainer):
        if int(os.environ.get("LOCAL_RANK", 0)) != 0:
            return
        epoch = trainer.epoch + 1
        if epoch % 5 != 0:
            return
        _refresh_aliases()
        print(f"[modal_train_yolo] Committing volume after epoch {epoch} (aliases refreshed) …")
        data_volume.commit()

    model.add_callback("on_fit_epoch_end", _commit_after_checkpoint)

    # DDP across 2× A100-80GB. Ultralytics splits `batch` evenly per rank
    # (batch=64 → 32 images/GPU, ≈75 GB VRAM/rank — fits 80 GB comfortably).
    # Pass a list of device indices to enable DDP; if only 1 GPU is allocated,
    # fall back to single-device training (Ultralytics will auto-grad-accum).
    n_gpu = torch.cuda.device_count()
    device = list(range(n_gpu)) if n_gpu > 1 else 0
    print(f"[modal_train_yolo] Devices  : {device}  (n_gpu={n_gpu})")

    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        patience=patience,
        resume=trainer_resume,
        device=device,
        workers=8,            # per-rank dataloader workers — saturates CPU pool during mosaic/mixup at 1280px
        cache="ram",
        # ── Optimizer: paper-canonical YOLO transfer-learning recipe ──
        # Pin SGD explicitly — `optimizer="auto"` silently discards lr0/momentum.
        # These values are calibrated for `nbs=64`, which matches our effective
        # batch (32/GPU × 2 ranks), so no linear-scaling-rule adjustment is needed.
        optimizer="SGD",
        lr0=0.01,
        momentum=0.937,
        weight_decay=5e-4,
        warmup_epochs=5,
        nbs=64,               # explicit: nominal batch == effective batch → no implicit grad accumulation
        # ── Speed levers that do NOT trade mAP ──
        amp=True,             # FP16 autocast forward, FP32 master weights — ~1.5–2× step speedup, neutral mAP
        cos_lr=True,          # cosine LR decay — converges to a slightly better minimum in fewer effective steps
        close_mosaic=10,      # canonical YOLO recipe: disable mosaic in last 10 epochs (faster + better fine-tune)
        deterministic=False,  # let cuDNN benchmark pick fastest non-deterministic kernels (~3–5% step speedup)
        plots=False,          # skip per-epoch plot rendering; final plots are still emitted at training end
        save_json=False,      # skip COCO-JSON export every validation (only ever need it at the end)
        project=str(weights_path),
        name="run",
        exist_ok=True,
        val=True,
        save=True,
        save_period=5,        # checkpoint every 5 epochs for tight recovery window (cheap, ~50 MB each)
        # ── Augmentation: tuned for fixed-angle CCTV, small occluded handguns ──
        # Paper refs: Olmos 2018 (hard negatives), Hnoohom 2022 (multi-scale),
        #             Yellapragada 2023 CCTV-Gun (cross-dataset collapse)
        mosaic=1.0,       # multi-image composition — critical for small objects
        mixup=0.1,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=5.0,      # slight rotation — CCTV cameras have fixed but imperfect angles
        translate=0.1,
        scale=0.5,        # scale jitter handles varying gun-to-frame-size ratios
        fliplr=0.5,
        # copy_paste requires segmentation masks — omitted
        # SAHI 2×2 tiling applied at inference time, not here
        label_smoothing=0.0,
    )

    _commit_stop.set()
    _commit_thread.join(timeout=10)

    # Final alias refresh + ancillary file copy. last.pt / best.pt have been
    # kept in sync throughout training; this is the post-training catch-up.
    _refresh_aliases()
    for src, dst in ((run_last_pt, alias_last_pt), (run_best_pt, alias_best_pt)):
        if dst.exists():
            print(f"[modal_train_yolo] Final alias  {dst}")

    for fname in ("results.csv", "args.yaml"):
        src = vol_run_dir / fname
        if src.exists():
            shutil.copy2(src, weights_path / fname)
            print(f"[modal_train_yolo] Saved        {weights_path / fname}")

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
    # See run_training(): default True so Modal preemption restarts auto-recover
    # from the last checkpoint. Pass --no-resume on the CLI to force a fresh start.
    resume:     bool = True,
) -> None:
    """
    Submit a YOLO fine-tuning job to Modal.

    Usage (recommended — survives terminal/network disconnects)::

        modal run --detach training/modal_train_yolo.py
        modal run --detach training/modal_train_yolo.py --resume
        modal run --detach training/modal_train_yolo.py --epochs 50
        modal run --detach training/modal_train_yolo.py --base_model /vol/weights/last.pt

        modal app logs yolo-finetune          # tail a detached run
        modal app stop yolo-finetune          # cancel a detached run

    To use a different GPU, change DEFAULT_GPU and DEFAULT_BATCH at the top:
        2× A100-80GB    : imgsz=1280 batch=64  ← default (DDP, ~4-5 h, ~$22)
        1× A100-80GB    : imgsz=1280 batch=64  (auto-reduces to 32, ~7-9 h, ~$15)
        2× A100-40GB    : DOES NOT FIT — per-rank activations need >40 GB at imgsz=1280
        1× A10G   (24GB): imgsz=640  batch=16  (significant accuracy hit, not recommended)
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
    print(f"  dataset=/vol/yolo  weights → /vol/weights/")
    print(f"")
    print(f"  TIP: this run takes ~4-5 h on 2x A100-80GB. If you did NOT")
    print(f"       launch with `modal run --detach ...`, a client disconnect")
    print(f"       will kill the job. To recover, re-launch with `--detach")
    print(f"       --resume` and training will pick up at the last checkpoint.")

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
