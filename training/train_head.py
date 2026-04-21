"""
train_head.py
=============
Train the Dense MLP classification head on pre-extracted YAMNet embeddings.

Reads train/val splits from ``data/processed/splits/``, trains with early
stopping and class-weight correction for the ~1:4 gunshot/not_gunshot
imbalance, and saves the best model weights to ``models/saved_weights/``.
A per-run JSON with full metrics and training history is written to
``experiments/runs/``.

Usage
-----
    # All defaults (recommended first run)
    python -m training.train_head

    # Custom hyperparameters
    python -m training.train_head --epochs 150 --batch_size 64 --patience 15

Prerequisites
-------------
1. Run ``pipeline/extract_embeddings.py`` to produce embeddings.
2. Run ``pipeline/split_dataset.py`` to produce train/val/test splits.
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf

from models.head_dense import build_dense_head, get_model_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SPLITS_DIR_DEFAULT = Path("data/processed/splits")
OUTPUT_DIR_DEFAULT = Path("models/saved_weights")
RUNS_DIR_DEFAULT = Path("experiments/runs")
WEIGHTS_FILENAME = "dense_head_best.keras"
DEFAULT_EPOCHS: int = 100
DEFAULT_BATCH_SIZE: int = 32
DEFAULT_PATIENCE: int = 10
DEFAULT_LR: float = 3e-4
DEFAULT_DROPOUT: float = 0.3
DEFAULT_UNITS: int = 256
DEFAULT_THRESHOLD: float = 0.5
RANDOM_SEED: int = 42


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_splits(
    splits_dir: Path,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load train and validation splits from ``splits_dir``.

    Parameters
    ----------
    splits_dir : Path
        Directory produced by ``pipeline/split_dataset.py``. Must contain
        ``X_train.npy``, ``y_train.npy``, ``X_val.npy``, ``y_val.npy``.

    Returns
    -------
    X_train, y_train, X_val, y_val : np.ndarray
        Float32 arrays. X arrays have shape ``(N, 1024)``,
        y arrays have shape ``(N,)`` with values in ``{0.0, 1.0}``.
    """
    if not splits_dir.is_dir():
        logger.error(
            "Splits directory not found: '%s'. "
            "Run pipeline/split_dataset.py first.",
            splits_dir,
        )
        sys.exit(1)

    required = ["X_train.npy", "y_train.npy", "X_val.npy", "y_val.npy"]
    for name in required:
        if not (splits_dir / name).exists():
            logger.error(
                "Missing file: '%s'. Re-run pipeline/split_dataset.py.",
                splits_dir / name,
            )
            sys.exit(1)

    X_train = np.load(splits_dir / "X_train.npy")
    y_train = np.load(splits_dir / "y_train.npy")
    X_val = np.load(splits_dir / "X_val.npy")
    y_val = np.load(splits_dir / "y_val.npy")

    # Shape validation
    for name, X, y in [("train", X_train, y_train), ("val", X_val, y_val)]:
        if X.ndim != 2:
            raise ValueError(f"X_{name} must be 2-D, got shape {X.shape}")
        if X.shape[1] != 1024:
            logger.warning("X_%s second dim is %d (expected 1024)", name, X.shape[1])
        if y.ndim != 1:
            raise ValueError(f"y_{name} must be 1-D, got shape {y.shape}")
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"X_{name} and y_{name} length mismatch: "
                f"{X.shape[0]} vs {y.shape[0]}"
            )

    logger.info(
        "Loaded splits — train: %d samples, val: %d samples",
        len(X_train), len(X_val),
    )
    logger.info(
        "  train: %d gunshot, %d not_gunshot",
        int(np.sum(y_train == 1.0)), int(np.sum(y_train == 0.0)),
    )
    logger.info(
        "  val:   %d gunshot, %d not_gunshot",
        int(np.sum(y_val == 1.0)), int(np.sum(y_val == 0.0)),
    )

    return X_train, y_train, X_val, y_val


# ---------------------------------------------------------------------------
# Class weights
# ---------------------------------------------------------------------------


def compute_class_weights(y_train: np.ndarray) -> Dict[int, float]:
    """
    Compute balanced class weights to correct for the ~1:4 imbalance.

    Uses sklearn's ``"balanced"`` mode:
    ``weight_c = n_samples / (n_classes * count_c)``

    For 4,621 gunshot vs 19,523 not_gunshot the gunshot class gets ~4.2×
    higher weight, making its gradient contribution equal to not_gunshot.

    Parameters
    ----------
    y_train : np.ndarray
        1-D binary label array from the training split.

    Returns
    -------
    dict
        ``{0: float, 1: float}`` — Keras ``class_weight`` argument requires
        integer keys.
    """
    classes = np.array([0, 1])
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train.astype(int),
    )
    class_weight_dict = {0: float(weights[0]), 1: float(weights[1])}
    logger.info(
        "Class weights — not_gunshot: %.4f, gunshot: %.4f",
        class_weight_dict[0], class_weight_dict[1],
    )
    return class_weight_dict


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------


def build_callbacks(output_dir: Path, patience: int) -> list:
    """
    Build the standard training callbacks.

    - **EarlyStopping** — stops when ``val_loss`` stops improving and
      restores the best weights so post-training evaluation uses the optimal
      checkpoint automatically.
    - **ModelCheckpoint** — saves the best model to disk independently (belt
      and suspenders).
    - **ReduceLROnPlateau** — halves the learning rate after
      ``patience // 2`` non-improving epochs, giving the optimizer a chance
      to escape a plateau before early stopping fires.

    Parameters
    ----------
    output_dir : Path
        Directory where ``dense_head_best.keras`` will be saved.
    patience : int
        EarlyStopping patience. ReduceLROnPlateau uses ``patience // 2``.

    Returns
    -------
    list
        Three Keras callback objects.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=patience,
        restore_best_weights=True,
        verbose=1,
    )
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=str(output_dir / WEIGHTS_FILENAME),
        monitor="val_loss",
        save_best_only=True,
        verbose=1,
    )
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=max(1, patience // 2),
        min_lr=1e-7,
        verbose=1,
    )
    return [early_stop, checkpoint, reduce_lr]


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate_on_val(
    model: tf.keras.Model,
    X_val: np.ndarray,
    y_val: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Compute exact validation metrics using sklearn on the full val set.

    Keras streaming metrics are approximations; sklearn on the full dataset
    is authoritative for the final report.

    Parameters
    ----------
    model : tf.keras.Model
        Trained model (with best weights already restored by EarlyStopping).
    X_val : np.ndarray
        Validation embeddings, shape ``(N_val, 1024)``.
    y_val : np.ndarray
        Ground-truth labels, shape ``(N_val,)``.
    threshold : float
        Decision threshold applied to sigmoid output. Default: 0.5.

    Returns
    -------
    dict
        Keys: accuracy, f1, precision, recall, auc_roc — all Python floats.
    """
    y_prob = model.predict(X_val, verbose=0).squeeze()
    y_pred = (y_prob >= threshold).astype(int)
    y_true = y_val.astype(int)

    try:
        auc_roc = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        logger.warning(
            "roc_auc_score failed (all predictions same class). "
            "Returning auc_roc=0.5 (random baseline)."
        )
        auc_roc = 0.5

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "auc_roc": auc_roc,
    }


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def save_run_json(
    runs_dir: Path,
    timestamp: str,
    config: dict,
    class_weights: Dict[int, float],
    n_train: int,
    n_val: int,
    n_train_gunshot: int,
    n_train_not_gunshot: int,
    history: tf.keras.callbacks.History,
    metrics: Dict[str, float],
    best_epoch: int,
    weights_path: str,
) -> Path:
    """
    Write the full per-run record to ``runs_dir/run_{timestamp}.json``.

    Parameters
    ----------
    runs_dir : Path
        Destination directory for run JSON files.
    timestamp : str
        UTC timestamp string (e.g. ``"20260419T143022Z"``).
    config : dict
        Model + training hyperparameters from ``get_model_config()``.
    class_weights : dict
        ``{0: float, 1: float}`` as computed by ``compute_class_weights()``.
    n_train, n_val : int
        Total sample counts per split.
    n_train_gunshot, n_train_not_gunshot : int
        Per-class counts in the training split.
    history : tf.keras.callbacks.History
        Object returned by ``model.fit()``.
    metrics : dict
        Final validation metrics from ``evaluate_on_val()``.
    best_epoch : int
        1-indexed epoch with the lowest val_loss.
    weights_path : str
        Absolute or relative path to the saved ``.keras`` file.

    Returns
    -------
    Path
        Path to the written JSON file.
    """
    runs_dir.mkdir(parents=True, exist_ok=True)

    total_epochs = len(history.history["loss"])
    early_stopped = total_epochs < config.get("epochs_requested", DEFAULT_EPOCHS)

    # Convert numpy float32 lists to Python floats for JSON serialization.
    history_serializable = {
        k: [float(v) for v in vals]
        for k, vals in history.history.items()
    }

    run = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "run_id": f"run_{timestamp}",
        "config": config,
        "class_weights": {str(k): v for k, v in class_weights.items()},
        "data": {
            "n_train": n_train,
            "n_val": n_val,
            "n_train_gunshot": n_train_gunshot,
            "n_train_not_gunshot": n_train_not_gunshot,
        },
        "training": {
            "best_epoch": best_epoch,
            "total_epochs_run": total_epochs,
            "early_stopped": early_stopped,
        },
        "val_metrics": metrics,
        "history": history_serializable,
        "weights_path": weights_path,
    }

    run_path = runs_dir / f"run_{timestamp}.json"
    with open(run_path, "w", encoding="utf-8") as f:
        json.dump(run, f, indent=2)

    logger.info("Run record saved to: %s", run_path)
    return run_path


# ---------------------------------------------------------------------------
# Summary display
# ---------------------------------------------------------------------------


def print_summary_table(
    metrics: Dict[str, float],
    class_weights: Dict[int, float],
    best_epoch: int,
    threshold: float,
) -> None:
    """Print a clean ASCII summary table of the final validation metrics."""
    sep = "-" * 52
    print(sep)
    print(f"Validation Metrics  (Dense Head — best epoch: {best_epoch})")
    print(sep)
    print(f"  Threshold  :  {threshold}")
    print(f"  Accuracy   :  {metrics['accuracy']:.4f}")
    print(f"  F1 Score   :  {metrics['f1']:.4f}")
    print(f"  AUC-ROC    :  {metrics['auc_roc']:.4f}")
    print(f"  Precision  :  {metrics['precision']:.4f}")
    print(f"  Recall     :  {metrics['recall']:.4f}")
    print(sep)
    print(
        f"  Class weights — not_gunshot: {class_weights[0]:.4f}, "
        f"gunshot: {class_weights[1]:.4f}"
    )
    print(sep)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train the Dense MLP head on pre-extracted YAMNet embeddings.\n"
            "Reads train/val splits from --splits_dir.\n"
            "Saves best weights to --output_dir/dense_head_best.keras.\n"
            "Saves run JSON to --runs_dir/run_<timestamp>.json."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--splits_dir", type=Path, default=SPLITS_DIR_DEFAULT,
        help=f"Directory with X/y train/val .npy files. Default: {SPLITS_DIR_DEFAULT}",
    )
    parser.add_argument(
        "--output_dir", type=Path, default=OUTPUT_DIR_DEFAULT,
        help=f"Directory to save best model weights. Default: {OUTPUT_DIR_DEFAULT}",
    )
    parser.add_argument(
        "--runs_dir", type=Path, default=RUNS_DIR_DEFAULT,
        help=f"Directory to save per-run JSON. Default: {RUNS_DIR_DEFAULT}",
    )
    parser.add_argument(
        "--epochs", type=int, default=DEFAULT_EPOCHS,
        help=f"Maximum training epochs. Default: {DEFAULT_EPOCHS}",
    )
    parser.add_argument(
        "--batch_size", type=int, default=DEFAULT_BATCH_SIZE,
        help=f"Mini-batch size. Default: {DEFAULT_BATCH_SIZE}",
    )
    parser.add_argument(
        "--patience", type=int, default=DEFAULT_PATIENCE,
        help=f"EarlyStopping patience (val_loss). Default: {DEFAULT_PATIENCE}",
    )
    parser.add_argument(
        "--lr", type=float, default=DEFAULT_LR,
        help=f"Adam learning rate. Default: {DEFAULT_LR}",
    )
    parser.add_argument(
        "--dropout", type=float, default=DEFAULT_DROPOUT,
        help=f"Dropout rate. Default: {DEFAULT_DROPOUT}",
    )
    parser.add_argument(
        "--units", type=int, default=DEFAULT_UNITS,
        help=f"Units in the hidden Dense layer. Default: {DEFAULT_UNITS}",
    )
    parser.add_argument(
        "--threshold", type=float, default=DEFAULT_THRESHOLD,
        help=f"Decision threshold on sigmoid output for val evaluation. Default: {DEFAULT_THRESHOLD}",
    )
    parser.add_argument(
        "--class_weight_gunshot", type=float, default=None,
        help="Manual gunshot class weight override. Default: auto-computed (balanced).",
    )

    args = parser.parse_args()

    tf.random.set_seed(RANDOM_SEED)

    # Load data.
    X_train, y_train, X_val, y_val = load_splits(args.splits_dir)

    n_train = len(X_train)
    n_val = len(X_val)
    n_train_gunshot = int(np.sum(y_train == 1.0))
    n_train_not_gunshot = int(np.sum(y_train == 0.0))

    # Class weights for imbalance correction — manual override for recall optimisation.
    class_weights = compute_class_weights(y_train)
    if args.class_weight_gunshot is not None:
        class_weights[1] = args.class_weight_gunshot
        logger.info("Overriding gunshot class weight → %.4f", class_weights[1])

    # Timestamp for this run.
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    # Build model.
    model = build_dense_head(
        units=args.units,
        dropout_rate=args.dropout,
        learning_rate=args.lr,
    )
    model.summary(print_fn=logger.info)

    # Config snapshot (includes training hyperparameters for the run JSON).
    config = get_model_config(
        units=args.units,
        dropout_rate=args.dropout,
        learning_rate=args.lr,
    )
    config.update({
        "epochs_requested": args.epochs,
        "batch_size": args.batch_size,
        "patience": args.patience,
        "threshold": args.threshold,
        "class_weight_gunshot_override": args.class_weight_gunshot,
        "splits_dir": str(args.splits_dir),
        "output_dir": str(args.output_dir),
    })

    # Callbacks.
    callbacks = build_callbacks(args.output_dir, args.patience)

    # Train.
    logger.info("Starting training ...")
    try:
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=args.epochs,
            batch_size=args.batch_size,
            class_weight=class_weights,
            callbacks=callbacks,
            verbose=1,
        )
    except Exception as exc:
        logger.error("Training failed: %s", exc)
        sys.exit(1)

    best_epoch = int(np.argmin(history.history["val_loss"])) + 1

    # Final evaluation on validation set (exact sklearn metrics).
    metrics = evaluate_on_val(model, X_val, y_val, threshold=args.threshold)

    # Save run record.
    weights_path = str(args.output_dir / WEIGHTS_FILENAME)
    save_run_json(
        runs_dir=args.runs_dir,
        timestamp=timestamp,
        config=config,
        class_weights=class_weights,
        n_train=n_train,
        n_val=n_val,
        n_train_gunshot=n_train_gunshot,
        n_train_not_gunshot=n_train_not_gunshot,
        history=history,
        metrics=metrics,
        best_epoch=best_epoch,
        weights_path=weights_path,
    )

    print_summary_table(metrics, class_weights, best_epoch, threshold=args.threshold)
    logger.info("Best weights saved to: %s", weights_path)


if __name__ == "__main__":
    main()
