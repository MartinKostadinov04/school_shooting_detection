"""
evaluate_test.py
================
Evaluate the trained Dense head on the held-out test split.

Loads the best saved weights and the test split produced by
pipeline/split_dataset.py, computes exact sklearn metrics, prints a summary
table, and appends the results to experiments/runs/test_results.json.

Usage
-----
    python -m training.evaluate_test

    # Custom paths
    python -m training.evaluate_test \\
        --weights  models/saved_weights/dense_head_best.keras \\
        --splits_dir data/processed/splits \\
        --runs_dir experiments/runs
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
)
import tensorflow as tf

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

WEIGHTS_DEFAULT = Path("models/saved_weights/dense_head_best.keras")
SPLITS_DIR_DEFAULT = Path("data/processed/splits")
RUNS_DIR_DEFAULT = Path("experiments/runs")
TEST_RESULTS_FILE = "test_results.json"
THRESHOLD = 0.5


def load_test_split(splits_dir: Path):
    for name in ("X_test.npy", "y_test.npy"):
        if not (splits_dir / name).exists():
            logger.error("Missing: %s", splits_dir / name)
            sys.exit(1)
    X_test = np.load(splits_dir / "X_test.npy")
    y_test = np.load(splits_dir / "y_test.npy")
    logger.info(
        "Test split — %d samples (%d gunshot, %d not_gunshot)",
        len(y_test), int(np.sum(y_test == 1.0)), int(np.sum(y_test == 0.0)),
    )
    return X_test, y_test


def compute_metrics(model, X_test, y_test, threshold=THRESHOLD):
    y_prob = model.predict(X_test, verbose=0).squeeze()
    y_pred = (y_prob >= threshold).astype(int)
    y_true = y_test.astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    return {
        "accuracy":  float(accuracy_score(y_true, y_pred)),
        "f1":        float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":    float(recall_score(y_true, y_pred, zero_division=0)),
        "auc_roc":   float(roc_auc_score(y_true, y_prob)),
        "threshold": threshold,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def print_summary(metrics, n_test):
    cm = metrics["confusion_matrix"]
    sep = "-" * 52
    print(sep)
    print(f"Test Metrics  (Dense Head — threshold: {metrics['threshold']})")
    print(sep)
    print(f"  Samples    :  {n_test}")
    print(f"  Accuracy   :  {metrics['accuracy']:.4f}")
    print(f"  F1 Score   :  {metrics['f1']:.4f}")
    print(f"  AUC-ROC    :  {metrics['auc_roc']:.4f}")
    print(f"  Precision  :  {metrics['precision']:.4f}")
    print(f"  Recall     :  {metrics['recall']:.4f}")
    print(sep)
    print(f"  Confusion Matrix:")
    print(f"    TP={cm['tp']}  FP={cm['fp']}")
    print(f"    FN={cm['fn']}  TN={cm['tn']}")
    print(sep)


def save_results(runs_dir: Path, weights_path: Path, metrics: dict, n_test: int):
    runs_dir.mkdir(parents=True, exist_ok=True)
    out_path = runs_dir / TEST_RESULTS_FILE

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "weights": str(weights_path),
        "n_test": n_test,
        "test_metrics": metrics,
    }

    # Append to existing file or create new list.
    if out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            records = json.load(f)
        if not isinstance(records, list):
            records = [records]
    else:
        records = []

    records.append(record)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    logger.info("Test results saved to: %s", out_path)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate the Dense head on the held-out test split.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--weights", type=Path, default=WEIGHTS_DEFAULT,
        help=f"Path to saved .keras weights. Default: {WEIGHTS_DEFAULT}",
    )
    parser.add_argument(
        "--splits_dir", type=Path, default=SPLITS_DIR_DEFAULT,
        help=f"Directory with X_test/y_test .npy files. Default: {SPLITS_DIR_DEFAULT}",
    )
    parser.add_argument(
        "--runs_dir", type=Path, default=RUNS_DIR_DEFAULT,
        help=f"Directory to save test_results.json. Default: {RUNS_DIR_DEFAULT}",
    )
    parser.add_argument(
        "--threshold", type=float, default=THRESHOLD,
        help=f"Decision threshold on sigmoid output. Default: {THRESHOLD}",
    )
    args = parser.parse_args()

    if not args.weights.exists():
        logger.error("Weights file not found: %s", args.weights)
        sys.exit(1)

    logger.info("Loading model from %s ...", args.weights)
    model = tf.keras.models.load_model(str(args.weights))

    X_test, y_test = load_test_split(args.splits_dir)

    metrics = compute_metrics(model, X_test, y_test, threshold=args.threshold)
    print_summary(metrics, n_test=len(y_test))
    save_results(args.runs_dir, args.weights, metrics, n_test=len(y_test))


if __name__ == "__main__":
    main()
