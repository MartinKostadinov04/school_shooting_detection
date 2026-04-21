"""
threshold_sweep.py
==================
Sweep decision thresholds on the saved Dense head and save diagnostic plots.

Loads the best saved weights + test split, evaluates at thresholds 0.50–0.94
(step 0.02), and writes five outputs to experiments/plots/threshold_sweep/:

  1. precision_recall_vs_threshold.png
  2. f1_vs_threshold.png
  3. precision_recall_curve.png
  4. roc_curve.png
  5. metrics_table.png  (full table including TN)

Each plot subtitle shows the model config and dataset split used.

Usage
-----
    python -m experiments.threshold_sweep

    # Custom paths
    python -m experiments.threshold_sweep \\
        --weights  models/saved_weights/dense_head_best.keras \\
        --splits_dir data/processed/splits \\
        --runs_dir experiments/runs \\
        --out_dir experiments/plots/threshold_sweep
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_recall_curve,
    roc_curve,
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
)
import tensorflow as tf

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

WEIGHTS_DEFAULT  = Path("models/saved_weights/dense_head_best.keras")
SPLITS_DIR_DEFAULT = Path("data/processed/splits")
RUNS_DIR_DEFAULT   = Path("experiments/runs")
OUT_DIR_DEFAULT    = Path("experiments/plots/threshold_sweep")
THRESHOLDS = np.round(np.arange(0.02, 1.00, 0.02), 2)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_test_split(splits_dir: Path):
    for name in ("X_test.npy", "y_test.npy"):
        if not (splits_dir / name).exists():
            logger.error("Missing: %s", splits_dir / name)
            sys.exit(1)
    X = np.load(splits_dir / "X_test.npy")
    y = np.load(splits_dir / "y_test.npy")
    logger.info(
        "Test split: %d samples (%d gunshot, %d not_gunshot)",
        len(y), int(np.sum(y == 1)), int(np.sum(y == 0)),
    )
    return X, y


def load_run_config(runs_dir: Path) -> dict:
    """Return config from the most recent run JSON."""
    jsons = sorted(runs_dir.glob("run_*.json"))
    if not jsons:
        return {}
    with open(jsons[-1], encoding="utf-8") as f:
        run = json.load(f)
    return run.get("config", {})


def build_subtitle(config: dict, n_test: int, n_gunshot: int) -> str:
    arch   = config.get("architecture", "Input(1024)→Dense(256,relu)→Dropout(0.3)→Dense(1,sigmoid)")
    lr     = config.get("learning_rate", "—")
    bs     = config.get("batch_size", "—")
    do     = config.get("dropout_rate", "—")
    units  = config.get("units", "—")
    cw     = config.get("class_weight_gunshot_override", "auto")
    return (
        f"arch: {arch}  |  lr={lr}  bs={bs}  dropout={do}  units={units}  "
        f"cw_gunshot={cw}\n"
        f"test set: {n_test} samples  ({n_gunshot} gunshot / {n_test - n_gunshot} not_gunshot)"
    )


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------

def sweep(y_prob: np.ndarray, y_true: np.ndarray):
    rows = []
    for t in THRESHOLDS:
        y_pred = (y_prob >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        rows.append({
            "threshold":  t,
            "precision":  float(precision_score(y_true, y_pred, zero_division=0)),
            "recall":     float(recall_score(y_true, y_pred, zero_division=0)),
            "f1":         float(f1_score(y_true, y_pred, zero_division=0)),
            "accuracy":   float(accuracy_score(y_true, y_pred)),
            "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        })
        logger.info(
            "  t=%.2f  P=%.3f  R=%.3f  F1=%.3f  Acc=%.3f",
            t, rows[-1]["precision"], rows[-1]["recall"],
            rows[-1]["f1"], rows[-1]["accuracy"],
        )
    return rows


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

STYLE = {
    "figure.facecolor": "white",
    "axes.facecolor":   "white",
    "axes.grid":        True,
    "grid.alpha":       0.3,
    "font.size":        10,
}

def _save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", path)


def plot_pr_vs_threshold(rows, subtitle: str, out_dir: Path):
    thresholds = [r["threshold"] for r in rows]
    precisions = [r["precision"] for r in rows]
    recalls    = [r["recall"]    for r in rows]
    f1s        = [r["f1"]        for r in rows]

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(thresholds, precisions, "o-", color="#2196F3", label="Precision", linewidth=2)
        ax.plot(thresholds, recalls,    "s-", color="#F44336", label="Recall",    linewidth=2)
        ax.plot(thresholds, f1s,        "^-", color="#4CAF50", label="F1",        linewidth=2)
        ax.axvline(0.5, color="grey", linestyle="--", linewidth=1, label="default threshold (0.5)")
        ax.set_xlabel("Decision Threshold")
        ax.set_ylabel("Score")
        ax.set_title("Precision / Recall / F1  vs  Threshold\n" + subtitle, fontsize=9)
        ax.legend()
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
        _save(fig, out_dir / "precision_recall_vs_threshold.png")


def plot_f1_vs_threshold(rows, subtitle: str, out_dir: Path):
    thresholds = [r["threshold"] for r in rows]
    f1s        = [r["f1"]        for r in rows]
    best_idx   = int(np.argmax(f1s))
    best_t     = thresholds[best_idx]
    best_f1    = f1s[best_idx]

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(thresholds, f1s, "o-", color="#4CAF50", linewidth=2)
        ax.axvline(best_t, color="#FF9800", linestyle="--", linewidth=1.5,
                   label=f"best threshold={best_t:.2f}  (F1={best_f1:.3f})")
        ax.axvline(0.5, color="grey", linestyle=":", linewidth=1, label="default (0.5)")
        ax.set_xlabel("Decision Threshold")
        ax.set_ylabel("F1 Score")
        ax.set_title("F1 Score  vs  Threshold\n" + subtitle, fontsize=9)
        ax.legend()
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
        _save(fig, out_dir / "f1_vs_threshold.png")


def plot_precision_recall_curve(y_prob, y_true, subtitle: str, out_dir: Path):
    precision, recall, thresh = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall, precision)

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.plot(recall, precision, color="#9C27B0", linewidth=2,
                label=f"PR curve  (AUC={pr_auc:.3f})")
        # Mark every 0.1 threshold step on the curve
        for t in np.arange(0.1, 1.0, 0.1):
            idx = np.searchsorted(thresh, t)
            if idx < len(recall) - 1:
                ax.annotate(
                    f"{t:.1f}",
                    (recall[idx], precision[idx]),
                    fontsize=7, color="grey",
                    xytext=(4, 2), textcoords="offset points",
                )
                ax.plot(recall[idx], precision[idx], ".", color="grey", markersize=6)
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Precision-Recall Curve\n" + subtitle, fontsize=9)
        ax.legend()
        ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.02)
        _save(fig, out_dir / "precision_recall_curve.png")


def plot_metrics_table(rows, subtitle: str, out_dir: Path):
    headers = ["Threshold", "Precision", "Recall", "F1", "Accuracy", "TP", "FP", "FN", "TN"]
    table_data = []
    best_f1 = max(r["f1"] for r in rows)
    for r in rows:
        cm = r["confusion_matrix"]
        table_data.append([
            f"{r['threshold']:.2f}",
            f"{r['precision']:.3f}",
            f"{r['recall']:.3f}",
            f"{r['f1']:.3f}",
            f"{r['accuracy']:.3f}",
            str(cm["tp"]),
            str(cm["fp"]),
            str(cm["fn"]),
            str(cm["tn"]),
        ])

    n_rows = len(table_data)
    fig_h = max(4, 0.35 * n_rows + 2)
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(13, fig_h))
        ax.axis("off")
        tbl = ax.table(
            cellText=table_data,
            colLabels=headers,
            loc="center",
            cellLoc="center",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9)
        tbl.scale(1, 1.4)

        # Header styling
        for j in range(len(headers)):
            tbl[0, j].set_facecolor("#37474F")
            tbl[0, j].set_text_props(color="white", fontweight="bold")

        # Row styling — highlight best F1
        for i, r in enumerate(rows, start=1):
            color = "#E8F5E9" if abs(r["f1"] - best_f1) < 1e-9 else ("white" if i % 2 == 0 else "#FAFAFA")
            for j in range(len(headers)):
                tbl[i, j].set_facecolor(color)

        fig.suptitle("Threshold Sweep — Full Metrics Table\n" + subtitle, fontsize=8, y=0.98)
        _save(fig, out_dir / "metrics_table.png")


def plot_roc_curve(y_prob, y_true, subtitle: str, out_dir: Path):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.plot(fpr, tpr, color="#FF5722", linewidth=2,
                label=f"ROC curve  (AUC={roc_auc:.3f})")
        ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="random baseline")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve\n" + subtitle, fontsize=9)
        ax.legend()
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
        _save(fig, out_dir / "roc_curve.png")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Sweep thresholds and save diagnostic plots.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--weights",    type=Path, default=WEIGHTS_DEFAULT)
    parser.add_argument("--splits_dir", type=Path, default=SPLITS_DIR_DEFAULT)
    parser.add_argument("--runs_dir",   type=Path, default=RUNS_DIR_DEFAULT)
    parser.add_argument("--out_dir",    type=Path, default=OUT_DIR_DEFAULT)
    args = parser.parse_args()

    if not args.weights.exists():
        logger.error("Weights not found: %s", args.weights)
        sys.exit(1)

    logger.info("Loading model from %s ...", args.weights)
    model = tf.keras.models.load_model(str(args.weights))

    X_test, y_test = load_test_split(args.splits_dir)
    y_true = y_test.astype(int)
    y_prob = model.predict(X_test, verbose=0).squeeze()

    config   = load_run_config(args.runs_dir)
    n_gun    = int(np.sum(y_true == 1))
    subtitle = build_subtitle(config, len(y_true), n_gun)

    logger.info("Sweeping thresholds %s ...", THRESHOLDS.tolist())
    rows = sweep(y_prob, y_true)

    plot_pr_vs_threshold(rows, subtitle, args.out_dir)
    plot_f1_vs_threshold(rows, subtitle, args.out_dir)
    plot_precision_recall_curve(y_prob, y_true, subtitle, args.out_dir)
    plot_roc_curve(y_prob, y_true, subtitle, args.out_dir)
    plot_metrics_table(rows, subtitle, args.out_dir)

    logger.info("All plots saved to: %s", args.out_dir)


if __name__ == "__main__":
    main()
