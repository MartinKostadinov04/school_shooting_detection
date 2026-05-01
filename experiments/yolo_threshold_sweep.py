"""
yolo_threshold_sweep.py
=======================
Sweep confidence thresholds on the fine-tuned YOLOv11s model over the
held-out vision test set and save diagnostic plots.

Strategy
--------
1. Run inference ONCE at conf=0.001 (capture every raw detection).
2. For each threshold in [0.05 … 0.95] (step 0.05):
   - Filter detections by confidence.
   - Match each prediction to a ground-truth box (IoU ≥ 0.50, greedy
     highest-conf-first; each GT matched at most once per image).
   - Accumulate TP, FP, FN across the full test set.
3. Generate five plots to experiments/plots/yolo_threshold_sweep/:
     precision_recall_vs_threshold.png
     f1_vs_threshold.png
     precision_recall_curve.png     (detection PR curve)
     f1_iou_grid.png                (F1 at multiple IoU thresholds)
     metrics_table.png
4. Save raw numbers to experiments/runs/yolo_sweep_results.json.

Usage
-----
    python -m experiments.yolo_threshold_sweep

    # Custom paths
    python -m experiments.yolo_threshold_sweep \\
        --weights  models/yolo_finetuned/best.pt \\
        --data     data/vision_yolo/data.yaml \\
        --out_dir  experiments/plots/yolo_threshold_sweep
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
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
WEIGHTS_DEFAULT = ROOT / "models/yolo_finetuned/best.pt"
DATA_YAML_DEFAULT = ROOT / "data/vision_yolo/data.yaml"
OUT_DIR_DEFAULT = ROOT / "experiments/plots/yolo_threshold_sweep"
RUNS_DIR_DEFAULT = ROOT / "experiments/runs"

CONF_COLLECT = 0.001        # inference pass — collect everything
NMS_IOU = 0.45              # NMS IoU used during inference
IOU_MATCH = 0.50            # IoU to count a prediction as TP
IMGSZ = 640                 # inference image size (640 saves GPU memory vs 1280)
THRESHOLDS = np.round(np.arange(0.05, 1.00, 0.05), 2)
IOU_VARIANTS = [0.40, 0.50, 0.60, 0.75]   # for the F1-vs-IoU grid


# ---------------------------------------------------------------------------
# Ground-truth loader
# ---------------------------------------------------------------------------

def load_ground_truth(test_images_dir: Path, labels_dir: Path):
    """Return dict[image_stem -> list[np.ndarray shape (4,)]] of GT boxes (xyxy, pixel coords)."""
    gt = {}
    for img_path in sorted(test_images_dir.glob("*")):
        if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            continue
        label_path = labels_dir / (img_path.stem + ".txt")
        boxes = []
        if label_path.exists():
            img = Image.open(img_path)
            W, H = img.size
            with open(label_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    # class cx cy w h  (normalised)
                    cx, cy, bw, bh = map(float, parts[1:5])
                    x1 = (cx - bw / 2) * W
                    y1 = (cy - bh / 2) * H
                    x2 = (cx + bw / 2) * W
                    y2 = (cy + bh / 2) * H
                    boxes.append(np.array([x1, y1, x2, y2], dtype=np.float32))
        gt[img_path] = boxes
    logger.info(
        "Loaded GT for %d images (%d positives, %d negatives)",
        len(gt),
        sum(1 for v in gt.values() if v),
        sum(1 for v in gt.values() if not v),
    )
    return gt


# ---------------------------------------------------------------------------
# Inference pass
# ---------------------------------------------------------------------------

def run_inference(model, gt: dict, device: str = "cpu"):
    """
    Run YOLO on all test images at CONF_COLLECT.
    Returns dict[img_path -> list[dict(conf, xyxy)]] of raw predictions.
    Processes images in small batches to avoid GPU OOM.
    """
    raw_preds = {}
    img_paths = list(gt.keys())
    batch_size = 8 if device != "cpu" else 16

    logger.info(
        "Running inference on %d images at conf=%.3f  imgsz=%d  device=%s  batch=%d …",
        len(img_paths), CONF_COLLECT, IMGSZ, device, batch_size,
    )
    for start in range(0, len(img_paths), batch_size):
        batch_paths = img_paths[start : start + batch_size]
        results = model.predict(
            source=[str(p) for p in batch_paths],
            conf=CONF_COLLECT,
            iou=NMS_IOU,
            imgsz=IMGSZ,
            device=device,
            verbose=False,
            stream=False,
        )
        for i, result in enumerate(results):
            img_path = batch_paths[i]
            preds = []
            if result.boxes is not None and len(result.boxes):
                boxes_xyxy = result.boxes.xyxy.cpu().numpy()
                confs = result.boxes.conf.cpu().numpy()
                for j in range(len(confs)):
                    preds.append({"conf": float(confs[j]), "xyxy": boxes_xyxy[j]})
            raw_preds[img_path] = preds

        done = min(start + batch_size, len(img_paths))
        if done % 200 < batch_size or done == len(img_paths):
            logger.info("  … %d / %d done", done, len(img_paths))

    total_dets = sum(len(v) for v in raw_preds.values())
    logger.info("Inference complete — %d total raw detections", total_dets)
    return raw_preds


# ---------------------------------------------------------------------------
# IoU helper
# ---------------------------------------------------------------------------

def iou_box(a: np.ndarray, b: np.ndarray) -> float:
    """Compute IoU between two xyxy boxes."""
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0.0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


# ---------------------------------------------------------------------------
# Threshold sweep
# ---------------------------------------------------------------------------

def sweep_threshold(raw_preds: dict, gt: dict, thresholds=THRESHOLDS, iou_thresh=IOU_MATCH):
    """
    For each confidence threshold, match predictions to GT and return metrics.
    Also collect all (confidence, is_tp) pairs for the PR curve.
    """
    rows = []
    # Collect ALL predictions above CONF_COLLECT with their TP flag for PR curve
    all_confs = []
    all_tp_flags = []

    # Pre-sort predictions per image by confidence descending (shared across thresholds)
    sorted_preds = {
        img_path: sorted(preds, key=lambda p: p["conf"], reverse=True)
        for img_path, preds in raw_preds.items()
    }

    for thresh in thresholds:
        total_tp = total_fp = total_fn = 0

        for img_path, gt_boxes in gt.items():
            preds = [p for p in sorted_preds[img_path] if p["conf"] >= thresh]
            matched_gt = [False] * len(gt_boxes)

            for pred in preds:
                best_iou = 0.0
                best_j = -1
                for j, gt_box in enumerate(gt_boxes):
                    if matched_gt[j]:
                        continue
                    iou = iou_box(pred["xyxy"], gt_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_j = j

                if best_iou >= iou_thresh:
                    total_tp += 1
                    matched_gt[best_j] = True
                else:
                    total_fp += 1

            total_fn += sum(1 for m in matched_gt if not m)

        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        recall    = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        f1        = (2 * precision * recall / (precision + recall)
                     if (precision + recall) > 0 else 0.0)

        rows.append({
            "threshold": float(thresh),
            "precision": round(precision, 4),
            "recall":    round(recall, 4),
            "f1":        round(f1, 4),
            "tp":        int(total_tp),
            "fp":        int(total_fp),
            "fn":        int(total_fn),
        })
        logger.info(
            "  t=%.2f  P=%.3f  R=%.3f  F1=%.3f  TP=%d  FP=%d  FN=%d",
            thresh, precision, recall, f1, total_tp, total_fp, total_fn,
        )

    # Build full (conf, is_tp) list at the finest granularity (all raw predictions)
    for img_path, gt_boxes in gt.items():
        preds = sorted_preds[img_path]
        matched_gt = [False] * len(gt_boxes)
        for pred in preds:
            best_iou = 0.0
            best_j = -1
            for j, gt_box in enumerate(gt_boxes):
                if matched_gt[j]:
                    continue
                iou = iou_box(pred["xyxy"], gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_j = j
            is_tp = int(best_iou >= iou_thresh)
            if is_tp:
                matched_gt[best_j] = True
            all_confs.append(pred["conf"])
            all_tp_flags.append(is_tp)

    return rows, np.array(all_confs), np.array(all_tp_flags)


def build_pr_curve(all_confs, all_tp_flags, total_gt_boxes):
    """
    Construct detection PR curve by sorting predictions by confidence and
    computing cumulative precision / recall.
    """
    order = np.argsort(-all_confs)
    sorted_tp = all_tp_flags[order]
    cum_tp = np.cumsum(sorted_tp)
    cum_fp = np.cumsum(1 - sorted_tp)
    precisions = cum_tp / (cum_tp + cum_fp + 1e-9)
    recalls    = cum_tp / (total_gt_boxes + 1e-9)
    # Append sentinel point at (0, 1) for clean curve start
    precisions = np.concatenate([[1.0], precisions])
    recalls    = np.concatenate([[0.0], recalls])
    return precisions, recalls


# ---------------------------------------------------------------------------
# Plot helpers
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


def plot_pr_vs_threshold(rows, out_dir: Path):
    thresholds = [r["threshold"] for r in rows]
    precisions = [r["precision"] for r in rows]
    recalls    = [r["recall"]    for r in rows]
    f1s        = [r["f1"]        for r in rows]

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(thresholds, precisions, "o-", color="#2196F3", label="Precision", linewidth=2)
        ax.plot(thresholds, recalls,    "s-", color="#F44336", label="Recall",    linewidth=2)
        ax.plot(thresholds, f1s,        "^-", color="#4CAF50", label="F1",        linewidth=2)
        ax.axvline(0.5, color="grey", linestyle="--", linewidth=1, label="conf=0.5 (YOLO default)")
        ax.set_xlabel("Confidence Threshold")
        ax.set_ylabel("Score")
        ax.set_title(
            "YOLOv11s  —  Precision / Recall / F1 vs Confidence Threshold\n"
            f"Vision test set  |  IoU-match={IOU_MATCH:.2f}  |  NMS-IoU={NMS_IOU:.2f}",
            fontsize=9,
        )
        ax.legend()
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
        _save(fig, out_dir / "precision_recall_vs_threshold.png")


def plot_f1_vs_threshold(rows, out_dir: Path):
    thresholds = [r["threshold"] for r in rows]
    f1s        = [r["f1"]        for r in rows]
    best_idx   = int(np.argmax(f1s))
    best_t     = thresholds[best_idx]
    best_f1    = f1s[best_idx]

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(thresholds, f1s, "o-", color="#4CAF50", linewidth=2)
        ax.axvline(best_t, color="#FF9800", linestyle="--", linewidth=1.5,
                   label=f"best threshold = {best_t:.2f}  (F1 = {best_f1:.3f})")
        ax.axvline(0.5, color="grey", linestyle=":", linewidth=1, label="default conf=0.5")
        ax.set_xlabel("Confidence Threshold")
        ax.set_ylabel("F1 Score")
        ax.set_title(
            "YOLOv11s  —  F1 Score vs Confidence Threshold\n"
            f"Vision test set  |  IoU-match={IOU_MATCH:.2f}",
            fontsize=9,
        )
        ax.legend()
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
        _save(fig, out_dir / "f1_vs_threshold.png")

    return best_t, best_f1


def plot_pr_curve(precisions, recalls, pr_auc: float, out_dir: Path):
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.plot(recalls, precisions, color="#9C27B0", linewidth=2,
                label=f"PR curve  (AP={pr_auc:.3f})")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(
            "YOLOv11s  —  Precision-Recall Curve (detection)\n"
            f"Vision test set  |  IoU-match={IOU_MATCH:.2f}",
            fontsize=9,
        )
        ax.legend()
        ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.02)
        _save(fig, out_dir / "precision_recall_curve.png")


def plot_metrics_table(rows, out_dir: Path):
    headers = ["Threshold", "Precision", "Recall", "F1", "TP", "FP", "FN"]
    best_f1 = max(r["f1"] for r in rows)
    table_data = []
    for r in rows:
        table_data.append([
            f"{r['threshold']:.2f}",
            f"{r['precision']:.3f}",
            f"{r['recall']:.3f}",
            f"{r['f1']:.3f}",
            str(r["tp"]),
            str(r["fp"]),
            str(r["fn"]),
        ])

    n_rows = len(table_data)
    fig_h = max(4, 0.35 * n_rows + 2)
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(11, fig_h))
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
        for j in range(len(headers)):
            tbl[0, j].set_facecolor("#37474F")
            tbl[0, j].set_text_props(color="white", fontweight="bold")
        for i, r in enumerate(rows, start=1):
            color = "#E8F5E9" if abs(r["f1"] - best_f1) < 1e-9 else ("white" if i % 2 == 0 else "#FAFAFA")
            for j in range(len(headers)):
                tbl[i, j].set_facecolor(color)
        fig.suptitle(
            f"YOLOv11s — Threshold Sweep Metrics  |  IoU-match={IOU_MATCH:.2f}",
            fontsize=9, y=0.98,
        )
        _save(fig, out_dir / "metrics_table.png")


def plot_f1_iou_grid(raw_preds, gt, out_dir: Path):
    """Plot F1 vs threshold for multiple IoU-match values on one axes."""
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#F44336"]
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(9, 5))
        for iou_t, color in zip(IOU_VARIANTS, colors):
            rows_iou, _, _ = sweep_threshold(raw_preds, gt, thresholds=THRESHOLDS, iou_thresh=iou_t)
            thresholds = [r["threshold"] for r in rows_iou]
            f1s        = [r["f1"]        for r in rows_iou]
            best_f1    = max(f1s)
            ax.plot(thresholds, f1s, "-o", color=color, linewidth=2, markersize=4,
                    label=f"IoU≥{iou_t:.2f}  (best F1={best_f1:.3f})")
        ax.set_xlabel("Confidence Threshold")
        ax.set_ylabel("F1 Score")
        ax.set_title("YOLOv11s  —  F1 at Multiple IoU-match Thresholds\nVision test set", fontsize=9)
        ax.legend()
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
        _save(fig, out_dir / "f1_iou_grid.png")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Sweep YOLO confidence thresholds on the vision test set.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--weights",  type=Path, default=WEIGHTS_DEFAULT)
    parser.add_argument("--data",     type=Path, default=DATA_YAML_DEFAULT)
    parser.add_argument("--out_dir",  type=Path, default=OUT_DIR_DEFAULT)
    parser.add_argument("--runs_dir", type=Path, default=RUNS_DIR_DEFAULT)
    parser.add_argument("--device",   type=str,  default="cpu",
                        help="Inference device: 'cpu', '0', 'cuda:0', etc.")
    args = parser.parse_args()

    if not args.weights.exists():
        logger.error("Weights not found: %s", args.weights)
        sys.exit(1)

    # --- resolve test images/labels from data.yaml ---
    import yaml
    with open(args.data) as f:
        data_cfg = yaml.safe_load(f)
    dataset_root = Path(data_cfg.get("path", args.data.parent))
    test_images_dir = dataset_root / data_cfg.get("test", "images/test")
    labels_dir      = dataset_root / "labels" / "test"

    if not test_images_dir.exists():
        logger.error("Test images dir not found: %s", test_images_dir)
        sys.exit(1)

    gt = load_ground_truth(test_images_dir, labels_dir)
    total_gt_boxes = sum(len(v) for v in gt.values())
    logger.info("Total ground-truth boxes in test set: %d", total_gt_boxes)

    # --- load model ---
    from ultralytics import YOLO
    logger.info("Loading YOLO model from %s …", args.weights)
    model = YOLO(str(args.weights))

    # --- inference ---
    raw_preds = run_inference(model, gt, device=args.device)

    # --- primary sweep at IOU_MATCH=0.50 ---
    logger.info("Sweeping thresholds (IoU-match=%.2f) …", IOU_MATCH)
    rows, all_confs, all_tp_flags = sweep_threshold(raw_preds, gt)

    # --- PR curve ---
    precisions, recalls = build_pr_curve(all_confs, all_tp_flags, total_gt_boxes)
    pr_auc = float(np.trapz(precisions, recalls))   # recalls is increasing → positive area

    # --- plots ---
    args.out_dir.mkdir(parents=True, exist_ok=True)
    plot_pr_vs_threshold(rows, args.out_dir)
    best_t, best_f1 = plot_f1_vs_threshold(rows, args.out_dir)
    plot_pr_curve(precisions, recalls, pr_auc, args.out_dir)
    plot_metrics_table(rows, args.out_dir)
    logger.info("Building F1-IoU grid (this runs the sweep 4× more) …")
    plot_f1_iou_grid(raw_preds, gt, args.out_dir)

    # --- save JSON ---
    args.runs_dir.mkdir(parents=True, exist_ok=True)
    out_json = args.runs_dir / "yolo_sweep_results.json"
    result = {
        "model": str(args.weights),
        "test_images_dir": str(test_images_dir),
        "n_test_images":   len(gt),
        "n_positives":     sum(1 for v in gt.values() if v),
        "n_negatives":     sum(1 for v in gt.values() if not v),
        "total_gt_boxes":  total_gt_boxes,
        "iou_match":       IOU_MATCH,
        "nms_iou":         NMS_IOU,
        "pr_auc":          round(pr_auc, 4),
        "best_threshold":  best_t,
        "best_f1":         round(best_f1, 4),
        "sweep":           rows,
    }
    with open(out_json, "w") as f:
        json.dump(result, f, indent=2)
    logger.info("Results saved to: %s", out_json)

    # --- summary ---
    logger.info("=" * 60)
    logger.info("BEST THRESHOLD : %.2f", best_t)
    logger.info("BEST F1        : %.3f", best_f1)
    logger.info("PR-AUC         : %.3f", pr_auc)
    logger.info("All plots in   : %s", args.out_dir)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
