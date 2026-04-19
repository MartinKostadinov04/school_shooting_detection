"""
split_dataset.py
================
Stratified train / validation / test split of pre-extracted YAMNet embeddings.

Reads ``X_embeddings.npy`` and ``y_labels.npy`` produced by
``extract_embeddings.py`` and writes six split arrays plus a JSON record of
the split indices and class distributions for full reproducibility.

Usage
-----
    python -m pipeline.split_dataset \\
        --embeddings_dir data/processed/embeddings \\
        --output_dir data/processed/splits

Flags
-----
``--embeddings_dir``
    Directory containing ``X_embeddings.npy`` and ``y_labels.npy``.
    Default: ``data/processed/embeddings``.

``--output_dir``
    Destination for the six split arrays and ``split_info.json``.
    Default: ``data/processed/splits``.

Pipeline contract
-----------------
  Reads  : ``<embeddings_dir>/X_embeddings.npy``  shape (N, 1024) float32
             ``<embeddings_dir>/y_labels.npy``       shape (N,)      float32
  Writes : ``<output_dir>/X_train.npy``, ``X_val.npy``, ``X_test.npy``
             ``<output_dir>/y_train.npy``, ``y_val.npy``, ``y_test.npy``
             ``<output_dir>/split_info.json``

Split ratios: 70% train / 15% validation / 15% test (stratified, seed=42).

FUTURE PLACEHOLDERS
-------------------
- ``--stratify_weapon_type``: once weapon-type sub-labels are available,
  enable fine-grained stratification on (binary_label, weapon_type) jointly.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from sklearn.model_selection import train_test_split

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_VAL_SIZE: float = 0.15
DEFAULT_TEST_SIZE: float = 0.15
RANDOM_STATE: int = 42
MIN_SAMPLES_PER_CLASS_PER_SPLIT: int = 5  # warn if any split has fewer than this


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_embeddings(
    embeddings_dir: Path,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load ``X_embeddings.npy`` and ``y_labels.npy`` from ``embeddings_dir``.

    Parameters
    ----------
    embeddings_dir : Path
        Directory containing the two ``.npy`` files.

    Returns
    -------
    X : np.ndarray
        Shape ``(N, 1024)``, dtype ``float32``.
    y : np.ndarray
        Shape ``(N,)``, dtype ``float32``.

    Raises
    ------
    FileNotFoundError
        If either file is missing. The error message includes a suggestion
        to run ``extract_embeddings.py`` first.
    ValueError
        If ``X.shape[0] != y.shape[0]`` (shapes are inconsistent).
    """
    x_path = embeddings_dir / "X_embeddings.npy"
    y_path = embeddings_dir / "y_labels.npy"

    for p in (x_path, y_path):
        if not p.exists():
            raise FileNotFoundError(
                f"Required file not found: '{p}'. "
                "Run pipeline/extract_embeddings.py first to generate embeddings."
            )

    X = np.load(str(x_path))
    y = np.load(str(y_path))

    if X.shape[0] != y.shape[0]:
        raise ValueError(
            f"Shape mismatch: X has {X.shape[0]} rows but y has {y.shape[0]} rows. "
            "The embedding files may be corrupted or from different extraction runs."
        )

    logger.info("Loaded X: shape=%s  dtype=%s", X.shape, X.dtype)
    logger.info("Loaded y: shape=%s  dtype=%s", y.shape, y.dtype)
    return X, y


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------


def stratified_split(
    X: np.ndarray,
    y: np.ndarray,
    val_size: float = DEFAULT_VAL_SIZE,
    test_size: float = DEFAULT_TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> Dict[str, np.ndarray]:
    """
    Perform a stratified 70 / 15 / 15 train / val / test split.

    Stratification ensures that the class balance in each split mirrors
    the overall dataset balance, which is critical for reliable evaluation
    when the dataset may be slightly imbalanced.

    Two-stage approach
    ------------------
    1. Split the full dataset into **train** (70%) and **temp** (30%),
       stratified on ``y``.
    2. Split **temp** 50/50 into **val** (15%) and **test** (15%),
       stratified on the temp labels.

    Both stages use the same ``random_state`` for determinism.

    Parameters
    ----------
    X : np.ndarray
        Shape ``(N, 1024)``, dtype ``float32``.
    y : np.ndarray
        Shape ``(N,)``, dtype ``float32``.
    val_size : float
        Fraction of the total dataset for validation. Default: 0.15.
    test_size : float
        Fraction of the total dataset for test. Default: 0.15.
    random_state : int
        Random seed for reproducibility. Default: 42.

    Returns
    -------
    dict with keys:
        ``X_train``, ``X_val``, ``X_test`` : np.ndarray
        ``y_train``, ``y_val``, ``y_test`` : np.ndarray
        ``idx_train``, ``idx_val``, ``idx_test`` : np.ndarray of int
            Original row indices in ``X`` / ``y`` for each split.
            Stored in ``split_info.json`` to allow exact reconstruction.

    Raises
    ------
    ValueError
        If the dataset is too small for stratified splitting (fewer than
        2 * num_classes samples per class required by sklearn).
    """
    n = len(y)
    all_indices = np.arange(n)
    temp_fraction = val_size + test_size  # 0.30

    # Stage 1: train vs temp.
    idx_train, idx_temp = train_test_split(
        all_indices,
        test_size=temp_fraction,
        stratify=y,
        random_state=random_state,
    )

    # Stage 2: val vs test from temp.
    # relative test_size within temp = test_size / temp_fraction = 0.5
    relative_test = test_size / temp_fraction
    idx_val, idx_test = train_test_split(
        idx_temp,
        test_size=relative_test,
        stratify=y[idx_temp],
        random_state=random_state,
    )

    splits = {
        "X_train": X[idx_train],
        "X_val": X[idx_val],
        "X_test": X[idx_test],
        "y_train": y[idx_train],
        "y_val": y[idx_val],
        "y_test": y[idx_test],
        "idx_train": idx_train,
        "idx_val": idx_val,
        "idx_test": idx_test,
    }

    # Warn if any split has very few samples of either class.
    for split_name in ("train", "val", "test"):
        split_y = splits[f"y_{split_name}"]
        for class_val, class_name in [(1.0, "gunshot"), (0.0, "not_gunshot")]:
            count = int(np.sum(split_y == class_val))
            if count < MIN_SAMPLES_PER_CLASS_PER_SPLIT:
                logger.warning(
                    "Split '%s' has only %d '%s' samples. "
                    "Consider collecting more data or adjusting split ratios.",
                    split_name, count, class_name,
                )

    return splits


# ---------------------------------------------------------------------------
# Class distribution helper
# ---------------------------------------------------------------------------


def compute_class_distribution(y: np.ndarray) -> Dict[str, int]:
    """
    Return a dict with the count of each class in label array ``y``.

    Parameters
    ----------
    y : np.ndarray
        1-D float32 array of labels (0.0 = not_gunshot, 1.0 = gunshot).

    Returns
    -------
    dict
        ``{"gunshot": int, "not_gunshot": int, "total": int}``
    """
    gunshot = int(np.sum(y == 1.0))
    not_gunshot = int(np.sum(y == 0.0))
    return {"gunshot": gunshot, "not_gunshot": not_gunshot, "total": len(y)}


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------


def save_splits(
    splits: Dict[str, np.ndarray],
    output_dir: Path,
) -> None:
    """
    Save the six split arrays and ``split_info.json`` to ``output_dir``.

    ``split_info.json`` schema::

        {
            "random_state": 42,
            "split_ratios": {"train": 0.70, "val": 0.15, "test": 0.15},
            "counts": {
                "train": {"gunshot": N, "not_gunshot": N, "total": N},
                "val":   {...},
                "test":  {...}
            },
            "indices": {
                "train": [0, 3, 5, ...],
                "val":   [...],
                "test":  [...]
            }
        }

    The ``indices`` field allows any future run to reconstruct the exact
    same splits from the original embedding arrays.

    Parameters
    ----------
    splits : dict
        Output of ``stratified_split()``.
    output_dir : Path
        Destination directory. Created if it does not exist.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save the six numpy arrays.
    for name in ("X_train", "X_val", "X_test", "y_train", "y_val", "y_test"):
        np.save(str(output_dir / f"{name}.npy"), splits[name])
        logger.info("Saved %-12s  shape=%s", f"{name}.npy", splits[name].shape)

    # Build and save split_info.json.
    split_info = {
        "random_state": RANDOM_STATE,
        "split_ratios": {
            "train": round(1.0 - DEFAULT_VAL_SIZE - DEFAULT_TEST_SIZE, 4),
            "val": DEFAULT_VAL_SIZE,
            "test": DEFAULT_TEST_SIZE,
        },
        "counts": {
            split_name: compute_class_distribution(splits[f"y_{split_name}"])
            for split_name in ("train", "val", "test")
        },
        "indices": {
            split_name: splits[f"idx_{split_name}"].tolist()
            for split_name in ("train", "val", "test")
        },
    }

    info_path = output_dir / "split_info.json"
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(split_info, f, indent=2)
    logger.info("Saved split_info.json   path=%s", info_path)


# ---------------------------------------------------------------------------
# Summary printing
# ---------------------------------------------------------------------------


def print_summary(splits: Dict[str, np.ndarray]) -> None:
    """
    Print a human-readable table of clip counts and class balance per split.

    Example output::

        ┌─────────┬────────┬──────────┬──────────────┬─────────┐
        │  Split  │  Total │  Gunshot │  Not-Gunshot │ Balance │
        ├─────────┼────────┼──────────┼──────────────┼─────────┤
        │  train  │    294 │      147 │          147 │  50.0%  │
        │  val    │     63 │       32 │           31 │  50.8%  │
        │  test   │     63 │       31 │           32 │  49.2%  │
        └─────────┴────────┴──────────┴──────────────┴─────────┘

    Parameters
    ----------
    splits : dict
        Output of ``stratified_split()``.
    """
    header = f"{'Split':<8}  {'Total':>6}  {'Gunshot':>8}  {'Not-Gunshot':>12}  {'Balance':>8}"
    sep = "-" * len(header)
    print()
    print(sep)
    print(header)
    print(sep)

    total_overall = 0
    for split_name in ("train", "val", "test"):
        y_split = splits[f"y_{split_name}"]
        dist = compute_class_distribution(y_split)
        balance_pct = 100.0 * dist["gunshot"] / dist["total"] if dist["total"] > 0 else 0.0
        print(
            f"{split_name:<8}  {dist['total']:>6}  {dist['gunshot']:>8}  "
            f"{dist['not_gunshot']:>12}  {balance_pct:>7.1f}%"
        )
        total_overall += dist["total"]

    print(sep)
    print(f"{'TOTAL':<8}  {total_overall:>6}")
    print()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """
    Command-line entry point for dataset splitting.

    Example
    -------
    .. code-block:: bash

        python -m pipeline.split_dataset \\
            --embeddings_dir data/processed/embeddings \\
            --output_dir data/processed/splits

    FUTURE PLACEHOLDER
    ------------------
    ``--stratify_weapon_type`` flag: once weapon-type sub-labels are
    available, enable fine-grained stratification on
    ``(binary_label, weapon_type)`` jointly to ensure each split has
    balanced representation of pistol / semi-automatic / automatic clips.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Create stratified 70/15/15 train/val/test splits from pre-extracted\n"
            "YAMNet embeddings. Run extract_embeddings.py first.\n\n"
            "Reads : X_embeddings.npy, y_labels.npy from <embeddings_dir>\n"
            "Writes: X_train.npy, X_val.npy, X_test.npy,\n"
            "        y_train.npy, y_val.npy, y_test.npy,\n"
            "        split_info.json  to <output_dir>"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--embeddings_dir",
        type=Path,
        default=Path("data/processed/embeddings"),
        help="Directory containing X_embeddings.npy and y_labels.npy. "
             "Default: data/processed/embeddings",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("data/processed/splits"),
        help="Destination for split arrays and split_info.json. "
             "Default: data/processed/splits",
    )
    # FUTURE PLACEHOLDER: --stratify_weapon_type flag goes here.

    args = parser.parse_args()

    try:
        X, y = load_embeddings(args.embeddings_dir)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        sys.exit(1)
    except ValueError as exc:
        logger.error("Data validation error: %s", exc)
        sys.exit(1)

    splits = stratified_split(X, y, random_state=RANDOM_STATE)
    save_splits(splits, args.output_dir)
    print_summary(splits)

    logger.info(
        "Done. Run training/train_head.py next (once implemented) to train "
        "a classification head on the splits."
    )


if __name__ == "__main__":
    main()
