"""
extract_embeddings.py
=====================
Extract YAMNet embeddings from raw gunshot/not_gunshot WAV files and save
the resulting arrays to disk for downstream head training.

This script runs YAMNet **once** on every clip and caches the results.
Head training (train_head.py) reads these cached embeddings — YAMNet does
**not** run during the training loop, which makes training fast.

Usage
-----
    python -m pipeline.extract_embeddings \\
        --data_dir data/raw \\
        --output_dir data/processed/embeddings \\
        [--force]

Flags
-----
``--data_dir``
    Path to the raw data root. Must contain ``gunshot/`` and ``not_gunshot/``
    subdirectories populated with WAV files. Default: ``data/raw``.

``--output_dir``
    Destination directory for output files. Default:
    ``data/processed/embeddings``.

``--force``
    Overwrite existing output files if they already exist. Without this
    flag the script exits cleanly when outputs are found, to prevent
    accidental re-extraction.

Pipeline contract
-----------------
  Reads  : ``<data_dir>/gunshot/*.wav``
             ``<data_dir>/not_gunshot/*.wav``
  Writes : ``<output_dir>/X_embeddings.npy``  shape (N, 1024)  float32
             ``<output_dir>/y_labels.npy``      shape (N,)       float32
             ``<output_dir>/metadata.json``

YAMNet input contract
---------------------
- 1-D float32 tensor, shape ``(32000,)``
- Values in ``[-1.0, +1.0]``
- 16 kHz mono
- YAMNet handles its own mel spectrogram internally — do NOT pre-compute it.

FUTURE PLACEHOLDERS
-------------------
- ``--zero_shot``: also compute and save zero-shot class-427 scores for the
  entire dataset as ``zero_shot_scores.npy`` (Experiment 1 baseline).
- Per-frame embedding mode (needed by BiLSTM head): save raw
  ``(num_frames, 1024)`` embeddings instead of mean-pooled vectors.
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

from pipeline.preprocessing import preprocess_clip

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

YAMNET_URL: str = "https://tfhub.dev/google/yamnet/1"
GUNSHOT_CLASS_IDX: int = 427  # AudioSet class: "Gunshot, gunfire"
LABEL_MAP: Dict[str, float] = {"gunshot": 1.0, "not_gunshot": 0.0}
LOG_EVERY_N: int = 50  # log progress every N files


# ---------------------------------------------------------------------------
# YAMNet helpers
# ---------------------------------------------------------------------------


def load_yamnet(url: str = YAMNET_URL) -> object:
    """
    Load YAMNet from TensorFlow Hub and return the model.

    The model is frozen — its weights are not trainable and will not be
    updated during any downstream head training.

    Parameters
    ----------
    url : str
        TensorFlow Hub URL for YAMNet.
        Default: ``"https://tfhub.dev/google/yamnet/1"``.

    Returns
    -------
    TF-Hub SavedModel
        Callable: ``scores, embeddings, log_mel = model(waveform_tensor)``

    Notes
    -----
    The returned model is a TF-Hub SavedModel, not a ``tf.keras.Model``.
    It is callable directly with a 1-D float32 tensor.
    """
    logger.info("Loading YAMNet from %s ...", url)
    model = hub.load(url)
    logger.info("YAMNet loaded successfully.")
    return model


def extract_embedding(
    audio: np.ndarray,
    yamnet_model: object,
) -> np.ndarray:
    """
    Run YAMNet on a preprocessed audio array and return a single clip-level
    embedding by mean-pooling over the time (frame) axis.

    YAMNet internally divides the waveform into overlapping 0.96-second windows
    (0.48-second hop), computes a mel spectrogram for each window, and returns
    a 1024-dimensional embedding per window. We reduce these to a single
    ``(1024,)`` vector with ``tf.reduce_mean``.

    Parameters
    ----------
    audio : np.ndarray
        Shape ``(32000,)``, dtype ``float32``, values in ``[-1.0, +1.0]``.
        This must be the direct output of ``preprocess_clip()``.
    yamnet_model : TF-Hub SavedModel
        Loaded YAMNet model returned by ``load_yamnet()``.

    Returns
    -------
    np.ndarray
        Shape ``(1024,)``, dtype ``float32``. Clip-level embedding.

    Raises
    ------
    ValueError
        If YAMNet returns zero frames (e.g. waveform too short internally).
    """
    waveform = tf.constant(audio, dtype=tf.float32)
    _, embeddings, _ = yamnet_model(waveform)

    if embeddings.shape[0] == 0:
        raise ValueError(
            "YAMNet returned 0 frames. The waveform may be too short for "
            "YAMNet's internal windowing (minimum ~0.96 s recommended)."
        )

    clip_embedding = tf.reduce_mean(embeddings, axis=0).numpy()
    return clip_embedding.astype(np.float32)


def extract_zero_shot_score(
    audio: np.ndarray,
    yamnet_model: object,
) -> float:
    """
    Extract YAMNet's raw AudioSet class-427 (Gunshot, gunfire) score as a
    scalar, for use as a zero-shot baseline without any trained head.

    YAMNet returns one score vector per frame; we average across frames to
    get a single clip-level score. This score can be thresholded directly
    for Experiment 1 (zero-shot baseline) without training any head.

    Parameters
    ----------
    audio : np.ndarray
        Shape ``(32000,)``, dtype ``float32``, values in ``[-1.0, +1.0]``.
    yamnet_model : TF-Hub SavedModel
        Loaded YAMNet model returned by ``load_yamnet()``.

    Returns
    -------
    float
        Scalar in ``[0.0, 1.0]`` — YAMNet's mean per-frame confidence that
        the clip contains a gunshot (AudioSet class 427).

    Notes
    -----
    This function does **not** use the trained classification head.
    Threshold tuning for this score is handled in a future training session.

    To save zero-shot scores for the full dataset, use the ``--zero_shot``
    flag (FUTURE PLACEHOLDER — not yet implemented).
    """
    waveform = tf.constant(audio, dtype=tf.float32)
    scores, _, _ = yamnet_model(waveform)
    # scores shape: (num_frames, 521)
    gunshot_scores = scores[:, GUNSHOT_CLASS_IDX]
    return float(tf.reduce_mean(gunshot_scores).numpy())


# ---------------------------------------------------------------------------
# Core extraction logic
# ---------------------------------------------------------------------------


def build_embedding_matrix(
    data_dir: Path,
    yamnet_model: object,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Iterate ``gunshot/`` and ``not_gunshot/`` subdirectories under ``data_dir``,
    extract a YAMNet embedding for each clip, and return stacked arrays plus
    metadata.

    Parameters
    ----------
    data_dir : Path
        Root directory containing ``gunshot/`` and ``not_gunshot/``
        subdirectories.
    yamnet_model : TF-Hub SavedModel
        Loaded YAMNet model.

    Returns
    -------
    X : np.ndarray
        Shape ``(N, 1024)``, dtype ``float32``. One row per clip.
    y : np.ndarray
        Shape ``(N,)``, dtype ``float32``.
        ``1.0`` = gunshot, ``0.0`` = not_gunshot.
    metadata : dict
        Contains processing statistics (see ``save_outputs`` for schema).

    Notes
    -----
    Embeddings are accumulated in a Python list and stacked once at the end.
    This avoids repeatedly copying large arrays during ``np.concatenate``.

    Any file that raises an exception is skipped and recorded in
    ``metadata["skipped_files"]`` with the error message.
    """
    # Validate that both class directories exist.
    for class_name in LABEL_MAP:
        class_dir = data_dir / class_name
        if not class_dir.is_dir():
            logger.error(
                "Expected class directory not found: '%s'. "
                "Make sure your raw data is organized as "
                "data/raw/gunshot/ and data/raw/not_gunshot/.",
                class_dir,
            )
            sys.exit(1)

    embeddings_list: List[np.ndarray] = []
    labels_list: List[float] = []
    skipped_files: List[Dict] = []
    gunshot_count = 0
    not_gunshot_count = 0
    files_processed = 0

    for class_name, label in LABEL_MAP.items():
        class_dir = data_dir / class_name
        wav_files = sorted(list(class_dir.rglob("*.wav")) + list(class_dir.rglob("*.WAV")))

        # Deduplicate (case-insensitive filesystems like Windows).
        seen = set()
        unique_wavs = []
        for f in wav_files:
            k = str(f).lower()
            if k not in seen:
                seen.add(k)
                unique_wavs.append(f)

        logger.info("Processing %d files in '%s/' ...", len(unique_wavs), class_name)

        for wav_path in unique_wavs:
            files_processed += 1
            if files_processed % LOG_EVERY_N == 0:
                logger.info(
                    "  ... %d files processed so far (%d skipped)",
                    files_processed,
                    len(skipped_files),
                )

            try:
                audio = preprocess_clip(wav_path)
                embedding = extract_embedding(audio, yamnet_model)
            except Exception as exc:
                reason = f"{type(exc).__name__}: {exc}"
                logger.warning("Skipping '%s': %s", wav_path, reason)
                skipped_files.append({"path": str(wav_path), "reason": reason})
                continue

            embeddings_list.append(embedding)
            labels_list.append(label)

            if class_name == "gunshot":
                gunshot_count += 1
            else:
                not_gunshot_count += 1

    if not embeddings_list:
        logger.error("No embeddings were extracted. Check your data directory.")
        sys.exit(1)

    X = np.stack(embeddings_list, axis=0).astype(np.float32)
    y = np.array(labels_list, dtype=np.float32)

    metadata: Dict = {
        "total_clips": len(embeddings_list),
        "gunshot_count": gunshot_count,
        "not_gunshot_count": not_gunshot_count,
        "skipped_files": skipped_files,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "yamnet_url": YAMNET_URL,
        "embedding_shape": list(X.shape),
        "label_dtype": str(y.dtype),
    }

    logger.info(
        "Extraction complete: %d clips (%d gunshot, %d not_gunshot, %d skipped).",
        metadata["total_clips"],
        gunshot_count,
        not_gunshot_count,
        len(skipped_files),
    )
    return X, y, metadata


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def _outputs_exist(output_dir: Path) -> bool:
    """Return True if all three output files are already present."""
    return (
        (output_dir / "X_embeddings.npy").exists()
        and (output_dir / "y_labels.npy").exists()
        and (output_dir / "metadata.json").exists()
    )


def save_outputs(
    X: np.ndarray,
    y: np.ndarray,
    metadata: Dict,
    output_dir: Path,
) -> None:
    """
    Save ``X_embeddings.npy``, ``y_labels.npy``, and ``metadata.json`` to
    ``output_dir``.

    Parameters
    ----------
    X : np.ndarray
        Shape ``(N, 1024)``, dtype ``float32``.
    y : np.ndarray
        Shape ``(N,)``, dtype ``float32``.
    metadata : dict
        Extraction metadata (see ``build_embedding_matrix``).
    output_dir : Path
        Destination directory. Created if it does not exist.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    x_path = output_dir / "X_embeddings.npy"
    y_path = output_dir / "y_labels.npy"
    meta_path = output_dir / "metadata.json"

    np.save(str(x_path), X)
    np.save(str(y_path), y)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Saved X_embeddings.npy  shape=%s  dtype=%s", X.shape, X.dtype)
    logger.info("Saved y_labels.npy       shape=%s  dtype=%s", y.shape, y.dtype)
    logger.info("Saved metadata.json      path=%s", meta_path)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """
    Command-line entry point for embedding extraction.

    Example
    -------
    .. code-block:: bash

        # First run (extracts and saves embeddings)
        python -m pipeline.extract_embeddings \\
            --data_dir data/raw \\
            --output_dir data/processed/embeddings

        # Re-run and overwrite existing outputs
        python -m pipeline.extract_embeddings \\
            --data_dir data/raw \\
            --output_dir data/processed/embeddings \\
            --force

    FUTURE PLACEHOLDER
    ------------------
    ``--zero_shot`` flag: when implemented, also compute and save
    ``zero_shot_scores.npy`` (shape ``(N,)``) — YAMNet's raw class-427 score
    per clip — for use as the Experiment 1 zero-shot baseline without any
    trained head.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Extract YAMNet embeddings from raw WAV files and save to disk.\n"
            "Reads from <data_dir>/gunshot/ and <data_dir>/not_gunshot/.\n"
            "Writes X_embeddings.npy, y_labels.npy, metadata.json to <output_dir>."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=Path("data/raw"),
        help="Root directory containing gunshot/ and not_gunshot/ subdirs. "
             "Default: data/raw",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("data/processed/embeddings"),
        help="Destination for output files. Default: data/processed/embeddings",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing output files. Without this flag, the script "
             "exits cleanly if outputs already exist.",
    )
    # FUTURE PLACEHOLDER: add --zero_shot flag here when implemented.

    args = parser.parse_args()

    # Check for existing outputs BEFORE loading YAMNet (fast-fail path).
    if _outputs_exist(args.output_dir) and not args.force:
        logger.info(
            "Output files already exist in '%s'. "
            "Pass --force to overwrite. Exiting.",
            args.output_dir,
        )
        sys.exit(0)

    # Load YAMNet (downloads ~17 MB on first run; cached locally afterwards).
    yamnet_model = load_yamnet(YAMNET_URL)

    # Extract embeddings.
    X, y, metadata = build_embedding_matrix(args.data_dir, yamnet_model)

    # Save outputs.
    save_outputs(X, y, metadata, args.output_dir)

    logger.info("Done. Run split_dataset.py next to create train/val/test splits.")


if __name__ == "__main__":
    main()
