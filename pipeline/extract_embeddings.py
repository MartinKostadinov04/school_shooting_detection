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
        [--force] [--workers 8]

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

``--workers``
    Number of threads for parallel WAV loading. While the GPU runs YAMNet
    on the current clip, worker threads prefetch the next N files from disk.
    Default: ``os.cpu_count()``.

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
import collections
import itertools
import json
import logging
import os
import sys
import wave
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Generator, List, Tuple

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

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
GUNSHOT_CLASS_NAME: str = "Gunshot, gunfire"
LABEL_MAP: Dict[str, float] = {"gunshot": 1.0, "not_gunshot": 0.0}
LOG_EVERY_N: int = 500        # log progress every N files
PREFETCH_SIZE: int = 32       # how many files to keep loaded ahead of GPU


# ---------------------------------------------------------------------------
# Fast WAV loading (no librosa — files are already 16 kHz mono 16-bit)
# ---------------------------------------------------------------------------


def _load_wav_direct(path: Path) -> np.ndarray:
    """
    Load a WAV file using the stdlib ``wave`` module and return a float32
    waveform in ``[-1.0, +1.0]``.

    Assumes the file is already 16 kHz, mono, 16-bit PCM — which is true for
    all clips in this dataset (confirmed by wav_info.py audit). Using the
    stdlib avoids librosa's resampling overhead, making this ~10× faster.

    Parameters
    ----------
    path : Path
        Path to the WAV file.

    Returns
    -------
    np.ndarray
        Shape ``(32000,)``, dtype ``float32``, values in ``[-1.0, +1.0]``.
    """
    with wave.open(str(path), "rb") as wf:
        raw = wf.readframes(wf.getnframes())
    # int16 PCM → float32 in [-1, +1]
    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    return samples


# ---------------------------------------------------------------------------
# Prefetch pipeline
# ---------------------------------------------------------------------------


def _iter_with_prefetch(
    items: List[Tuple],
    n_workers: int,
) -> Generator[Tuple, None, None]:
    """
    Yield ``(item, audio)`` pairs while prefetching the next ``PREFETCH_SIZE``
    WAV files in a thread pool — overlapping disk I/O with GPU inference.

    The main thread always has a ready-loaded waveform waiting; it never
    stalls on disk I/O between YAMNet calls.

    Parameters
    ----------
    items : list of (Path, label)
        Ordered list of (wav_path, float_label) pairs to process.
    n_workers : int
        Number of threads for parallel file loading.

    Yields
    ------
    item : tuple
        The original ``(wav_path, label)`` pair.
    audio : np.ndarray
        Shape ``(32000,)``, float32 — loaded waveform ready for YAMNet.
    """
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        pending: collections.deque = collections.deque()
        it = iter(items)

        # Seed the prefetch window.
        for item in itertools.islice(it, PREFETCH_SIZE):
            pending.append((item, executor.submit(_load_wav_direct, item[0])))

        # Slide the window: for each new item submitted, yield the oldest.
        for item in it:
            pending.append((item, executor.submit(_load_wav_direct, item[0])))
            orig_item, future = pending.popleft()
            yield orig_item, future.result()

        # Drain remaining futures.
        while pending:
            orig_item, future = pending.popleft()
            yield orig_item, future.result()


# ---------------------------------------------------------------------------
# YAMNet helpers
# ---------------------------------------------------------------------------


def resolve_gunshot_class_idx(class_names: List[str]) -> int:
    """
    Return the index of ``"Gunshot, gunfire"`` in the YAMNet class map.

    Parameters
    ----------
    class_names : list of str
        521-entry list returned by ``load_class_map()``.

    Returns
    -------
    int
        Zero-based index of ``GUNSHOT_CLASS_NAME`` in ``class_names``.

    Raises
    ------
    ValueError
        If ``GUNSHOT_CLASS_NAME`` is absent from the class map — which would
        indicate the model asset has changed in an unexpected way.
    """
    try:
        return class_names.index(GUNSHOT_CLASS_NAME)
    except ValueError:
        raise ValueError(
            f"'{GUNSHOT_CLASS_NAME}' not found in YAMNet class map "
            f"({len(class_names)} classes). The model asset may have changed."
        )


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


def load_class_map(yamnet_model: object) -> List[str]:
    """
    Load the AudioSet class name list bundled with YAMNet.

    YAMNet outputs numerical scores for 521 classes. The human-readable names
    (e.g. ``"Gunshot, gunfire"`` for index 427) live in a CSV file that ships
    as a model asset alongside the SavedModel weights.

    Parameters
    ----------
    yamnet_model : TF-Hub SavedModel
        Already-loaded YAMNet model (from ``load_yamnet()``).

    Returns
    -------
    list of str
        521 class display names, ordered by class index.
        Use ``resolve_gunshot_class_idx(class_names)`` to find the correct
        index for ``"Gunshot, gunfire"`` — do not assume it is 427.
    """
    import csv
    class_map_path = yamnet_model.class_map_path().numpy().decode("utf-8")
    class_names: List[str] = []
    with open(class_map_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            class_names.append(row["display_name"])
    gunshot_idx = resolve_gunshot_class_idx(class_names)
    logger.info(
        "Loaded %d AudioSet class names. '%s' is at index %d.",
        len(class_names), GUNSHOT_CLASS_NAME, gunshot_idx,
    )
    return class_names


def _yamnet_infer(
    audio: np.ndarray,
    yamnet_model: object,
    class_names: List[str],
    gunshot_class_idx: int,
) -> Tuple[np.ndarray, float, int, float, str]:
    """
    Run a **single** YAMNet forward pass and return all outputs at once.

    YAMNet outputs numerical float scores for 521 AudioSet classes — it does
    NOT return strings directly. We resolve the top class index to a
    human-readable name using ``class_names`` (loaded via ``load_class_map()``).

    Parameters
    ----------
    audio : np.ndarray
        Shape ``(32000,)``, float32, values in ``[-1.0, +1.0]``.
    yamnet_model : TF-Hub SavedModel
    class_names : list of str
        521 AudioSet display names from ``load_class_map()``.
        ``class_names[idx]`` gives the string for class index ``idx``.

    Returns
    -------
    embedding : np.ndarray
        Shape ``(1024,)``, float32 — mean-pooled clip embedding.
    zero_shot_score : float
        Mean ``"Gunshot, gunfire"`` class score — YAMNet's gunshot confidence
        in ``[0, 1]``. Uses ``gunshot_class_idx``, not the hardcoded 427.
    top_class_idx : int
        AudioSet class index (0–520) with the highest mean score.
    top_class_score : float
        Confidence of the top predicted class.
    top_class_name : str
        Human-readable display name for ``top_class_idx``,
        e.g. ``"Gunshot, gunfire"``, ``"Fireworks"``, ``"Dog"``.

    Raises
    ------
    ValueError
        If YAMNet returns 0 frames.
    """
    waveform = tf.constant(audio, dtype=tf.float32)
    scores, embeddings, _ = yamnet_model(waveform)

    if embeddings.shape[0] == 0:
        raise ValueError(
            "YAMNet returned 0 frames. The waveform may be too short for "
            "YAMNet's internal windowing (minimum ~0.96 s recommended)."
        )

    mean_scores = tf.reduce_mean(scores, axis=0).numpy()                       # (521,)
    embedding = tf.reduce_mean(embeddings, axis=0).numpy().astype(np.float32)  # (1024,)
    zero_shot_score = float(mean_scores[gunshot_class_idx])
    top_class_idx = int(np.argmax(mean_scores))
    top_class_score = float(mean_scores[top_class_idx])
    top_class_name = class_names[top_class_idx]

    return embedding, zero_shot_score, top_class_idx, top_class_score, top_class_name


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
    gunshot_class_idx: int,
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
    """
    waveform = tf.constant(audio, dtype=tf.float32)
    scores, _, _ = yamnet_model(waveform)
    # scores shape: (num_frames, 521)
    gunshot_scores = scores[:, gunshot_class_idx]
    return float(tf.reduce_mean(gunshot_scores).numpy())


def extract_yamnet_classification(
    audio: np.ndarray,
    yamnet_model: object,
) -> Tuple[int, float]:
    """
    Return YAMNet's top predicted AudioSet class for a clip.

    Averages scores across all frames and returns the class with the
    highest mean score, along with its confidence value.

    Parameters
    ----------
    audio : np.ndarray
        Shape ``(32000,)``, dtype ``float32``, values in ``[-1.0, +1.0]``.
    yamnet_model : TF-Hub SavedModel
        Loaded YAMNet model returned by ``load_yamnet()``.

    Returns
    -------
    top_class_idx : int
        AudioSet class index (0–520) with the highest mean score.
    top_class_score : float
        Confidence of that class, in ``[0.0, 1.0]``.
    """
    waveform = tf.constant(audio, dtype=tf.float32)
    scores, _, _ = yamnet_model(waveform)
    # Mean over frames → (521,)
    mean_scores = tf.reduce_mean(scores, axis=0).numpy()
    top_class_idx = int(np.argmax(mean_scores))
    top_class_score = float(mean_scores[top_class_idx])
    return top_class_idx, top_class_score


# ---------------------------------------------------------------------------
# Core extraction logic
# ---------------------------------------------------------------------------


def build_embedding_matrix(
    data_dir: Path,
    yamnet_model: object,
    n_workers: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict]:
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

    Parameters
    ----------
    n_workers : int
        Number of threads for parallel WAV prefetching. ``0`` means use
        ``os.cpu_count()``.

    Notes
    -----
    WAV files are loaded in parallel by a thread pool while the GPU runs
    YAMNet on the previously loaded clip. Each clip runs a single YAMNet
    forward pass (``_yamnet_infer``) that returns the embedding, zero-shot
    score, and top class simultaneously — avoiding 3× redundant inference.

    Any file that raises an exception is skipped and recorded in
    ``metadata["skipped_files"]`` with the error message.
    """
    n_workers = n_workers or os.cpu_count() or 4

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

    # Build a flat ordered list of (wav_path, label, class_name) across both classes.
    all_items: List[Tuple[Path, float, str]] = []
    for class_name, label in LABEL_MAP.items():
        class_dir = data_dir / class_name
        wav_files = sorted(list(class_dir.rglob("*.wav")) + list(class_dir.rglob("*.WAV")))
        seen: set = set()
        for f in wav_files:
            k = str(f).lower()
            if k not in seen:
                seen.add(k)
                all_items.append((f, label, class_name))

    logger.info(
        "Starting extraction: %d files, %d prefetch threads, prefetch window=%d",
        len(all_items), n_workers, PREFETCH_SIZE,
    )

    # Load class names once — maps integer index → display string.
    class_names = load_class_map(yamnet_model)
    gunshot_class_idx = resolve_gunshot_class_idx(class_names)

    embeddings_list: List[np.ndarray] = []
    labels_list: List[float] = []
    zero_shot_scores_list: List[float] = []
    top_class_idx_list: List[int] = []
    top_class_score_list: List[float] = []
    top_class_name_list: List[str] = []
    skipped_files: List[Dict] = []
    gunshot_count = 0
    not_gunshot_count = 0

    # _iter_with_prefetch yields ((wav_path, label, class_name), audio_array)
    for files_processed, ((wav_path, label, class_name), audio) in enumerate(
        _iter_with_prefetch(all_items, n_workers), start=1
    ):
        if files_processed % LOG_EVERY_N == 0:
            logger.info(
                "  %d / %d  (skipped: %d)",
                files_processed, len(all_items), len(skipped_files),
            )

        try:
            embedding, zero_shot, top_idx, top_score, top_name = _yamnet_infer(
                audio, yamnet_model, class_names, gunshot_class_idx
            )
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"
            logger.warning("Skipping '%s': %s", wav_path, reason)
            skipped_files.append({"path": str(wav_path), "reason": reason})
            continue

        embeddings_list.append(embedding)
        labels_list.append(label)
        zero_shot_scores_list.append(zero_shot)
        top_class_idx_list.append(top_idx)
        top_class_score_list.append(top_score)
        top_class_name_list.append(top_name)

        if class_name == "gunshot":
            gunshot_count += 1
        else:
            not_gunshot_count += 1

    if not embeddings_list:
        logger.error("No embeddings were extracted. Check your data directory.")
        sys.exit(1)

    X = np.stack(embeddings_list, axis=0).astype(np.float32)
    y = np.array(labels_list, dtype=np.float32)
    zero_shot_scores = np.array(zero_shot_scores_list, dtype=np.float32)
    top_class_indices = np.array(top_class_idx_list, dtype=np.int32)
    top_class_scores = np.array(top_class_score_list, dtype=np.float32)
    top_class_names = np.array(top_class_name_list, dtype=object)  # string array

    metadata: Dict = {
        "total_clips": len(embeddings_list),
        "gunshot_count": gunshot_count,
        "not_gunshot_count": not_gunshot_count,
        "skipped_files": skipped_files,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "yamnet_url": YAMNET_URL,
        "embedding_shape": list(X.shape),
        "label_dtype": str(y.dtype),
        "yamnet_class_count": len(class_names),
        "gunshot_audioset_class_idx": gunshot_class_idx,
        "gunshot_audioset_class_name": class_names[gunshot_class_idx],
        "class_map": class_names,  # full 521-entry list for reference
    }

    logger.info(
        "Extraction complete: %d clips (%d gunshot, %d not_gunshot, %d skipped).",
        metadata["total_clips"],
        gunshot_count,
        not_gunshot_count,
        len(skipped_files),
    )
    return X, y, zero_shot_scores, top_class_indices, top_class_scores, top_class_names, metadata


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def _outputs_exist(output_dir: Path) -> bool:
    """Return True if all output files are already present."""
    return all(
        (output_dir / name).exists()
        for name in (
            "X_embeddings.npy",
            "y_labels.npy",
            "zero_shot_scores.npy",
            "yamnet_top_class_indices.npy",
            "yamnet_top_class_scores.npy",
            "yamnet_top_class_names.npy",
            "metadata.json",
        )
    )


def save_outputs(
    X: np.ndarray,
    y: np.ndarray,
    zero_shot_scores: np.ndarray,
    top_class_indices: np.ndarray,
    top_class_scores: np.ndarray,
    top_class_names: np.ndarray,
    metadata: Dict,
    output_dir: Path,
) -> None:
    """
    Save all extraction outputs to ``output_dir``.

    Files written
    -------------
    ``X_embeddings.npy``
        Shape ``(N, 1024)``, float32 — YAMNet mean-pooled embeddings.
    ``y_labels.npy``
        Shape ``(N,)``, float32 — ground-truth labels (1.0 = gunshot, 0.0 = not).
    ``zero_shot_scores.npy``
        Shape ``(N,)``, float32 — YAMNet's mean class-427 score per clip.
        Use this directly as a zero-shot gunshot detector (no head needed).
    ``yamnet_top_class_indices.npy``
        Shape ``(N,)``, int32 — AudioSet class index with the highest mean
        score per clip (YAMNet's top prediction, ignoring our labels).
    ``yamnet_top_class_scores.npy``
        Shape ``(N,)``, float32 — confidence of the top predicted class.
    ``yamnet_top_class_names.npy``
        Shape ``(N,)``, object (str) — display name for the top predicted class,
        e.g. ``"Gunshot, gunfire"``, ``"Fireworks"``, ``"Dog"``.
    ``metadata.json``
        Processing statistics, run provenance, and full 521-entry class map.

    Parameters
    ----------
    X : np.ndarray
    y : np.ndarray
    zero_shot_scores : np.ndarray
    top_class_indices : np.ndarray
    top_class_scores : np.ndarray
    metadata : dict
    output_dir : Path
        Destination directory. Created if it does not exist.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(str(output_dir / "X_embeddings.npy"), X)
    np.save(str(output_dir / "y_labels.npy"), y)
    np.save(str(output_dir / "zero_shot_scores.npy"), zero_shot_scores)
    np.save(str(output_dir / "yamnet_top_class_indices.npy"), top_class_indices)
    np.save(str(output_dir / "yamnet_top_class_scores.npy"), top_class_scores)
    np.save(str(output_dir / "yamnet_top_class_names.npy"), top_class_names)

    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Saved X_embeddings.npy             shape=%s", X.shape)
    logger.info("Saved y_labels.npy                 shape=%s", y.shape)
    logger.info("Saved zero_shot_scores.npy         shape=%s", zero_shot_scores.shape)
    logger.info("Saved yamnet_top_class_indices.npy  shape=%s", top_class_indices.shape)
    logger.info("Saved yamnet_top_class_scores.npy   shape=%s", top_class_scores.shape)
    logger.info("Saved yamnet_top_class_names.npy    shape=%s", top_class_names.shape)
    logger.info("Saved metadata.json                (includes full 521-class map)")


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
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Threads for parallel WAV prefetching. 0 = os.cpu_count() (default).",
    )

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

    # Extract embeddings and YAMNet classifications.
    X, y, zero_shot_scores, top_class_indices, top_class_scores, top_class_names, metadata = \
        build_embedding_matrix(args.data_dir, yamnet_model, n_workers=args.workers)

    # Save outputs.
    save_outputs(X, y, zero_shot_scores, top_class_indices, top_class_scores,
                 top_class_names, metadata, args.output_dir)

    logger.info("Done. Run split_dataset.py next to create train/val/test splits.")


if __name__ == "__main__":
    main()
