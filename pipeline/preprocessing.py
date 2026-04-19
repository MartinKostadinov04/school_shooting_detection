"""
preprocessing.py
================
Audio preprocessing for the YAMNet-based gunshot detection pipeline.

Operates exclusively with librosa and numpy — no TensorFlow, no soundfile.
This keeps the module reusable in non-TF contexts (e.g. data auditing scripts,
server-side preprocessing, testing environments without a GPU).

Pipeline contract
-----------------
  Input  : path to any WAV file at any sample rate, any number of channels
  Output : np.ndarray  shape (32000,)  dtype float32  values in [-1.0, +1.0]

The output is the exact format required by YAMNet:
  - 16 kHz sample rate
  - Mono (single channel)
  - float32 dtype
  - Amplitude range [-1.0, +1.0]
  - Exactly 32000 samples (2.0 seconds)

YAMNet handles its own internal mel spectrogram computation — do NOT pass
spectrograms to YAMNet. This module produces raw audio waveforms only.
"""

import logging
import warnings
from pathlib import Path
from typing import Dict, List, Union

import librosa
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TARGET_SR: int = 16_000
"""YAMNet's required input sample rate (Hz)."""

CLIP_LENGTH: int = 32_000
"""Fixed clip length in samples = 2.0 seconds at 16 kHz."""

MIN_CLIP_SAMPLES: int = 8_000
"""Minimum acceptable clip length = 0.5 seconds at 16 kHz.
Clips shorter than this after loading are flagged as potentially bad recordings."""

SILENCE_THRESHOLD: float = 1e-6
"""Maximum peak amplitude below which a clip is considered nearly silent.
Used in audit_dataset() for data quality reporting."""

_NORMALIZE_GUARD: float = 1e-9
"""Floor used when normalizing to prevent division by zero on silent clips."""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def preprocess_clip(path: Union[str, Path]) -> np.ndarray:
    """
    Load, resample, mono-mix, normalize, and fix the length of a single audio clip.

    This is the canonical preprocessing function for the YAMNet pipeline.
    Its output is passed directly to YAMNet — no further preprocessing needed.

    Parameters
    ----------
    path : str or Path
        Filesystem path to a WAV file. The file may be at any sample rate
        and any number of channels; both are handled internally.

    Returns
    -------
    np.ndarray
        Shape ``(32000,)``, dtype ``float32``, values in ``[-1.0, +1.0]``.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist on disk.
    Exception
        Any exception raised by librosa (e.g. corrupt file, unsupported
        codec) propagates unchanged so the calling script can log the
        failure in ``skipped_files``.

    Warns
    -----
    RuntimeWarning
        If the clip is shorter than 0.5 s (< 8000 samples at 16 kHz) after
        loading and resampling. The file is flagged in the log and then
        center-padded to 32000 samples so the pipeline can continue.
        Short clips often indicate bad recordings; inspect them manually.

    Notes
    -----
    Processing steps (in order):

    1. **Load & resample** — ``librosa.load(path, sr=16000, mono=False)``.
       The ``sr=16000`` argument forces explicit resampling regardless of the
       source sample rate. We load with ``mono=False`` so we can handle the
       channel axis ourselves (step 2).

    2. **Ensure mono** — if the audio is multi-channel (e.g. stereo), we
       average all channels: ``np.mean(audio, axis=0)``. librosa with
       ``mono=False`` returns shape ``(channels, samples)`` for multi-channel
       or ``(samples,)`` for single-channel.

    3. **Cast to float32** — ensures consistent dtype regardless of source
       format.

    4. **Short-clip warning** — if fewer than 8000 samples remain after
       resampling, emit a ``RuntimeWarning`` and log at WARNING level.
       Processing continues (clip is padded in step 6); the warning allows
       the caller to track problematic files.

    5. **Normalize to [-1, +1]** — divide by the peak absolute amplitude.
       If the peak is below ``_NORMALIZE_GUARD`` (effectively silent), skip
       division and return zeros to avoid producing NaN values.

    6. **Fix length to 32000 samples**:
       - *Short clips*: center-pad with zeros.
         ``pad_before = total_pad // 2``, ``pad_after = total_pad - pad_before``
       - *Long clips*: center-trim.
         ``start = (len - 32000) // 2``, then slice.
       - *Exact length*: pass through unchanged.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    # Step 1: Load and resample to 16 kHz.
    # mono=False preserves channels so we can average them ourselves (step 2).
    audio, _ = librosa.load(str(path), sr=TARGET_SR, mono=False)

    # Step 2: Ensure mono.
    # librosa returns (samples,) for mono and (channels, samples) for multi-channel.
    if audio.ndim == 2:
        audio = np.mean(audio, axis=0)

    # Step 3: Cast to float32.
    audio = audio.astype(np.float32)

    # Step 4: Short-clip warning.
    if len(audio) < MIN_CLIP_SAMPLES:
        msg = (
            f"Clip at '{path}' is {len(audio)} samples "
            f"({len(audio) / TARGET_SR:.3f} s) which is below the minimum "
            f"of {MIN_CLIP_SAMPLES} samples ({MIN_CLIP_SAMPLES / TARGET_SR:.1f} s). "
            "This may indicate a bad recording. The clip will be padded, but "
            "you should inspect it manually."
        )
        warnings.warn(msg, RuntimeWarning, stacklevel=2)
        logger.warning(msg)

    # Step 5: Normalize to [-1, +1].
    peak = float(np.max(np.abs(audio)))
    if peak >= _NORMALIZE_GUARD:
        audio = audio / peak
    # If peak < _NORMALIZE_GUARD the clip is effectively silent; leave as zeros.

    # Step 6: Fix length to exactly CLIP_LENGTH samples.
    n = len(audio)
    if n == CLIP_LENGTH:
        pass  # already correct length
    elif n < CLIP_LENGTH:
        # Center-pad with zeros.
        total_pad = CLIP_LENGTH - n
        pad_before = total_pad // 2
        pad_after = total_pad - pad_before
        audio = np.pad(audio, (pad_before, pad_after), mode="constant", constant_values=0.0)
    else:
        # Center-trim.
        start = (n - CLIP_LENGTH) // 2
        audio = audio[start : start + CLIP_LENGTH]

    assert audio.shape == (CLIP_LENGTH,), f"Unexpected shape after fix-length: {audio.shape}"
    assert audio.dtype == np.float32, f"Unexpected dtype after processing: {audio.dtype}"

    return audio


def audit_dataset(directory: Union[str, Path]) -> Dict:
    """
    Scan a directory tree of WAV files and report data quality counts **before**
    any processing occurs.

    This is a pre-flight check. It reads every WAV file in the directory (and
    all subdirectories), loads it at 16 kHz, and tallies how many files fall
    into each quality category. It does **not** modify any file.

    Run this once before ``extract_embeddings.py`` to catch data issues early.

    Parameters
    ----------
    directory : str or Path
        Root directory to scan recursively for ``*.wav`` files
        (case-insensitive — both ``*.wav`` and ``*.WAV`` are matched).

    Returns
    -------
    dict
        Keys:

        ``total_files`` : int
            Total number of WAV files found.
        ``too_short`` : int
            Files with fewer than ``MIN_CLIP_SAMPLES`` (8000) samples at 16 kHz.
        ``too_long`` : int
            Files with more than ``CLIP_LENGTH`` (32000) samples at 16 kHz.
        ``nearly_silent`` : int
            Files where ``max(abs(audio)) < SILENCE_THRESHOLD``.
        ``has_nan`` : int
            Files containing NaN values after loading.
        ``unreadable`` : int
            Files that raised an exception during loading.
        ``details`` : list[dict]
            One entry per *flagged* file (any issue detected). Each entry has::

                {
                    "path": str,            # absolute path
                    "issues": list[str],    # list of issue labels
                    "duration_samples": int # samples at 16 kHz (-1 if unreadable)
                }

    Notes
    -----
    A single file can appear in multiple issue categories simultaneously
    (e.g. a file can be both ``too_short`` and ``nearly_silent``). The
    ``details`` list captures all issues for each flagged file.
    """
    directory = Path(directory)

    # Collect WAV files — case-insensitive by collecting both lowercase and uppercase.
    wav_files: List[Path] = sorted(
        list(directory.rglob("*.wav")) + list(directory.rglob("*.WAV"))
    )
    # Deduplicate in case the filesystem is case-insensitive (Windows).
    seen = set()
    unique_wav_files = []
    for f in wav_files:
        key = str(f).lower()
        if key not in seen:
            seen.add(key)
            unique_wav_files.append(f)
    wav_files = unique_wav_files

    counts: Dict = {
        "total_files": len(wav_files),
        "too_short": 0,
        "too_long": 0,
        "nearly_silent": 0,
        "has_nan": 0,
        "unreadable": 0,
        "details": [],
    }

    for wav_path in wav_files:
        try:
            audio, _ = librosa.load(str(wav_path), sr=TARGET_SR, mono=True)
        except Exception as exc:
            counts["unreadable"] += 1
            counts["details"].append(
                {
                    "path": str(wav_path),
                    "issues": ["unreadable"],
                    "duration_samples": -1,
                    "error": str(exc),
                }
            )
            logger.warning("Could not load '%s': %s", wav_path, exc)
            continue

        n_samples = len(audio)
        issues: List[str] = []

        if n_samples < MIN_CLIP_SAMPLES:
            counts["too_short"] += 1
            issues.append("too_short")

        if n_samples > CLIP_LENGTH:
            counts["too_long"] += 1
            issues.append("too_long")

        peak = float(np.max(np.abs(audio))) if n_samples > 0 else 0.0
        if peak < SILENCE_THRESHOLD:
            counts["nearly_silent"] += 1
            issues.append("nearly_silent")

        if np.any(np.isnan(audio)):
            counts["has_nan"] += 1
            issues.append("has_nan")

        if issues:
            counts["details"].append(
                {
                    "path": str(wav_path),
                    "issues": issues,
                    "duration_samples": n_samples,
                }
            )

    logger.info(
        "Audit complete — %d files: %d too_short, %d too_long, "
        "%d nearly_silent, %d has_nan, %d unreadable",
        counts["total_files"],
        counts["too_short"],
        counts["too_long"],
        counts["nearly_silent"],
        counts["has_nan"],
        counts["unreadable"],
    )
    return counts
