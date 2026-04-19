"""
test_preprocessing.py
=====================
Unit tests for pipeline/preprocessing.py.

Run with:
    python -m pytest tests/test_preprocessing.py -v

These tests use synthetic audio arrays written to temporary WAV files via
soundfile (available as a librosa transitive dependency), so no real WAV
files from the dataset are required.
"""

import warnings
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from pipeline.preprocessing import (
    CLIP_LENGTH,
    MIN_CLIP_SAMPLES,
    TARGET_SR,
    audit_dataset,
    preprocess_clip,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_wav(tmp_path: Path, audio: np.ndarray, sr: int = TARGET_SR) -> Path:
    """Write a numpy array as a WAV file and return the path."""
    p = tmp_path / "clip.wav"
    sf.write(str(p), audio, sr)
    return p


# ---------------------------------------------------------------------------
# preprocess_clip tests
# ---------------------------------------------------------------------------


class TestPreprocessClip:
    def test_output_shape_and_dtype(self, tmp_path):
        """A standard 1-s, 44.1 kHz mono clip should produce (32000,) float32."""
        audio = np.random.uniform(-0.5, 0.5, 44100).astype(np.float32)
        wav = _write_wav(tmp_path, audio, sr=44100)
        result = preprocess_clip(wav)
        assert result.shape == (CLIP_LENGTH,)
        assert result.dtype == np.float32

    def test_output_range(self, tmp_path):
        """All values must be in [-1.0, +1.0]."""
        audio = np.random.uniform(-0.5, 0.5, 16000).astype(np.float32)
        wav = _write_wav(tmp_path, audio)
        result = preprocess_clip(wav)
        assert np.all(result >= -1.0), "Values below -1.0 found"
        assert np.all(result <= 1.0), "Values above +1.0 found"

    def test_stereo_converted_to_mono(self, tmp_path):
        """Stereo input should be averaged to mono."""
        stereo = np.random.uniform(-0.3, 0.3, (16000, 2)).astype(np.float32)
        wav = tmp_path / "stereo.wav"
        sf.write(str(wav), stereo, TARGET_SR)
        result = preprocess_clip(wav)
        assert result.shape == (CLIP_LENGTH,)

    def test_short_clip_warns_and_pads(self, tmp_path):
        """A 0.25-s clip should trigger RuntimeWarning and still produce (32000,)."""
        short_audio = np.zeros(int(TARGET_SR * 0.25), dtype=np.float32) + 0.1
        wav = _write_wav(tmp_path, short_audio)
        with pytest.warns(RuntimeWarning, match="below the minimum"):
            result = preprocess_clip(wav)
        assert result.shape == (CLIP_LENGTH,)

    def test_silent_clip_no_nan_no_exception(self, tmp_path):
        """A fully silent clip should return all zeros without NaN or exception."""
        silent = np.zeros(16000, dtype=np.float32)
        wav = _write_wav(tmp_path, silent)
        result = preprocess_clip(wav)
        assert result.shape == (CLIP_LENGTH,)
        assert not np.any(np.isnan(result)), "NaN found in silent clip output"
        assert np.all(result == 0.0), "Silent clip should produce all-zero output"

    def test_long_clip_center_trimmed(self, tmp_path):
        """A 5-s clip should be center-trimmed to exactly (32000,)."""
        long_audio = np.random.uniform(-0.5, 0.5, TARGET_SR * 5).astype(np.float32)
        wav = _write_wav(tmp_path, long_audio)
        result = preprocess_clip(wav)
        assert result.shape == (CLIP_LENGTH,)

    def test_exact_length_unchanged(self, tmp_path):
        """A clip already at exactly 32000 samples should not be padded or trimmed."""
        exact = np.random.uniform(-0.5, 0.5, CLIP_LENGTH).astype(np.float32)
        # Normalize manually so we know the expected output
        exact = exact / np.max(np.abs(exact))
        wav = _write_wav(tmp_path, exact)
        result = preprocess_clip(wav)
        assert result.shape == (CLIP_LENGTH,)
        # Max amplitude should be ~1.0 (already normalized)
        assert abs(float(np.max(np.abs(result))) - 1.0) < 1e-5

    def test_file_not_found_raises(self, tmp_path):
        """Missing file should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            preprocess_clip(tmp_path / "nonexistent.wav")

    def test_normalization_peak_is_one(self, tmp_path):
        """After normalization, max(abs(output)) should be 1.0 for non-silent clips."""
        audio = np.random.uniform(-0.3, 0.3, 16000).astype(np.float32)
        wav = _write_wav(tmp_path, audio)
        result = preprocess_clip(wav)
        assert abs(float(np.max(np.abs(result))) - 1.0) < 1e-5

    def test_resampling_from_non_16k(self, tmp_path):
        """Input at 22050 Hz should be resampled to produce (32000,) output."""
        audio = np.random.uniform(-0.5, 0.5, 22050).astype(np.float32)
        wav = _write_wav(tmp_path, audio, sr=22050)
        result = preprocess_clip(wav)
        assert result.shape == (CLIP_LENGTH,)


# ---------------------------------------------------------------------------
# audit_dataset tests
# ---------------------------------------------------------------------------


class TestAuditDataset:
    def test_audit_empty_directory(self, tmp_path):
        """An empty directory should return all zeros."""
        result = audit_dataset(tmp_path)
        assert result["total_files"] == 0
        assert result["too_short"] == 0
        assert result["too_long"] == 0
        assert result["nearly_silent"] == 0
        assert result["has_nan"] == 0

    def test_audit_detects_short_file(self, tmp_path):
        """A file shorter than MIN_CLIP_SAMPLES should be counted as too_short."""
        short = np.zeros(100, dtype=np.float32) + 0.1
        sf.write(str(tmp_path / "short.wav"), short, TARGET_SR)
        result = audit_dataset(tmp_path)
        assert result["total_files"] == 1
        assert result["too_short"] == 1

    def test_audit_detects_long_file(self, tmp_path):
        """A file longer than CLIP_LENGTH should be counted as too_long."""
        long_audio = np.random.uniform(-0.5, 0.5, CLIP_LENGTH * 3).astype(np.float32)
        sf.write(str(tmp_path / "long.wav"), long_audio, TARGET_SR)
        result = audit_dataset(tmp_path)
        assert result["too_long"] == 1

    def test_audit_detects_silent_file(self, tmp_path):
        """A silent file should be counted as nearly_silent."""
        silent = np.zeros(16000, dtype=np.float32)
        sf.write(str(tmp_path / "silent.wav"), silent, TARGET_SR)
        result = audit_dataset(tmp_path)
        assert result["nearly_silent"] == 1

    def test_audit_does_not_modify_files(self, tmp_path):
        """audit_dataset must not create, modify, or delete any files."""
        audio = np.random.uniform(-0.5, 0.5, 16000).astype(np.float32)
        wav = tmp_path / "test.wav"
        sf.write(str(wav), audio, TARGET_SR)
        mtime_before = wav.stat().st_mtime
        audit_dataset(tmp_path)
        mtime_after = wav.stat().st_mtime
        assert mtime_before == mtime_after, "File was modified during audit"

    def test_audit_counts_total_files(self, tmp_path):
        """total_files should equal the number of WAV files in the directory."""
        for i in range(3):
            audio = np.random.uniform(-0.5, 0.5, 16000).astype(np.float32)
            sf.write(str(tmp_path / f"clip_{i}.wav"), audio, TARGET_SR)
        result = audit_dataset(tmp_path)
        assert result["total_files"] == 3

    def test_audit_details_list_populated(self, tmp_path):
        """details list should have one entry per flagged file."""
        short = np.zeros(100, dtype=np.float32) + 0.1
        sf.write(str(tmp_path / "short.wav"), short, TARGET_SR)
        result = audit_dataset(tmp_path)
        assert len(result["details"]) >= 1
        detail = result["details"][0]
        assert "path" in detail
        assert "issues" in detail
        assert "too_short" in detail["issues"]
