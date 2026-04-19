"""
test_extract_embeddings.py
==========================
Unit tests for pipeline/extract_embeddings.py.

These tests mock the YAMNet model to avoid requiring a TensorFlow Hub
download in CI, while fully exercising the extraction and I/O logic.

Run with:
    python -m pytest tests/test_extract_embeddings.py -v
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import soundfile as sf

from pipeline.extract_embeddings import (
    GUNSHOT_CLASS_IDX,
    _outputs_exist,
    build_embedding_matrix,
    extract_embedding,
    extract_zero_shot_score,
    save_outputs,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_fake_yamnet():
    """
    Return a mock YAMNet model that produces plausible output shapes.
    scores: (3, 521), embeddings: (3, 1024), log_mel: (3, 64, 96)
    """
    mock_model = MagicMock()
    fake_scores = np.random.rand(3, 521).astype(np.float32)
    fake_embeddings = np.random.rand(3, 1024).astype(np.float32)
    fake_log_mel = np.random.rand(3, 64, 96).astype(np.float32)

    import tensorflow as tf
    mock_model.return_value = (
        tf.constant(fake_scores),
        tf.constant(fake_embeddings),
        tf.constant(fake_log_mel),
    )
    return mock_model


@pytest.fixture
def fake_yamnet():
    return _make_fake_yamnet()


@pytest.fixture
def data_dir(tmp_path):
    """Create a minimal data/raw directory with one WAV per class."""
    for class_name in ("gunshot", "not_gunshot"):
        class_dir = tmp_path / class_name
        class_dir.mkdir()
        audio = np.random.uniform(-0.5, 0.5, 16000).astype(np.float32)
        sf.write(str(class_dir / "clip_0.wav"), audio, 16000)
    return tmp_path


# ---------------------------------------------------------------------------
# extract_embedding tests
# ---------------------------------------------------------------------------


class TestExtractEmbedding:
    def test_output_shape(self, fake_yamnet):
        """extract_embedding should return shape (1024,)."""
        audio = np.random.rand(32000).astype(np.float32)
        result = extract_embedding(audio, fake_yamnet)
        assert result.shape == (1024,)

    def test_output_dtype(self, fake_yamnet):
        """Output should be float32."""
        audio = np.random.rand(32000).astype(np.float32)
        result = extract_embedding(audio, fake_yamnet)
        assert result.dtype == np.float32

    def test_zero_frames_raises(self):
        """If YAMNet returns 0 frames, a ValueError should be raised."""
        import tensorflow as tf
        mock_model = MagicMock()
        mock_model.return_value = (
            tf.constant(np.zeros((0, 521), dtype=np.float32)),
            tf.constant(np.zeros((0, 1024), dtype=np.float32)),
            tf.constant(np.zeros((0, 64, 96), dtype=np.float32)),
        )
        audio = np.random.rand(32000).astype(np.float32)
        with pytest.raises(ValueError, match="0 frames"):
            extract_embedding(audio, mock_model)


# ---------------------------------------------------------------------------
# extract_zero_shot_score tests
# ---------------------------------------------------------------------------


class TestExtractZeroShotScore:
    def test_output_is_float(self, fake_yamnet):
        """extract_zero_shot_score should return a Python float."""
        audio = np.random.rand(32000).astype(np.float32)
        score = extract_zero_shot_score(audio, fake_yamnet)
        assert isinstance(score, float)

    def test_output_in_range(self, fake_yamnet):
        """Score should be in [0.0, 1.0] since YAMNet outputs softmax scores."""
        audio = np.random.rand(32000).astype(np.float32)
        score = extract_zero_shot_score(audio, fake_yamnet)
        assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# _outputs_exist tests
# ---------------------------------------------------------------------------


class TestOutputsExist:
    def test_returns_false_when_missing(self, tmp_path):
        assert _outputs_exist(tmp_path) is False

    def test_returns_true_when_all_present(self, tmp_path):
        (tmp_path / "X_embeddings.npy").touch()
        (tmp_path / "y_labels.npy").touch()
        (tmp_path / "metadata.json").touch()
        assert _outputs_exist(tmp_path) is True

    def test_returns_false_when_partial(self, tmp_path):
        (tmp_path / "X_embeddings.npy").touch()
        assert _outputs_exist(tmp_path) is False


# ---------------------------------------------------------------------------
# save_outputs tests
# ---------------------------------------------------------------------------


class TestSaveOutputs:
    def test_files_created(self, tmp_path):
        """save_outputs should create all three output files."""
        X = np.random.rand(10, 1024).astype(np.float32)
        y = np.array([1.0] * 5 + [0.0] * 5, dtype=np.float32)
        metadata = {
            "total_clips": 10,
            "gunshot_count": 5,
            "not_gunshot_count": 5,
            "skipped_files": [],
            "timestamp": "2026-04-19T00:00:00+00:00",
            "yamnet_url": "https://tfhub.dev/google/yamnet/1",
            "embedding_shape": [10, 1024],
            "label_dtype": "float32",
        }
        save_outputs(X, y, metadata, tmp_path)
        assert (tmp_path / "X_embeddings.npy").exists()
        assert (tmp_path / "y_labels.npy").exists()
        assert (tmp_path / "metadata.json").exists()

    def test_saved_arrays_loadable_and_correct(self, tmp_path):
        """Arrays saved by save_outputs should load back correctly."""
        X = np.random.rand(4, 1024).astype(np.float32)
        y = np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float32)
        save_outputs(X, y, {"total_clips": 4, "skipped_files": []}, tmp_path)
        X_loaded = np.load(str(tmp_path / "X_embeddings.npy"))
        y_loaded = np.load(str(tmp_path / "y_labels.npy"))
        np.testing.assert_array_almost_equal(X, X_loaded)
        np.testing.assert_array_equal(y, y_loaded)

    def test_metadata_json_parseable(self, tmp_path):
        """metadata.json should be valid JSON with required keys."""
        metadata = {
            "total_clips": 2,
            "gunshot_count": 1,
            "not_gunshot_count": 1,
            "skipped_files": [],
            "timestamp": "2026-04-19T00:00:00+00:00",
            "yamnet_url": "https://tfhub.dev/google/yamnet/1",
            "embedding_shape": [2, 1024],
            "label_dtype": "float32",
        }
        save_outputs(
            np.zeros((2, 1024), dtype=np.float32),
            np.array([1.0, 0.0], dtype=np.float32),
            metadata,
            tmp_path,
        )
        with open(tmp_path / "metadata.json") as f:
            loaded = json.load(f)
        for key in ("total_clips", "gunshot_count", "not_gunshot_count",
                    "skipped_files", "timestamp", "yamnet_url"):
            assert key in loaded


# ---------------------------------------------------------------------------
# build_embedding_matrix tests
# ---------------------------------------------------------------------------


class TestBuildEmbeddingMatrix:
    def test_returns_correct_shapes(self, data_dir, fake_yamnet):
        """build_embedding_matrix should return (2, 1024) X and (2,) y."""
        X, y, metadata = build_embedding_matrix(data_dir, fake_yamnet)
        assert X.shape == (2, 1024)
        assert y.shape == (2,)

    def test_labels_are_valid(self, data_dir, fake_yamnet):
        """y should only contain 0.0 and 1.0."""
        _, y, _ = build_embedding_matrix(data_dir, fake_yamnet)
        assert set(y.tolist()).issubset({0.0, 1.0})

    def test_metadata_counts_match(self, data_dir, fake_yamnet):
        """gunshot_count + not_gunshot_count should equal total_clips."""
        _, _, metadata = build_embedding_matrix(data_dir, fake_yamnet)
        assert (
            metadata["gunshot_count"] + metadata["not_gunshot_count"]
            == metadata["total_clips"]
        )

    def test_exits_on_missing_class_dir(self, tmp_path, fake_yamnet):
        """Should exit with sys.exit(1) if class directory is missing."""
        # Only create gunshot/, not not_gunshot/
        (tmp_path / "gunshot").mkdir()
        with pytest.raises(SystemExit):
            build_embedding_matrix(tmp_path, fake_yamnet)
