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
from unittest.mock import MagicMock

import numpy as np
import pytest
import soundfile as sf

from pipeline.extract_embeddings import (
    GUNSHOT_CLASS_NAME,
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
    Also mocks class_map_path() for load_class_map().
    """
    import tensorflow as tf
    import tempfile, csv, os

    mock_model = MagicMock()
    fake_scores = np.random.rand(3, 521).astype(np.float32)
    fake_embeddings = np.random.rand(3, 1024).astype(np.float32)
    fake_log_mel = np.random.rand(3, 64, 96).astype(np.float32)

    mock_model.return_value = (
        tf.constant(fake_scores),
        tf.constant(fake_embeddings),
        tf.constant(fake_log_mel),
    )

    # Write a minimal class map CSV so load_class_map() works with this mock.
    csv_path = os.path.join(tempfile.mkdtemp(), "yamnet_class_map.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["index", "mid", "display_name"])
        writer.writeheader()
        for i in range(521):
            # Put the real gunshot class name at index 421 so resolve_gunshot_class_idx() works.
            name = "Gunshot, gunfire" if i == 421 else f"Class_{i}"
            writer.writerow({"index": i, "mid": f"/m/{i:05d}", "display_name": name})

    mock_model.class_map_path.return_value = tf.constant(csv_path.encode())
    return mock_model


def _make_dummy_outputs(n: int = 4):
    """Return dummy arrays and metadata matching current save_outputs signature."""
    X = np.random.rand(n, 1024).astype(np.float32)
    y = np.array([1.0, 0.0] * (n // 2), dtype=np.float32)
    zero_shot = np.random.rand(n).astype(np.float32)
    top_idx = np.zeros(n, dtype=np.int32)
    top_scores = np.random.rand(n).astype(np.float32)
    top_names = np.array([f"Class_{i}" for i in range(n)], dtype=object)
    metadata = {
        "total_clips": n,
        "gunshot_count": n // 2,
        "not_gunshot_count": n // 2,
        "skipped_files": [],
        "timestamp": "2026-04-19T00:00:00+00:00",
        "yamnet_url": "https://tfhub.dev/google/yamnet/1",
        "embedding_shape": [n, 1024],
        "label_dtype": "float32",
        "yamnet_class_count": 521,
        "gunshot_audioset_class_idx": 0,  # dummy index; real value resolved via resolve_gunshot_class_idx()
        "gunshot_audioset_class_name": "Gunshot, gunfire",
        "class_map": [f"Class_{i}" for i in range(521)],
    }
    return X, y, zero_shot, top_idx, top_scores, top_names, metadata


@pytest.fixture
def fake_yamnet():
    return _make_fake_yamnet()


@pytest.fixture
def data_dir(tmp_path):
    """Minimal data/raw with one 16kHz mono WAV per class."""
    for class_name in ("gunshot", "not_gunshot"):
        class_dir = tmp_path / class_name
        class_dir.mkdir()
        audio = np.random.uniform(-0.5, 0.5, 32000).astype(np.float32)
        sf.write(str(class_dir / "clip_0.wav"), audio, 16000)
    return tmp_path


# ---------------------------------------------------------------------------
# extract_embedding
# ---------------------------------------------------------------------------


class TestExtractEmbedding:
    def test_output_shape(self, fake_yamnet):
        audio = np.random.rand(32000).astype(np.float32)
        result = extract_embedding(audio, fake_yamnet)
        assert result.shape == (1024,)

    def test_output_dtype(self, fake_yamnet):
        audio = np.random.rand(32000).astype(np.float32)
        assert extract_embedding(audio, fake_yamnet).dtype == np.float32

    def test_zero_frames_raises(self):
        import tensorflow as tf
        mock_model = MagicMock()
        mock_model.return_value = (
            tf.constant(np.zeros((0, 521), dtype=np.float32)),
            tf.constant(np.zeros((0, 1024), dtype=np.float32)),
            tf.constant(np.zeros((0, 64, 96), dtype=np.float32)),
        )
        with pytest.raises(ValueError, match="0 frames"):
            extract_embedding(np.random.rand(32000).astype(np.float32), mock_model)


# ---------------------------------------------------------------------------
# extract_zero_shot_score
# ---------------------------------------------------------------------------


class TestExtractZeroShotScore:
    def test_output_is_float(self, fake_yamnet):
        score = extract_zero_shot_score(np.random.rand(32000).astype(np.float32), fake_yamnet, 0)
        assert isinstance(score, float)

    def test_output_in_range(self, fake_yamnet):
        score = extract_zero_shot_score(np.random.rand(32000).astype(np.float32), fake_yamnet, 0)
        assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# _outputs_exist
# ---------------------------------------------------------------------------


class TestOutputsExist:
    def test_returns_false_when_missing(self, tmp_path):
        assert _outputs_exist(tmp_path) is False

    def test_returns_true_when_all_present(self, tmp_path):
        for name in (
            "X_embeddings.npy", "y_labels.npy", "zero_shot_scores.npy",
            "yamnet_top_class_indices.npy", "yamnet_top_class_scores.npy",
            "yamnet_top_class_names.npy", "metadata.json",
        ):
            (tmp_path / name).touch()
        assert _outputs_exist(tmp_path) is True

    def test_returns_false_when_partial(self, tmp_path):
        (tmp_path / "X_embeddings.npy").touch()
        assert _outputs_exist(tmp_path) is False


# ---------------------------------------------------------------------------
# save_outputs
# ---------------------------------------------------------------------------


class TestSaveOutputs:
    def test_all_files_created(self, tmp_path):
        save_outputs(*_make_dummy_outputs(), output_dir=tmp_path)
        for name in (
            "X_embeddings.npy", "y_labels.npy", "zero_shot_scores.npy",
            "yamnet_top_class_indices.npy", "yamnet_top_class_scores.npy",
            "yamnet_top_class_names.npy", "metadata.json",
        ):
            assert (tmp_path / name).exists(), f"{name} not found"

    def test_arrays_roundtrip(self, tmp_path):
        X, y, zs, ti, ts, tn, meta = _make_dummy_outputs(4)
        save_outputs(X, y, zs, ti, ts, tn, meta, tmp_path)
        np.testing.assert_array_almost_equal(X, np.load(tmp_path / "X_embeddings.npy"))
        np.testing.assert_array_equal(y, np.load(tmp_path / "y_labels.npy"))
        np.testing.assert_array_almost_equal(zs, np.load(tmp_path / "zero_shot_scores.npy"))

    def test_metadata_has_required_keys(self, tmp_path):
        save_outputs(*_make_dummy_outputs(), output_dir=tmp_path)
        with open(tmp_path / "metadata.json") as f:
            loaded = json.load(f)
        for key in ("total_clips", "gunshot_count", "not_gunshot_count",
                    "skipped_files", "timestamp", "yamnet_url",
                    "gunshot_audioset_class_name", "class_map"):
            assert key in loaded, f"'{key}' missing from metadata.json"

    def test_class_map_in_metadata(self, tmp_path):
        save_outputs(*_make_dummy_outputs(), output_dir=tmp_path)
        with open(tmp_path / "metadata.json") as f:
            loaded = json.load(f)
        assert len(loaded["class_map"]) == 521
        # Dummy class map uses "Class_N" names; real map contains GUNSHOT_CLASS_NAME.
        assert all(isinstance(n, str) for n in loaded["class_map"])


# ---------------------------------------------------------------------------
# build_embedding_matrix
# ---------------------------------------------------------------------------


class TestBuildEmbeddingMatrix:
    def test_returns_correct_shapes(self, data_dir, fake_yamnet):
        X, y, zs, ti, ts, tn, meta = build_embedding_matrix(data_dir, fake_yamnet)
        assert X.shape == (2, 1024)
        assert y.shape == (2,)
        assert zs.shape == (2,)
        assert ti.shape == (2,)
        assert ts.shape == (2,)
        assert tn.shape == (2,)

    def test_labels_are_valid(self, data_dir, fake_yamnet):
        _, y, *_ = build_embedding_matrix(data_dir, fake_yamnet)
        assert set(y.tolist()).issubset({0.0, 1.0})

    def test_metadata_counts_match(self, data_dir, fake_yamnet):
        *_, meta = build_embedding_matrix(data_dir, fake_yamnet)
        assert meta["gunshot_count"] + meta["not_gunshot_count"] == meta["total_clips"]

    def test_top_class_names_are_strings(self, data_dir, fake_yamnet):
        *_, tn, _ = build_embedding_matrix(data_dir, fake_yamnet)
        assert all(isinstance(n, str) for n in tn)

    def test_exits_on_missing_class_dir(self, tmp_path, fake_yamnet):
        (tmp_path / "gunshot").mkdir()
        with pytest.raises(SystemExit):
            build_embedding_matrix(tmp_path, fake_yamnet)
