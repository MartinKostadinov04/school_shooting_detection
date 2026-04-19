"""
test_split_dataset.py
=====================
Unit tests for pipeline/split_dataset.py.

Run with:
    python -m pytest tests/test_split_dataset.py -v
"""

import json
from pathlib import Path

import numpy as np
import pytest

from pipeline.split_dataset import (
    RANDOM_STATE,
    compute_class_distribution,
    load_embeddings,
    print_summary,
    save_splits,
    stratified_split,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_dummy_embeddings(n: int = 100, seed: int = 0):
    """Generate synthetic (N, 1024) embeddings and balanced labels."""
    rng = np.random.default_rng(seed)
    X = rng.random((n, 1024)).astype(np.float32)
    y = np.array([1.0] * (n // 2) + [0.0] * (n // 2), dtype=np.float32)
    rng.shuffle(y)
    return X, y


@pytest.fixture
def embeddings_dir(tmp_path):
    X, y = _make_dummy_embeddings(100)
    np.save(str(tmp_path / "X_embeddings.npy"), X)
    np.save(str(tmp_path / "y_labels.npy"), y)
    return tmp_path


# ---------------------------------------------------------------------------
# load_embeddings tests
# ---------------------------------------------------------------------------


class TestLoadEmbeddings:
    def test_loads_correctly(self, embeddings_dir):
        X, y = load_embeddings(embeddings_dir)
        assert X.shape == (100, 1024)
        assert y.shape == (100,)

    def test_raises_if_x_missing(self, tmp_path):
        y = np.zeros(10, dtype=np.float32)
        np.save(str(tmp_path / "y_labels.npy"), y)
        with pytest.raises(FileNotFoundError, match="X_embeddings.npy"):
            load_embeddings(tmp_path)

    def test_raises_if_y_missing(self, tmp_path):
        X = np.zeros((10, 1024), dtype=np.float32)
        np.save(str(tmp_path / "X_embeddings.npy"), X)
        with pytest.raises(FileNotFoundError, match="y_labels.npy"):
            load_embeddings(tmp_path)

    def test_raises_on_shape_mismatch(self, tmp_path):
        np.save(str(tmp_path / "X_embeddings.npy"), np.zeros((10, 1024), dtype=np.float32))
        np.save(str(tmp_path / "y_labels.npy"), np.zeros(9, dtype=np.float32))
        with pytest.raises(ValueError, match="Shape mismatch"):
            load_embeddings(tmp_path)


# ---------------------------------------------------------------------------
# stratified_split tests
# ---------------------------------------------------------------------------


class TestStratifiedSplit:
    def test_total_counts_match(self):
        """train + val + test should equal the input N."""
        X, y = _make_dummy_embeddings(100)
        splits = stratified_split(X, y)
        total = (
            splits["X_train"].shape[0]
            + splits["X_val"].shape[0]
            + splits["X_test"].shape[0]
        )
        assert total == 100

    def test_indices_non_overlapping(self):
        """train, val, test index sets must be disjoint."""
        X, y = _make_dummy_embeddings(100)
        splits = stratified_split(X, y)
        idx_train = set(splits["idx_train"].tolist())
        idx_val = set(splits["idx_val"].tolist())
        idx_test = set(splits["idx_test"].tolist())
        assert idx_train & idx_val == set()
        assert idx_train & idx_test == set()
        assert idx_val & idx_test == set()

    def test_indices_cover_all_data(self):
        """Union of all indices should cover the full range [0, N)."""
        X, y = _make_dummy_embeddings(100)
        splits = stratified_split(X, y)
        all_idx = (
            set(splits["idx_train"].tolist())
            | set(splits["idx_val"].tolist())
            | set(splits["idx_test"].tolist())
        )
        assert all_idx == set(range(100))

    def test_deterministic_with_same_seed(self):
        """Same seed should produce identical splits on repeated calls."""
        X, y = _make_dummy_embeddings(100)
        splits_a = stratified_split(X, y, random_state=42)
        splits_b = stratified_split(X, y, random_state=42)
        np.testing.assert_array_equal(splits_a["idx_train"], splits_b["idx_train"])
        np.testing.assert_array_equal(splits_a["idx_val"], splits_b["idx_val"])

    def test_different_seeds_produce_different_splits(self):
        """Different seeds should (almost certainly) produce different splits."""
        X, y = _make_dummy_embeddings(200)
        splits_a = stratified_split(X, y, random_state=0)
        splits_b = stratified_split(X, y, random_state=99)
        # It is astronomically unlikely the train indices are identical
        assert not np.array_equal(splits_a["idx_train"], splits_b["idx_train"])

    def test_approximate_70_15_15_ratio(self):
        """Split sizes should be approximately 70/15/15."""
        X, y = _make_dummy_embeddings(200)
        splits = stratified_split(X, y)
        n_train = splits["X_train"].shape[0]
        n_val = splits["X_val"].shape[0]
        n_test = splits["X_test"].shape[0]
        assert abs(n_train - 140) <= 2, f"Expected ~140 train, got {n_train}"
        assert abs(n_val - 30) <= 2, f"Expected ~30 val, got {n_val}"
        assert abs(n_test - 30) <= 2, f"Expected ~30 test, got {n_test}"

    def test_both_classes_in_all_splits(self):
        """Each split should contain both classes (stratification)."""
        X, y = _make_dummy_embeddings(100)
        splits = stratified_split(X, y)
        for name in ("train", "val", "test"):
            split_y = splits[f"y_{name}"]
            assert 1.0 in split_y, f"No gunshot samples in {name}"
            assert 0.0 in split_y, f"No not_gunshot samples in {name}"


# ---------------------------------------------------------------------------
# compute_class_distribution tests
# ---------------------------------------------------------------------------


class TestComputeClassDistribution:
    def test_balanced(self):
        y = np.array([1.0, 1.0, 0.0, 0.0], dtype=np.float32)
        dist = compute_class_distribution(y)
        assert dist == {"gunshot": 2, "not_gunshot": 2, "total": 4}

    def test_all_gunshot(self):
        y = np.ones(5, dtype=np.float32)
        dist = compute_class_distribution(y)
        assert dist["gunshot"] == 5
        assert dist["not_gunshot"] == 0


# ---------------------------------------------------------------------------
# save_splits tests
# ---------------------------------------------------------------------------


class TestSaveSplits:
    def test_all_files_created(self, tmp_path):
        X, y = _make_dummy_embeddings(100)
        splits = stratified_split(X, y)
        save_splits(splits, tmp_path)
        for name in ("X_train", "X_val", "X_test", "y_train", "y_val", "y_test"):
            assert (tmp_path / f"{name}.npy").exists(), f"{name}.npy not found"
        assert (tmp_path / "split_info.json").exists()

    def test_split_info_json_valid(self, tmp_path):
        X, y = _make_dummy_embeddings(100)
        splits = stratified_split(X, y)
        save_splits(splits, tmp_path)
        with open(tmp_path / "split_info.json") as f:
            info = json.load(f)
        assert "random_state" in info
        assert "split_ratios" in info
        assert "counts" in info
        assert "indices" in info
        for split_name in ("train", "val", "test"):
            assert split_name in info["counts"]
            assert split_name in info["indices"]

    def test_saved_counts_match_arrays(self, tmp_path):
        """Counts in split_info.json should match the saved numpy arrays."""
        X, y = _make_dummy_embeddings(100)
        splits = stratified_split(X, y)
        save_splits(splits, tmp_path)
        with open(tmp_path / "split_info.json") as f:
            info = json.load(f)
        for split_name in ("train", "val", "test"):
            y_loaded = np.load(str(tmp_path / f"y_{split_name}.npy"))
            assert info["counts"][split_name]["total"] == len(y_loaded)

    def test_indices_non_overlapping_from_json(self, tmp_path):
        """Indices in split_info.json should be non-overlapping."""
        X, y = _make_dummy_embeddings(100)
        splits = stratified_split(X, y)
        save_splits(splits, tmp_path)
        with open(tmp_path / "split_info.json") as f:
            info = json.load(f)
        idx_train = set(info["indices"]["train"])
        idx_val = set(info["indices"]["val"])
        idx_test = set(info["indices"]["test"])
        assert idx_train & idx_val == set()
        assert idx_train & idx_test == set()
        assert idx_val & idx_test == set()


# ---------------------------------------------------------------------------
# print_summary tests (smoke test only — output is visual)
# ---------------------------------------------------------------------------


class TestPrintSummary:
    def test_does_not_raise(self, capsys):
        X, y = _make_dummy_embeddings(100)
        splits = stratified_split(X, y)
        print_summary(splits)  # Should not raise
        captured = capsys.readouterr()
        assert "train" in captured.out
        assert "val" in captured.out
        assert "test" in captured.out
