"""
test_yamnet_integration.py
==========================
Integration test: load real YAMNet from TF Hub and run inference on one
actual WAV file from each class (gunshot and not_gunshot).

This test is skipped automatically if:
  - The data directories don't exist or are empty (WAV files not present)
  - TensorFlow / TF-Hub is not installed

Run explicitly:
    python -m pytest tests/test_yamnet_integration.py -v -s

Skip in CI (fast unit tests only):
    python -m pytest tests/ -v --ignore=tests/test_yamnet_integration.py
"""

import random
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).parent.parent
GUNSHOT_DIR = PROJECT_ROOT / "data" / "raw" / "gunshot"
NOT_GUNSHOT_DIR = PROJECT_ROOT / "data" / "raw" / "not_gunshot"


# ---------------------------------------------------------------------------
# Skip conditions
# ---------------------------------------------------------------------------

def _wav_files(directory: Path):
    return list(directory.glob("*.wav")) if directory.is_dir() else []


pytestmark = pytest.mark.skipif(
    not _wav_files(GUNSHOT_DIR) or not _wav_files(NOT_GUNSHOT_DIR),
    reason="data/raw/gunshot/ or data/raw/not_gunshot/ is empty — skipping integration test",
)

tf = pytest.importorskip("tensorflow", reason="TensorFlow not installed")
hub = pytest.importorskip("tensorflow_hub", reason="tensorflow-hub not installed")


# ---------------------------------------------------------------------------
# Fixture: load YAMNet once for the whole module
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def yamnet_and_classes():
    """Load real YAMNet and class map once; reuse across all tests in this module."""
    from pipeline.extract_embeddings import load_yamnet, load_class_map, YAMNET_URL
    model = load_yamnet(YAMNET_URL)
    class_names = load_class_map(model)
    return model, class_names


@pytest.fixture(scope="module")
def sample_clips():
    """Pick one random WAV from each class (reproducible via seed)."""
    rng = random.Random(42)
    gunshot_wav = rng.choice(_wav_files(GUNSHOT_DIR))
    not_gunshot_wav = rng.choice(_wav_files(NOT_GUNSHOT_DIR))
    return {"gunshot": gunshot_wav, "not_gunshot": not_gunshot_wav}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestYAMNetIntegration:

    def test_class_map_length(self, yamnet_and_classes):
        """YAMNet class map should have exactly 521 entries."""
        _, class_names = yamnet_and_classes
        assert len(class_names) == 521

    def test_gunshot_class_in_map(self, yamnet_and_classes):
        """'Gunshot, gunfire' must appear somewhere in the 521-class map."""
        from pipeline.extract_embeddings import GUNSHOT_CLASS_NAME, resolve_gunshot_class_idx
        _, class_names = yamnet_and_classes
        idx = resolve_gunshot_class_idx(class_names)
        assert class_names[idx] == GUNSHOT_CLASS_NAME, (
            f"resolve_gunshot_class_idx returned {idx}, "
            f"but class_names[{idx}] = '{class_names[idx]}'"
        )

    def test_gunshot_clip_inference(self, yamnet_and_classes, sample_clips):
        """Run full _yamnet_infer on a real gunshot clip and check output contracts."""
        from pipeline.extract_embeddings import _load_wav_direct, _yamnet_infer, resolve_gunshot_class_idx

        model, class_names = yamnet_and_classes
        wav_path = sample_clips["gunshot"]

        print(f"\n  [gunshot]     {wav_path.name}")
        audio = _load_wav_direct(wav_path)
        gunshot_idx = resolve_gunshot_class_idx(class_names)
        embedding, zero_shot, top_idx, top_score, top_name = _yamnet_infer(
            audio, model, class_names, gunshot_idx
        )

        print(f"  embedding     shape={embedding.shape}  dtype={embedding.dtype}")
        print(f"  zero_shot     {zero_shot:.4f}  (gunshot class score)")
        print(f"  top_class     [{top_idx}] '{top_name}'  score={top_score:.4f}")

        assert embedding.shape == (1024,)
        assert embedding.dtype == np.float32
        assert 0.0 <= zero_shot <= 1.0
        assert 0 <= top_idx <= 520
        assert 0.0 <= top_score <= 1.0
        assert isinstance(top_name, str) and len(top_name) > 0
        assert top_name == class_names[top_idx]

    def test_not_gunshot_clip_inference(self, yamnet_and_classes, sample_clips):
        """Run full _yamnet_infer on a real not_gunshot clip and check output contracts."""
        from pipeline.extract_embeddings import _load_wav_direct, _yamnet_infer, resolve_gunshot_class_idx

        model, class_names = yamnet_and_classes
        wav_path = sample_clips["not_gunshot"]

        print(f"\n  [not_gunshot] {wav_path.name}")
        audio = _load_wav_direct(wav_path)
        gunshot_idx = resolve_gunshot_class_idx(class_names)
        embedding, zero_shot, top_idx, top_score, top_name = _yamnet_infer(
            audio, model, class_names, gunshot_idx
        )

        print(f"  embedding     shape={embedding.shape}  dtype={embedding.dtype}")
        print(f"  zero_shot     {zero_shot:.4f}  (gunshot class score)")
        print(f"  top_class     [{top_idx}] '{top_name}'  score={top_score:.4f}")

        assert embedding.shape == (1024,)
        assert embedding.dtype == np.float32
        assert 0.0 <= zero_shot <= 1.0
        assert 0 <= top_idx <= 520
        assert 0.0 <= top_score <= 1.0
        assert isinstance(top_name, str) and len(top_name) > 0
        assert top_name == class_names[top_idx]

    def test_embeddings_differ_between_classes(self, yamnet_and_classes, sample_clips):
        """Embeddings for a gunshot and a non-gunshot clip should not be identical."""
        from pipeline.extract_embeddings import _load_wav_direct, _yamnet_infer, resolve_gunshot_class_idx

        model, class_names = yamnet_and_classes
        gunshot_idx = resolve_gunshot_class_idx(class_names)

        emb_g, *_ = _yamnet_infer(
            _load_wav_direct(sample_clips["gunshot"]), model, class_names, gunshot_idx
        )
        emb_ng, *_ = _yamnet_infer(
            _load_wav_direct(sample_clips["not_gunshot"]), model, class_names, gunshot_idx
        )

        assert not np.allclose(emb_g, emb_ng), \
            "Gunshot and not_gunshot embeddings are identical — something is wrong"

    def test_zero_shot_score_higher_for_gunshot(self, yamnet_and_classes, sample_clips):
        """
        The zero-shot "Gunshot, gunfire" class score should tend to be higher for
        gunshot clips.

        NOTE: This is a soft sanity check on a single sample pair, not a
        precision/recall evaluation. It can occasionally fail for ambiguous clips
        (e.g. a very quiet gunshot or a very impulsive non-gunshot). If it fails
        consistently, investigate the specific clips selected by the seed.
        """
        from pipeline.extract_embeddings import _load_wav_direct, _yamnet_infer, resolve_gunshot_class_idx

        model, class_names = yamnet_and_classes
        gunshot_idx = resolve_gunshot_class_idx(class_names)

        _, score_g, *_ = _yamnet_infer(
            _load_wav_direct(sample_clips["gunshot"]), model, class_names, gunshot_idx
        )
        _, score_ng, *_ = _yamnet_infer(
            _load_wav_direct(sample_clips["not_gunshot"]), model, class_names, gunshot_idx
        )

        print(f"\n  zero_shot gunshot={score_g:.4f}  not_gunshot={score_ng:.4f}")
        assert score_g > score_ng, (
            f"Expected gunshot score ({score_g:.4f}) > not_gunshot score ({score_ng:.4f}). "
            "This may be a hard sample — check the specific clips."
        )
