"""
modal_extract.py
================
Run YAMNet embedding extraction on Modal cloud with GPU.

This script mirrors pipeline/extract_embeddings.py but executes remotely
inside a Modal container with a GPU. The WAV files are read from (and
embeddings written back to) a Modal persistent Volume.

Setup (one-time)
----------------
1. Install Modal and authenticate::

       pip install modal
       modal token new          # saves credentials to ~/.modal.toml

2. Create the persistent volume::

       modal volume create gunshot-data

3. Upload your WAV files into the volume::

       modal volume put gunshot-data data/raw/gunshot   gunshot
       modal volume put gunshot-data data/raw/not_gunshot not_gunshot

Run
---
::

    modal run pipeline/modal_extract.py

    # Force re-extraction if embeddings already exist
    modal run pipeline/modal_extract.py --force

    # Override worker count
    modal run pipeline/modal_extract.py --workers 12

Download results
----------------
::

    modal volume get gunshot-data embeddings data/processed/embeddings

Volume layout (inside /vol)
----------------------------
::

    /vol/
      gunshot/          ← uploaded WAV files (label 1)
      not_gunshot/      ← uploaded WAV files (label 0)
      embeddings/       ← written by this script
        X_embeddings.npy
        y_labels.npy
        zero_shot_scores.npy
        yamnet_top_class_indices.npy
        yamnet_top_class_scores.npy
        metadata.json

Authentication
--------------
Modal reads credentials from ``~/.modal.toml`` (set by ``modal token new``)
or from the environment variables named in ``configs/yamnet_pipeline.yaml``::

    MODAL_TOKEN_ID      ← token ID
    MODAL_TOKEN_SECRET  ← token secret

Never commit actual token values. The config stores only the env var names.
"""

import os
import sys
from pathlib import Path

import modal
import yaml

# ---------------------------------------------------------------------------
# Load config
# ---------------------------------------------------------------------------

# Local:  pipeline/modal_extract.py  → configs/ is two levels up
# Remote: /root/modal_extract.py     → /root/configs/ is one level up
_CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "yamnet_pipeline.yaml"
if not _CONFIG_PATH.exists():
    _CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "yamnet_pipeline.yaml"

with open(_CONFIG_PATH, "r", encoding="utf-8") as _f:
    _CFG = yaml.safe_load(_f)

_MODAL = _CFG["modal"]

APP_NAME: str          = _MODAL["app_name"]
VOLUME_NAME: str       = _MODAL["volume_name"]
GPU_TYPE: str          = _MODAL["gpu"]
TIMEOUT_S: int         = _MODAL["timeout_seconds"]
MEMORY_MB: int         = _MODAL["memory_mb"]
DEFAULT_WORKERS: int   = _MODAL["workers"]
VOLUME_MOUNT: str      = _MODAL["volume_mount"]
REMOTE_DATA_DIR: str   = _MODAL["remote_data_dir"]
REMOTE_OUTPUT_DIR: str = _MODAL["remote_output_dir"]

YAMNET_URL: str = _CFG["yamnet_url"]

# ---------------------------------------------------------------------------
# Modal app definition
# ---------------------------------------------------------------------------

app = modal.App(APP_NAME)

# Container image — TensorFlow with CUDA support + project dependencies.
# Local pipeline/ and configs/ directories are baked into the image at build time.
extraction_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "tensorflow[and-cuda]>=2.13.0",
        "tensorflow-hub>=0.15.0",
        "numpy>=1.24.0",
        "pyyaml>=6.0",
    )
    .add_local_dir(
        Path(__file__).parent,
        remote_path="/root/pipeline",
    )
    .add_local_dir(
        Path(__file__).parent.parent / "configs",
        remote_path="/root/configs",
    )
)

# Persistent volume — survives between runs.
# Create with: modal volume create gunshot-data
data_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)


# ---------------------------------------------------------------------------
# Remote function
# ---------------------------------------------------------------------------

@app.function(
    image=extraction_image,
    gpu=GPU_TYPE,
    volumes={VOLUME_MOUNT: data_volume},
    timeout=TIMEOUT_S,
    memory=MEMORY_MB,
)
def run_extraction(
    data_dir: str = REMOTE_DATA_DIR,
    output_dir: str = REMOTE_OUTPUT_DIR,
    workers: int = DEFAULT_WORKERS,
    force: bool = False,
) -> dict:
    """
    Extract YAMNet embeddings from WAV files stored in the Modal volume.

    Parameters
    ----------
    data_dir : str
        Path inside the volume to the directory containing ``gunshot/`` and
        ``not_gunshot/`` subdirectories.
    output_dir : str
        Path inside the volume to write output ``.npy`` files and
        ``metadata.json``.
    workers : int
        Number of threads for parallel WAV prefetching.
    force : bool
        If ``False`` and output files already exist, skip extraction.

    Returns
    -------
    dict
        Extraction metadata (same as ``metadata.json``).
    """
    import sys
    sys.path.insert(0, "/root")

    # Imports from our pipeline package (mounted above).
    from pipeline.extract_embeddings import (
        _outputs_exist,
        build_embedding_matrix,
        load_yamnet,
        save_outputs,
    )

    data_path = Path(data_dir)
    output_path = Path(output_dir)

    if _outputs_exist(output_path) and not force:
        print(
            f"[modal_extract] Output files already exist in '{output_path}'. "
            "Pass force=True to overwrite."
        )
        import json
        with open(output_path / "metadata.json") as f:
            return json.load(f)

    print(f"[modal_extract] GPU: {GPU_TYPE}")
    print(f"[modal_extract] Data dir: {data_path}")
    print(f"[modal_extract] Output dir: {output_path}")
    print(f"[modal_extract] Prefetch workers: {workers}")

    # Verify GPU is visible to TensorFlow.
    import tensorflow as tf
    gpus = tf.config.list_physical_devices("GPU")
    print(f"[modal_extract] TensorFlow sees {len(gpus)} GPU(s): {gpus}")

    # Load YAMNet (cached in container image layer after first pull).
    yamnet_model = load_yamnet(YAMNET_URL)

    # Run extraction with disk-IO prefetch + single-pass GPU inference.
    X, y, zero_shot, top_idx, top_scores, top_names, metadata = build_embedding_matrix(
        data_path, yamnet_model, n_workers=workers
    )

    # Save to volume.
    save_outputs(X, y, zero_shot, top_idx, top_scores, top_names, metadata, output_path)

    # Flush volume writes so they persist after the container exits.
    data_volume.commit()

    print(
        f"[modal_extract] Done — {metadata['total_clips']} clips "
        f"({metadata['gunshot_count']} gunshot, "
        f"{metadata['not_gunshot_count']} not_gunshot, "
        f"{len(metadata['skipped_files'])} skipped)."
    )
    return metadata


# ---------------------------------------------------------------------------
# Local entrypoint (called by `modal run pipeline/modal_extract.py`)
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main(
    force: bool = False,
    workers: int = DEFAULT_WORKERS,
) -> None:
    """
    Submit the extraction job to Modal.

    Usage::

        modal run pipeline/modal_extract.py
        modal run pipeline/modal_extract.py --force
        modal run pipeline/modal_extract.py --workers 12
    """
    # Validate that Modal credentials are available before submitting.
    token_id_env  = _MODAL["token_id_env"]
    token_sec_env = _MODAL["token_secret_env"]

    if not os.environ.get(token_id_env) and not Path("~/.modal.toml").expanduser().exists():
        print(
            f"WARNING: Neither '{token_id_env}' env var nor ~/.modal.toml found.\n"
            "Run `modal token new` to authenticate, or set:\n"
            f"  export {token_id_env}=...\n"
            f"  export {token_sec_env}=..."
        )
        sys.exit(1)

    print(f"Submitting extraction job to Modal (GPU={GPU_TYPE}) ...")
    metadata = run_extraction.remote(force=force, workers=workers)
    print(f"\nExtraction complete. {metadata['total_clips']} clips processed.")
    print(
        f"\nTo download results run:\n"
        f"  modal volume get {VOLUME_NAME} embeddings data/processed/embeddings"
    )
