"""
inference/config.py
===================
Centralized inference configuration — single source of truth for model paths.

Resolution order for the YOLO gun-detection weights:
    1. The ``YOLO_WEIGHTS_PATH`` environment variable, if it points at an
       existing file (lets you pin an ad-hoc checkpoint without code changes).
    2. ``models/yolo_finetuned/best.pt`` — the output location of the current
       Modal training pipeline (see ``training/modal_train_yolo.py`` and the
       ``modal volume get vision-train weights/best.pt …`` instruction).
    3. ``YOLO_hugging-main/best.pt`` — the legacy off-the-shelf checkpoint
       used before our own fine-tune existed.

If none of those files exist on disk yet, ``YOLO_WEIGHTS_PATH`` still points
at the *new* fine-tuned location (option 2) so error messages tell users
where to drop the file.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

# Ordered candidates — first existing one wins.
_FALLBACK_CANDIDATES: tuple[str, ...] = (
    "models/yolo_finetuned/best.pt",   # current Modal training output
    "YOLO_hugging-main/best.pt",       # legacy fallback
)


def _resolve_yolo_weights() -> Path:
    """Resolve the YOLO weights path against the candidate chain."""
    env_override = os.environ.get("YOLO_WEIGHTS_PATH")
    candidates: Iterable[str] = (
        (env_override,) if env_override else ()
    ) + _FALLBACK_CANDIDATES

    for c in candidates:
        if c and Path(c).exists():
            return Path(c)

    # Nothing on disk — return the preferred new location so the resulting
    # FileNotFoundError points users at the right place.
    return Path(_FALLBACK_CANDIDATES[0])


YOLO_WEIGHTS_PATH: Path = _resolve_yolo_weights()
