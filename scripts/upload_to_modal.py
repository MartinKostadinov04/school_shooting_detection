"""
upload_to_modal.py
==================
Create the Modal volume and upload the prepared YOLO dataset.

Usage
-----
    python scripts/upload_to_modal.py

Expects data/vision_yolo/ to already exist (run prepare_vision_data.py first).
Uploads in parallel: images/train, images/val, images/test, labels/, data.yaml.
"""

import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

VOLUME_NAME = "vision-train"
LOCAL_ROOT  = Path("data/vision_yolo")

# Each entry: (local_path, remote_dest)
# Using per-split parallelism on the image side since images/train is ~2 GB.
# modal volume put <vol> <local_dir> <remote_dest/> places <local_dir> inside
# <remote_dest>, so images/train → yolo/images/train, labels/train → yolo/labels/train.
UPLOADS = [
    (LOCAL_ROOT / "images" / "train",  "yolo/images/"),
    (LOCAL_ROOT / "images" / "val",    "yolo/images/"),
    (LOCAL_ROOT / "images" / "test",   "yolo/images/"),
    (LOCAL_ROOT / "labels" / "train",  "yolo/labels/"),
    (LOCAL_ROOT / "labels" / "val",    "yolo/labels/"),
    (LOCAL_ROOT / "labels" / "test",   "yolo/labels/"),
    (LOCAL_ROOT / "data.yaml",         "yolo/"),
]


def _run(local: Path, remote: str) -> tuple[str, float]:
    label = f"{local.parent.name}/{local.name}" if local.is_dir() else local.name
    t0 = time.time()
    result = subprocess.run(
        ["modal", "volume", "put", "--force", VOLUME_NAME, str(local), remote],
        capture_output=True,
        text=True,
    )
    elapsed = time.time() - t0
    if result.returncode != 0:
        raise RuntimeError(
            f"upload failed for {label}:\n{result.stderr.strip()}"
        )
    return label, elapsed


def create_volume() -> None:
    print(f"Creating volume '{VOLUME_NAME}' …")
    result = subprocess.run(
        ["modal", "volume", "create", VOLUME_NAME],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        # "already exists" is fine
        if "already exists" in result.stderr or "already exists" in result.stdout:
            print(f"  Volume already exists, continuing.")
        else:
            print(f"ERROR: {result.stderr.strip()}", file=sys.stderr)
            sys.exit(1)
    else:
        print(f"  Created.")


def main() -> None:
    # Validate local data
    for local, _ in UPLOADS:
        if not local.exists():
            print(f"ERROR: {local} not found. Run scripts/prepare_vision_data.py first.")
            sys.exit(1)

    create_volume()

    print(f"\nUploading {len(UPLOADS)} paths in parallel → {VOLUME_NAME} …")
    print(f"  (this will take a few minutes for ~3 GB of images)\n")

    failed = []
    with ThreadPoolExecutor(max_workers=len(UPLOADS)) as pool:
        futures = {pool.submit(_run, local, remote): (local, remote)
                   for local, remote in UPLOADS}
        for future in as_completed(futures):
            local, remote = futures[future]
            label = f"{local.parent.name}/{local.name}" if local.is_dir() else local.name
            try:
                _, elapsed = future.result()
                print(f"  ✓  {label:<30}  ({elapsed:.0f}s)")
            except RuntimeError as exc:
                print(f"  ✗  {label:<30}  FAILED: {exc}")
                failed.append(label)

    if failed:
        print(f"\nFailed uploads: {failed}")
        sys.exit(1)

    print(f"\nDone. Verify with:")
    print(f"  modal volume ls {VOLUME_NAME} yolo")
    print(f"\nStart training:")
    print(f"  modal run training/modal_train_yolo.py")


if __name__ == "__main__":
    main()
