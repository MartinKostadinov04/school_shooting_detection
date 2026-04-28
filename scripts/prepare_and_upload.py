#!/usr/bin/env python3
"""
prepare_and_upload.py
=====================
Phase 1 — Parallel extraction (4 threads, one per dataset):
  CCTV, Monash, YouTube-GDD, DatasetNinja → gun/ and no_gun/ folders

Phase 2 — Parallel Modal upload (3 subprocesses, one per split):
  modal volume put vision-train <out>/train/  train/
  modal volume put vision-train <out>/val/    val/
  modal volume put vision-train <out>/test/   test/

Output
------
data/vision_classify/
    train/  gun/   no_gun/
    val/    gun/   no_gun/
    test/   gun/   no_gun/

Gun images  — any image with at least one gun/pistol annotation
No-gun images — images with zero gun annotations (negatives):
  • Monash empty-label frames
  • YouTube-GDD person-only frames
  • DatasetNinja frames with only knife / phone / wallet / etc.

Usage
-----
  python scripts/prepare_and_upload.py                     # full run
  python scripts/prepare_and_upload.py --dry-run           # count only
  python scripts/prepare_and_upload.py --skip-upload       # extract, no upload
  python scripts/prepare_and_upload.py --output data/out   # custom output dir
"""

import argparse
import json
import random
import subprocess
import sys
import tarfile
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# ── Config ───────────────────────────────────────────────────────────────────

SOURCES = {
    "cctv":   "CCTV Gun Detector.v1i.yolov11.zip",
    "monash": "The Monash Guns Dataset.v2i.yolov11.zip",
    "ygdd":   "archive.zip",
    "dninja": "od-weapondetection_-sohas-detection-DatasetNinja.tar",
}

DNINJA_GUN_CLASSES = {"pistol", "gun"}
YGDD_GUN_CLASS_IDS = {1}
DNINJA_VAL_FRACTION = 0.15
SEED = 42
MODAL_VOLUME = "vision-train"

_print_lock = threading.Lock()

def log(tag: str, msg: str) -> None:
    with _print_lock:
        print(f"  [{tag}] {msg}", flush=True)


# ── File I/O ─────────────────────────────────────────────────────────────────

def write_image(dest: Path, data: bytes, dry_run: bool) -> None:
    if not dry_run:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)


def img_dest(out_root: Path, split: str, cls: str, prefix: str, name: str) -> Path:
    return out_root / split / cls / f"{prefix}_{name}"


# ── Dataset processors ───────────────────────────────────────────────────────

def process_roboflow_zip(
    zip_path: Path, prefix: str, out_root: Path, dry_run: bool
) -> dict:
    """
    Roboflow YOLOv11 zip (CCTV and Monash).
    Layout: train/images/ + train/labels/  valid/  test/

    • non-empty label → gun/
    • empty label     → no_gun/
    """
    split_map = {"train": "train", "valid": "val", "test": "test"}
    stats = {s: {"gun": 0, "no_gun": 0} for s in ("train", "val", "test")}

    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()

        label_idx: dict[str, dict[str, str]] = {s: {} for s in split_map}
        for n in names:
            for src, _ in split_map.items():
                if n.startswith(f"{src}/labels/") and n.endswith(".txt"):
                    label_idx[src][Path(n).stem] = n

        for n in names:
            if not n.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            for src, dst in split_map.items():
                if not n.startswith(f"{src}/images/"):
                    continue
                stem = Path(n).stem
                lbl_path = label_idx[src].get(stem)
                if lbl_path is None:
                    continue

                has_gun = bool(zf.read(lbl_path).strip())
                cls = "gun" if has_gun else "no_gun"
                fname = Path(n).name

                write_image(img_dest(out_root, dst, cls, prefix, fname),
                            zf.read(n), dry_run)
                stats[dst][cls] += 1

    return stats


def process_ygdd_zip(zip_path: Path, prefix: str, out_root: Path, dry_run: bool) -> dict:
    """
    YouTube-GDD (archive.zip).
    Layout: YouTube-GDD-clean/YouTube-GDD-clean/images/train|val|test/
             .../labels/train|val/   (no labels for test)

    Class 0=person (drop), class 1=gun.
    Images with ≥1 gun annotation → gun/  else → no_gun/
    Test split skipped (no labels).
    """
    stats = {s: {"gun": 0, "no_gun": 0} for s in ("train", "val", "test")}

    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()

        label_idx: dict[str, dict[str, str]] = {"train": {}, "val": {}}
        for n in names:
            for split in ("train", "val"):
                if f"/labels/{split}/" in n and n.endswith(".txt"):
                    label_idx[split][Path(n).stem] = n

        for n in names:
            if not n.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            for split in ("train", "val"):
                if f"/images/{split}/" not in n:
                    continue
                stem = Path(n).stem
                lbl_path = label_idx[split].get(stem)
                if lbl_path is None:
                    continue

                raw = zf.read(lbl_path).decode().strip()
                has_gun = any(
                    int(l.split()[0]) in YGDD_GUN_CLASS_IDS
                    for l in raw.splitlines() if l.strip()
                )
                cls = "gun" if has_gun else "no_gun"
                write_image(img_dest(out_root, split, cls, prefix, Path(n).name),
                            zf.read(n), dry_run)
                stats[split][cls] += 1

    return stats


def process_dninja_tar(tar_path: Path, prefix: str, out_root: Path, dry_run: bool) -> dict:
    """
    DatasetNinja Supervisely tar.
    Layout: train/img/ + train/ann/  test/img/ + test/ann/
    Annotations are JSON with absolute-pixel bboxes.

    • any pistol/gun object → gun/
    • zero pistol objects   → no_gun/
    Carves DNINJA_VAL_FRACTION of train as val.
    """
    rng = random.Random(SEED)
    stats = {s: {"gun": 0, "no_gun": 0} for s in ("train", "val", "test")}

    # ── Pass 1: classify every stem ───────────────────────────────────────────
    log(prefix, "scanning annotations …")
    stem_class: dict[str, dict[str, str]] = {"train": {}, "test": {}}

    with tarfile.open(tar_path, "r") as tf:
        for member in tf:
            if not member.isfile() or "/ann/" not in member.name:
                continue
            parts = member.name.split("/")
            if len(parts) < 3 or parts[0] not in stem_class:
                continue
            src_split = parts[0]
            ann_name = parts[-1]
            for ext in (".jpg.json", ".jpeg.json", ".JPG.json", ".JPEG.json"):
                if ann_name.lower().endswith(ext.lower()):
                    stem = ann_name[: -len(ext)]
                    break
            else:
                stem = ann_name.rsplit(".", 2)[0]

            f = tf.extractfile(member)
            if f is None:
                continue
            try:
                ann = json.load(f)
            except (json.JSONDecodeError, KeyError):
                continue

            has_gun = any(
                obj.get("classTitle", "").lower() in DNINJA_GUN_CLASSES
                for obj in ann.get("objects", [])
            )
            stem_class[src_split][stem] = "gun" if has_gun else "no_gun"

    # Carve val from train stems
    train_stems = sorted(stem_class["train"])
    n_val = int(len(train_stems) * DNINJA_VAL_FRACTION)
    val_set = set(rng.sample(train_stems, n_val))

    # Build final split→stem→cls mapping
    routing: dict[str, dict[str, str]] = {
        "train": {s: c for s, c in stem_class["train"].items() if s not in val_set},
        "val":   {s: stem_class["train"][s] for s in val_set},
        "test":  stem_class["test"],
    }
    for dst, mapping in routing.items():
        for cls in ("gun", "no_gun"):
            stats[dst][cls] = sum(1 for c in mapping.values() if c == cls)

    log(prefix, (
        f"gun  → train={stats['train']['gun']} val={stats['val']['gun']} test={stats['test']['gun']}  |  "
        f"no_gun → train={stats['train']['no_gun']} val={stats['val']['no_gun']} test={stats['test']['no_gun']}"
    ))

    if dry_run:
        return stats

    # ── Pass 2: stream images to their destinations ───────────────────────────
    log(prefix, "copying images …")

    # Flatten: (stem, src_split) → (dst_split, cls)
    stem_route: dict[tuple[str, str], tuple[str, str]] = {}
    for dst_split, mapping in routing.items():
        for stem, cls in mapping.items():
            # DatasetNinja train/img → may route to train or val dst
            src = "train" if dst_split in ("train", "val") else "test"
            stem_route[(stem, src)] = (dst_split, cls)

    with tarfile.open(tar_path, "r") as tf:
        for member in tf:
            if not member.isfile() or "/img/" not in member.name:
                continue
            if not member.name.lower().endswith((".jpg", ".jpeg")):
                continue
            parts = member.name.split("/")
            src_split = parts[0]
            fname = parts[-1]
            for ext in (".jpg", ".jpeg", ".JPG", ".JPEG"):
                if fname.lower().endswith(ext.lower()):
                    stem = fname[: -len(ext)]
                    break
            else:
                continue

            route = stem_route.get((stem, src_split))
            if route is None:
                continue
            dst_split, cls = route

            f = tf.extractfile(member)
            if f is None:
                continue
            write_image(
                img_dest(out_root, dst_split, cls, prefix, fname),
                f.read(),
                dry_run=False,
            )

    return stats


# ── Upload ────────────────────────────────────────────────────────────────────

def upload_split(split: str, local_split_dir: Path, volume: str) -> tuple[str, int, str]:
    """
    Run `modal volume put <volume> <local_split_dir> <split>/` as a subprocess.
    Returns (split, returncode, output).
    """
    cmd = ["modal", "volume", "put", "--force", volume,
           str(local_split_dir), f"{split}/"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    output = (result.stdout + result.stderr).strip()
    return split, result.returncode, output


def upload_parallel(out_dir: Path, volume: str) -> None:
    splits = [s for s in ("train", "val", "test") if (out_dir / s).exists()]
    print(f"\nUploading {len(splits)} splits in parallel → modal volume: {volume}")

    with ThreadPoolExecutor(max_workers=len(splits)) as pool:
        futures = {pool.submit(upload_split, s, out_dir / s, volume): s for s in splits}
        for future in as_completed(futures):
            split, rc, output = future.result()
            status = "✓" if rc == 0 else "✗ FAILED"
            with _print_lock:
                print(f"  [{split}] {status}")
                if output:
                    for line in output.splitlines()[-5:]:   # last 5 lines
                        print(f"    {line}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input",       default="data/vision",
                        help="Directory containing the raw zips/tar")
    parser.add_argument("--output",      default="data/vision_classify",
                        help="Output directory")
    parser.add_argument("--dry-run",     action="store_true",
                        help="Count only — write nothing, no upload")
    parser.add_argument("--skip-upload", action="store_true",
                        help="Extract locally but skip the Modal upload")
    args = parser.parse_args()

    in_dir  = Path(args.input)
    out_dir = Path(args.output)
    dry_run = args.dry_run

    if not in_dir.exists():
        sys.exit(f"ERROR: input directory not found: {in_dir}")

    if not dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    # ── Phase 1: parallel extraction ──────────────────────────────────────────
    tasks = [
        ("cctv",   process_roboflow_zip,
         in_dir / SOURCES["cctv"],   "cctv"),
        ("monash", process_roboflow_zip,
         in_dir / SOURCES["monash"], "monash"),
        ("ygdd",   process_ygdd_zip,
         in_dir / SOURCES["ygdd"],   "ygdd"),
        ("dninja", process_dninja_tar,
         in_dir / SOURCES["dninja"], "dninja"),
    ]

    all_stats: dict[str, dict] = {}
    t0 = time.time()

    print(f"\nPhase 1 — Extracting {len(tasks)} datasets in parallel …\n")
    with ThreadPoolExecutor(max_workers=4) as pool:
        future_map = {}
        for tag, fn, src, prefix in tasks:
            if not src.exists():
                print(f"  [{tag}] not found — skipping ({src})")
                continue
            mb = src.stat().st_size / 1e6
            print(f"  [{tag}] queued  ({mb:.0f} MB)")
            future_map[pool.submit(fn, src, prefix, out_dir, dry_run)] = tag

        for future in as_completed(future_map):
            tag = future_map[future]
            try:
                stats = future.result()
                all_stats[tag] = stats
                g = {s: stats[s]["gun"]    for s in stats}
                n = {s: stats[s]["no_gun"] for s in stats}
                log(tag, f"done  gun={g}  no_gun={n}")
            except Exception as exc:
                log(tag, f"FAILED: {exc}")
                raise

    elapsed = time.time() - t0

    # ── Summary ───────────────────────────────────────────────────────────────
    totals: dict[str, dict[str, int]] = {
        s: {"gun": 0, "no_gun": 0} for s in ("train", "val", "test")
    }
    for stats in all_stats.values():
        for split, counts in stats.items():
            for cls, n in counts.items():
                totals[split][cls] += n

    print(f"\n{'─'*62}")
    print("SUMMARY")
    print(f"{'─'*62}")
    print(f"  {'split':<8}  {'gun':>7}  {'no_gun':>7}  {'total':>7}")
    print(f"  {'─'*8}  {'─'*7}  {'─'*7}  {'─'*7}")
    grand = 0
    for split in ("train", "val", "test"):
        g = totals[split]["gun"]
        n = totals[split]["no_gun"]
        t = g + n
        grand += t
        print(f"  {split:<8}  {g:>7}  {n:>7}  {t:>7}")
    print(f"  {'─'*8}  {'─'*7}  {'─'*7}  {'─'*7}")
    print(f"  {'TOTAL':<8}  {sum(totals[s]['gun'] for s in totals):>7}  "
          f"{sum(totals[s]['no_gun'] for s in totals):>7}  {grand:>7}")
    print(f"{'─'*62}")
    print(f"  Extraction time: {elapsed:.0f}s")

    if dry_run:
        print("\n[DRY RUN — no files written, no upload]")
        return

    if args.skip_upload:
        print(f"\nFiles written to {out_dir}")
        print("Skipping Modal upload (--skip-upload).")
        return

    # ── Phase 2: parallel Modal upload ────────────────────────────────────────
    print(f"\nPhase 2 — Uploading to Modal volume '{MODAL_VOLUME}' …")
    upload_parallel(out_dir, MODAL_VOLUME)
    print("\nDone. Browse the volume:")
    print(f"  modal volume ls {MODAL_VOLUME}")


if __name__ == "__main__":
    main()
