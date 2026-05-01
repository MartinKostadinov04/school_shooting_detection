#!/usr/bin/env python3
"""
prepare_vision_data.py
======================
Converts and merges all raw vision datasets into a single YOLO-format
dataset ready for upload to Modal.

Sources  (data/vision/)
-----------------------
  CCTV Gun Detector.v1i.yolov11.zip          Roboflow, native YOLO, class: gun
  The Monash Guns Dataset.v2i.yolov11.zip    Roboflow, native YOLO, class: pistol
  archive.zip                                YouTube-GDD, YOLO, classes: person(0) gun(1)
  od-weapondetection_-sohas-detection-DatasetNinja.tar  Supervisely JSON, class: pistol

Output  (data/vision_yolo/)
---------------------------
  images/train/   images/val/   images/test/
  labels/train/   labels/val/   labels/test/
  data.yaml

All gun classes unified to class 0 = "gun".
Hard negatives (images with no gun) are included with empty label files —
the single highest-impact FP-reduction technique in the literature
(Olmos & Tabik 2018, Pérez-Hernández 2020).

Usage
-----
  python scripts/prepare_vision_data.py
  python scripts/prepare_vision_data.py --output data/vision_yolo
  python scripts/prepare_vision_data.py --dry-run
"""

import argparse
import json
import random
import sys
import tarfile
import zipfile
from pathlib import Path


# ── Constants ────────────────────────────────────────────────────────────────

SOURCES = {
    "cctv":   "CCTV Gun Detector.v1i.yolov11.zip",
    "monash": "The Monash Guns Dataset.v2i.yolov11.zip",
    "ygdd":   "archive.zip",
    "dninja": "od-weapondetection_-sohas-detection-DatasetNinja.tar",
}

DNINJA_GUN_CLASSES  = {"pistol", "gun"}
YGDD_GUN_CLASS_IDS  = {1}
DNINJA_VAL_FRACTION = 0.15
SEED = 42


# ── Helpers ──────────────────────────────────────────────────────────────────

def _write_label(dest: Path, lines: list[str], dry_run: bool) -> None:
    """
    Write YOLO label file.  Empty lines → empty file = hard-negative example.
    An empty label file tells YOLO there is nothing to detect in this image,
    suppressing false positives on visually similar objects.
    """
    if not dry_run:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text("\n".join(lines) + "\n" if lines else "")


def _write_image(dest: Path, data: bytes, dry_run: bool) -> None:
    if not dry_run:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)


def _bbox_to_yolo(x1: float, y1: float, x2: float, y2: float,
                  W: int, H: int) -> str:
    """Convert absolute [x1,y1,x2,y2] to YOLO normalised cx cy w h."""
    cx = max(0.0, min(1.0, (x1 + x2) / 2 / W))
    cy = max(0.0, min(1.0, (y1 + y2) / 2 / H))
    w  = max(0.0, min(1.0, (x2 - x1) / W))
    h  = max(0.0, min(1.0, (y2 - y1) / H))
    return f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def _stem(filename: str) -> str:
    return Path(filename).stem


def _make_stats() -> dict:
    return {"train": {"pos": 0, "neg": 0}, "val": {"pos": 0, "neg": 0},
            "test":  {"pos": 0, "neg": 0}, "skipped": 0}


# ── Per-dataset converters ────────────────────────────────────────────────────

def convert_roboflow_zip(
    zip_path: Path, prefix: str, out_root: Path, dry_run: bool
) -> dict:
    """
    Roboflow YOLOv11 zips (CCTV and Monash).

    Layout: train/images/ + train/labels/   valid/   test/
    Classes are already gun-only at class 0 — copy straight through.
    Images with empty label files → hard negatives (empty label written).
    """
    split_map = {"train": "train", "valid": "val", "test": "test"}
    stats = _make_stats()

    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()

        label_lookup: dict[str, dict[str, str]] = {s: {} for s in split_map}
        for n in names:
            for src in split_map:
                if n.startswith(f"{src}/labels/") and n.endswith(".txt"):
                    label_lookup[src][_stem(n)] = n

        for n in names:
            if not n.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            for src, dst in split_map.items():
                if not n.startswith(f"{src}/images/"):
                    continue
                stem      = _stem(n)
                lbl_path  = label_lookup[src].get(stem)
                if lbl_path is None:
                    stats["skipped"] += 1
                    continue

                raw   = zf.read(lbl_path).decode("utf-8").strip()
                lines = [l for l in raw.splitlines() if l.strip()]
                name  = f"{prefix}_{stem}"

                _write_image(out_root / "images" / dst / f"{name}{Path(n).suffix.lower()}",
                             zf.read(n), dry_run)
                _write_label(out_root / "labels" / dst / f"{name}.txt", lines, dry_run)

                key = "pos" if lines else "neg"
                stats[dst][key] += 1

    return stats


def convert_ygdd_zip(
    zip_path: Path, prefix: str, out_root: Path, dry_run: bool
) -> dict:
    """
    YouTube-GDD (archive.zip) — double-nested layout.

    Class 0=person (dropped), class 1=gun → remapped to 0.
    Images with label files but zero gun annotations → hard negatives.
    Test split skipped (no labels).
    """
    split_map = {"train": "train", "val": "val"}
    stats = _make_stats()

    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()

        label_lookup: dict[str, dict[str, str]] = {"train": {}, "val": {}}
        for n in names:
            for split in split_map:
                if f"/labels/{split}/" in n and n.endswith(".txt"):
                    label_lookup[split][_stem(n)] = n

        for n in names:
            if not n.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            for src, dst in split_map.items():
                if f"/images/{src}/" not in n:
                    continue
                stem     = _stem(n)
                lbl_path = label_lookup[src].get(stem)
                if lbl_path is None:
                    stats["skipped"] += 1
                    continue

                raw = zf.read(lbl_path).decode().strip()
                gun_lines = [
                    "0 " + " ".join(l.split()[1:])
                    for l in raw.splitlines()
                    if l.strip() and int(l.split()[0]) in YGDD_GUN_CLASS_IDS
                ]
                name = f"{prefix}_{stem}"

                _write_image(out_root / "images" / dst / f"{name}{Path(n).suffix.lower()}",
                             zf.read(n), dry_run)
                _write_label(out_root / "labels" / dst / f"{name}.txt", gun_lines, dry_run)

                key = "pos" if gun_lines else "neg"
                stats[dst][key] += 1

    return stats


def convert_dninja_tar(
    tar_path: Path, prefix: str, out_root: Path, dry_run: bool
) -> dict:
    """
    DatasetNinja / Supervisely tar.

    Layout: train/img/ + train/ann/  test/img/ + test/ann/
    Annotation JSON: absolute-pixel rectangles.

    Pistol/gun objects → YOLO bbox labels.
    All other images (knife, phone, wallet…) → hard negatives (empty label).
    Val split carved from ALL train stems for balanced representation.
    """
    rng   = random.Random(SEED)
    stats = _make_stats()

    # ── Pass 1: read every annotation, build stem→yolo_lines map ─────────────
    print("  [dninja] scanning annotations …", flush=True)

    # gun_labels[src_split][stem] = [yolo lines]  (empty list = no pistol found)
    gun_labels: dict[str, dict[str, list[str]]] = {"train": {}, "test": {}}

    with tarfile.open(tar_path, "r") as tf:
        for member in tf:
            if not member.isfile() or "/ann/" not in member.name:
                continue
            parts     = member.name.split("/")
            src_split = parts[0]
            if src_split not in gun_labels or len(parts) < 3:
                continue

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

            H, W = ann["size"]["height"], ann["size"]["width"]
            lines = []
            for obj in ann.get("objects", []):
                if obj.get("classTitle", "").lower() not in DNINJA_GUN_CLASSES:
                    continue
                pts = obj["points"]["exterior"]
                x1, y1 = pts[0]; x2, y2 = pts[1]
                lines.append(_bbox_to_yolo(x1, y1, x2, y2, W, H))

            # Always record the stem; empty lines = hard negative
            gun_labels[src_split][stem] = lines

    # ── Carve val from ALL train stems (gun + non-gun) ────────────────────────
    all_train = sorted(gun_labels["train"])
    n_val     = int(len(all_train) * DNINJA_VAL_FRACTION)
    val_stems = set(rng.sample(all_train, n_val))

    n_gun_train = sum(1 for l in gun_labels["train"].values() if l)
    n_gun_test  = sum(1 for l in gun_labels["test"].values()  if l)
    print(f"  [dninja] total train={len(all_train)} (gun={n_gun_train} neg={len(all_train)-n_gun_train})  "
          f"val carved={n_val}  test={len(gun_labels['test'])} (gun={n_gun_test})", flush=True)

    # ── Write label files ─────────────────────────────────────────────────────
    # Build routing: stem → (dst_split)
    routing: dict[tuple[str, str], str] = {}  # (src_split, stem) → dst_split
    for stem in gun_labels["train"]:
        dst = "val" if stem in val_stems else "train"
        routing[("train", stem)] = dst
        key = "pos" if gun_labels["train"][stem] else "neg"
        stats[dst][key] += 1
    for stem in gun_labels["test"]:
        routing[("test", stem)] = "test"
        key = "pos" if gun_labels["test"][stem] else "neg"
        stats["test"][key] += 1

    if not dry_run:
        for (src, stem), dst in routing.items():
            lines = gun_labels[src][stem]
            _write_label(out_root / "labels" / dst / f"{prefix}_{stem}.txt",
                         lines, dry_run=False)

    # ── Pass 2: stream images ─────────────────────────────────────────────────
    print("  [dninja] copying images …", flush=True)
    if not dry_run:
        with tarfile.open(tar_path, "r") as tf:
            for member in tf:
                if not member.isfile() or "/img/" not in member.name:
                    continue
                if not member.name.lower().endswith((".jpg", ".jpeg")):
                    continue

                parts     = member.name.split("/")
                src_split = parts[0]
                fname     = parts[-1]

                for ext in (".jpg", ".jpeg", ".JPG", ".JPEG"):
                    if fname.lower().endswith(ext.lower()):
                        stem = fname[: -len(ext)]
                        break
                else:
                    continue

                dst = routing.get((src_split, stem))
                if dst is None:
                    continue

                f = tf.extractfile(member)
                if f is None:
                    continue
                _write_image(
                    out_root / "images" / dst / f"{prefix}_{stem}{Path(fname).suffix.lower()}",
                    f.read(), dry_run=False,
                )

    return stats


# ── data.yaml ────────────────────────────────────────────────────────────────

def write_data_yaml(out_root: Path, dry_run: bool) -> None:
    content = (
        f"path: {out_root.resolve()}\n"
        "train: images/train\n"
        "val:   images/val\n"
        "test:  images/test\n"
        "\n"
        "nc: 1\n"
        "names: ['gun']\n"
    )
    if not dry_run:
        (out_root / "data.yaml").write_text(content)
    print(content)


# ── Main ─────────────────────────────────────────────────────────────────────

def _print_stats(tag: str, s: dict) -> None:
    for split in ("train", "val", "test"):
        p, n = s[split]["pos"], s[split]["neg"]
        if p + n:
            print(f"    {split:5}  pos={p:>5}  neg={n:>4}  total={p+n:>5}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input",   default="data/vision",    help="Raw zips/tar directory")
    parser.add_argument("--output",  default="data/vision_yolo", help="Output YOLO dataset")
    parser.add_argument("--dry-run", action="store_true",      help="Count only, write nothing")
    args = parser.parse_args()

    in_dir  = Path(args.input)
    out_dir = Path(args.output)
    dry_run = args.dry_run

    if not in_dir.exists():
        sys.exit(f"ERROR: input directory not found: {in_dir}")
    if not dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    totals = _make_stats()

    converters = [
        ("1/4", "CCTV",    SOURCES["cctv"],   "cctv",   convert_roboflow_zip),
        ("2/4", "Monash",  SOURCES["monash"], "monash", convert_roboflow_zip),
        ("3/4", "YT-GDD",  SOURCES["ygdd"],   "ygdd",   convert_ygdd_zip),
        ("4/4", "DNinja",  SOURCES["dninja"], "dninja", convert_dninja_tar),
    ]

    for num, name, fname, prefix, fn in converters:
        src = in_dir / fname
        if not src.exists():
            print(f"\n[{num}] {name} not found — skipping")
            continue
        print(f"\n[{num}] {name}  ({src.stat().st_size/1e6:.0f} MB)")
        s = fn(src, prefix, out_dir, dry_run)
        _print_stats(name, s)
        for split in ("train", "val", "test"):
            totals[split]["pos"] += s[split]["pos"]
            totals[split]["neg"] += s[split]["neg"]
        totals["skipped"] += s.get("skipped", 0)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "─" * 62)
    print("MERGED DATASET SUMMARY")
    print("─" * 62)
    print(f"  {'split':<6}  {'positives':>10}  {'negatives':>10}  {'total':>7}")
    print(f"  {'─'*6}  {'─'*10}  {'─'*10}  {'─'*7}")
    grand_pos = grand_neg = 0
    for split in ("train", "val", "test"):
        p, n = totals[split]["pos"], totals[split]["neg"]
        grand_pos += p; grand_neg += n
        print(f"  {split:<6}  {p:>10}  {n:>10}  {p+n:>7}")
    print(f"  {'─'*6}  {'─'*10}  {'─'*10}  {'─'*7}")
    print(f"  {'TOTAL':<6}  {grand_pos:>10}  {grand_neg:>10}  {grand_pos+grand_neg:>7}")
    print(f"  neg ratio: {grand_neg/(grand_pos+grand_neg)*100:.1f}%  "
          f"(target 20-30% per Olmos 2018)")
    print("─" * 62)

    if dry_run:
        print("\n[DRY RUN — no files written]\n")
        print("data.yaml preview:")
        write_data_yaml(out_dir, dry_run=True)
        return

    write_data_yaml(out_dir, dry_run=False)
    print(f"\nDone → {out_dir}")
    print("Upload to Modal:")
    print(f"  modal volume put --force vision-train {out_dir}/images yolo/")
    print(f"  modal volume put --force vision-train {out_dir}/labels yolo/")
    print(f"  modal volume put --force vision-train {out_dir}/data.yaml yolo/")


if __name__ == "__main__":
    main()
