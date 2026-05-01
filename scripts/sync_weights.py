#!/usr/bin/env python3
"""
sync_weights.py
===============
Mirror the latest YOLO checkpoint files from the Modal volume to local disk
on a schedule, so ``models/yolo_finetuned/best.pt`` is always fresh while
training is in flight.

The training script's background commit thread refreshes the volume's
``weights/best.pt`` and ``weights/last.pt`` aliases every 5 minutes (and
every 5 epochs via the on_fit_epoch_end callback). This poller mirrors
those aliases down to your machine, skipping the rename whenever the
remote file is unchanged so we don't burn bandwidth on idle polls.

Default behaviour
-----------------
    Polls every 300 s, mirrors ``weights/best.pt`` and ``weights/last.pt``
    on the ``vision-train`` Modal volume into ``models/yolo_finetuned/``.

Usage
-----
    # Run alongside training in a separate terminal:
    python scripts/sync_weights.py

    # Faster polling (more bandwidth, freshness within ~60 s):
    python scripts/sync_weights.py --interval 60

    # Only mirror best.pt:
    python scripts/sync_weights.py --skip-last

    # Custom output directory (e.g. mirror somewhere else):
    python scripts/sync_weights.py --output some/other/dir

The script exits cleanly on Ctrl+C. It is stateless beyond local file
hashes — restart it anytime; it picks up where it left off.
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
import time
from pathlib import Path

VOLUME = "vision-train"


def _file_hash(path: Path) -> str:
    """Return the SHA-256 of a file, or ``""`` if it doesn't exist."""
    if not path.exists():
        return ""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(remote: str, local: Path) -> tuple[bool, str]:
    """
    Download ``remote`` from the Modal volume to ``local``.

    Writes to a sibling ``.tmp`` file first, then compares hashes and
    only renames over the destination if the content actually changed.
    Returns ``(changed, status)`` where ``status`` is ``"updated"``,
    ``"unchanged"``, ``"missing"`` (remote not present yet), or
    ``"error"``.
    """
    local.parent.mkdir(parents=True, exist_ok=True)
    tmp = local.with_suffix(local.suffix + ".tmp")

    try:
        result = subprocess.run(
            [
                "modal", "volume", "get", "--force",
                VOLUME, remote, str(tmp),
            ],
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        # `modal` CLI not on PATH — bail loudly so the user knows.
        print("ERROR: `modal` CLI not found on PATH. Install with `pip install modal`.")
        sys.exit(1)

    if result.returncode != 0:
        # Most common reason: remote file doesn't exist yet (training
        # hasn't completed its first validation). Surface other errors.
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        stderr = (result.stderr or "").strip()
        if "not found" in stderr.lower() or "no such" in stderr.lower():
            return False, "missing"
        return False, f"error: {stderr.splitlines()[-1] if stderr else 'unknown'}"

    new_hash = _file_hash(tmp)
    old_hash = _file_hash(local)
    if new_hash and new_hash == old_hash:
        tmp.unlink(missing_ok=True)
        return False, "unchanged"

    tmp.replace(local)
    return True, "updated"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--interval", type=int, default=300,
        help="Seconds between polls. Default 300 (matches the training "
             "background commit cadence). Lower = fresher local copy "
             "but more bandwidth.",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("models/yolo_finetuned"),
        help="Local directory to mirror weights into. "
             "Default models/yolo_finetuned (where YOLO_WEIGHTS_PATH "
             "in inference/config.py looks for them).",
    )
    parser.add_argument(
        "--skip-last", action="store_true",
        help="Only mirror best.pt; skip last.pt.",
    )
    parser.add_argument(
        "--once", action="store_true",
        help="Run a single sync pass and exit — useful in CI.",
    )
    args = parser.parse_args()

    # Force line-buffering so users running this in PowerShell / cmd.exe see
    # progress messages immediately rather than after the entire poll loop.
    # (Python's default on Windows is full buffering when stdout isn't a TTY.)
    try:
        sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
    except Exception:
        pass

    targets: list[tuple[str, Path]] = [
        ("weights/best.pt", args.output / "best.pt"),
    ]
    if not args.skip_last:
        targets.append(("weights/last.pt", args.output / "last.pt"))

    print(f"Polling Modal volume '{VOLUME}' every {args.interval}s", flush=True)
    print(f"Mirroring  → {args.output.resolve()}", flush=True)
    for remote, _ in targets:
        print(f"           · {remote}", flush=True)
    print("Press Ctrl+C to stop.\n", flush=True)

    seen_missing: set[str] = set()

    def _pass() -> None:
        for remote, local in targets:
            changed, status = _download(remote, local)
            ts = time.strftime("%H:%M:%S")
            name = Path(remote).name
            if status == "updated":
                size_mb = local.stat().st_size / 1e6
                print(
                    f"  [{ts}] ✓ {name:<8}  updated ({size_mb:.1f} MB)  → {local}",
                    flush=True,
                )
                seen_missing.discard(remote)
            elif status == "missing":
                # Only print "waiting" once per remote until the file shows up,
                # to keep the log clean during long pre-validation periods.
                if remote not in seen_missing:
                    print(
                        f"  [{ts}]   {name:<8}  waiting — remote file not present yet",
                        flush=True,
                    )
                    seen_missing.add(remote)
            elif status.startswith("error"):
                print(f"  [{ts}] ! {name:<8}  {status}", flush=True)

    try:
        if args.once:
            _pass()
            return
        while True:
            _pass()
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
