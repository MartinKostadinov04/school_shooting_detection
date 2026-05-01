"""
wav_info.py
===========
Inspect WAV file characteristics.

Usage
-----
    # Single file
    python wav_info.py path/to/file.wav

    # Multiple files
    python wav_info.py file1.wav file2.wav file3.wav

    # Entire directory (recursive)
    python wav_info.py path/to/folder/

    # Directory, show only files that differ from a target spec
    python wav_info.py path/to/folder/ --check-yamnet

Reports
-------
For every file:
  - Sample rate (Hz)
  - Channels (1 = mono, 2 = stereo, …)
  - Bit depth (8, 16, 24, 32 bit)
  - Duration (seconds and sample count)
  - File size (KB)

If librosa is installed, also reports:
  - Peak amplitude
  - RMS amplitude
  - Whether the clip is nearly silent
  - Whether it contains NaN values

With --check-yamnet, flags any file that does NOT already match YAMNet's
required input format (16 kHz, mono, float32-compatible). Useful before
running extract_embeddings.py.
"""

import argparse
import os
import struct
import sys
import wave
from pathlib import Path
from typing import Dict, List, Optional

# librosa is optional — used only for waveform statistics
try:
    import numpy as np
    import librosa
    _LIBROSA_AVAILABLE = True
except ImportError:
    _LIBROSA_AVAILABLE = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

YAMNET_SR = 16_000
YAMNET_CHANNELS = 1
SILENCE_THRESHOLD = 1e-4   # peak amplitude below which clip is "nearly silent"

# ANSI colour codes (disabled on Windows if not supported)
_USE_COLOR = sys.stdout.isatty() and os.name != "nt" or os.environ.get("FORCE_COLOR")

def _c(text: str, code: str) -> str:
    if _USE_COLOR:
        return f"\033[{code}m{text}\033[0m"
    return text

RED    = lambda t: _c(t, "31")
YELLOW = lambda t: _c(t, "33")
GREEN  = lambda t: _c(t, "32")
BOLD   = lambda t: _c(t, "1")
DIM    = lambda t: _c(t, "2")


# ---------------------------------------------------------------------------
# Core inspection logic
# ---------------------------------------------------------------------------


def inspect_wav(path: Path) -> Dict:
    """
    Read WAV metadata and (optionally) waveform statistics for a single file.

    Parameters
    ----------
    path : Path
        Path to the WAV file.

    Returns
    -------
    dict with keys:
        path, sample_rate, channels, bit_depth, n_frames, duration_s,
        file_size_kb, error (str or None),
        peak, rms, is_silent, has_nan  (None if librosa unavailable)
    """
    result = {
        "path": path,
        "sample_rate": None,
        "channels": None,
        "bit_depth": None,
        "n_frames": None,
        "duration_s": None,
        "file_size_kb": None,
        "error": None,
        # Waveform stats — populated only if librosa is available
        "peak": None,
        "rms": None,
        "is_silent": None,
        "has_nan": None,
    }

    # File size
    try:
        result["file_size_kb"] = path.stat().st_size / 1024
    except OSError:
        pass

    # Metadata via stdlib wave
    try:
        with wave.open(str(path), "rb") as wf:
            result["sample_rate"] = wf.getframerate()
            result["channels"] = wf.getnchannels()
            result["bit_depth"] = wf.getsampwidth() * 8
            result["n_frames"] = wf.getnframes()
            sr = result["sample_rate"]
            result["duration_s"] = result["n_frames"] / sr if sr > 0 else 0.0
    except wave.Error as exc:
        result["error"] = f"wave.Error: {exc}"
        return result
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        return result

    # Waveform statistics via librosa (optional)
    if _LIBROSA_AVAILABLE:
        try:
            audio, _ = librosa.load(str(path), sr=None, mono=True)
            peak = float(np.max(np.abs(audio))) if len(audio) > 0 else 0.0
            rms = float(np.sqrt(np.mean(audio ** 2))) if len(audio) > 0 else 0.0
            result["peak"] = peak
            result["rms"] = rms
            result["is_silent"] = peak < SILENCE_THRESHOLD
            result["has_nan"] = bool(np.any(np.isnan(audio)))
        except Exception as exc:
            result["error"] = (result["error"] or "") + f" | librosa: {exc}"

    return result


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def _fmt_duration(seconds: float) -> str:
    if seconds < 1.0:
        return f"{seconds * 1000:.1f} ms"
    if seconds < 60.0:
        return f"{seconds:.3f} s"
    m, s = divmod(seconds, 60)
    return f"{int(m)}m {s:.1f}s"


def _yamnet_issues(info: Dict) -> List[str]:
    """Return a list of human-readable issues for YAMNet compatibility."""
    issues = []
    if info["sample_rate"] is not None and info["sample_rate"] != YAMNET_SR:
        issues.append(f"sample rate {info['sample_rate']} Hz (need {YAMNET_SR})")
    if info["channels"] is not None and info["channels"] != YAMNET_CHANNELS:
        issues.append(f"{info['channels']} channels (need mono)")
    return issues


def print_info(info: Dict, check_yamnet: bool = False, verbose: bool = False) -> None:
    """Print a single file's characteristics to stdout."""
    path = info["path"]
    print(BOLD(f"\n{path.name}"))
    print(DIM(f"  {path}"))

    if info["error"]:
        print(RED(f"  ERROR: {info['error']}"))
        return

    # Core metadata
    sr = info["sample_rate"]
    ch = info["channels"]
    bd = info["bit_depth"]
    dur = info["duration_s"]
    n = info["n_frames"]
    kb = info["file_size_kb"]

    ch_label = {1: "mono", 2: "stereo"}.get(ch, f"{ch}-ch") if ch else "?"
    print(f"  Sample rate  : {sr:,} Hz")
    print(f"  Channels     : {ch} ({ch_label})")
    print(f"  Bit depth    : {bd}-bit")
    print(f"  Duration     : {_fmt_duration(dur)}  ({n:,} samples)")
    print(f"  File size    : {kb:.1f} KB")

    # Waveform stats
    if info["peak"] is not None:
        silent_tag = RED("  [SILENT]") if info["is_silent"] else ""
        nan_tag    = RED("  [HAS NaN]") if info["has_nan"] else ""
        print(f"  Peak amp.    : {info['peak']:.6f}{silent_tag}")
        print(f"  RMS amp.     : {info['rms']:.6f}{nan_tag}")

    # YAMNet compatibility check
    if check_yamnet:
        issues = _yamnet_issues(info)
        if issues:
            print(YELLOW("  YAMNet       : needs preprocessing — " + "; ".join(issues)))
        else:
            print(GREEN("  YAMNet       : OK (already 16 kHz mono)"))


def print_summary(infos: List[Dict], check_yamnet: bool = False) -> None:
    """Print a summary table for multiple files."""
    n_total = len(infos)
    n_errors = sum(1 for i in infos if i["error"])
    n_ok = n_total - n_errors

    print(BOLD(f"\n{'─' * 60}"))
    print(BOLD(f"Summary: {n_total} file(s)"))
    print(f"  Readable    : {n_ok}")
    if n_errors:
        print(RED(f"  Errors      : {n_errors}"))

    readable = [i for i in infos if not i["error"]]
    if not readable:
        return

    # Sample rate breakdown
    sr_counts: Dict[int, int] = {}
    for i in readable:
        sr_counts[i["sample_rate"]] = sr_counts.get(i["sample_rate"], 0) + 1
    if len(sr_counts) == 1:
        print(f"  Sample rate : {list(sr_counts)[0]:,} Hz (all files)")
    else:
        rates = ", ".join(f"{sr:,} Hz × {cnt}" for sr, cnt in sorted(sr_counts.items()))
        print(YELLOW(f"  Sample rates: {rates}"))

    # Channel breakdown
    ch_counts: Dict[int, int] = {}
    for i in readable:
        ch_counts[i["channels"]] = ch_counts.get(i["channels"], 0) + 1
    ch_parts = []
    for ch, cnt in sorted(ch_counts.items()):
        label = {1: "mono", 2: "stereo"}.get(ch, f"{ch}-ch")
        ch_parts.append(f"{cnt} × {label}")
    print(f"  Channels    : {', '.join(ch_parts)}")

    # Bit depth breakdown
    bd_counts: Dict[int, int] = {}
    for i in readable:
        bd_counts[i["bit_depth"]] = bd_counts.get(i["bit_depth"], 0) + 1
    bd_parts = ", ".join(f"{bd}-bit × {cnt}" for bd, cnt in sorted(bd_counts.items()))
    print(f"  Bit depth   : {bd_parts}")

    # Duration stats
    durations = [i["duration_s"] for i in readable]
    total_dur = sum(durations)
    print(f"  Duration    : min {_fmt_duration(min(durations))}  "
          f"max {_fmt_duration(max(durations))}  "
          f"total {_fmt_duration(total_dur)}")

    # Waveform stats (if librosa available)
    if _LIBROSA_AVAILABLE:
        n_silent = sum(1 for i in readable if i["is_silent"])
        n_nan    = sum(1 for i in readable if i["has_nan"])
        if n_silent:
            print(YELLOW(f"  Silent clips: {n_silent}"))
        if n_nan:
            print(RED(f"  Has NaN     : {n_nan}"))

    # YAMNet compatibility
    if check_yamnet:
        n_issues = sum(1 for i in readable if _yamnet_issues(i))
        if n_issues == 0:
            print(GREEN(f"  YAMNet      : all {n_ok} files are 16 kHz mono"))
        else:
            print(YELLOW(f"  YAMNet      : {n_issues} file(s) need resampling/mixing"))


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------


def collect_wav_files(targets: List[Path]) -> List[Path]:
    """Expand a list of files/directories into a flat list of .wav paths."""
    result = []
    for target in targets:
        if target.is_dir():
            found = sorted(list(target.rglob("*.wav")) + list(target.rglob("*.WAV")))
            # Deduplicate on Windows case-insensitive filesystems
            seen = set()
            for f in found:
                k = str(f).lower()
                if k not in seen:
                    seen.add(k)
                    result.append(f)
        elif target.is_file():
            result.append(target)
        else:
            print(RED(f"Not found: {target}"), file=sys.stderr)
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect WAV file characteristics.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python wav_info.py recording.wav
  python wav_info.py data/raw/gunshot/
  python wav_info.py data/raw/ --check-yamnet
  python wav_info.py file1.wav file2.wav --no-detail
        """,
    )
    parser.add_argument(
        "targets",
        nargs="+",
        type=Path,
        help="WAV file(s) or directory/ies to inspect.",
    )
    parser.add_argument(
        "--check-yamnet",
        action="store_true",
        help="Flag files that are not already in YAMNet format (16 kHz, mono).",
    )
    parser.add_argument(
        "--no-detail",
        action="store_true",
        help="Skip per-file output; show summary table only.",
    )
    parser.add_argument(
        "--errors-only",
        action="store_true",
        help="Only print files that have errors or quality issues.",
    )
    args = parser.parse_args()

    wav_files = collect_wav_files(args.targets)

    if not wav_files:
        print("No WAV files found.")
        sys.exit(1)

    if not _LIBROSA_AVAILABLE:
        print(DIM("(librosa not installed — waveform statistics unavailable)\n"))

    infos = []
    for path in wav_files:
        info = inspect_wav(path)
        infos.append(info)

        if args.errors_only:
            has_issues = (
                info["error"]
                or info.get("is_silent")
                or info.get("has_nan")
                or (args.check_yamnet and _yamnet_issues(info))
            )
            if has_issues:
                print_info(info, check_yamnet=args.check_yamnet)
        elif not args.no_detail:
            print_info(info, check_yamnet=args.check_yamnet)

    if len(infos) > 1 or args.no_detail:
        print_summary(infos, check_yamnet=args.check_yamnet)


if __name__ == "__main__":
    main()
