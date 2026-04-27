"""
cascade.py
==========
Interactive two-stage detection pipeline.

  Stage 1 (Audio):  YAMNet + Dense head  — gunshot detection (sliding window, 75% overlap)
  Stage 2 (Video):  YOLOv11s             — visual gun confirmation

Ably events emitted
  audio:detected:{location}:{prob}   mic turns red, incident created
  video:detected:{location}:{conf}   camera turns red, police alert WITH visual reference
  video:negative:{location}          no camera trigger, police alert WITHOUT visual reference

────────────────────────────────────────────────────────────────────────────
REPL mode (default)
────────────────────────────────────────────────────────────────────────────
  python -m inference.cascade

  cascade [Cafeteria]> shot.wav                 # use default location
  cascade [Cafeteria]> shot.wav Gymnasium        # override location for this run

  On gunshot detection the engine pauses and asks:
    Video path (Enter to skip): clip.mp4

  After YOLO runs:
    gun found  → video:detected published  → police alert WITH visual reference
    no gun     → video:negative published  → police alert WITHOUT visual reference

────────────────────────────────────────────────────────────────────────────
Live-mic mode  (--live)
────────────────────────────────────────────────────────────────────────────
  python -m inference.cascade --live [--location "Cafeteria"]

  Runs the mic continuously in the background (same windowing/overlap as
  live_inference.py).  On detection the terminal prompts for a video path.

────────────────────────────────────────────────────────────────────────────
Common flags
────────────────────────────────────────────────────────────────────────────
  --location          default location label          (default: Cafeteria)
  --audio_threshold   gunshot probability threshold   (default: 0.64)
  --video_threshold   YOLO confidence threshold       (default: 0.60)
  --audio_model       path to .keras weights
  --video_model       path to .pt weights
  --ably_key / ABLY_API_KEY
  --show              open OpenCV window during video stage
"""

import argparse
import json
import logging
import os
import queue
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import tensorflow as tf

from pipeline.extract_embeddings import load_yamnet, extract_embedding
from inference.live_inference import (
    AblyPublisher,
    AudioCapture,
    DEFAULT_CHANNEL,
    DEFAULT_LOCATION,
    DEFAULT_THRESHOLD,
    ALERT_COOLDOWN_SECS,
    SAMPLE_RATE,
    CLIP_SAMPLES,
    CHUNK_SAMPLES,
)
from vision.live_inference import VideoCapture

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

AUDIO_DEFAULT_MODEL  = Path("models/saved_weights/dense_head_best.keras")
AUDIO_DEFAULT_LOG    = Path("inference/cascade_detections.jsonl")
VIDEO_DEFAULT_MODEL  = Path("YOLO_hugging-main/best.pt")
VIDEO_DEFAULT_LOG    = Path("vision/cascade_detections.jsonl")
VIDEO_DEFAULT_THRESH = 0.60
VIDEO_DEFAULT_IOU    = 0.45
VIDEO_DEFAULT_IMGSZ  = 1280


# ---------------------------------------------------------------------------
# Core pipeline helpers
# ---------------------------------------------------------------------------

def _run_audio_inference(buffer: np.ndarray, yamnet, head: tf.keras.Model) -> float:
    emb = extract_embedding(buffer, yamnet)
    return float(head.predict(emb[np.newaxis], verbose=0)[0, 0])


def infer_audio_file(
    path: Path,
    location: str,
    yamnet,
    head: tf.keras.Model,
    threshold: float,
    publisher: "AblyPublisher | None",
    log_file: Path,
) -> tuple[bool, float]:
    """
    Process an audio file through the sliding-window pipeline.
    Returns (detected, max_prob).
    Publishes audio:detected on the first threshold crossing (5 s cooldown).
    """
    import librosa
    audio, _ = librosa.load(str(path), sr=SAMPLE_RATE, mono=True)
    n_chunks  = len(audio) // CHUNK_SAMPLES
    chunk_dur = CHUNK_SAMPLES / SAMPLE_RATE

    buffer     = np.zeros(CLIP_SAMPLES, dtype=np.float32)
    max_prob   = 0.0
    detected   = False
    last_alert = 0.0

    print(f"\n  ▶  '{path.name}'  ({len(audio) / SAMPLE_RATE:.1f} s  ·  "
          f"{n_chunks} chunks  ·  window={CLIP_SAMPLES/SAMPLE_RATE:.1f}s  "
          f"hop={chunk_dur:.1f}s)\n")

    for i in range(n_chunks):
        t0 = time.perf_counter()

        chunk = audio[i * CHUNK_SAMPLES : (i + 1) * CHUNK_SAMPLES].astype(np.float32)
        buffer = np.roll(buffer, -CHUNK_SAMPLES)
        buffer[-CHUNK_SAMPLES:] = chunk

        prob     = _run_audio_inference(buffer, yamnet, head)
        max_prob = max(max_prob, prob)

        print(f"  [{i * chunk_dur:5.1f}s]  prob={prob:.4f}", end="\r")

        now = time.monotonic()
        if prob >= threshold and now - last_alert >= ALERT_COOLDOWN_SECS:
            last_alert = now
            detected   = True
            ts         = datetime.now(timezone.utc).isoformat()

            print(f"\n\n  🔴  GUNSHOT DETECTED  prob={prob:.3f}  loc={location}\n")

            if publisher:
                publisher.publish("audio:detected", f"audio:detected:{location}:{prob:.4f}")
                logger.info("Ably  →  audio:detected:%s  prob=%.4f", location, prob)

            try:
                log_file.parent.mkdir(parents=True, exist_ok=True)
                with open(log_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps({
                        "event": "gunshot_detected",
                        "timestamp": ts,
                        "probability": round(prob, 4),
                        "location": location,
                    }) + "\n")
            except Exception as exc:
                logger.warning("Log write failed: %s", exc)

        elapsed = time.perf_counter() - t0
        if chunk_dur - elapsed > 0:
            time.sleep(chunk_dur - elapsed)

    if not detected:
        print(f"\n\n  ■  No gunshot detected  (max_prob={max_prob:.4f})\n")

    return detected, max_prob


def infer_video_file(
    path: Path,
    location: str,
    video_model: Path,
    threshold: float,
    iou: float,
    imgsz: int,
    publisher: "AblyPublisher | None",
    log_file: Path,
    s3_bucket: "str | None",
    aws_region: str,
    show: bool,
) -> tuple[bool, float]:
    """
    Process a video file through YOLO.
    Returns (detected, max_conf).
    Publishes video:detected if gun found; caller publishes video:negative if not.
    """
    cap = VideoCapture(
        model_path=video_model,
        threshold=threshold,
        iou=iou,
        imgsz=imgsz,
        location=location,
        log_file=log_file,
        publisher=publisher,
        s3_bucket=s3_bucket,
        aws_region=aws_region,
        source=str(path),
        show=show,
    )
    print(f"\n  ▶  STAGE-2  '{path.name}'  threshold={threshold:.2f}\n")
    cap.start()
    detected, max_conf = cap.result
    if detected:
        print(f"\n  🔴  GUN DETECTED  conf={max_conf:.3f}  loc={location}")
        print(f"  → Police alert WITH visual reference\n")
    else:
        print(f"\n  ■  No gun detected  (max_conf={max_conf:.3f})\n")
    return detected, max_conf


def publish_video_negative(location: str, publisher: "AblyPublisher | None") -> None:
    if publisher:
        publisher.publish("video:negative", f"video:negative:{location}")
        logger.info("Ably  →  video:negative:%s", location)
    print(f"  ⚪  Police alert WITHOUT visual reference\n")


# ---------------------------------------------------------------------------
# Video prompt — shared by REPL and live-mic modes
# ---------------------------------------------------------------------------

def prompt_and_run_video(
    location: str,
    video_model: Path,
    threshold: float,
    iou: float,
    imgsz: int,
    publisher: "AblyPublisher | None",
    log_file: Path,
    s3_bucket: "str | None",
    aws_region: str,
    show: bool,
) -> None:
    try:
        path_str = input("  Video path for visual confirmation (Enter to skip): ").strip()
    except (EOFError, KeyboardInterrupt):
        publish_video_negative(location, publisher)
        return

    if not path_str:
        publish_video_negative(location, publisher)
        return

    path = Path(path_str)
    if not path.exists():
        print(f"  File not found: {path}")
        publish_video_negative(location, publisher)
        return

    detected, _ = infer_video_file(
        path=path,
        location=location,
        video_model=video_model,
        threshold=threshold,
        iou=iou,
        imgsz=imgsz,
        publisher=publisher,
        log_file=log_file,
        s3_bucket=s3_bucket,
        aws_region=aws_region,
        show=show,
    )
    if not detected:
        publish_video_negative(location, publisher)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interactive two-stage gunshot + visual gun detection pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--audio_model",     type=Path,  default=AUDIO_DEFAULT_MODEL)
    parser.add_argument("--audio_threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--video_model",     type=Path,  default=VIDEO_DEFAULT_MODEL)
    parser.add_argument("--video_threshold", type=float, default=VIDEO_DEFAULT_THRESH)
    parser.add_argument("--iou",             type=float, default=VIDEO_DEFAULT_IOU)
    parser.add_argument("--imgsz",           type=int,   default=VIDEO_DEFAULT_IMGSZ)
    parser.add_argument("--location",        type=str,   default=DEFAULT_LOCATION)
    parser.add_argument("--log_file",        type=Path,  default=AUDIO_DEFAULT_LOG)
    parser.add_argument("--video_log_file",  type=Path,  default=VIDEO_DEFAULT_LOG)
    parser.add_argument("--channel",         type=str,   default=DEFAULT_CHANNEL)
    parser.add_argument("--ably_key",        type=str,   default=None)
    parser.add_argument("--s3_bucket",       type=str,   default=None)
    parser.add_argument("--aws_region",      type=str,   default="us-east-1")
    parser.add_argument("--show",            action="store_true",
                        help="Open OpenCV window during video stage.")
    parser.add_argument("--live",            action="store_true",
                        help="Run live mic in background instead of REPL file mode.")
    parser.add_argument("--device",          type=int,   default=None,
                        help="sounddevice mic index (--live only).")
    args = parser.parse_args()

    for label, path in [("Audio model", args.audio_model), ("Video model", args.video_model)]:
        if not path.exists():
            logger.error("%s not found: %s", label, path)
            sys.exit(1)

    ably_key  = args.ably_key or os.environ.get("ABLY_API_KEY")
    publisher = None
    if ably_key:
        try:
            publisher = AblyPublisher(ably_key, args.channel)
        except Exception as exc:
            logger.warning("Ably connection failed: %s — running without WS alerts", exc)
    else:
        logger.warning("No Ably key — WS alerts disabled (set --ably_key or ABLY_API_KEY)")

    # Shared kwargs forwarded to every video stage call
    video_kwargs = dict(
        video_model=args.video_model,
        threshold=args.video_threshold,
        iou=args.iou,
        imgsz=args.imgsz,
        publisher=publisher,
        log_file=args.video_log_file,
        s3_bucket=args.s3_bucket,
        aws_region=args.aws_region,
        show=args.show,
    )

    logger.info("Loading YAMNet ...")
    yamnet = load_yamnet()
    logger.info("Loading audio head from %s ...", args.audio_model)
    head = tf.keras.models.load_model(str(args.audio_model))

    print(
        f"\n  TacticalEye — Cascade Pipeline\n"
        f"  ══════════════════════════════════════════════════════\n"
        f"  Audio:    threshold={args.audio_threshold:.2f}  "
        f"window={CLIP_SAMPLES/SAMPLE_RATE:.1f}s  hop={CHUNK_SAMPLES/SAMPLE_RATE:.1f}s\n"
        f"  Video:    threshold={args.video_threshold:.2f}  model={args.video_model.name}\n"
        f"  Location: {args.location}\n"
        f"  Mode:     {'live mic' if args.live else 'REPL (file submission)'}\n"
        f"  ══════════════════════════════════════════════════════\n"
    )

    # ── LIVE MIC MODE ────────────────────────────────────────────────────────
    if args.live:
        detection_q: "queue.Queue[tuple[float, str, str]]" = queue.Queue()

        capture = AudioCapture(
            yamnet_model=yamnet,
            head_model=head,
            threshold=args.audio_threshold,
            location=args.location,
            log_file=args.log_file,
            publisher=publisher,
            s3_bucket=args.s3_bucket,
            aws_region=args.aws_region,
            device=args.device,
            on_detection=lambda prob, loc, ts: detection_q.put((prob, loc, ts)),
        )
        capture.start()
        print(f"  🎙  Live mic active — monitoring '{args.location}' ...\n"
              f"  Press Ctrl+C to stop.\n")

        try:
            while True:
                try:
                    prob, location, _ = detection_q.get(timeout=0.5)
                except queue.Empty:
                    continue
                prompt_and_run_video(location=location, **video_kwargs)
        except KeyboardInterrupt:
            print("\nStopped.")
        finally:
            capture.stop()

    # ── REPL FILE MODE ───────────────────────────────────────────────────────
    else:
        print(f"  Type an audio file path to run Stage-1 inference.")
        print(f"  Optionally append the location:  shot.wav Gymnasium")
        print(f"  Type 'quit' or Ctrl+C to exit.\n")

        try:
            while True:
                try:
                    line = input(f"  cascade [{args.location}]> ").strip()
                except (EOFError, KeyboardInterrupt):
                    break

                if not line:
                    continue
                if line.lower() in ("q", "quit", "exit"):
                    break

                # Parse: <audio_path> [location override]
                parts    = line.split(None, 1)
                path     = Path(parts[0])
                location = parts[1].strip("'\"") if len(parts) > 1 else args.location

                if not path.exists():
                    print(f"  File not found: {path}\n")
                    continue

                detected, _ = infer_audio_file(
                    path=path,
                    location=location,
                    yamnet=yamnet,
                    head=head,
                    threshold=args.audio_threshold,
                    publisher=publisher,
                    log_file=args.log_file,
                )

                if detected:
                    prompt_and_run_video(location=location, **video_kwargs)

        except KeyboardInterrupt:
            pass

        print("\nStopped.")

    if publisher:
        publisher.close()
    logger.info("Done.")


if __name__ == "__main__":
    main()
