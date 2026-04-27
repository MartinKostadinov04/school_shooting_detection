"""
cascade.py
==========
Gated two-stage detection pipeline.

Stage 1 (always active):  Audio — YAMNet + Dense head (gunshot detection)
Stage 2 (gated):          Video — YOLOv11s (gun detection)
                          Activated ONLY when Stage 1 fires a positive event.

After each audio trigger the video stage runs for --video_window seconds,
then returns to standby. This keeps the compute-heavy vision model idle when
there is no alert and cuts the false-positive rate.

Usage
-----
  Live mic + webcam:
    python -m inference.cascade --location "Cafeteria"

  Demo — audio file feeds Stage 1, video file feeds Stage 2:
    python -m inference.cascade \\
        --audio_file   shot.wav \\
        --video_source path/to/clip.mp4 \\
        --location "Cafeteria"

Optional flags:
    --audio_threshold 0.64
    --video_threshold 0.60
    --video_window    15        seconds of video to analyze per audio trigger
    --ably_key        KEY       (or set ABLY_API_KEY env var)
    --channel         gunshot-detection
    --log_file        inference/cascade_detections.jsonl
    --show                      open OpenCV window during the video stage
    --audio_model     models/saved_weights/dense_head_best.keras
    --video_model     YOLO_hugging-main/best.pt
"""

import argparse
import logging
import os
import sys
import threading
import time
from pathlib import Path

import tensorflow as tf

from pipeline.extract_embeddings import load_yamnet
from inference.live_inference import (
    AudioCapture,
    AblyPublisher,
    DEFAULT_CHANNEL,
    DEFAULT_LOCATION,
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
AUDIO_DEFAULT_THRESH = 0.64
VIDEO_DEFAULT_MODEL  = Path("YOLO_hugging-main/best.pt")
VIDEO_DEFAULT_THRESH = 0.60
VIDEO_DEFAULT_IOU    = 0.45
VIDEO_DEFAULT_IMGSZ  = 1280
VIDEO_DEFAULT_WINDOW = 15.0   # seconds of video per audio trigger


class _GatedVideoStage:
    """
    Wraps VideoCapture in a background thread.
    Stays idle until trigger() is called, then runs the video source
    for window_secs seconds before returning to standby.
    A re-trigger while the window is active is ignored — the current
    window completes first.
    """

    def __init__(
        self,
        model_path:  Path,
        source:      "int | str",
        threshold:   float,
        iou:         float,
        imgsz:       int,
        location:    str,
        log_file:    Path,
        publisher:   "AblyPublisher | None",
        s3_bucket:   "str | None",
        aws_region:  str,
        show:        bool,
        window_secs: float,
    ):
        self._vc_kwargs  = dict(
            model_path=model_path,
            threshold=threshold,
            iou=iou,
            imgsz=imgsz,
            location=location,
            log_file=log_file,
            publisher=publisher,
            s3_bucket=s3_bucket,
            aws_region=aws_region,
            source=source,
            show=show,
        )
        self._window_secs = window_secs
        self._trigger     = threading.Event()
        self._shutdown    = threading.Event()
        self._active      = False
        self._thread      = threading.Thread(
            target=self._loop, daemon=True, name="video-stage"
        )
        self._thread.start()

    def trigger(self, audio_prob: float) -> None:
        """Called by the audio detection callback to wake the video stage."""
        if self._active:
            logger.debug("Stage-2 already active — ignoring re-trigger")
            return
        logger.info(
            "Stage-1 fired (prob=%.3f) — activating Stage-2 for %.0fs",
            audio_prob, self._window_secs,
        )
        self._trigger.set()

    def _loop(self) -> None:
        while not self._shutdown.is_set():
            if not self._trigger.wait(timeout=1.0):
                continue
            self._trigger.clear()
            self._active = True

            print(
                f"\n  ▶  [STAGE-2 VIDEO]  window={self._window_secs:.0f}s  "
                f"source={self._vc_kwargs['source']}  "
                f"threshold={self._vc_kwargs['threshold']:.2f}\n"
            )

            cap = VideoCapture(**self._vc_kwargs)
            # Schedule a clean stop after window_secs by signalling the loop event
            stop_timer = threading.Timer(self._window_secs, cap.request_stop)
            stop_timer.start()
            try:
                cap.start()
            finally:
                stop_timer.cancel()

            self._active = False
            print("\n  ■  [STAGE-2 VIDEO]  window closed — Stage-1 audio monitoring ...\n")

    def stop(self) -> None:
        self._shutdown.set()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Gated two-stage gunshot + visual gun detection pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Stage 1 — audio
    parser.add_argument("--audio_model",     type=Path,  default=AUDIO_DEFAULT_MODEL,
                        help="Path to Dense head .keras weights.")
    parser.add_argument("--audio_threshold", type=float, default=AUDIO_DEFAULT_THRESH,
                        help="Gunshot probability threshold for Stage 1. Default: %(default)s")
    parser.add_argument("--audio_file",      type=Path,  default=None,
                        help="Audio file (WAV/MP3) for Stage 1. Omit to use live mic.")
    parser.add_argument("--device",          type=int,   default=None,
                        help="sounddevice input device index (mic mode only).")

    # Stage 2 — video
    parser.add_argument("--video_model",     type=Path,  default=VIDEO_DEFAULT_MODEL,
                        help="Path to YOLO .pt weights.")
    parser.add_argument("--video_threshold", type=float, default=VIDEO_DEFAULT_THRESH,
                        help="YOLO confidence threshold for Stage 2. Default: %(default)s")
    parser.add_argument("--video_source",    default=0,
                        help="Stage-2 video source: 0 = webcam, or path to video file.")
    parser.add_argument("--video_window",    type=float, default=VIDEO_DEFAULT_WINDOW,
                        help="Seconds of video to analyze per audio trigger. Default: %(default)s")
    parser.add_argument("--iou",             type=float, default=VIDEO_DEFAULT_IOU,
                        help="NMS IoU threshold for YOLO. Default: %(default)s")
    parser.add_argument("--imgsz",           type=int,   default=VIDEO_DEFAULT_IMGSZ,
                        help="YOLO inference image size. Default: %(default)s")
    parser.add_argument("--show",            action="store_true",
                        help="Open an OpenCV window during the video stage.")

    # Shared
    parser.add_argument("--location",   type=str,  default=DEFAULT_LOCATION,
                        help="Location label attached to every Ably message.")
    parser.add_argument("--log_file",   type=Path, default=AUDIO_DEFAULT_LOG)
    parser.add_argument("--channel",    type=str,  default=DEFAULT_CHANNEL)
    parser.add_argument("--ably_key",   type=str,  default=None,
                        help="Ably API key. Defaults to ABLY_API_KEY env var.")
    parser.add_argument("--s3_bucket",  type=str,  default=None)
    parser.add_argument("--aws_region", type=str,  default="us-east-1")
    args = parser.parse_args()

    # Validate paths
    for label, path in [
        ("Audio model", args.audio_model),
        ("Video model", args.video_model),
    ]:
        if not path.exists():
            logger.error("%s not found: %s", label, path)
            sys.exit(1)
    if args.audio_file and not args.audio_file.exists():
        logger.error("Audio file not found: %s", args.audio_file)
        sys.exit(1)

    # Coerce --video_source to int when it looks like a device index
    video_source = args.video_source
    if isinstance(video_source, str) and video_source.isdigit():
        video_source = int(video_source)

    # Shared Ably publisher (both stages publish to the same channel)
    ably_key  = args.ably_key or os.environ.get("ABLY_API_KEY")
    publisher = None
    if ably_key:
        try:
            publisher = AblyPublisher(ably_key, args.channel)
        except Exception as exc:
            logger.warning("Ably connection failed: %s — running without WS alerts", exc)
    else:
        logger.warning("No Ably key — WS alerts disabled (set --ably_key or ABLY_API_KEY)")

    # Stage 2: start idle, waiting for Stage 1 to fire
    video_stage = _GatedVideoStage(
        model_path=args.video_model,
        source=video_source,
        threshold=args.video_threshold,
        iou=args.iou,
        imgsz=args.imgsz,
        location=args.location,
        log_file=args.log_file,
        publisher=publisher,
        s3_bucket=args.s3_bucket,
        aws_region=args.aws_region,
        show=args.show,
        window_secs=args.video_window,
    )

    # Stage 1: load models and wire the detection callback to stage 2
    logger.info("Loading YAMNet ...")
    yamnet = load_yamnet()
    logger.info("Loading audio head from %s ...", args.audio_model)
    head = tf.keras.models.load_model(str(args.audio_model))

    audio_capture = AudioCapture(
        yamnet_model=yamnet,
        head_model=head,
        threshold=args.audio_threshold,
        location=args.location,
        log_file=args.log_file,
        publisher=publisher,
        s3_bucket=args.s3_bucket,
        aws_region=args.aws_region,
        device=args.device,
        on_detection=lambda prob, loc, ts: video_stage.trigger(prob),
    )

    audio_src = f"file: {args.audio_file}" if args.audio_file else "live mic"
    print(
        f"\n  TacticalEye — Cascade Pipeline\n"
        f"  ══════════════════════════════════════════════════════\n"
        f"  Stage 1 (Audio):  threshold={args.audio_threshold:.2f}  {audio_src}\n"
        f"  Stage 2 (Video):  threshold={args.video_threshold:.2f}  "
        f"window={args.video_window:.0f}s  source={video_source}\n"
        f"  Location: {args.location}\n"
        f"  ══════════════════════════════════════════════════════\n"
        f"  Stage 2 is IDLE — waiting for Stage 1 to fire ...\n"
    )

    try:
        if args.audio_file:
            audio_capture.run_demo_file_inprocess(args.audio_file)
        else:
            audio_capture.start()
            print("  Press Enter to stop ...\n")
            input()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        audio_capture.stop()
        video_stage.stop()
        if publisher:
            publisher.close()
        logger.info("Detections saved to: %s", args.log_file)


if __name__ == "__main__":
    main()
