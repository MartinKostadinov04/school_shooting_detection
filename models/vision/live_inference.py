"""
live_inference.py  (vision)
===========================
Real-time gun detection from a camera or video file using a fine-tuned YOLOv11s model.

Pipeline per frame:
  camera / video file (BGR frame via OpenCV)
    → YOLO model → confidence scores
    → if max_conf >= threshold:
        • console log
        • JSONL log file
        • Ably WS  →  "video:detected:{location}"
        • (optional) annotated frame  →  S3  →  "video:segment:{location}:{presigned_url}"

Ably credentials: set ABLY_API_KEY env var (or pass --ably_key).
AWS credentials:  set AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / AWS_DEFAULT_REGION
                  (standard boto3 env vars — never hardcode).

Usage
-----
  Webcam (default):
    python -m vision.live_inference --location "Main Entrance"

  Video file:
    python -m vision.live_inference --source path/to/video.mp4 --location "Gymnasium"

  Optional flags:
    --threshold 0.6  --iou 0.45  --imgsz 1280
    --log_file vision/detections.jsonl
    --ably_key KEY   --channel gunshot-detection
    --s3_bucket my-bucket  --aws_region eu-west-1
"""

import argparse
import asyncio
import io
import json
import logging
import os
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_MODEL_PATH = Path("YOLO_hugging-main/best.pt")
DEFAULT_LOG_FILE   = Path("vision/detections.jsonl")
DEFAULT_THRESHOLD  = 0.6
DEFAULT_IOU        = 0.45
DEFAULT_IMG_SIZE   = 1280
DEFAULT_CHANNEL    = "gunshot-detection"
DEFAULT_LOCATION   = "unknown"
S3_PRESIGN_EXPIRY  = 3600    # seconds


# ---------------------------------------------------------------------------
# Async Ably publisher
# ---------------------------------------------------------------------------
# cv2 capture loops run in the main thread — AblyPublisher bridges to its own
# asyncio loop via run_coroutine_threadsafe(), identical pattern to audio component.

class AblyPublisher:
    """
    Manages a persistent Ably Realtime connection on a background asyncio loop.
    Thread-safe: call publish() from any thread.
    """

    def __init__(self, api_key: str, channel_name: str):
        self._api_key      = api_key
        self._channel_name = channel_name
        self._loop         = asyncio.new_event_loop()
        self._channel      = None
        self._client       = None
        self._ready        = threading.Event()

        t = threading.Thread(target=self._run_loop, daemon=True)
        t.start()
        connected = self._ready.wait(timeout=15)
        if not connected:
            raise RuntimeError("Ably: timed out waiting for connection")

    def _run_loop(self):
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._connect())
        self._loop.run_forever()

    async def _connect(self):
        from ably import AblyRealtime
        self._client  = AblyRealtime(self._api_key)
        await self._client.connection.once_async("connected")
        self._channel = self._client.channels.get(self._channel_name)
        logger.info("Ably connected  →  channel='%s'", self._channel_name)
        self._ready.set()

    def publish(self, name: str, data: str):
        asyncio.run_coroutine_threadsafe(
            self._channel.publish(name, data),
            self._loop,
        )

    def close(self):
        if self._client:
            asyncio.run_coroutine_threadsafe(
                self._client.close(),
                self._loop,
            )


# ---------------------------------------------------------------------------
# S3 frame upload
# ---------------------------------------------------------------------------

def _frame_to_jpeg_bytes(frame: np.ndarray) -> bytes:
    """Encode a BGR OpenCV frame to JPEG bytes."""
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    return buf.tobytes()


def _s3_upload_frame(frame: np.ndarray, location: str, timestamp: str,
                     bucket: str, region: str) -> str:
    import boto3
    jpeg_bytes = _frame_to_jpeg_bytes(frame)
    safe_loc   = location.replace("/", "_")
    key        = f"video-snapshots/{safe_loc}/{timestamp}.jpg"

    s3 = boto3.client("s3", region_name=region)
    s3.upload_fileobj(io.BytesIO(jpeg_bytes), bucket, key,
                      ExtraArgs={"ContentType": "image/jpeg"})

    url = s3.generate_presigned_url(
        "get_object",
        Params={"Bucket": bucket, "Key": key},
        ExpiresIn=S3_PRESIGN_EXPIRY,
    )
    logger.info("Frame uploaded  →  s3://%s/%s", bucket, key)
    return url


# ---------------------------------------------------------------------------
# Alert
# ---------------------------------------------------------------------------

def _alert(
    conf:       float,
    frame:      np.ndarray,
    timestamp:  str,
    threshold:  float,
    location:   str,
    log_file:   Path,
    publisher:  "AblyPublisher | None",
    s3_bucket:  "str | None",
    aws_region: str,
) -> None:
    record = {
        "event":       "gun_detected",
        "timestamp":   timestamp,
        "confidence":  round(float(conf), 4),
        "threshold":   threshold,
        "location":    location,
    }

    # Console
    print(f"\n🔴  GUN DETECTED  conf={conf:.3f}  loc={location}  [{timestamp}]")

    # JSONL log
    try:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except Exception as exc:
        logger.warning("Log write failed: %s", exc)

    if publisher is None:
        return

    # Ably: detection message
    publisher.publish("video:detected", f"video:detected:{location}")
    logger.info("Ably  →  video:detected:%s", location)

    # Ably: frame snapshot (upload async so we don't block the capture loop)
    if s3_bucket:
        def _upload_and_publish():
            try:
                url = _s3_upload_frame(frame.copy(), location, timestamp, s3_bucket, aws_region)
                publisher.publish("video:segment", f"video:segment:{location}:{url}")
                logger.info("Ably  →  video:segment:%s:<url>", location)
            except Exception as exc:
                logger.warning("S3 upload failed: %s", exc)

        threading.Thread(target=_upload_and_publish, daemon=True).start()


# ---------------------------------------------------------------------------
# Video capture + inference
# ---------------------------------------------------------------------------

class VideoCapture:
    def __init__(
        self,
        model_path:  Path,
        threshold:   float,
        iou:         float,
        imgsz:       int,
        location:    str,
        log_file:    Path,
        publisher:   "AblyPublisher | None",
        s3_bucket:   "str | None",
        aws_region:  str,
        source:      "int | str",
    ):
        from ultralytics import YOLO
        logger.info("Loading YOLO model from %s ...", model_path)
        self._model     = YOLO(str(model_path))
        self._threshold = threshold
        self._iou       = iou
        self._imgsz     = imgsz
        self._location  = location
        self._log_file  = log_file
        self._publisher = publisher
        self._s3_bucket = s3_bucket
        self._aws_region = aws_region
        self._source    = source
        self._cap       = None

    def _run_inference(self, frame: np.ndarray) -> tuple[float, np.ndarray]:
        """Run YOLO on a single frame. Returns (max_confidence, annotated_frame)."""
        results = self._model.predict(
            source=frame,
            conf=self._threshold,
            iou=self._iou,
            imgsz=self._imgsz,
            verbose=False,
        )
        result = results[0]
        annotated = result.plot()  # BGR frame with bounding boxes drawn
        boxes = result.boxes
        max_conf = float(boxes.conf.max()) if len(boxes) > 0 else 0.0
        return max_conf, annotated

    def _process_frame(self, frame: np.ndarray) -> float:
        """Run inference and fire alert if a gun is detected. Returns max confidence."""
        max_conf, annotated = self._run_inference(frame)
        logger.debug("conf=%.4f", max_conf)

        if max_conf >= self._threshold:
            ts = datetime.now(timezone.utc).isoformat()
            _alert(
                conf=max_conf,
                frame=annotated,
                timestamp=ts,
                threshold=self._threshold,
                location=self._location,
                log_file=self._log_file,
                publisher=self._publisher,
                s3_bucket=self._s3_bucket,
                aws_region=self._aws_region,
            )
        return max_conf

    def start(self) -> None:
        """Open the video source and process frames until stopped or source ends."""
        self._cap = cv2.VideoCapture(self._source)
        if not self._cap.isOpened():
            logger.error("Cannot open video source: %s", self._source)
            sys.exit(1)

        logger.info(
            "Capturing  source=%s  threshold=%.2f  iou=%.2f  imgsz=%d  location=%s",
            self._source, self._threshold, self._iou, self._imgsz, self._location,
        )
        print("\nPress Ctrl+C to stop ...\n")

        try:
            while True:
                ok, frame = self._cap.read()
                if not ok:
                    logger.info("Video source ended.")
                    break
                conf = self._process_frame(frame)
                print(f"  conf={conf:.4f}", end="\r")
        except KeyboardInterrupt:
            print("\nStopped.")
        finally:
            self.stop()

    def stop(self) -> None:
        if self._cap:
            self._cap.release()

    def run_demo_file(self, file_path: Path) -> None:
        """Process a video file frame-by-frame (no webcam required)."""
        self._source = str(file_path)
        self.start()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Real-time gun detection from camera with Ably WS alerts.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model_path", type=Path,  default=DEFAULT_MODEL_PATH)
    parser.add_argument("--threshold",  type=float, default=DEFAULT_THRESHOLD,
                        help="YOLO confidence threshold (default: %(default)s)")
    parser.add_argument("--iou",        type=float, default=DEFAULT_IOU,
                        help="NMS IoU threshold (default: %(default)s)")
    parser.add_argument("--imgsz",      type=int,   default=DEFAULT_IMG_SIZE,
                        help="Inference image size (default: %(default)s)")
    parser.add_argument("--source",     default=0,
                        help="Video source: 0 = webcam, or path to video file (default: 0)")
    parser.add_argument("--log_file",   type=Path,  default=DEFAULT_LOG_FILE)
    parser.add_argument("--location",   type=str,   default=DEFAULT_LOCATION,
                        help="Location name/code sent in every message")
    parser.add_argument("--channel",    type=str,   default=DEFAULT_CHANNEL,
                        help=f"Ably channel name (default: {DEFAULT_CHANNEL})")
    parser.add_argument("--ably_key",   type=str,   default=None,
                        help="Ably API key. Defaults to ABLY_API_KEY env var.")
    parser.add_argument("--s3_bucket",  type=str,   default=None,
                        help="S3 bucket for annotated frame upload. Omit to skip.")
    parser.add_argument("--aws_region", type=str,   default="us-east-1",
                        help="AWS region for S3 (default: us-east-1)")
    args = parser.parse_args()

    # Coerce --source to int when it looks like a device index
    source = args.source
    if isinstance(source, str) and source.isdigit():
        source = int(source)

    if not args.model_path.exists():
        logger.error("Model weights not found: %s", args.model_path)
        sys.exit(1)

    ably_key  = args.ably_key or os.environ.get("ABLY_API_KEY")
    publisher = None
    if ably_key:
        try:
            publisher = AblyPublisher(ably_key, args.channel)
        except Exception as exc:
            logger.warning("Ably connection failed: %s — running without WS alerts", exc)
    else:
        logger.warning("No Ably key provided (--ably_key / ABLY_API_KEY) — WS alerts disabled")

    capture = VideoCapture(
        model_path=args.model_path,
        threshold=args.threshold,
        iou=args.iou,
        imgsz=args.imgsz,
        location=args.location,
        log_file=args.log_file,
        publisher=publisher,
        s3_bucket=args.s3_bucket,
        aws_region=args.aws_region,
        source=source,
    )

    try:
        capture.start()
    finally:
        if publisher:
            publisher.close()
        logger.info("Detections saved to: %s", args.log_file)


if __name__ == "__main__":
    main()
