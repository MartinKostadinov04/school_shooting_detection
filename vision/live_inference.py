"""
live_inference.py  (vision)
===========================
Real-time gun detection from a camera or video file using a fine-tuned YOLOv11s model.

Three-layer FP-reduction stack (no retraining required):
  1. Temporal k-of-n gate  — require K positive frames in a rolling window of N
                             before firing an alert. Eliminates single-frame FPs
                             (Olmos MULTICAST: 80% FP reduction).
  2. SAHI tiled inference  — slice each frame into overlapping tiles, detect on
                             each tile, merge results. 10× mAP improvement on
                             small/distant guns in CCTV footage (Hnoohom 2022).
                             Falls back to standard inference if sahi not installed.
  3. Pose-overlap check    — require every gun bbox to overlap a detected hand
                             region. Stationary desk/object FPs have no associated
                             hand and are suppressed (Lamas WeDePE 2022: +4-18 AP).
                             Falls back to no constraint if mediapipe not installed.

Pipeline per frame:
  camera / video file (BGR frame via OpenCV)
    -> SAHI tiled YOLO -> confidence scores
    -> pose-overlap filter
    -> temporal k-of-n gate
    -> if gate trips and cooldown elapsed:
        * console log
        * JSONL log file
        * Ably WS  ->  "video:detected:{location}"
        * (optional) annotated frame  ->  S3  ->  "video:segment:{location}:{presigned_url}"

Ably credentials: set ABLY_API_KEY env var (or pass --ably_key).
AWS credentials:  set AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / AWS_DEFAULT_REGION
                  (standard boto3 env vars -- never hardcode).

Usage
-----
  Webcam (default):
    python -m vision.live_inference --location "Main Entrance"

  Video file:
    python -m vision.live_inference --source path/to/video.mp4 --location "Gymnasium"

  Optional flags:
    --threshold 0.35  --iou 0.45  --imgsz 1280
    --no_sahi        disable tiled inference (faster, less accurate on small guns)
    --no_pose        disable pose-overlap constraint
    --kofn_k 3       detections required in rolling window (default: 3)
    --kofn_n 4       rolling window size in frames (default: 4)
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
import re
import sys
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

# Optional dependencies — degrade gracefully if absent
try:
    from sahi import AutoDetectionModel
    from sahi.predict import get_sliced_prediction
    _SAHI_AVAILABLE = True
except ImportError:
    _SAHI_AVAILABLE = False

try:
    import mediapipe as mp
    _MEDIAPIPE_AVAILABLE = True
except ImportError:
    _MEDIAPIPE_AVAILABLE = False

from inference.config import YOLO_WEIGHTS_PATH

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_MODEL_PATH  = YOLO_WEIGHTS_PATH
DEFAULT_LOG_FILE    = Path("vision/detections.jsonl")
DEFAULT_THRESHOLD   = 0.35  # threshold sweep on test set: best F1=0.915 at conf=0.35
DEFAULT_IOU         = 0.45
DEFAULT_IMG_SIZE    = 1280
DEFAULT_CHANNEL     = "gunshot-detection"
DEFAULT_LOCATION    = "Cafeteria"
S3_PRESIGN_EXPIRY   = 3600   # seconds
ALERT_COOLDOWN_SECS = 5.0   # minimum seconds between consecutive alerts
S3_UPLOAD_WORKERS   = 4     # max concurrent S3 upload threads

# Temporal k-of-n defaults (Olmos MULTICAST: 80% FP reduction)
DEFAULT_KOFN_K      = 3     # positive frames required
DEFAULT_KOFN_N      = 4     # rolling window size

# SAHI tiling defaults (Hnoohom 2022: 10× mAP on small CCTV guns)
SAHI_SLICE_H        = 512
SAHI_SLICE_W        = 512
SAHI_OVERLAP        = 0.2

# Pose: hand bbox expansion factor relative to wrist-to-MCP distance
POSE_HAND_EXPAND    = 2.0


# ---------------------------------------------------------------------------
# Async Ably publisher
# ---------------------------------------------------------------------------
# cv2 capture loops run in the main thread -- AblyPublisher bridges to its own
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
        logger.info("Ably connected  ->  channel='%s'", self._channel_name)
        self._ready.set()

    def publish(self, name: str, data: str):
        asyncio.run_coroutine_threadsafe(
            self._channel.publish(name, data),
            self._loop,
        )

    def close(self):
        if self._client:
            future = asyncio.run_coroutine_threadsafe(
                self._client.close(),
                self._loop,
            )
            try:
                future.result(timeout=3)
            except Exception as exc:
                logger.warning("Ably close error: %s", exc)


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
    safe_loc   = re.sub(r"[^\w-]", "_", location)
    key        = f"video-snapshots/{safe_loc}/{timestamp}.jpg"

    s3 = boto3.client("s3", region_name=region)
    s3.upload_fileobj(io.BytesIO(jpeg_bytes), bucket, key,
                      ExtraArgs={"ContentType": "image/jpeg"})

    url = s3.generate_presigned_url(
        "get_object",
        Params={"Bucket": bucket, "Key": key},
        ExpiresIn=S3_PRESIGN_EXPIRY,
    )
    logger.info("Frame uploaded  ->  s3://%s/%s", bucket, key)
    return url


# ---------------------------------------------------------------------------
# Alert
# ---------------------------------------------------------------------------

def _alert(
    conf:       float,
    count:      int,
    boxes:      list,
    frame:      np.ndarray,
    timestamp:  str,
    threshold:  float,
    location:   str,
    log_file:   Path,
    publisher:  "AblyPublisher | None",
    s3_bucket:  "str | None",
    aws_region: str,
    executor:   ThreadPoolExecutor,
) -> None:
    """
    Fire a single gun-detection alert. ``count`` is the number of boxes that
    survived the pose+threshold filter on the alerting frame; ``boxes`` is the
    list of ``(x1, y1, x2, y2, conf)`` tuples for those survivors.
    """
    record = {
        "event":       "gun_detected",
        "timestamp":   timestamp,
        "confidence":  round(float(conf), 4),
        "count":       int(count),
        "boxes": [
            {
                "x1":   round(float(b[0]), 2),
                "y1":   round(float(b[1]), 2),
                "x2":   round(float(b[2]), 2),
                "y2":   round(float(b[3]), 2),
                "conf": round(float(b[4]), 4),
            }
            for b in boxes
        ],
        "threshold":   threshold,
        "location":    location,
    }

    # Console
    print(
        f"\n\U0001f534  GUN DETECTED  conf={conf:.3f}  count={count}  "
        f"loc={location}  [{timestamp}]"
    )

    # JSONL log
    try:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except Exception as exc:
        logger.warning("Log write failed: %s", exc)

    if publisher is None:
        return

    # Ably: detection message — appended count keeps the format backwards
    # compatible (existing parsers strip a single numeric tail).
    # Format: video:detected:{location}:{conf}:{count}
    publisher.publish(
        "video:detected",
        f"video:detected:{location}:{conf:.4f}:{count}",
    )
    logger.info("Ably  ->  video:detected:%s  conf=%.4f  count=%d", location, conf, count)

    # Ably: frame snapshot -- submitted to bounded thread pool so the capture
    # loop is never blocked and thread count stays capped under rapid detection.
    if s3_bucket:
        snapshot = frame.copy()

        def _upload_and_publish():
            try:
                url = _s3_upload_frame(snapshot, location, timestamp, s3_bucket, aws_region)
                publisher.publish("video:segment", f"video:segment:{location}:{url}")
                logger.info("Ably  ->  video:segment:%s:<url>", location)
            except Exception as exc:
                logger.warning("S3 upload failed: %s", exc)

        executor.submit(_upload_and_publish)


# ---------------------------------------------------------------------------
# Pose helper — hand-region extraction via MediaPipe
# ---------------------------------------------------------------------------

def _build_hand_detector():
    """Return a MediaPipe Hands instance, or None if unavailable.

    mediapipe >= 0.10.10 removed the legacy ``mp.solutions.hands`` API in
    favour of the new ``mp.tasks.vision.HandLandmarker`` interface. Rather
    than rewrite the call site for both APIs, we just fall back to None on
    AttributeError — the rest of the pipeline already degrades gracefully
    when the hand detector is missing (pose-overlap filter is bypassed,
    SAHI + temporal-k-of-n still gate detections).
    """
    if not _MEDIAPIPE_AVAILABLE:
        return None
    try:
        return mp.solutions.hands.Hands(
            static_image_mode=True,
            max_num_hands=4,
            min_detection_confidence=0.4,
        )
    except AttributeError:
        logger.warning(
            "mediapipe %s does not expose mp.solutions.hands — pose-overlap "
            "filter disabled. Install mediapipe<0.10.10 (or pass --no_pose) "
            "to silence this warning.",
            getattr(mp, "__version__", "unknown"),
        )
        return None


def _hand_boxes(frame_rgb: np.ndarray, hands_detector) -> List[Tuple[int, int, int, int]]:
    """
    Return a list of (x1,y1,x2,y2) hand bounding boxes in pixel coords.
    Each box is derived from the 21 MediaPipe hand landmarks, expanded by
    POSE_HAND_EXPAND to account for partially visible hands.
    """
    if hands_detector is None:
        return []
    result = hands_detector.process(frame_rgb)
    if not result.multi_hand_landmarks:
        return []

    h, w = frame_rgb.shape[:2]
    boxes = []
    for hand_lms in result.multi_hand_landmarks:
        xs = [lm.x * w for lm in hand_lms.landmark]
        ys = [lm.y * h for lm in hand_lms.landmark]
        cx, cy = sum(xs) / len(xs), sum(ys) / len(ys)
        half = max(max(xs) - min(xs), max(ys) - min(ys)) * POSE_HAND_EXPAND / 2
        boxes.append((
            int(max(0, cx - half)), int(max(0, cy - half)),
            int(min(w, cx + half)), int(min(h, cy + half)),
        ))
    return boxes


def _gun_box_has_hand(gun_box: Tuple[float, float, float, float, float],
                      hand_boxes: List[Tuple[int, int, int, int]]) -> bool:
    """True if the gun bounding box overlaps any hand region.

    ``gun_box`` is ``(x1, y1, x2, y2, conf)`` — only the spatial coords are
    used here, but accepting the full tuple keeps callers from re-packing.
    """
    gx1, gy1, gx2, gy2 = gun_box[0], gun_box[1], gun_box[2], gun_box[3]
    for hx1, hy1, hx2, hy2 in hand_boxes:
        if gx1 < hx2 and gx2 > hx1 and gy1 < hy2 and gy2 > hy1:
            return True
    return False


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
        show:        bool = False,
        use_sahi:    bool = True,
        use_pose:    bool = True,
        kofn_k:      int  = DEFAULT_KOFN_K,
        kofn_n:      int  = DEFAULT_KOFN_N,
    ):
        from ultralytics import YOLO
        logger.info("Loading YOLO model from %s ...", model_path)
        self._yolo            = YOLO(str(model_path))
        self._threshold       = threshold
        self._iou             = iou
        self._imgsz           = imgsz
        self._location        = location
        self._log_file        = log_file
        self._publisher       = publisher
        self._s3_bucket       = s3_bucket
        self._aws_region      = aws_region
        self._source          = source
        self._show            = show
        self._cap             = None
        self._last_alert_time = 0.0
        self._s3_executor     = ThreadPoolExecutor(max_workers=S3_UPLOAD_WORKERS)
        self._stop_event      = threading.Event()
        self._run_detected    = False
        self._run_max_conf    = 0.0
        self._run_max_count   = 0   # peak number of simultaneously visible guns

        # Annotated-MP4 writer state — populated lazily by start() when the
        # source is a file path. None for webcam captures.
        self._annotated_writer: "cv2.VideoWriter | None" = None
        self._annotated_path:   "Path | None"            = None

        # --- Layer 1: temporal k-of-n gate ---
        self._kofn_k          = kofn_k
        self._kofn_n          = kofn_n
        self._detection_window: deque = deque(maxlen=kofn_n)

        # --- Layer 2: SAHI tiled inference ---
        self._use_sahi = use_sahi and _SAHI_AVAILABLE
        if use_sahi and not _SAHI_AVAILABLE:
            logger.warning("sahi not installed — falling back to standard inference. "
                           "Install with: pip install sahi")
        if self._use_sahi:
            self._sahi_model = AutoDetectionModel.from_pretrained(
                "ultralytics",
                model_path=str(model_path),
                confidence_threshold=threshold,
                device="cpu",
            )
            logger.info("SAHI tiled inference enabled  (slice=%dx%d overlap=%.0f%%)",
                        SAHI_SLICE_H, SAHI_SLICE_W, SAHI_OVERLAP * 100)

        # --- Layer 3: pose-overlap constraint ---
        self._use_pose    = use_pose and _MEDIAPIPE_AVAILABLE
        self._hand_detect = _build_hand_detector() if self._use_pose else None
        if use_pose and not _MEDIAPIPE_AVAILABLE:
            logger.warning("mediapipe not installed — pose-overlap constraint disabled. "
                           "Install with: pip install mediapipe")
        # _build_hand_detector returns None when the installed mediapipe
        # has dropped the legacy ``mp.solutions.hands`` API. In that case
        # the constraint can't run, so reflect that in _use_pose.
        if self._use_pose and self._hand_detect is None:
            self._use_pose = False
        if self._use_pose:
            logger.info("Pose-overlap constraint enabled")

        active = []
        if self._use_sahi:  active.append("SAHI-tiling")
        if self._use_pose:  active.append("pose-overlap")
        active.append(f"temporal-{kofn_k}of{kofn_n}")
        logger.info("FP-reduction stack: %s", " + ".join(active))

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------

    def _run_inference_standard(self, frame: np.ndarray) -> Tuple[float, np.ndarray, list]:
        """
        Standard single-shot YOLO inference.
        Returns (max_conf, annotated, raw_boxes) where each raw_boxes entry is
        ``(x1, y1, x2, y2, conf)`` — confidence is preserved per-box so the
        downstream count/filter logic can keep the right max after dropping
        boxes via the pose filter.
        """
        results   = self._yolo.predict(
            source=frame, conf=self._threshold, iou=self._iou,
            imgsz=self._imgsz, verbose=False,
        )
        result    = results[0]
        annotated = result.plot()                         # ultralytics draws box+label+conf
        boxes     = result.boxes
        if len(boxes) > 0:
            xyxy_arr = boxes.xyxy.cpu().numpy()           # (N, 4)
            conf_arr = boxes.conf.cpu().numpy()           # (N,)
            raw_boxes = [
                (float(x1), float(y1), float(x2), float(y2), float(c))
                for (x1, y1, x2, y2), c in zip(xyxy_arr, conf_arr)
            ]
            max_conf = float(conf_arr.max())
        else:
            raw_boxes = []
            max_conf  = 0.0
        return max_conf, annotated, raw_boxes

    def _run_inference_sahi(self, frame: np.ndarray) -> Tuple[float, np.ndarray, list]:
        """
        SAHI tiled inference — better recall on small/distant guns.
        Returns ``(max_conf, annotated, raw_boxes)`` with boxes as
        ``(x1, y1, x2, y2, conf)``. SAHI does not provide a built-in plot()
        equivalent so we draw the boxes + per-box confidence labels here.
        """
        import PIL.Image
        pil_img = PIL.Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        result  = get_sliced_prediction(
            pil_img,
            self._sahi_model,
            slice_height=SAHI_SLICE_H,
            slice_width=SAHI_SLICE_W,
            overlap_height_ratio=SAHI_OVERLAP,
            overlap_width_ratio=SAHI_OVERLAP,
            verbose=0,
        )
        raw_boxes: list[tuple[float, float, float, float, float]] = []
        max_conf  = 0.0
        for pred in result.object_prediction_list:
            score = float(pred.score.value)
            if score >= self._threshold:
                bb = pred.bbox
                raw_boxes.append(
                    (float(bb.minx), float(bb.miny), float(bb.maxx), float(bb.maxy), score)
                )
                if score > max_conf:
                    max_conf = score

        # Draw bounding boxes + conf labels on frame for display / annotated MP4 / S3 upload
        annotated = frame.copy()
        for x1, y1, x2, y2, c in raw_boxes:
            p1 = (int(x1), int(y1))
            p2 = (int(x2), int(y2))
            cv2.rectangle(annotated, p1, p2, (0, 0, 255), 2)
            cv2.putText(
                annotated, f"gun {c:.2f}", (p1[0], max(p1[1] - 6, 12)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA,
            )

        return max_conf, annotated, raw_boxes

    # ------------------------------------------------------------------
    # Per-frame pipeline
    # ------------------------------------------------------------------

    def _process_frame(self, frame: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Full three-layer FP-reduction pipeline.
        Returns ``(reported_conf, annotated_frame)``. ``reported_conf`` is the
        post-pose-filter maximum confidence of any surviving box; it is > 0
        only when at least one box clears every layer.
        """
        # Layer 2: SAHI vs standard inference. raw_boxes is a list of
        # (x1, y1, x2, y2, conf) tuples — keeping conf alongside the box lets
        # the pose filter recompute the max correctly after dropping boxes.
        if self._use_sahi:
            max_conf, annotated, raw_boxes = self._run_inference_sahi(frame)
        else:
            max_conf, annotated, raw_boxes = self._run_inference_standard(frame)

        # Layer 3: pose-overlap filter — discard boxes with no associated hand.
        # ``filtered`` is the authoritative survivor list; everything below
        # (count, max_conf, alert payload) is computed from it.
        filtered: list = list(raw_boxes)
        if self._use_pose and raw_boxes and self._hand_detect is not None:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h_boxes   = _hand_boxes(frame_rgb, self._hand_detect)
            if h_boxes:
                filtered = [b for b in raw_boxes if _gun_box_has_hand(b, h_boxes)]
                if not filtered:
                    logger.debug(
                        "pose-overlap: all %d boxes suppressed (no hand overlap)",
                        len(raw_boxes),
                    )
            else:
                # No hands detected in frame at all — suppress everything.
                logger.debug(
                    "pose-overlap: no hands in frame, suppressing %d boxes",
                    len(raw_boxes),
                )
                filtered = []

        # Recompute count + max_conf from the survivor set.
        count    = len(filtered)
        max_conf = max((b[4] for b in filtered), default=0.0)

        # Layer 1: temporal k-of-n gate
        self._detection_window.append(max_conf >= self._threshold)
        gate_open = sum(self._detection_window) >= self._kofn_k

        # Track run-level stats regardless of gate. ``max_count`` is the peak
        # number of simultaneously visible guns across the whole run — that's
        # what we report to the police/school view.
        self._run_max_conf  = max(self._run_max_conf, max_conf)
        self._run_max_count = max(self._run_max_count, count)
        if gate_open and max_conf >= self._threshold:
            self._run_detected = True

        if gate_open and max_conf >= self._threshold:
            now = time.monotonic()
            if now - self._last_alert_time >= ALERT_COOLDOWN_SECS:
                self._last_alert_time = now
                ts = datetime.now(timezone.utc).isoformat()
                _alert(
                    conf=max_conf,
                    count=count,
                    boxes=filtered,
                    frame=annotated,
                    timestamp=ts,
                    threshold=self._threshold,
                    location=self._location,
                    log_file=self._log_file,
                    publisher=self._publisher,
                    s3_bucket=self._s3_bucket,
                    aws_region=self._aws_region,
                    executor=self._s3_executor,
                )

        return (max_conf if gate_open else 0.0), annotated

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def result(self) -> Tuple[bool, float, int]:
        """``(gun_detected, max_confidence, max_count)`` — valid after start() returns.

        ``max_count`` is the peak number of simultaneously visible guns
        across the whole run, not the count at any single alert frame.
        """
        return (self._run_detected, self._run_max_conf, self._run_max_count)

    @property
    def annotated_path(self) -> "Path | None":
        """Path to the annotated MP4 written during the last run, or None.

        ``None`` is returned for webcam captures (no on-disk source) or when
        the writer could not be opened (codec missing, permission denied,
        etc.). Callers should fall back to the original input in that case.
        """
        return self._annotated_path

    def request_stop(self) -> None:
        self._stop_event.set()

    def start(self) -> None:
        """Open the video source and process frames until stopped or source ends.

        When the source is an on-disk file path we also open a ``cv2.VideoWriter``
        and persist every annotated frame to ``<input>.annotated.mp4`` next to
        the source. The annotated video has bounding boxes + per-box confidence
        labels baked into the pixels, so the police/school front-end can show
        the model's detections by playing this file back through a stock HTML
        ``<video>`` element with no canvas overlay required.
        """
        self._stop_event.clear()
        self._run_detected     = False
        self._run_max_conf     = 0.0
        self._run_max_count    = 0
        self._annotated_writer = None
        self._annotated_path   = None
        self._detection_window.clear()
        self._cap = cv2.VideoCapture(self._source)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {self._source}")

        # Open the annotated-MP4 writer only for on-disk sources. For webcam
        # captures (source=0) we skip it — there's no natural file location.
        if isinstance(self._source, str):
            try:
                src    = Path(self._source)
                dest   = src.with_name(f"{src.stem}.annotated.mp4")
                fps    = float(self._cap.get(cv2.CAP_PROP_FPS) or 30.0)
                width  = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
                height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
                if width > 0 and height > 0:
                    # avc1 / H.264 is universally playable in browsers and
                    # Cursor's Chromium engine. mp4v (MPEG-4 Part 2) is NOT
                    # supported by any browser natively → video stuck at 0:00.
                    # Try avc1 first; fall back to mp4v if the codec is absent
                    # (we re-encode to H.264 with ffmpeg in stop() anyway).
                    for fourcc_str in ("avc1", "mp4v"):
                        fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
                        writer = cv2.VideoWriter(str(dest), fourcc, fps, (width, height))
                        if writer.isOpened():
                            self._annotated_writer = writer
                            self._annotated_path   = dest
                            logger.info(
                                "Annotated MP4 writer  ->  %s  (%dx%d @ %.1f fps, codec=%s)",
                                dest, width, height, fps, fourcc_str,
                            )
                            break
                        writer.release()
                    else:
                        logger.warning(
                            "Could not open VideoWriter for %s — annotated MP4 disabled",
                            dest,
                        )
            except Exception:
                logger.exception("Failed to set up annotated MP4 writer; continuing without it")

        logger.info(
            "Capturing  source=%s  threshold=%.2f  iou=%.2f  imgsz=%d  location=%s",
            self._source, self._threshold, self._iou, self._imgsz, self._location,
        )
        print("\nPress 'q' in the video window to stop …\n" if self._show
              else "\nPress Ctrl+C to stop …\n")

        try:
            while not self._stop_event.is_set():
                ok, frame = self._cap.read()
                if not ok:
                    logger.info("Video source ended.")
                    break
                conf, annotated = self._process_frame(frame)
                # Persist the annotated frame regardless of whether the gate
                # has tripped — every frame goes into the output MP4 so the
                # police view sees the entire clip with detections.
                if self._annotated_writer is not None:
                    try:
                        self._annotated_writer.write(annotated)
                    except Exception:
                        logger.exception("VideoWriter.write failed; disabling annotated MP4")
                        self._annotated_writer.release()
                        self._annotated_writer = None
                        self._annotated_path   = None
                window_hits = sum(self._detection_window)
                print(f"  conf={conf:.4f}  gate={window_hits}/{self._kofn_k}", end="\r")
                if self._show:
                    cv2.imshow("Vision — gun detection", annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        print("\nStopped.")
                        break
        except KeyboardInterrupt:
            print("\nStopped.")
        finally:
            self.stop()

    def stop(self) -> None:
        self._stop_event.set()
        if self._cap:
            self._cap.release()
            self._cap = None
        if self._annotated_writer is not None:
            self._annotated_writer.release()
            self._annotated_writer = None
            # Re-encode to H.264 so the file is playable in every browser and
            # Chromium-based IDE (mp4v / MPEG-4 Part 2 is not browser-supported).
            # ffmpeg overwrites a temp file then renames atomically.
            if self._annotated_path and self._annotated_path.exists():
                self._reencode_h264(self._annotated_path)
        if self._show:
            cv2.destroyAllWindows()
        self._s3_executor.shutdown(wait=False)

    @staticmethod
    def _reencode_h264(path: Path) -> None:
        """Re-encode ``path`` in-place to H.264/AAC MP4 using ffmpeg.

        Writes to a sibling ``.tmp.mp4`` first, then replaces the original.
        Silently skips if ffmpeg is not on PATH.
        """
        import shutil
        import subprocess

        ffmpeg_bin = shutil.which("ffmpeg")
        if not ffmpeg_bin:
            logger.debug("ffmpeg not found on PATH — skipping H.264 re-encode")
            return

        tmp = path.with_suffix(".tmp.mp4")
        try:
            result = subprocess.run(
                [
                    ffmpeg_bin,
                    "-y",                          # overwrite without asking
                    "-i", str(path),               # input: raw annotated MP4
                    "-vcodec", "libx264",          # H.264 video
                    "-preset", "fast",             # fast encode, reasonable size
                    "-crf", "23",                  # quality (lower = better)
                    "-pix_fmt", "yuv420p",         # required for browser compat
                    "-an",                         # no audio track
                    "-movflags", "+faststart",     # moov atom at front → instant play
                    str(tmp),
                ],
                capture_output=True,
                timeout=120,
            )
            if result.returncode == 0 and tmp.exists():
                tmp.replace(path)
                logger.info("Re-encoded to H.264  ->  %s", path)
            else:
                stderr = result.stderr.decode(errors="replace").strip()
                logger.warning("ffmpeg re-encode failed: %s", stderr[-300:] if stderr else "unknown")
                tmp.unlink(missing_ok=True)
        except Exception:
            logger.exception("ffmpeg re-encode error; keeping original file")
            tmp.unlink(missing_ok=True)


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
    parser.add_argument("--show",       action="store_true",
                        help="Open a window showing the annotated video feed (press 'q' to stop).")
    # FP-reduction stack flags
    parser.add_argument("--no_sahi",    action="store_true",
                        help="Disable SAHI tiled inference (faster, lower recall on small guns).")
    parser.add_argument("--no_pose",    action="store_true",
                        help="Disable pose-overlap constraint.")
    parser.add_argument("--kofn_k",    type=int, default=DEFAULT_KOFN_K,
                        help=f"Frames required in temporal gate (default: {DEFAULT_KOFN_K})")
    parser.add_argument("--kofn_n",    type=int, default=DEFAULT_KOFN_N,
                        help=f"Temporal gate window size in frames (default: {DEFAULT_KOFN_N})")
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
            logger.warning("Ably connection failed: %s -- running without WS alerts", exc)
    else:
        logger.warning("No Ably key provided (--ably_key / ABLY_API_KEY) -- WS alerts disabled")

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
        show=args.show,
        use_sahi=not args.no_sahi,
        use_pose=not args.no_pose,
        kofn_k=args.kofn_k,
        kofn_n=args.kofn_n,
    )

    try:
        capture.start()
    except RuntimeError as exc:
        logger.error("%s", exc)
        sys.exit(1)
    finally:
        if publisher:
            publisher.close()
        logger.info("Detections saved to: %s", args.log_file)


if __name__ == "__main__":
    main()
