# vision/live_inference.py — Component Breakdown

---

## 1. Standard library imports (lines 34–47)

```python
import argparse    # builds the --flag CLI interface
import asyncio     # runs the Ably WebSocket connection on its own event loop
import io          # wraps bytes in a file-like object for boto3's upload_fileobj
import json        # serialises detection records to JSONL
import logging     # structured timestamped console output
import os          # reads ABLY_API_KEY from environment
import re          # sanitises location strings for safe S3 key segments
import sys         # sys.exit() on fatal errors
import threading   # runs the Ably event loop on a background daemon thread
import time        # time.monotonic() for alert cooldown tracking
from concurrent.futures import ThreadPoolExecutor  # bounded pool for S3 uploads
from datetime import datetime, timezone   # UTC timestamps on every detection
from pathlib import Path                  # cross-platform file paths
```

---

## 2. Third-party imports (lines 49–50)

```python
import cv2    # OpenCV — reads camera frames and encodes them to JPEG
import numpy as np  # frame data is a NumPy array (height × width × 3 BGR values)
```

Three heavy libraries (`ably`, `boto3`, `ultralytics`) are **not** imported at the top level — they're imported lazily inside functions so the script starts fast and doesn't crash if an optional library is missing (e.g. you can run without S3 if you omit `--s3_bucket`).

---

## 3. Logging setup (lines 52–57)

```python
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  ...")
logger = logging.getLogger(__name__)
```

Sets up a module-level logger. `INFO` messages appear in the terminal (model loading, Ably connection, S3 uploads). `DEBUG` messages (per-frame confidence) are silent unless you change the level.

---

## 4. Constants (lines 63–72)

```python
DEFAULT_MODEL_PATH  = Path("YOLO_hugging-main/best.pt")  # fine-tuned gun detector
DEFAULT_LOG_FILE    = Path("vision/detections.jsonl")     # where detections are written
DEFAULT_THRESHOLD   = 0.6     # min YOLO confidence to trigger an alert
DEFAULT_IOU         = 0.45    # NMS overlap threshold (suppresses duplicate boxes)
DEFAULT_IMG_SIZE    = 1280    # inference resolution — higher = catches small guns
DEFAULT_CHANNEL     = "gunshot-detection"  # shared Ably channel with audio component
DEFAULT_LOCATION    = "unknown"
S3_PRESIGN_EXPIRY   = 3600   # presigned URL valid for 1 hour
ALERT_COOLDOWN_SECS = 5.0   # minimum seconds between consecutive alerts
S3_UPLOAD_WORKERS   = 4     # max concurrent S3 upload threads
```

`ALERT_COOLDOWN_SECS` prevents alert/S3 spam when a gun is visible across many frames — at 30fps a 5-second cooldown means at most 1 alert per 150 frames. `S3_UPLOAD_WORKERS` caps the thread pool so sustained detections can't grow memory unboundedly.

---

## 5. `AblyPublisher` class (lines 79–123)

Ably is the real-time WebSocket service that pushes detections to the dashboard. The problem it solves: the OpenCV frame loop runs in the main thread, but Ably's Python SDK is async (requires an `asyncio` event loop). You can't just call `await channel.publish()` from a non-async loop.

**Solution — a dedicated background thread with its own event loop:**

```
Main thread (frame loop)
    │
    │  publisher.publish("video:detected", data)
    │
    ▼
run_coroutine_threadsafe()  ──────────────────────►  Background daemon thread
                                                        asyncio event loop
                                                        AblyRealtime connection
                                                        channel.publish()  (async)
```

- `__init__` — spins up the daemon thread, waits up to 15 seconds for the connection to confirm it's live before returning. If the connection times out, it raises immediately so you know Ably failed at startup.
- `_run_loop` — the entry point for the background thread; sets the event loop and blocks forever with `run_forever()`.
- `_connect` — async: creates the `AblyRealtime` client, waits for `"connected"`, grabs the channel object, signals `_ready`.
- `publish(name, data)` — the only method you call from outside. Thread-safe because `run_coroutine_threadsafe` is designed exactly for this.
- `close()` — schedules `client.close()` on the background loop and waits up to 3 seconds for it to complete (`.result(timeout=3)`), ensuring the WebSocket is torn down gracefully before the process exits.

---

## 6. S3 helpers (lines 130–155)

**`_frame_to_jpeg_bytes(frame)`** — takes a NumPy BGR frame (what OpenCV produces) and compresses it to a JPEG byte string at 90% quality. `cv2.imencode` does this in memory — no temp file is written to disk.

**`_s3_upload_frame(frame, location, timestamp, bucket, region)`** — the actual upload:
1. Converts the frame to JPEG bytes
2. Sanitises the location string with `re.sub(r"[^\w-]", "_", location)` — replaces spaces, slashes, and all other special characters so the result is safe as an S3 key segment
3. Builds the S3 key: `video-snapshots/{safe_location}/{timestamp}.jpg`
4. Uploads via `boto3.upload_fileobj` (streams from memory, no disk write)
5. Generates a **presigned URL** — a time-limited public link (1 hour) that the frontend uses to display the snapshot image
6. Returns that URL so `_alert` can publish it to Ably

`boto3` is imported **inside** this function (lazy import) because it's only needed if `--s3_bucket` was passed.

---

## 7. `_alert()` function (lines 162–213)

Called when a detection occurs and the cooldown has elapsed. Takes an `executor` (the `ThreadPoolExecutor` owned by `VideoCapture`) and does four things in order:

**① Console print** — immediate visual feedback:
```
🔴  GUN DETECTED  conf=0.847  loc=Main Entrance  [2026-04-26T10:32:11+00:00]
```

**② JSONL log** — appends one JSON record per alert to `vision/detections.jsonl`:
```json
{"event": "gun_detected", "timestamp": "...", "confidence": 0.847, "threshold": 0.6, "location": "Main Entrance"}
```
Each detection is one line. The file grows over time and can be parsed by any tool.

**③ Ably — detection event** — publishes immediately to the dashboard:
```
event name: "video:detected"
data:        "video:detected:Main Entrance"
```
The frontend sees this and creates a new incident card in the UI.

**④ Ably — frame upload** (only if `--s3_bucket` was given) — submitted to the bounded `ThreadPoolExecutor` so the capture loop is never blocked and at most `S3_UPLOAD_WORKERS` uploads run concurrently:
```
event name: "video:segment"
data:        "video:segment:Main Entrance:https://s3.amazonaws.com/..."
```
The frontend sees this and attaches the image URL to the existing incident card.

`frame.copy()` is called before submitting to the executor — this snapshots the numpy array so the capture loop can't overwrite it before the upload thread reads it.

---

## 8. `VideoCapture` class (lines 220–312)

The core engine. Parallel to the audio component's `AudioCapture`.

**`__init__`** — loads the YOLO model immediately (`ultralytics.YOLO(model_path)`). Stores all config. Also initialises:
- `_cap = None` — the OpenCV camera handle, created in `start()`
- `_last_alert_time = 0.0` — monotonic timestamp of the last fired alert, used for cooldown
- `_s3_executor = ThreadPoolExecutor(max_workers=S3_UPLOAD_WORKERS)` — bounded pool for S3 uploads

**`_run_inference(frame)`** — the actual model call:
```python
results = self._model.predict(source=frame, conf=0.6, iou=0.45, imgsz=1280, verbose=False)
result    = results[0]       # one result per input image
annotated = result.plot()    # YOLO draws bounding boxes + labels on the frame (BGR array)
max_conf  = boxes.conf.max() # highest confidence among all detected boxes in this frame
```
Returns `(max_confidence, annotated_frame)`. The annotated frame (with boxes drawn) is what gets uploaded to S3, not the raw frame. If no boxes are returned (nothing detected), `max_conf = 0.0`.

**`_process_frame(frame)`** — calls `_run_inference`, then fires a **throttled** alert:
```python
if max_conf > 0.0:                                         # any detection above threshold
    if time.monotonic() - self._last_alert_time >= ALERT_COOLDOWN_SECS:
        self._last_alert_time = time.monotonic()
        _alert(...)
```
Using `time.monotonic()` (not `datetime.now()`) makes the cooldown immune to system clock changes.

**`start()`** — the main loop:
```
cv2.VideoCapture(source)   ← opens webcam (int) or video file (str)
    └─ while True:
        cap.read()          ← grabs next frame as NumPy array
        _process_frame()    ← run YOLO, alert if cooldown allows
        print conf=X.XXXX  ← live readout (overwrites same line with \r)
        if ok == False:     ← video file ended, or webcam disconnected
            break
```
If the source can't be opened, raises `RuntimeError` (caught in `main()`) instead of calling `sys.exit()` directly — keeping the class testable. Ctrl+C triggers `KeyboardInterrupt`, caught by the `except` block. The `finally` always calls `stop()`.

**`stop()`** — releases the OpenCV capture handle, sets `self._cap = None` to prevent double-release, and shuts down the S3 executor with `wait=False` (in-flight uploads may complete in the background; no new ones are accepted).

**`run_demo_file(file_path)`** — convenience wrapper: sets `_source` to the file path string, then calls `start()`. OpenCV handles both webcam indexes and file paths through the same `VideoCapture` API.

---

## 9. `main()` and CLI (lines 319–406)

**Argument parsing:**

| Flag | Type | Default | What it controls |
|---|---|---|---|
| `--model_path` | Path | `YOLO_hugging-main/best.pt` | Which weights file to load |
| `--threshold` | float | `0.6` | Min YOLO confidence to trigger alert |
| `--iou` | float | `0.45` | NMS deduplication aggressiveness |
| `--imgsz` | int | `1280` | Frame resize before YOLO inference |
| `--source` | str/int | `0` | `0` = webcam, or a file path |
| `--log_file` | Path | `vision/detections.jsonl` | Where to write detection records |
| `--location` | str | `"unknown"` | Label sent in every Ably message |
| `--channel` | str | `"gunshot-detection"` | Ably channel name |
| `--ably_key` | str | `None` | Overrides `ABLY_API_KEY` env var |
| `--s3_bucket` | str | `None` | Omit entirely to skip S3 uploads |
| `--aws_region` | str | `"us-east-1"` | AWS region for boto3 |

**Startup sequence (after parsing):**
1. `--source` coercion — `argparse` returns everything as a string. If it looks like a digit (`"0"`), convert it to an int so OpenCV interprets it as a device index, not a filename.
2. Model path check — if `best.pt` doesn't exist, exit immediately with a clear error.
3. Ably setup — tries to connect; if it fails (wrong key, no internet), logs a warning and continues without WebSocket alerts rather than crashing.
4. `VideoCapture(...)` — instantiated with all config; YOLO model loads here.
5. `capture.start()` — blocks until the video ends or Ctrl+C. If the source can't be opened, catches the `RuntimeError` and exits with code 1.
6. `finally` block — always closes the Ably connection (waiting up to 3s) and logs where detections were saved.
