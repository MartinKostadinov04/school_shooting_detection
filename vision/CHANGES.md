# CHANGES.md

This document records every addition, modification, and deletion made to the repository after the initial handover. The base state was an audio-only detection system (`inference/` component). The changes below bring the repository to a two-component system — audio and vision — with supporting documentation and a rewritten project README.

---

## Added

### `CLAUDE.md`
A guidance file for Claude Code (AI coding assistant) placed at the project root. Contains:
- All commands needed to run, test, and build the project (backend, frontend, audio inference, vision inference, ML pipeline)
- High-level architecture diagram showing how both AI components connect to Ably and the frontend
- The exact Ably message format contract (`audio:detected`, `audio:snippet`, `video:detected`, `video:segment`) that the frontend parser relies on
- Environment variable reference table
- Demo credentials

This file is read automatically by Claude Code at the start of every session so it has full context without re-reading the codebase each time.

---

### `vision/__init__.py`
Empty Python package marker. Makes `vision/` importable as a module so the inference script can be run as `python -m vision.live_inference`.

---

### `vision/live_inference.py`
The core addition. A full production-grade inference module for real-time gun detection from a camera or video file, built to mirror the architecture of `inference/live_inference.py` exactly so both components can be integrated later.

**Structure (section by section):**

- **Constants** — default values for every configurable parameter: model path (`YOLO_hugging-main/best.pt`), confidence threshold (`0.6`), IoU threshold (`0.45`), inference resolution (`1280`), Ably channel (`gunshot-detection`), S3 presign expiry (`3600s`)

- **`AblyPublisher` class** — manages a persistent Ably Realtime WebSocket connection on a background asyncio daemon thread. Thread-safe: `publish(name, data)` can be called from any thread. Identical in design to the class in `inference/live_inference.py`.

- **`_frame_to_jpeg_bytes(frame)`** — encodes a BGR NumPy frame (OpenCV format) to JPEG bytes in memory at 90% quality. No temp file written to disk.

- **`_s3_upload_frame(frame, location, timestamp, bucket, region)`** — uploads an annotated frame (bounding boxes drawn) to S3 at key `video-snapshots/{location}/{timestamp}.jpg`, then generates and returns a presigned URL valid for 1 hour.

- **`_alert(conf, frame, timestamp, ...)`** — fires on every detection above threshold. In order: prints to console, appends a JSON record to `vision/detections.jsonl`, publishes `video:detected:{location}` to Ably, and (if S3 is configured) spawns a daemon thread to upload the frame and publish `video:segment:{location}:{url}` to Ably.

- **`VideoCapture` class** — the main engine:
  - `__init__` — loads the YOLO model from weights file at instantiation
  - `_run_inference(frame)` — calls `model.predict()` on a single frame, returns `(max_confidence, annotated_frame)` where the annotated frame has bounding boxes drawn via YOLO's built-in `.plot()` method
  - `_process_frame(frame)` — calls `_run_inference`, triggers `_alert` if confidence meets threshold, returns confidence for live display
  - `start()` — opens `cv2.VideoCapture(source)` and loops frames until the source ends or Ctrl+C
  - `stop()` — releases the OpenCV capture handle (always runs via `finally`)
  - `run_demo_file(file_path)` — convenience wrapper to process a video file

- **`main()` / CLI** — argparse interface with the following flags:

  | Flag | Default | Description |
  |---|---|---|
  | `--model_path` | `YOLO_hugging-main/best.pt` | Path to YOLO weights |
  | `--threshold` | `0.6` | Min confidence to trigger alert |
  | `--iou` | `0.45` | NMS IoU threshold |
  | `--imgsz` | `1280` | Inference resolution |
  | `--source` | `0` | `0` = webcam, or path to video file |
  | `--log_file` | `vision/detections.jsonl` | Detection log output |
  | `--location` | `unknown` | Location label in every Ably message |
  | `--channel` | `gunshot-detection` | Ably channel name |
  | `--ably_key` | — | Overrides `ABLY_API_KEY` env var |
  | `--s3_bucket` | — | Omit to skip S3 uploads entirely |
  | `--aws_region` | `us-east-1` | AWS region for boto3 |

**Run:**
```bash
# Webcam
python -m vision.live_inference --location "Main Entrance"

# Video file
python -m vision.live_inference --source YOLO_hugging-main/footage_worked/footage_pistol.avi --location "Gymnasium"
```

---

### `vision/inference_breakdown.md`
A component-by-component written explanation of `vision/live_inference.py` covering every import, constant, class, method, and CLI flag. Intended for teammates who need to understand, modify, or present the vision component without reading raw code.

---

## Modified

### `README.md`
Fully rewritten. The original README documented only the audio pipeline (Component A). The updated version covers both components as a unified system.

Specific changes:
- **Title** changed from "Gunshot Detection — Part A: YAMNet + Dense Head" to "Warden — School Shooting Detection System"
- **Introduction** rewritten to present both components from the top rather than treating the audio component as the whole product
- **Results section** now contains two tables — audio metrics (unchanged) and vision metrics (mAP50 0.984, Precision 0.995, Recall 0.980)
- **Repository layout** updated to show `vision/` as a first-class sibling of `inference/`, with `YOLO_hugging-main/` correctly described as training artefacts only
- **Quick start** updated — dev script section notes the 4th terminal needed for vision; manual startup lists all four services
- **Real-time detection flow** — new section with a diagram showing both components publishing to the same Ably channel and the full message format table
- **Component A** section retained and reorganised under the two-component framing
- **Component B** — new section covering YOLO architecture, dataset sources, live inference commands, CLI flag table, Modal training commands, and known limitations
- **Removed** the "Part A / Part B separate repo" framing — both components now live in this repository

---

## Deleted

None. No files were removed from the original repository.

---

## Unchanged

All other files from the original repository — `inference/`, `pipeline/`, `models/`, `training/`, `experiments/`, `api/`, `frontend/`, `configs/`, `tests/`, `scripts/`, `YOLO_hugging-main/`, `requirements.txt`, `.env.example`, `.gitignore` — are untouched.
