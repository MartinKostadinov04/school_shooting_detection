"""
live_inference.py
=================
Real-time gunshot detection from a microphone using a sliding window.

Pipeline per 0.5-second chunk:
  microphone (16 kHz mono float32)
    → ring buffer (last 2 s = 32 000 samples)
    → YAMNet → (1024,) embedding
    → Dense head → gunshot probability
    → if prob >= threshold: alert (console + JSONL log + optional webhook)

Usage
-----
    # Defaults (threshold 0.64, log to inference/detections.jsonl)
    python -m inference.live_inference

    # Custom
    python -m inference.live_inference \\
        --model_path  models/saved_weights/dense_head_best.keras \\
        --threshold   0.64 \\
        --log_file    inference/detections.jsonl \\
        --webhook_url https://your-app.com/api/gunshot-event \\
        --device      0
"""

import argparse
import json
import logging
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import tensorflow as tf

# Reuse YAMNet loader and embedding extractor from the pipeline.
from pipeline.extract_embeddings import load_yamnet, extract_embedding

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SAMPLE_RATE: int = 16000          # YAMNet requires exactly 16 kHz
CLIP_SAMPLES: int = 32000         # 2 seconds — YAMNet input length
CHUNK_SAMPLES: int = 8000         # 0.5 seconds — hop size / sounddevice block size
DEFAULT_MODEL_PATH = Path("models/saved_weights/dense_head_best.keras")
DEFAULT_LOG_FILE   = Path("inference/detections.jsonl")
DEFAULT_THRESHOLD  = 0.64


# ---------------------------------------------------------------------------
# Alert
# ---------------------------------------------------------------------------

def _alert(prob: float, threshold: float, log_file: Path, webhook_url: str | None) -> None:
    ts = datetime.now(timezone.utc).isoformat()
    record = {
        "event":       "gunshot_detected",
        "timestamp":   ts,
        "probability": round(float(prob), 4),
        "threshold":   threshold,
    }

    # Console
    print(f"\n🔴  GUNSHOT DETECTED  prob={prob:.3f}  [{ts}]")

    # JSONL log
    try:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except Exception as exc:
        logger.warning("Could not write to log file: %s", exc)

    # Webhook
    if webhook_url:
        try:
            import requests
            resp = requests.post(webhook_url, json=record, timeout=3)
            logger.info("Webhook POST → %s  status=%d", webhook_url, resp.status_code)
        except Exception as exc:
            logger.warning("Webhook POST failed: %s", exc)


# ---------------------------------------------------------------------------
# Audio capture + inference
# ---------------------------------------------------------------------------

class AudioCapture:
    """
    Wraps sounddevice.InputStream with a sliding ring buffer.

    Each callback receives CHUNK_SAMPLES new samples. The ring buffer always
    holds the most recent CLIP_SAMPLES (2 s). Inference runs on every chunk.
    """

    def __init__(
        self,
        yamnet_model,
        head_model: tf.keras.Model,
        threshold: float,
        log_file: Path,
        webhook_url: str | None,
        device: int | None,
    ):
        self._yamnet     = yamnet_model
        self._head       = head_model
        self._threshold  = threshold
        self._log_file   = log_file
        self._webhook    = webhook_url
        self._device     = device
        self._buffer     = np.zeros(CLIP_SAMPLES, dtype=np.float32)
        self._lock       = threading.Lock()
        self._stream     = None

    def _callback(self, indata, frames, time_info, status):
        if status:
            logger.warning("sounddevice status: %s", status)

        chunk = indata[:CHUNK_SAMPLES, 0].astype(np.float32)  # mono

        with self._lock:
            # Slide buffer left and append new chunk.
            self._buffer = np.roll(self._buffer, -CHUNK_SAMPLES)
            self._buffer[-CHUNK_SAMPLES:] = chunk

            buf = self._buffer.copy()

        # Run inference outside the lock so audio capture never stalls.
        prob = self._run_inference(buf)
        logger.debug("prob=%.4f", prob)

        if prob >= self._threshold:
            _alert(prob, self._threshold, self._log_file, self._webhook)

    def _run_inference(self, audio: np.ndarray) -> float:
        embedding = extract_embedding(audio, self._yamnet)          # (1024,)
        prob = self._head.predict(embedding[np.newaxis], verbose=0) # (1, 1)
        return float(prob[0, 0])

    def start(self):
        import sounddevice as sd
        self._stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=CHUNK_SAMPLES,
            device=self._device,
            callback=self._callback,
        )
        self._stream.start()
        logger.info(
            "Listening on device=%s  |  chunk=%.2fs  window=%.2fs  threshold=%.2f",
            self._device if self._device is not None else "default",
            CHUNK_SAMPLES / SAMPLE_RATE,
            CLIP_SAMPLES  / SAMPLE_RATE,
            self._threshold,
        )

    def stop(self):
        if self._stream:
            self._stream.stop()
            self._stream.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Real-time gunshot detection from microphone.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model_path", type=Path, default=DEFAULT_MODEL_PATH,
        help=f"Path to saved .keras weights. Default: {DEFAULT_MODEL_PATH}",
    )
    parser.add_argument(
        "--threshold", type=float, default=DEFAULT_THRESHOLD,
        help=f"Detection threshold on sigmoid output. Default: {DEFAULT_THRESHOLD}",
    )
    parser.add_argument(
        "--log_file", type=Path, default=DEFAULT_LOG_FILE,
        help=f"JSONL file to append detections to. Default: {DEFAULT_LOG_FILE}",
    )
    parser.add_argument(
        "--webhook_url", type=str, default=None,
        help="HTTP endpoint to POST detection events to. Optional.",
    )
    parser.add_argument(
        "--device", type=int, default=None,
        help="sounddevice input device index. Default: system default.",
    )
    args = parser.parse_args()

    if not args.model_path.exists():
        logger.error("Model weights not found: %s", args.model_path)
        sys.exit(1)

    logger.info("Loading YAMNet ...")
    yamnet = load_yamnet()

    logger.info("Loading head model from %s ...", args.model_path)
    head = tf.keras.models.load_model(str(args.model_path))

    capture = AudioCapture(
        yamnet_model=yamnet,
        head_model=head,
        threshold=args.threshold,
        log_file=args.log_file,
        webhook_url=args.webhook_url,
        device=args.device,
    )

    capture.start()
    print("\nPress Enter to stop ...\n")
    try:
        input()
    except KeyboardInterrupt:
        pass
    finally:
        capture.stop()
        logger.info("Stopped. Detections saved to: %s", args.log_file)


if __name__ == "__main__":
    main()
