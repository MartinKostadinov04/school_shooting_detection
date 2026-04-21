"""
live_inference.py
=================
Real-time gunshot detection from a microphone using a sliding window.

Pipeline per 0.5-second chunk:
  microphone (16 kHz mono float32)
    → ring buffer (last 2 s = 32 000 samples)
    → YAMNet → (1024,) embedding
    → Dense head → gunshot probability
    → if prob >= threshold:
        • console log
        • JSONL log file
        • Ably WS  →  "audio:detected:{location}"
        • (optional) S3 upload  →  "audio:snippet:{location}:{presigned_url}"

Ably credentials: set ABLY_API_KEY env var (or pass --ably_key).
AWS credentials:  set AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / AWS_DEFAULT_REGION
                  (standard boto3 env vars — never hardcode).

Usage
-----
  Live mic (default):
    python -m inference.live_inference --location "Gymnasium"

  Demo mode — two terminals:
    # Terminal 1: start inference engine (no mic, waits on UDP port 9999)
    python -m inference.live_inference --run --location "Gymnasium"

    # Terminal 2: stream an audio file into it
    python -m inference.live_inference --demo_file shot.wav

  Optional flags (all modes):
    --threshold 0.64  --log_file inference/detections.jsonl
    --ably_key KEY    --channel gunshot-detection
    --s3_bucket my-bucket  --aws_region eu-west-1
    --port 9999       (change UDP port for --run / --demo_file)
"""

import argparse
import asyncio
import io
import json
import logging
import os
import sys
import threading
import wave
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import tensorflow as tf

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

SAMPLE_RATE: int   = 16000
CLIP_SAMPLES: int  = 32000   # 2 s window
CHUNK_SAMPLES: int = 8000    # 0.5 s hop
DEFAULT_MODEL_PATH = Path("models/saved_weights/dense_head_best.keras")
DEFAULT_LOG_FILE   = Path("inference/detections.jsonl")
DEFAULT_THRESHOLD  = 0.64
DEFAULT_CHANNEL    = "gunshot-detection"
DEFAULT_LOCATION   = "unknown"
S3_PRESIGN_EXPIRY  = 3600    # seconds
DEFAULT_DEMO_PORT  = 9999    # UDP port for --run / --demo_file IPC
CHUNK_BYTES        = CHUNK_SAMPLES * 4  # float32 = 4 bytes per sample


# ---------------------------------------------------------------------------
# Async Ably publisher
# ---------------------------------------------------------------------------
# sounddevice callbacks run in a C thread — asyncio cannot be called directly.
# We start a dedicated event loop in a background thread and bridge to it via
# run_coroutine_threadsafe().

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
# S3 snippet upload
# ---------------------------------------------------------------------------

def _audio_to_wav_bytes(audio: np.ndarray) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)          # 16-bit PCM
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes((audio * 32768).astype(np.int16).tobytes())
    return buf.getvalue()


def _s3_upload(audio: np.ndarray, location: str, timestamp: str,
               bucket: str, region: str) -> str:
    import boto3
    wav_bytes = _audio_to_wav_bytes(audio)
    safe_loc  = location.replace("/", "_")
    key       = f"audio-snippets/{safe_loc}/{timestamp}.wav"

    s3 = boto3.client("s3", region_name=region)
    s3.upload_fileobj(io.BytesIO(wav_bytes), bucket, key)

    url = s3.generate_presigned_url(
        "get_object",
        Params={"Bucket": bucket, "Key": key},
        ExpiresIn=S3_PRESIGN_EXPIRY,
    )
    logger.info("Snippet uploaded  →  s3://%s/%s", bucket, key)
    return url


# ---------------------------------------------------------------------------
# Alert
# ---------------------------------------------------------------------------

def _alert(
    prob:      float,
    audio:     np.ndarray,
    timestamp: str,
    threshold: float,
    location:  str,
    log_file:  Path,
    publisher: "AblyPublisher | None",
    s3_bucket: "str | None",
    aws_region: str,
) -> None:
    record = {
        "event":       "gunshot_detected",
        "timestamp":   timestamp,
        "probability": round(float(prob), 4),
        "threshold":   threshold,
        "location":    location,
    }

    # Console
    print(f"\n🔴  GUNSHOT DETECTED  prob={prob:.3f}  loc={location}  [{timestamp}]")

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
    publisher.publish("audio:detected", f"audio:detected:{location}")
    logger.info("Ably  →  audio:detected:%s", location)

    # Ably: snippet message (upload async so we don't block the audio thread)
    if s3_bucket:
        def _upload_and_publish():
            try:
                url = _s3_upload(audio.copy(), location, timestamp, s3_bucket, aws_region)
                publisher.publish("audio:snippet", f"audio:snippet:{location}:{url}")
                logger.info("Ably  →  audio:snippet:%s:<url>", location)
            except Exception as exc:
                logger.warning("S3 upload failed: %s", exc)

        threading.Thread(target=_upload_and_publish, daemon=True).start()


# ---------------------------------------------------------------------------
# Audio capture + inference
# ---------------------------------------------------------------------------

class AudioCapture:
    def __init__(
        self,
        yamnet_model,
        head_model:  tf.keras.Model,
        threshold:   float,
        location:    str,
        log_file:    Path,
        publisher:   "AblyPublisher | None",
        s3_bucket:   "str | None",
        aws_region:  str,
        device:      "int | None",
    ):
        self._yamnet    = yamnet_model
        self._head      = head_model
        self._threshold = threshold
        self._location  = location
        self._log_file  = log_file
        self._publisher = publisher
        self._s3_bucket = s3_bucket
        self._aws_region = aws_region
        self._device    = device
        self._buffer    = np.zeros(CLIP_SAMPLES, dtype=np.float32)
        self._lock      = threading.Lock()
        self._stream    = None

    def _process_chunk(self, chunk: np.ndarray) -> float:
        """Core pipeline: update ring buffer, run inference, fire alert if triggered."""
        with self._lock:
            self._buffer = np.roll(self._buffer, -CHUNK_SAMPLES)
            self._buffer[-CHUNK_SAMPLES:] = chunk
            buf = self._buffer.copy()

        prob = self._run_inference(buf)
        logger.debug("prob=%.4f", prob)

        if prob >= self._threshold:
            ts = datetime.now(timezone.utc).isoformat()
            _alert(
                prob=prob,
                audio=buf,
                timestamp=ts,
                threshold=self._threshold,
                location=self._location,
                log_file=self._log_file,
                publisher=self._publisher,
                s3_bucket=self._s3_bucket,
                aws_region=self._aws_region,
            )
        return prob

    def _callback(self, indata, frames, time_info, status):
        if status:
            logger.warning("sounddevice: %s", status)
        chunk = indata[:CHUNK_SAMPLES, 0].astype(np.float32)
        self._process_chunk(chunk)

    def _run_inference(self, audio: np.ndarray) -> float:
        embedding = extract_embedding(audio, self._yamnet)
        prob = self._head.predict(embedding[np.newaxis], verbose=0)
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
            "Listening  device=%s  chunk=%.2fs  window=%.2fs  threshold=%.2f  location=%s",
            self._device if self._device is not None else "default",
            CHUNK_SAMPLES / SAMPLE_RATE,
            CLIP_SAMPLES  / SAMPLE_RATE,
            self._threshold,
            self._location,
        )

    def stop(self):
        if self._stream:
            self._stream.stop()
            self._stream.close()

    def run_demo_file_inprocess(self, file_path: Path) -> None:
        """Single-terminal demo: load file, play audio, run inference here."""
        import librosa
        import sounddevice as sd
        import time

        logger.info("Loading %s ...", file_path)
        audio, _ = librosa.load(str(file_path), sr=SAMPLE_RATE, mono=True)
        duration  = len(audio) / SAMPLE_RATE
        n_chunks  = len(audio) // CHUNK_SAMPLES
        chunk_dur = CHUNK_SAMPLES / SAMPLE_RATE

        print(f"\n▶  '{file_path.name}'  ({duration:.1f} s)  —  Ctrl+C to stop\n")
        sd.play(audio, samplerate=SAMPLE_RATE)

        for i in range(n_chunks):
            t0    = time.perf_counter()
            chunk = audio[i * CHUNK_SAMPLES : (i + 1) * CHUNK_SAMPLES]
            prob  = self._process_chunk(chunk)
            print(f"  [{i * chunk_dur:5.1f}s]  prob={prob:.4f}", end="\r")
            spent = time.perf_counter() - t0
            if chunk_dur - spent > 0:
                time.sleep(chunk_dur - spent)

        sd.wait()
        print(f"\n\nDone. {n_chunks} chunks processed.")

    def run_socket_listener(self, port: int) -> None:
        """
        --run mode: bind a UDP socket and process every chunk that arrives.
        Blocks until KeyboardInterrupt. The mic is NOT opened — all audio
        comes from whatever sends to 127.0.0.1:{port} (e.g. --demo_file).
        """
        import socket
        import time

        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind(("127.0.0.1", port))
        sock.settimeout(1.0)          # allows Ctrl+C to be caught cleanly

        logger.info(
            "Listening on UDP 127.0.0.1:%d  |  threshold=%.2f  |  location=%s",
            port, self._threshold, self._location,
        )
        print(f"\n  Inference engine ready — waiting for audio on port {port} ...\n"
              f"  Feed a file:  python -m inference.live_inference --demo_file <file>\n"
              f"  Ctrl+C to stop.\n")

        chunk_duration = CHUNK_SAMPLES / SAMPLE_RATE
        while True:
            try:
                data, _ = sock.recvfrom(CHUNK_BYTES + 64)
            except socket.timeout:
                continue

            if len(data) != CHUNK_BYTES:
                logger.warning("Unexpected chunk size %d bytes — skipping", len(data))
                continue

            chunk = np.frombuffer(data, dtype=np.float32).copy()
            prob  = self._process_chunk(chunk)
            elapsed_s = time.monotonic()
            print(f"  prob={prob:.4f}", end="\r")

        sock.close()


# ---------------------------------------------------------------------------
# Demo file sender (--demo_file, standalone — no model needed)
# ---------------------------------------------------------------------------

def _listener_reachable(port: int) -> bool:
    """Return True if a --run listener is accepting on 127.0.0.1:{port}."""
    import socket
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.settimeout(0.2)
        # Send a zero-length probe; the listener will discard it (wrong size)
        sock.sendto(b"", ("127.0.0.1", port))
        sock.close()
        # If the port is bound, sendto succeeds; if nothing is bound on UDP,
        # we may get an ICMP port-unreachable on some OSes but not on Windows.
        # So we do a second check: try to bind to the same port — if it fails,
        # something is already listening there.
        probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        probe.bind(("127.0.0.1", port))
        probe.close()
        return False   # bind succeeded → port is free → no listener
    except OSError:
        return True    # bind failed → port is taken → listener is up


def send_demo_file(file_path: Path, port: int) -> None:
    """
    Load an audio file, play it through the speakers, and stream 0.5 s chunks
    over UDP to the --run listener at 127.0.0.1:{port}.
    No model is loaded here — inference happens in the listener process.
    """
    import socket
    import time
    import librosa
    import sounddevice as sd

    logger.info("Loading %s ...", file_path)
    audio, _ = librosa.load(str(file_path), sr=SAMPLE_RATE, mono=True)
    duration   = len(audio) / SAMPLE_RATE
    n_chunks   = len(audio) // CHUNK_SAMPLES
    chunk_dur  = CHUNK_SAMPLES / SAMPLE_RATE

    print(f"\n▶  '{file_path.name}'  ({duration:.1f} s, {n_chunks} chunks)"
          f"  →  127.0.0.1:{port}\n")

    sd.play(audio, samplerate=SAMPLE_RATE)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    for i in range(n_chunks):
        t0    = time.perf_counter()
        chunk = audio[i * CHUNK_SAMPLES : (i + 1) * CHUNK_SAMPLES].astype(np.float32)
        sock.sendto(chunk.tobytes(), ("127.0.0.1", port))

        elapsed = i * chunk_dur
        print(f"  [{elapsed:5.1f}s / {duration:.1f}s]  sent chunk {i+1}/{n_chunks}", end="\r")

        spent = time.perf_counter() - t0
        if chunk_dur - spent > 0:
            time.sleep(chunk_dur - spent)

    sock.close()
    sd.wait()
    print(f"\n\n  Done — {n_chunks} chunks sent.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Real-time gunshot detection from microphone with Ably WS alerts.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model_path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--threshold",  type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--log_file",   type=Path,  default=DEFAULT_LOG_FILE)
    parser.add_argument("--device",     type=int,   default=None,
                        help="sounddevice input device index")
    parser.add_argument("--location",   type=str,   default=DEFAULT_LOCATION,
                        help="Location name/code sent in every message, e.g. 'building-a/entrance'")
    parser.add_argument("--channel",    type=str,   default=DEFAULT_CHANNEL,
                        help=f"Ably channel name. Default: {DEFAULT_CHANNEL}")
    parser.add_argument("--ably_key",   type=str,   default=None,
                        help="Ably API key. Defaults to ABLY_API_KEY env var.")
    parser.add_argument("--s3_bucket",  type=str,   default=None,
                        help="S3 bucket for audio snippet upload. Omit to skip.")
    parser.add_argument("--aws_region", type=str,   default="us-east-1",
                        help="AWS region for S3. Default: us-east-1")
    parser.add_argument("--run",        action="store_true",
                        help="Start the inference engine on a UDP socket (no mic).")
    parser.add_argument("--demo_file",  type=Path,  default=None,
                        help="Audio file (WAV/MP3) to stream to a --run listener.")
    parser.add_argument("--port",       type=int,   default=DEFAULT_DEMO_PORT,
                        help=f"UDP port for --run / --demo_file IPC. Default: {DEFAULT_DEMO_PORT}")
    args = parser.parse_args()

    # --demo_file: if a --run listener is up, act as sender only.
    # Otherwise load the model and run inference in-process (single terminal).
    if args.demo_file:
        if not args.demo_file.exists():
            logger.error("File not found: %s", args.demo_file)
            sys.exit(1)
        if _listener_reachable(args.port):
            logger.info("--run listener detected on port %d — acting as sender.", args.port)
            try:
                send_demo_file(args.demo_file, args.port)
            except KeyboardInterrupt:
                print("\nStopped.")
            return
        logger.info("No --run listener on port %d — running inference in-process.", args.port)
        # fall through to model loading below, then call run_demo_file_inprocess

    # All other modes need the model
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

    logger.info("Loading YAMNet ...")
    yamnet = load_yamnet()

    logger.info("Loading head model from %s ...", args.model_path)
    head = tf.keras.models.load_model(str(args.model_path))

    capture = AudioCapture(
        yamnet_model=yamnet,
        head_model=head,
        threshold=args.threshold,
        location=args.location,
        log_file=args.log_file,
        publisher=publisher,
        s3_bucket=args.s3_bucket,
        aws_region=args.aws_region,
        device=args.device,
    )

    # --demo_file in-process (no listener was running)
    if args.demo_file:
        try:
            capture.run_demo_file_inprocess(args.demo_file)
        except KeyboardInterrupt:
            print("\nStopped.")
        finally:
            if publisher:
                publisher.close()
            logger.info("Detections saved to: %s", args.log_file)
        return

    # --run: socket listener (blocks until Ctrl+C)
    if args.run:
        try:
            capture.run_socket_listener(args.port)
        except KeyboardInterrupt:
            print("\nStopped.")
        finally:
            if publisher:
                publisher.close()
            logger.info("Detections saved to: %s", args.log_file)
        return

    # Default: live microphone
    capture.start()
    print("\nPress Enter to stop ...\n")
    try:
        input()
    except KeyboardInterrupt:
        pass
    finally:
        capture.stop()
        if publisher:
            publisher.close()
        logger.info("Stopped. Detections saved to: %s", args.log_file)


if __name__ == "__main__":
    main()
