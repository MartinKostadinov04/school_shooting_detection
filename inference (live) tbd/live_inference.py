# FUTURE: Live microphone inference script
#
# This script will continuously capture audio from a microphone, run the full
# detection pipeline in real time, and trigger an alert when a gunshot is detected.
#
# Planned usage:
#   python -m inference.live_inference \
#       --model_path models/saved_weights/head_dense_v1.h5 \
#       --threshold 0.5 \
#       [--cascade]        # enable two-stage gate + head system
#
# Planned pipeline (per 2-second window):
#   1. Capture 32000 samples at 16 kHz from default microphone (pyaudio or sounddevice)
#   2. preprocess_clip() -> (32000,) float32 waveform
#   3. YAMNet -> (1024,) embedding (or per-frame embeddings for BiLSTM head)
#   4. Head model -> gunshot probability
#   6. If probability >= threshold: trigger alert (log, sound alarm, call Part B vision pipeline)
#
# FUTURE PLACEHOLDER: Integration with Part B (YOLO vision pipeline) happens here.
# When a gunshot is detected, this script should send a trigger signal to the
# Part B camera system.
#
# Dependencies (not yet in requirements.txt):
#   pyaudio>=0.2.13  OR  sounddevice>=0.4.6
