# CNN experiment mel spectrogram configuration
#
# WARNING: These parameters are used ONLY by the separate CNN-based experiment.
# Do NOT import or use these in the YAMNet pipeline (preprocessing.py,
# extract_embeddings.py, split_dataset.py). YAMNet handles its own internal
# mel computation and cannot be overridden.
#
# The YAMNet pipeline is configured via configs/yamnet_pipeline.yaml.

# Mel spectrogram settings for the CNN experiment
N_MELS: int = 128
N_FFT: int = 1024
HOP_LENGTH: int = 256
SAMPLE_RATE: int = 22050  # Note: different from YAMNet's required 16 kHz
