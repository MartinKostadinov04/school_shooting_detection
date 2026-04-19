# FUTURE: Head training script
#
# This script will train a classification head (Dense or BiLSTM) on top of
# the pre-extracted YAMNet embeddings saved by extract_embeddings.py.
#
# Planned usage:
#   python -m training.train_head --config configs/experiment_dense_v1.yaml
#
# Planned workflow:
#   1. Load X_train, y_train, X_val, y_val from data/processed/splits/
#   2. Build head model from config (head_type: "dense" or "bilstm")
#   3. Train with early stopping, save best weights to models/saved_weights/
#   4. Evaluate on X_val, log metrics (accuracy, F1, AUC-ROC) to experiments/runs/
#   5. Optionally run on X_test for held-out evaluation
#
# Each run saves a JSON to experiments/runs/<experiment_name>_<timestamp>.json
# containing: config snapshot, per-epoch metrics, final test metrics.
#
# FUTURE PLACEHOLDER: Two-stage cascade training will be a separate script
# (training/train_cascade.py) that jointly tunes gate and head thresholds.
