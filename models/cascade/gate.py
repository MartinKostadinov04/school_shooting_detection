# FUTURE: Two-stage cascade gate
#
# This module will implement the first stage of the cascade detection system.
# The gate uses YAMNet's raw AudioSet class-427 (Gunshot, gunfire) score as
# a cheap pre-filter: only clips that pass the gate threshold proceed to the
# trained classification head (head_dense.py or head_bilstm.py).
#
# Design intent:
#   Stage 1 (gate):  zero_shot_score >= gate_threshold  -> forward to head
#   Stage 2 (head):  head(embedding) >= head_threshold  -> GUNSHOT DETECTED
#
# The gate threshold should be tuned to maximize recall (minimize missed
# gunshots), accepting higher false positive rate at this stage. The head
# then filters out false positives.
#
# The extract_zero_shot_scores() function in pipeline/extract_embeddings.py
# already produces the scalar scores needed for gate-based filtering.
#
# Threshold tuning and precision-recall analysis are handled in a future
# training session.
