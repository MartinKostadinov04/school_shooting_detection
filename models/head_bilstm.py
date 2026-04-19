# FUTURE: BiLSTM classification head
#
# This module will implement a BiLSTM head that takes YAMNet per-frame
# embeddings as input (NOT mean-pooled) to capture temporal dynamics.
#
# NOTE: This head requires a different embedding extraction path than
# head_dense.py. Instead of mean-pooling embeddings over time, raw
# per-frame embeddings of shape (num_frames, 1024) must be passed.
# A flag or separate extraction mode in extract_embeddings.py will be
# needed before this head can be trained.
#
# Architecture (to be finalized in training session):
#   Input: (batch, num_frames, 1024) float32 per-frame embeddings
#   Bidirectional(LSTM(128, return_sequences=False))
#   Dense(64, activation='relu') + Dropout
#   Dense(1, activation='sigmoid')  -> gunshot probability
#
# Reference: Wu (DAML 2024) — YAMNet + BiLSTM, strong generalization
# on UrbanSound8K.
