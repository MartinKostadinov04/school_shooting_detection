# FUTURE: Dense MLP classification head
#
# This module will implement a Dense MLP head that takes YAMNet's (1024,)
# embeddings as input and outputs a binary gunshot probability.
#
# Architecture (to be finalized in training session):
#   Input: (batch, 1024) float32 embeddings from extract_embeddings.py
#   Dense(256, activation='relu') + Dropout
#   Dense(128, activation='relu') + Dropout
#   Dense(1, activation='sigmoid')  -> gunshot probability
#
# Training is handled by training/train_head.py.
# Experiment configs live in configs/experiment_template.yaml.
#
# Reference: Valliappan et al. (2024, IEEE Access) — YAMNet + Dense head,
# 94.96% accuracy on 12-class firearm ID.
