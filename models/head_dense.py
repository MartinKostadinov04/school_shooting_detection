"""
head_dense.py
=============
Dense MLP classification head for gunshot detection.

Takes YAMNet's mean-pooled (1024,) clip embeddings as input and outputs a
binary gunshot probability via a single hidden layer.

Architecture (Valliappan et al., 2024, IEEE Access):
    Input(1024) → Dense(256, relu) → Dropout(0.3) → Dense(1, sigmoid)

This module is importable standalone — live_inference.py loads it without any
training context. Class weights are NOT baked into the model; they are passed
to ``model.fit()`` by the training script.

Usage
-----
    from models.head_dense import build_dense_head

    model = build_dense_head()
    model.summary()
"""

import tensorflow as tf

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

INPUT_DIM: int = 1024
DEFAULT_UNITS: int = 256
DEFAULT_DROPOUT_RATE: float = 0.3
DEFAULT_LEARNING_RATE: float = 3e-4


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_dense_head(
    input_dim: int = INPUT_DIM,
    units: int = DEFAULT_UNITS,
    dropout_rate: float = DEFAULT_DROPOUT_RATE,
    learning_rate: float = DEFAULT_LEARNING_RATE,
) -> tf.keras.Model:
    """
    Build and compile the Dense MLP classification head.

    The model is compiled and ready to call ``model.fit()``. Class weights for
    handling the gunshot/not_gunshot imbalance (~1:4) must be passed to
    ``model.fit(class_weight=...)`` by the caller — they are not embedded here
    so that the saved model is usable during inference without any training
    context.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the input embedding. Must match YAMNet's output
        (1024). Only change if using a different backbone.
    units : int
        Number of units in the single hidden Dense layer. Default: 256.
    dropout_rate : float
        Dropout probability applied after the hidden layer. Default: 0.3.
    learning_rate : float
        Adam optimizer learning rate. Default: 3e-4.

    Returns
    -------
    tf.keras.Model
        Compiled Keras model. Input shape: ``(None, 1024)``. Output shape:
        ``(None, 1)`` — gunshot probability in ``[0, 1]``.
    """
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(input_dim,), dtype=tf.float32),
            tf.keras.layers.Dense(units, activation="relu"),
            tf.keras.layers.Dropout(dropout_rate),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ],
        name="gunshot_head",
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )

    return model


def get_model_config(
    input_dim: int = INPUT_DIM,
    units: int = DEFAULT_UNITS,
    dropout_rate: float = DEFAULT_DROPOUT_RATE,
    learning_rate: float = DEFAULT_LEARNING_RATE,
) -> dict:
    """
    Return a plain-Python dict of the model hyperparameters.

    Used by the training script to snapshot the exact configuration into the
    per-run JSON. Keeping the dict construction here avoids repeating argument
    names across files.

    Parameters
    ----------
    input_dim, units, dropout_rate, learning_rate
        Same meaning as in ``build_dense_head``.

    Returns
    -------
    dict
        JSON-serializable hyperparameter snapshot.
    """
    return {
        "input_dim": input_dim,
        "units": units,
        "dropout_rate": dropout_rate,
        "learning_rate": learning_rate,
        "architecture": (
            f"Input({input_dim}) -> Dense({units}, relu) "
            f"-> Dropout({dropout_rate}) -> Dense(1, sigmoid)"
        ),
    }
