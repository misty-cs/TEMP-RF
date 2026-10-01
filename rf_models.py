#!/usr/bin/env python3
"""
rf_models.py  —  Model definitions for RF fingerprinting.

Exports
-------
L2Normalize   — safe serialisable L2-norm Keras layer
create_DF     — build the DeepFingerprinting 1-D CNN
create_model  — factory wrapper (currently only 'DF' is supported)
"""

from __future__ import annotations

import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv1D, Dense, Dropout, Flatten,
    MaxPooling1D, BatchNormalization, ELU, Activation,
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model


# ---------------------------------------------------------------------------
# Custom L2-normalisation layer  (replaces Lambda for safe serialisation)
# ---------------------------------------------------------------------------

class L2Normalize(tf.keras.layers.Layer):
    """L2-normalise embeddings onto the unit hypersphere along axis=1.

    Cross-day amplitude drift shifts embedding vector *length* but not
    *direction*.  Normalising removes that length variation so same-device
    embeddings from different recording sessions still cluster together.
    """

    def call(self, inputs):
        return tf.math.l2_normalize(inputs, axis=1)

    def get_config(self):
        return super().get_config()


# ---------------------------------------------------------------------------
# DeepFingerprinting 1-D CNN
# ---------------------------------------------------------------------------

def create_DF(
    inp_shape,
    class_num:    int   = 10,
    emb_size:     int   = 64,
    classification: bool = True,
    activation:   str   = 'elu',
    dropout_conv: float = 0.3,
    dropout_fc:   float = 0.4,
    l2_reg:       float = 1e-4,
    l2_norm_emb:  bool  = True,
) -> Model:
    """
    Build the DeepFingerprinting 1-D CNN.

    Architecture
    ------------
    4 × conv-block  (Conv1D × 2 + BN × 2 + MaxPool + Dropout)
    Dense(128) bottleneck
    Dense(emb_size) embedding  [→ optional L2 normalisation]
    Dense(class_num, softmax)  [only when classification=True]

    Parameters
    ----------
    inp_shape     : (slice_len, n_channels)  — must be length-2 tuple
    class_num     : number of device classes
    emb_size      : embedding dimension (default 64)
    classification: if True, add softmax classifier head
    activation    : 'elu' or 'gelu'
    dropout_conv  : dropout rate after each conv block (default 0.3)
    dropout_fc    : dropout rate after Dense(128) (default 0.4)
    l2_reg        : L2 kernel regulariser weight (default 1e-4)
    l2_norm_emb   : project embedding onto unit hypersphere (default True)

    Returns
    -------
    tf.keras.Model
    """
    if len(inp_shape) != 2:
        raise ValueError(
            f"inp_shape must be (slice_len, channels), got {inp_shape}"
        )

    filters     = [32,  64,  128, 256]
    kernel_size = [8,   8,   8,   8  ]
    pool_size   = [8,   8,   8,   8  ]
    pool_stride = [4,   4,   4,   4  ]
    reg         = l2(l2_reg)

    def act(name: str):
        if activation == 'elu':
            return ELU(alpha=1.0, name=name)
        if activation == 'gelu':
            return Activation('gelu', name=name)
        raise ValueError(
            f"activation must be 'elu' or 'gelu', got {activation!r}"
        )

    def conv_block(x, filt, ksz, pool_sz, pool_str, block_id):
        x = Conv1D(filt, ksz, padding='same',
                   kernel_regularizer=reg, name=f'b{block_id}_conv1')(x)
        x = act(f'b{block_id}_act1')(x)
        x = BatchNormalization(name=f'b{block_id}_bn1')(x)
        x = Conv1D(filt, ksz, padding='same',
                   kernel_regularizer=reg, name=f'b{block_id}_conv2')(x)
        x = act(f'b{block_id}_act2')(x)
        x = BatchNormalization(name=f'b{block_id}_bn2')(x)
        x = MaxPooling1D(pool_size=pool_sz, strides=pool_str,
                         padding='same', name=f'b{block_id}_pool')(x)
        x = Dropout(dropout_conv, name=f'b{block_id}_drop')(x)
        return x

    inputs = Input(shape=inp_shape, name='iq_input')
    x = inputs
    for i in range(4):
        x = conv_block(
            x, filters[i], kernel_size[i],
            pool_size[i], pool_stride[i], block_id=i + 1,
        )

    x = Flatten(name='flatten')(x)
    x = Dense(128, kernel_regularizer=reg, name='fc1')(x)
    x = act('fc1_act')(x)
    x = Dropout(dropout_fc, name='fc1_drop')(x)

    x = Dense(emb_size, activation=None, name='embedding')(x)

    if l2_norm_emb:
        x = L2Normalize(name='emb_l2norm')(x)

    if classification:
        outputs = Dense(class_num, activation='softmax', name='classifier')(x)
    else:
        outputs = x

    return Model(inputs=inputs, outputs=outputs, name='DeepFingerprinting')


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_model(
    model_type: str,
    inp_shape,
    num_class:    int,
    emb_size:     int   = 64,
    classification: bool = True,
    activation:   str   = 'elu',
    dropout_conv: float = 0.3,
    dropout_fc:   float = 0.4,
    l2_reg:       float = 1e-4,
    l2_norm_emb:  bool  = True,
) -> Model:
    """
    Instantiate a model by name.

    Parameters
    ----------
    model_type : str   — currently only 'DF' is supported
    (all other kwargs forwarded to create_DF)

    Returns
    -------
    tf.keras.Model
    """
    print(
        f"Building model: {model_type}  inp_shape={inp_shape}  "
        f"num_class={num_class}  classification={classification}  "
        f"activation={activation}  dropout_conv={dropout_conv}  "
        f"dropout_fc={dropout_fc}  l2_reg={l2_reg}  "
        f"l2_norm_emb={l2_norm_emb}"
    )

    if model_type == 'DF':
        return create_DF(
            inp_shape      = inp_shape,
            class_num      = num_class,
            emb_size       = emb_size,
            classification = classification,
            activation     = activation,
            dropout_conv   = dropout_conv,
            dropout_fc     = dropout_fc,
            l2_reg         = l2_reg,
            l2_norm_emb    = l2_norm_emb,
        )

    raise ValueError(f"Unknown model type: {model_type!r}.  Supported: 'DF'")


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import numpy as np

    inp = (288, 2)

    for act in ('elu', 'gelu'):
        for norm in (True, False):
            print(f"\n=== activation={act}  l2_norm_emb={norm} ===")
            m = create_DF(inp_shape=inp, class_num=10,
                          classification=True, activation=act,
                          l2_norm_emb=norm)
            m.compile(optimizer='adam', loss='categorical_crossentropy',
                      metrics=['accuracy'])
            m.summary()

            dummy = np.random.randn(4, 288, 2).astype(np.float32)
            out   = m.predict(dummy, verbose=0)
            print(f"Output shape : {out.shape}")
            print(f"Row sums     : {out.sum(axis=1)}")

            emb_layer = 'emb_l2norm' if norm else 'embedding'
            emb_model = tf.keras.Model(
                inputs=m.input, outputs=m.get_layer(emb_layer).output
            )
            norms = np.linalg.norm(
                emb_model.predict(dummy, verbose=0), axis=1
            )
            print(f"Embedding norms (should be ~1 if l2_norm): {norms}")

    print("\nSelf-test passed.")
