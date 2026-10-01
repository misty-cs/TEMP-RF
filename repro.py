"""Reproducibility.

env(seed): the variables every launcher (main.py, tune_phase*.py) puts in the
environment of EVERY subprocess of a run (training, evaluation, audit), so they
are set before interpreter start and recorded identically in all manifests.
set_all(seed): call once in each process, before any TensorFlow computation.
"""
import os, random


def env(seed: int) -> dict:
    return {'PYTHONHASHSEED': str(seed), 'TF_DETERMINISTIC_OPS': '1'}


def set_all(seed: int) -> None:
    if os.environ.get('PYTHONHASHSEED') != str(seed):
        print(f'[repro] WARNING PYTHONHASHSEED={os.environ.get("PYTHONHASHSEED")} at start, expected {seed} '
              '(set it before interpreter start: launch through main.py / tune_phase*.py)')
    os.environ.update(env(seed))
    random.seed(seed)
    import numpy as np
    np.random.seed(seed)
    import tensorflow as tf
    tf.random.set_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception as e:                        # older TF: fall back to the env flag
        print(f'[repro] op determinism not available: {e}')
