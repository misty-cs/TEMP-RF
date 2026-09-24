#!/usr/bin/env python3
"""
measure_efficiency.py — Deployment-cost numbers for the paper.

Measures, from a completed seed's artifacts:
  1. Embedding inference latency per IQ slice (batch 1 and batch 128)
  2. Trajectory open-set classification latency per slice
  3. Prototype-bank memory footprint
  4. Per-day trajectory update time (EWMA + k-means prototype build)

    python measure_efficiency.py --seed_dir res_out_multiseed/seed_13
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument('--seed_dir', required=True)
    p.add_argument('--n_warm',   type=int, default=3)
    p.add_argument('--n_rep',    type=int, default=20)
    args = p.parse_args()

    import tensorflow as tf
    import rf_models
    from temporal_trajectory import TemporalTrajectory, extract_embeddings_from_model

    model_dir = os.path.join(args.seed_dir, 'modelDir')

    # Latest adapted model + trajectory
    traj_files = sorted(f for f in os.listdir(model_dir)
                        if f.startswith('phase3_trajectory_d'))
    if not traj_files:
        raise SystemExit('No phase3 trajectory found — run the pipeline first.')
    traj_day = int(traj_files[-1].split('_d')[1].split('.')[0])

    adapted = os.path.join(model_dir, f'phase3_bn_adapted_d{traj_day}.keras')
    if not os.path.exists(adapted):
        ft = sorted(f for f in os.listdir(model_dir)
                    if f.startswith('phase2_ft_d') and f.endswith('.keras'))
        adapted = os.path.join(model_dir, ft[-1])

    model = tf.keras.models.load_model(
        adapted, custom_objects={'L2Normalize': rf_models.L2Normalize})
    traj = TemporalTrajectory.load(
        os.path.join(model_dir, f'phase3_trajectory_d{traj_day}.npz'))
    known_ids = sorted(traj.known_device_ids)

    slice_len = model.input_shape[1]
    rng = np.random.default_rng(0)

    print(f'model: {os.path.basename(adapted)}   trajectory day: {traj_day}')
    print(f'devices in trajectory: {len(known_ids)}\n')

    # ── 1. Embedding latency ──────────────────────────────────────────────
    for bs in (1, 128):
        X = rng.normal(size=(bs, slice_len, 2)).astype(np.float32)
        for _ in range(args.n_warm):
            extract_embeddings_from_model(model, X, batch_size=bs)
        t0 = time.perf_counter()
        for _ in range(args.n_rep):
            extract_embeddings_from_model(model, X, batch_size=bs)
        dt = (time.perf_counter() - t0) / args.n_rep
        print(f'embedding latency  batch={bs:<4d}: '
              f'{dt*1e3:8.2f} ms/batch   {dt/bs*1e3:8.3f} ms/slice')

    # ── 2. Trajectory classification latency ─────────────────────────────
    emb_dim = None
    for dev in known_ids:
        proto, _ = traj.cosine_prototype_bank(dev)
        emb_dim = proto.shape[1]
        break
    Q = rng.normal(size=(1000, emb_dim)).astype(np.float32)
    Q /= np.linalg.norm(Q, axis=1, keepdims=True)
    for _ in range(args.n_warm):
        traj.classify_cosine_prototypes(Q, threshold=0.5, known_ids=known_ids)
    t0 = time.perf_counter()
    for _ in range(args.n_rep):
        traj.classify_cosine_prototypes(Q, threshold=0.5, known_ids=known_ids)
    dt = (time.perf_counter() - t0) / args.n_rep
    print(f'trajectory classify (1000 queries): '
          f'{dt*1e3:8.2f} ms   {dt:8.6f} ms/slice')

    # ── 3. Prototype-bank memory ──────────────────────────────────────────
    total_bytes, total_protos = 0, 0
    for dev in known_ids:
        proto, w = traj.cosine_prototype_bank(dev)
        total_bytes  += proto.nbytes + w.nbytes
        total_protos += len(proto)
    print(f'prototype bank: {total_protos} prototypes, '
          f'{total_bytes/1024:.1f} KiB total '
          f'({total_bytes/len(known_ids)/1024:.1f} KiB/device)')

    # ── 4. Per-day trajectory update time ────────────────────────────────
    n_per_dev = 1000
    emb_day = rng.normal(size=(n_per_dev*len(known_ids), emb_dim)).astype(np.float32)
    y_day   = np.repeat(known_ids, n_per_dev)
    t2 = TemporalTrajectory(ewma_alpha=0.8,
                            n_cosine_prototypes=traj.n_cosine_prototypes)
    t0 = time.perf_counter()
    t2.update(day_id=1, emb=emb_day, labels=y_day)
    dt = time.perf_counter() - t0
    print(f'trajectory update (1 day, {len(emb_day)} embeddings): {dt:.2f} s')

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
