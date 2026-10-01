#!/usr/bin/env python3
"""
xsession_finetune.py  —  Phase 2 variant: cross-session supervised contrast.

The default Phase 2 fine-tunes on one session at a time and pulls each
embedding toward a frozen centroid of the previous session. SupCon positives
there are two augmentations of the *same slice*, so nothing ever tells the
network that two slices of one device recorded on different days are the
same thing — which is exactly the invariance that fails across sessions.

This variant pools every session seen so far and adds a contrastive term
whose positives are the same device from a *different* session:

    pos(i, j)  =  y_i == y_j  and  session_i != session_j

Batches are built in pairs so every anchor has at least one cross-session
positive. Everything else (backbone, two-stage schedule, BN adaptation,
augmentation, z-scoring) is the baseline Phase 2.

Model selection uses the quantity Phase 4 actually measures: nearest-centroid
accuracy on the current session's held-out split with centroids built from
the *older* sessions only. The plain target-day softmax accuracy is logged
beside it for reference.

Enabled with T2R_XSESSION=1. Knobs: T2R_XS_LAMBDA (0.5), T2R_XS_TEMP (0.10).
"""
from __future__ import annotations

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import CosineDecay

import rf_models
import tta_bn
from finetune import (preprocess, compute_zscore, apply_zscore, _smooth_ce,
                      transfer_weights)

XS_LAMBDA = float(os.environ.get('T2R_XS_LAMBDA', '0.5'))
XS_TEMP   = float(os.environ.get('T2R_XS_TEMP',   '0.10'))


def xsession_supcon(emb: tf.Tensor, y: tf.Tensor, s: tf.Tensor,
                    temperature: float = XS_TEMP) -> tf.Tensor:
    """SupCon over a batch where a positive is the same device on another session."""
    emb  = tf.math.l2_normalize(emb, axis=1)
    sim  = tf.matmul(emb, emb, transpose_b=True) / temperature
    n    = tf.shape(emb)[0]
    eye  = tf.eye(n, dtype=tf.bool)
    pos  = tf.logical_and(tf.equal(y[:, None], y[None, :]),
                          tf.not_equal(s[:, None], s[None, :]))
    pos  = tf.logical_and(pos, tf.logical_not(eye))
    logits  = tf.where(eye, tf.fill(tf.shape(sim), -1e9), sim)
    logprob = logits - tf.reduce_logsumexp(logits, axis=1, keepdims=True)
    pos_f   = tf.cast(pos, tf.float32)
    n_pos   = tf.reduce_sum(pos_f, axis=1)
    per_i   = -tf.reduce_sum(logprob * pos_f, axis=1) / tf.maximum(n_pos, 1.0)
    has_pos = tf.cast(n_pos > 0, tf.float32)
    return tf.reduce_sum(per_i * has_pos) / tf.maximum(tf.reduce_sum(has_pos), 1.0)


class _PairSampler:
    """Anchor + cross-session positive of the same device, half a batch each."""

    def __init__(self, y: np.ndarray, s: np.ndarray, rng: np.random.Generator):
        self.rng = rng
        self.y, self.s = y, s
        self.by = {}
        for i, (yi, si) in enumerate(zip(y, s)):
            self.by.setdefault(int(yi), {}).setdefault(int(si), []).append(i)
        self.by = {d: {k: np.asarray(v) for k, v in ss.items()} for d, ss in self.by.items()}
        # anchors are only samples whose device exists on another session
        self.anchor_pool = np.array([
            i for i, (yi, si) in enumerate(zip(y, s))
            if len(self.by[int(yi)]) > 1], dtype=np.int64)
        assert len(self.anchor_pool) > 0, 'no device appears on two sessions'

    def batch(self, half: int) -> np.ndarray:
        a = self.rng.choice(self.anchor_pool, size=half, replace=len(self.anchor_pool) < half)
        p = np.empty_like(a)
        for k, i in enumerate(a):
            sess = self.by[int(self.y[i])]
            other = [t for t in sess if t != int(self.s[i])]
            pool = sess[other[self.rng.integers(len(other))]]
            p[k] = pool[self.rng.integers(len(pool))]
        return np.concatenate([a, p])


def _centroid_acc(emb_ref, y_ref, emb_q, y_q, num_class):
    c = np.zeros((num_class, emb_ref.shape[1]), np.float32)
    for k in range(num_class):
        m = y_ref == k
        if m.any():
            c[k] = emb_ref[m].mean(0)
    c /= np.linalg.norm(c, axis=1, keepdims=True) + 1e-8
    q = emb_q / (np.linalg.norm(emb_q, axis=1, keepdims=True) + 1e-8)
    return float(np.mean(np.argmax(q @ c.T, axis=1) == y_q))


def xsession_tune(cnn, X_old, y_old, s_old, X_cur, y_cur, s_cur,
                  X_te_cur, y_te_cur, NUM_CLASS, model_dir, log=print):
    """Fine-tune on pooled sessions with cross-session SupCon.

    X_old/y_old/s_old : training slices of every earlier session (session id per slice)
    X_cur/y_cur/s_cur : training slices of the session being fine-tuned to
    X_te_cur/y_te_cur : held-out slices of that session, reported only
    Returns the same dict keys phase2 expects from CNN.tune.
    """
    rng = np.random.default_rng(int(np.random.get_state()[1][0]))
    os.makedirs(model_dir, exist_ok=True)
    save_path = os.path.join(model_dir, f'tune_model_{NUM_CLASS}.keras')

    y_cur = y_cur.astype(np.int32); y_old = y_old.astype(np.int32)

    # stratified 15% of the current session held out for selection (as baseline)
    tr_idx, val_idx = [], []
    for cls in np.unique(y_cur):
        idx = np.flatnonzero(y_cur == cls)
        idx = idx[rng.permutation(len(idx))]
        nv = max(1, int(np.ceil(len(idx) * 0.15)))
        val_idx += idx[:nv].tolist(); tr_idx += idx[nv:].tolist()
    tr_idx = np.asarray(tr_idx); val_idx = np.asarray(val_idx)
    X_val, y_val = X_cur[val_idx], y_cur[val_idx]

    X_pool = np.concatenate([X_old, X_cur[tr_idx]])
    y_pool = np.concatenate([y_old, y_cur[tr_idx]]).astype(np.int32)
    s_pool = np.concatenate([s_old, s_cur[tr_idx]]).astype(np.int32)
    n_old  = len(X_old)
    log(f'  [xs] pooled train={len(X_pool)} from sessions {sorted(set(s_pool.tolist()))}'
        f'  (older={n_old}, current={len(tr_idx)})  val={len(val_idx)}'
        f'  lambda={XS_LAMBDA}  T={XS_TEMP}')

    # z-score on the pooled train set
    X_pool_pp = preprocess(X_pool, augment=False)
    zm, zs = compute_zscore(X_pool_pp)
    X_pool_ppz = apply_zscore(X_pool_pp, zm, zs)
    X_val_ppz  = apply_zscore(preprocess(X_val.copy(), augment=False), zm, zs)
    y_val_cat  = to_categorical(y_val, NUM_CLASS)

    # Stage 0 / 0.5 : pretrained weights, BN statistics adapted on the pool
    ref_model = cnn._load_pretrained(cnn.phase1_model_path if cnn.always_reinit
                                     else cnn.train_model_path)
    mk = lambda: rf_models.create_model(
        model_type=cnn.model_type, inp_shape=cnn.input_shape, num_class=NUM_CLASS,
        emb_size=cnn.emb_size, classification=True, activation=cnn.activation,
        dropout_conv=cnn.dropout_conv, dropout_fc=cnn.dropout_fc, l2_reg=cnn.l2_reg)
    adapt = mk(); adapt.compile(loss='categorical_crossentropy', optimizer='adam')
    transfer_weights(ref_model, adapt, freeze_until_name='fc1_drop', keep_bn_trainable=True)
    tta_bn.adapt_bn_statistics(adapt, X_pool_ppz, batch_size=cnn.batch_size, n_passes=3)
    tta_bn.reset_bn_statistics(ref_model, tta_bn.save_bn_statistics(adapt))

    model = mk()
    transfer_weights(ref_model, model, freeze_until_name='fc1_drop', keep_bn_trainable=True)
    emb_name = 'emb_l2norm' if any(l.name == 'emb_l2norm' for l in model.layers) else 'embedding'
    emb_sub  = tf.keras.Model(model.input, model.get_layer(emb_name).output)

    def xs_val_acc():
        """Centroids from older sessions, queries from the current session's val."""
        e_ref = emb_sub.predict(X_pool_ppz[:n_old], batch_size=256, verbose=0)
        e_q   = emb_sub.predict(X_val_ppz, batch_size=256, verbose=0)
        return _centroid_acc(e_ref, y_pool[:n_old], e_q, y_val, NUM_CLASS)

    def softmax_val_acc():
        p = model.predict(X_val_ppz, batch_size=256, verbose=0)
        return float(np.mean(np.argmax(p, 1) == y_val))

    sampler = _PairSampler(y_pool, s_pool, rng)
    half    = cnn.batch_size // 2
    n_batch = int(np.ceil(len(X_pool) / cnn.batch_size))

    # Stage 1 : head warm-up, frozen backbone, plain CE on the pool
    log('  [xs] stage 1: head warm-up on pooled sessions')
    opt1 = Adam(cnn.stage1_lr); best1 = -1.0; wait = 0
    for ep in range(cnn.stage1_epochs):
        Xa = apply_zscore(preprocess(X_pool.copy(), augment=True), zm, zs)
        perm = rng.permutation(len(Xa)); tot = 0.0
        for b in range(n_batch):
            sl = perm[b * cnn.batch_size:(b + 1) * cnn.batch_size]
            xb = tf.constant(Xa[sl]); yb = tf.constant(to_categorical(y_pool[sl], NUM_CLASS))
            with tf.GradientTape() as tape:
                ce = _smooth_ce(yb, model(xb, training=True), 0.05)
            g = tape.gradient(ce, model.trainable_variables)
            g, _ = tf.clip_by_global_norm(g, 5.0)
            opt1.apply_gradients(zip(g, model.trainable_variables)); tot += float(ce)
        v = softmax_val_acc()
        print(f'  Stage1 Epoch {ep+1:3d}/{cnn.stage1_epochs}  ce={tot/n_batch:.4f}  val_acc={v:.4f}')
        if v > best1:
            best1 = v; wait = 0; model.save_weights(save_path.replace('.keras', '_s1_best.weights.h5'))
        else:
            wait += 1
            if wait >= cnn.patience:
                break
    model.load_weights(save_path.replace('.keras', '_s1_best.weights.h5'))

    # Stage 2 : all layers, CE + cross-session SupCon on paired batches
    log('  [xs] stage 2: CE + cross-session SupCon')
    for l in model.layers:
        l.trainable = True
    sched = CosineDecay(cnn.stage2_lr, cnn.stage2_epochs * n_batch, alpha=1e-7)
    opt2 = Adam(cnn.stage2_lr); step = 0
    best_xs = xs_val_acc(); best_sm = softmax_val_acc(); wait = 0
    log(f'  [xs] before stage 2: xs_centroid_val={best_xs:.4f}  softmax_val={best_sm:.4f}')
    model.save_weights(save_path.replace('.keras', '_s2_best.weights.h5'))
    for ep in range(cnn.stage2_epochs):
        Xa = apply_zscore(preprocess(X_pool.copy(), augment=True), zm, zs)
        tot_ce = tot_xs = 0.0
        for b in range(n_batch):
            sl = sampler.batch(half)
            xb = tf.constant(Xa[sl])
            yb_i = tf.constant(y_pool[sl], tf.int32); sb = tf.constant(s_pool[sl], tf.int32)
            yb_c = tf.constant(to_categorical(y_pool[sl], NUM_CLASS))
            opt2.learning_rate.assign(float(sched(step)))
            with tf.GradientTape() as tape:
                logits = model(xb, training=True)
                ce = _smooth_ce(yb_c, logits, 0.05)
                xs = xsession_supcon(emb_sub(xb, training=True), yb_i, sb)
                total = ce + XS_LAMBDA * xs
            g = tape.gradient(total, model.trainable_variables)
            g, _ = tf.clip_by_global_norm(g, 5.0)
            opt2.apply_gradients(zip(g, model.trainable_variables))
            tot_ce += float(ce); tot_xs += float(xs); step += 1
        v_xs = xs_val_acc(); v_sm = softmax_val_acc()
        print(f'  Stage2 Epoch {ep+1:3d}/{cnn.stage2_epochs}  ce={tot_ce/n_batch:.4f}  '
              f'xs={tot_xs/n_batch:.4f}  xs_centroid_val={v_xs:.4f}  softmax_val={v_sm:.4f}')
        if v_xs > best_xs:
            best_xs = v_xs; best_sm = v_sm; wait = 0
            model.save_weights(save_path.replace('.keras', '_s2_best.weights.h5'))
        else:
            wait += 1
            if wait >= cnn.patience:
                print(f'  [Stage 2] Early stop  best xs_centroid_val={best_xs:.4f}')
                break
    model.load_weights(save_path.replace('.keras', '_s2_best.weights.h5'))
    model.save(save_path)
    log(f'  [xs] selected: xs_centroid_val={best_xs:.4f}  softmax_val={best_sm:.4f}')

    # held-out slices of the current session, for the phase-2 log line
    X_te_ppz = apply_zscore(preprocess(X_te_cur.copy(), augment=False), zm, zs)
    p = model.predict(X_te_ppz, batch_size=256, verbose=0)
    y_te = np.argmax(y_te_cur, 1) if np.asarray(y_te_cur).ndim == 2 else y_te_cur
    return {
        'softmax':        float(np.mean(np.argmax(p, 1) == y_te)),
        'bn_adapted':     float('nan'),
        'proto_agg1':     float('nan'),
        'xs_centroid_val': best_xs,
        'zscore_mean':    zm,
        'zscore_std':     zs,
    }
