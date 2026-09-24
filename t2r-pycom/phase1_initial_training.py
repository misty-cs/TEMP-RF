#!/usr/bin/env python3
"""
phase1_initial_training.py — Phase 1: Initial training on cfg.init_day.

KEY IMPROVEMENTS FOR STABLE CROSS-DAY EMBEDDINGS
-------------------------------------------------
1. Supervised Contrastive Loss (SupCon) added alongside CE loss.
   SupCon directly pulls same-device embeddings together and pushes
   different-device embeddings apart on the unit hypersphere.
   This is the single most effective change for cross-day stability:
   a model trained with SupCon produces tight, well-separated clusters
   that drift less across days because the geometry is explicitly enforced.

2. Two-view augmentation per sample for SupCon.
   Each slice is augmented twice with different random channel conditions.
   The model must map both views of the same device to nearby embeddings,
   which forces it to learn channel-invariant features from Day 1.

3. Combined loss = CE + lambda * SupCon (default lambda=0.5).
   CE keeps the classifier head accurate; SupCon shapes the embedding space.

4. Larger emb_size=128 (set in experiment_config.py) gives more capacity
   to separate 16 classes with tight clusters.

5. Cosine LR warmup + decay for more stable training.

Train on all_known_ids (16 devices) from the start.
"""

from __future__ import annotations

import os
import time

import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import CosineDecay

from experiment_config import ExperimentConfig
from data_utils import load_day_raw
import load_slice_IQ
import signal_preprocessing as sp
import rf_models


BATCH_SIZE   = 128
EPOCHS       = 100      # SupCon needs longer to converge than plain CE
LR           = 5e-4      # slightly lower — SupCon gradients are larger
WARMUP_STEPS = 200       # linear LR warmup scaled with longer training
PATIENCE     = 40        # more patience for the combined loss
N_INIT_RESTARTS = 3      # best-of-N random inits (probe briefly, keep best)
PROBE_EPOCHS    = 12     # epochs per init probe before selection
CROSS_CAPTURE_POS = os.environ.get('T2R_CROSS_CAPTURE_POS', '0') == '1'
SUPCON_LAMBDA = 0.5      # weight of SupCon loss relative to CE
TEMPERATURE   = 0.07     # SupCon temperature (lower = sharper clusters)


def shuffle_data(X, y):
    idx = np.random.permutation(len(X))
    return X[idx], y[idx]


# ---------------------------------------------------------------------------
# Supervised Contrastive Loss
# ---------------------------------------------------------------------------

def supervised_contrastive_loss(
    embeddings: tf.Tensor,
    labels:     tf.Tensor,
    temperature: float = TEMPERATURE,
    captures:   tf.Tensor = None,
) -> tf.Tensor:
    """
    SupCon loss over a batch of L2-normalised embeddings.

    For each anchor, positives = same-class embeddings in the batch,
    negatives = all other embeddings.

    embeddings : (N, D)  already L2-normalised
    labels     : (N,)    integer class labels
    """
    # Cosine similarity matrix (already normalised → dot product)
    sim = tf.matmul(embeddings, embeddings, transpose_b=True) / temperature
    # (N, N)

    N = tf.shape(embeddings)[0]

    # Mask: 1 where same class (excluding self)
    labels_col = tf.reshape(labels, [-1, 1])
    labels_row = tf.reshape(labels, [1, -1])
    pos_mask   = tf.cast(tf.equal(labels_col, labels_row), tf.float32)
    self_mask  = tf.eye(N, dtype=tf.float32)
    pos_mask   = pos_mask - self_mask      # remove diagonal

    if captures is not None:
        # Cross-capture positives only. With slices drawn from one short
        # window per capture, same-capture positives are near-duplicates, so
        # the loss is minimised by encoding the capture's channel state rather
        # than the device. Requiring positives to come from a different
        # capture makes invariance across captures the training objective.
        cap_col = tf.reshape(captures, [-1, 1])
        cap_row = tf.reshape(captures, [1, -1])
        diff_capture = tf.cast(tf.not_equal(cap_col, cap_row), tf.float32)
        pos_mask = pos_mask * diff_capture

    # For numerical stability subtract max from each row
    sim_max   = tf.stop_gradient(tf.reduce_max(sim, axis=1, keepdims=True))
    sim_exp   = tf.exp(sim - sim_max)
    sim_exp   = sim_exp * (1.0 - self_mask)   # zero out self

    # Log-softmax denominator = sum over all non-self
    denom = tf.reduce_sum(sim_exp, axis=1, keepdims=True) + 1e-8

    # log prob of each positive pair
    log_prob = (sim - sim_max) - tf.math.log(denom)

    # Average over positives
    n_positives = tf.reduce_sum(pos_mask, axis=1)           # (N,)
    has_pos     = tf.cast(n_positives > 0, tf.float32)
    mean_log_p  = tf.reduce_sum(pos_mask * log_prob, axis=1) \
                  / (n_positives + 1e-8)

    loss = -tf.reduce_sum(has_pos * mean_log_p) / (tf.reduce_sum(has_pos) + 1e-8)
    return loss


# ---------------------------------------------------------------------------
# Augment a batch twice for two-view SupCon training
# ---------------------------------------------------------------------------

def _augment_two_views(X_raw: np.ndarray, cfg: ExperimentConfig):
    """
    Return two independently augmented versions of the same batch.
    Both views go through CFO removal + RMS norm + random augmentation.
    """
    view1 = sp.preprocess_batch(
        X_raw,
        do_remove_cfo      = cfg.use_preprocessing,
        do_per_slice_norm  = cfg.use_preprocessing,
        augment            = True,
        phase_rot_range    = cfg.aug_phase_rot,
        amp_jitter_db      = cfg.aug_amp_db,
        noise_snr_db       = cfg.aug_snr_db,
        augment_apply_prob = cfg.aug_prob,
        multipath_taps     = cfg.aug_multipath_taps,
        multipath_mag      = cfg.aug_multipath_mag,
    )
    view2 = sp.preprocess_batch(
        X_raw,
        do_remove_cfo      = cfg.use_preprocessing,
        do_per_slice_norm  = cfg.use_preprocessing,
        augment            = True,
        phase_rot_range    = cfg.aug_phase_rot,
        amp_jitter_db      = cfg.aug_amp_db,
        noise_snr_db       = cfg.aug_snr_db,
        augment_apply_prob = cfg.aug_prob,
        multipath_taps     = cfg.aug_multipath_taps,
        multipath_mag      = cfg.aug_multipath_mag,
    )
    return view1, view2


# ---------------------------------------------------------------------------
# Main Phase 1 training
# ---------------------------------------------------------------------------

def run_phase1(cfg: ExperimentConfig) -> dict:
    print('\n' + '=' * 62)
    print(f'  PHASE 1 — Initial training on Day {cfg.init_day} (CE + SupCon)')
    print(f'  Devices: all_known_ids = {cfg.all_known_ids}  ({len(cfg.all_known_ids)} classes)')
    print(f'  emb_size={cfg.emb_size}  supcon_lambda={SUPCON_LAMBDA}  T={TEMPERATURE}')
    print('=' * 62)

    np.random.seed(cfg.seed)
    tf.random.set_seed(cfg.seed)

    # ── Load initial day as raw IQ. This phase owns preprocessing,
    # augmentation, and z-score fitting, so load_day must not normalize.
    _layout = getattr(cfg, 'dataset_layout', 'folder_iq')
    capture_split = (getattr(cfg, 'ft_split_mode', 'random') == 'capture'
                     or _layout == 'wisig_full')
    if _layout == 'wisig_full':
        # WiSig groups are bursts of near-identical repeat transmissions.
        from data_utils import wisig_full_day_with_groups
        X_all_raw, y_all, g_all = wisig_full_day_with_groups(
            cfg, cfg.init_day, cfg.all_known_ids)
    elif capture_split:
        # Three-way split by capture provenance:
        #   train  - slices from the training captures
        #   val    - held-out slices from those SAME captures (different offsets)
        #   test   - slices from captures never seen in training
        # Reporting val and test from one model separates "remembers this
        # capture" from "recognises this device": the gap between them is the
        # capture-memorisation effect.
        from data_utils import load_day_raw_with_groups
        X_all_raw, y_all, g_all = load_day_raw_with_groups(
            cfg, day_id=cfg.init_day, device_ids=cfg.all_known_ids
        )
    else:
        X_all_raw, y_all = load_day_raw(
            cfg, day_id=cfg.init_day, device_ids=cfg.all_known_ids
        )
        g_all = None
    X_tr_parts, y_tr_parts = [], []
    X_val_parts, y_val_parts = [], []
    X_test_parts, y_test_parts = [], []
    g_tr_parts = []
    for cls in np.unique(y_all):
        idx = np.where(y_all == cls)[0]
        if _layout == 'wisig_full':
            # burst-disjoint split: 15% of bursts to val, 15% to test
            groups = g_all[idx]
            uniq = np.unique(groups)
            rs = np.random.RandomState(cfg.seed * 1000 + int(cls))
            rs.shuffle(uniq)
            n_val_g  = max(1, int(0.15 * len(uniq)))
            n_test_g = max(1, int(0.15 * len(uniq)))
            val_g  = set(uniq[:n_val_g].tolist())
            test_g = set(uniq[n_val_g:n_val_g + n_test_g].tolist())
            val_idx  = idx[np.isin(groups, list(val_g))]
            test_idx = idx[np.isin(groups, list(test_g))]
            tr_idx   = idx[~np.isin(groups, list(val_g | test_g))]
            if min(len(tr_idx), len(val_idx), len(test_idx)) == 0:
                raise RuntimeError(f'Class {cls}: burst split left an empty part '
                                   f'({len(uniq)} bursts).')
        elif capture_split:
            files = g_all[idx, 0]
            uniq = np.unique(files)
            n_hold = max(1, min(2, len(uniq) - 2))
            if len(uniq) - n_hold < 1:
                raise RuntimeError(
                    f"Class {cls} has only {len(uniq)} capture(s); a "
                    "capture-disjoint Phase-1 split needs at least 2."
                )
            rs = np.random.RandomState(cfg.seed * 1000 + int(cls))
            held = set(rs.choice(uniq, size=n_hold, replace=False).tolist())
            test_idx = idx[np.isin(files, list(held))]
            rest = idx[~np.isin(files, list(held))]
            rest = rest[np.random.permutation(len(rest))]
            n_val = max(1, int(len(rest) * 0.15))
            val_idx = rest[:n_val]
            tr_idx = rest[n_val:]
            if len(tr_idx) == 0 or len(test_idx) == 0:
                raise RuntimeError(f"Class {cls}: empty split after capture holdout.")
        else:
            idx = idx[np.random.permutation(len(idx))]
            n_total = len(idx)
            n_train = int(n_total * 0.7)
            n_val = int(n_total * 0.15)
            n_test = n_total - n_train - n_val
            if n_train <= 0 or n_val <= 0 or n_test <= 0:
                raise RuntimeError(
                    f"Class {cls} has only {n_total} samples; Phase 1 needs enough "
                    "samples for non-empty train/validation/test splits."
                )
            tr_idx = idx[:n_train]
            val_idx = idx[n_train:n_train + n_val]
            test_idx = idx[n_train + n_val:]
        X_tr_parts.append(X_all_raw[tr_idx])
        y_tr_parts.append(y_all[tr_idx])
        if g_all is not None:
            # global capture id: file index made unique per class
            gid = g_all[tr_idx] if g_all.ndim == 1 else g_all[tr_idx, 0]
            g_tr_parts.append(gid.astype(np.int64) * 1000 + int(cls))
        X_val_parts.append(X_all_raw[val_idx])
        y_val_parts.append(y_all[val_idx])
        X_test_parts.append(X_all_raw[test_idx])
        y_test_parts.append(y_all[test_idx])

    X_tr_raw = np.concatenate(X_tr_parts, axis=0)
    y_tr = np.concatenate(y_tr_parts, axis=0)
    g_tr = np.concatenate(g_tr_parts, axis=0) if g_tr_parts else None
    if g_tr is not None:
        print(f'  capture ids in training set: {len(np.unique(g_tr))} distinct '
              f'(cross-capture positives={CROSS_CAPTURE_POS})')
    X_val_raw = np.concatenate(X_val_parts, axis=0)
    y_val = np.concatenate(y_val_parts, axis=0)
    X_test_raw = np.concatenate(X_test_parts, axis=0)
    y_test = np.concatenate(y_test_parts, axis=0)
    NUM_CLASS = len(cfg.all_known_ids)

    print(
        f'\n  X_tr={X_tr_raw.shape}  X_val={X_val_raw.shape}  '
        f'X_test={X_test_raw.shape}  NUM_CLASS={NUM_CLASS}'
    )
    assert NUM_CLASS == len(cfg.all_known_ids)

    X_tr_pp_clean = sp.preprocess_batch(
        X_tr_raw,
        do_remove_cfo=cfg.use_preprocessing,
        do_per_slice_norm=cfg.use_preprocessing,
        augment=False,
    )
    norm_mean, norm_std = load_slice_IQ.compute_normalization_stats(X_tr_pp_clean)

    # For validation/test we use preprocessed (no augment) normalised data.
    X_val_pp = sp.preprocess_batch(
        X_val_raw,
        do_remove_cfo=cfg.use_preprocessing,
        do_per_slice_norm=cfg.use_preprocessing,
        augment=False,
    )
    X_val_pp = load_slice_IQ.apply_normalization(X_val_pp, norm_mean, norm_std)
    y_val_cat = to_categorical(y_val, NUM_CLASS)

    X_test_pp = sp.preprocess_batch(
        X_test_raw,
        do_remove_cfo=cfg.use_preprocessing,
        do_per_slice_norm=cfg.use_preprocessing,
        augment=False,
    )
    X_test_pp = load_slice_IQ.apply_normalization(X_test_pp, norm_mean, norm_std)
    y_test_cat = to_categorical(y_test, NUM_CLASS)

    # Save normalisation stats
    norm_path = os.path.join(cfg.model_dir, f'phase1_norm_d{cfg.init_day}.npz')
    np.savez(norm_path, mean=norm_mean, std=norm_std)
    print(f'  Norm stats saved → {norm_path}')

    inp_shape = (X_tr_raw.shape[1], 2)

    # ── Build model ────────────────────────────────────────────────────────
    def _build_model():
        return rf_models.create_model(
            model_type     = cfg.model_type,
            inp_shape      = inp_shape,
            num_class      = NUM_CLASS,
            emb_size       = cfg.emb_size,
            classification = True,
            activation     = cfg.activation,
            dropout_conv   = cfg.dropout_conv,
            dropout_fc     = cfg.dropout_fc,
            l2_reg         = cfg.l2_reg,
        )

    def _make_emb_model(m):
        # Embedding sub-model for SupCon (taps emb_l2norm layer)
        names = [l.name for l in m.layers]
        emb_name = 'emb_l2norm' if 'emb_l2norm' in names else 'embedding'
        return tf.keras.Model(
            inputs  = m.input,
            outputs = m.get_layer(emb_name).output,
            name    = 'emb_model',
        )

    model     = _build_model()
    emb_model = _make_emb_model(model)

    model_path = os.path.join(cfg.model_dir, f'phase1_df_d{cfg.init_day}.keras')
    val_model_dir = os.path.join(cfg.model_dir, 'phase1_validation_models')
    test_model_dir = os.path.join(cfg.model_dir, 'phase1_test_models')
    os.makedirs(val_model_dir, exist_ok=True)
    os.makedirs(test_model_dir, exist_ok=True)
    val_model_path = os.path.join(
        val_model_dir, f'phase1_val_d{cfg.init_day}.keras'
    )
    test_model_path = os.path.join(
        test_model_dir, f'phase1_test_d{cfg.init_day}.keras'
    )

    # LR schedule: linear warmup then cosine decay
    total_steps  = EPOCHS * int(np.ceil(len(X_tr_raw) / BATCH_SIZE))
    lr_schedule  = CosineDecay(
        initial_learning_rate = LR,
        decay_steps           = total_steps - WARMUP_STEPS,
        alpha                 = 1e-6,
    )
    optimizer = Adam(learning_rate=LR)

    # ── Shared epoch runner (used by init probe and main loop) ────────────
    N_tr    = len(X_tr_raw)
    n_batch = int(np.ceil(N_tr / BATCH_SIZE))

    def _train_one_epoch(mdl, emb_mdl, opt, lr_fn, step0):
        """One epoch of CE + SupCon. Returns (mean_ce, mean_supcon, next_step)."""
        perm   = np.random.permutation(N_tr)
        X_shuf = X_tr_raw[perm]
        y_shuf = y_tr[perm]
        c_shuf = g_tr[perm] if g_tr is not None else None

        epoch_ce     = 0.0
        epoch_supcon = 0.0
        step         = step0

        for b in range(n_batch):
            sl      = slice(b * BATCH_SIZE, (b + 1) * BATCH_SIZE)
            Xb_raw  = X_shuf[sl]
            yb_int  = y_shuf[sl]
            cb_int  = c_shuf[sl] if c_shuf is not None else None

            opt.learning_rate.assign(lr_fn(step))
            step += 1

            # Two augmented views
            v1, v2  = _augment_two_views(Xb_raw, cfg)
            v1 = load_slice_IQ.apply_normalization(v1, norm_mean, norm_std)
            v2 = load_slice_IQ.apply_normalization(v2, norm_mean, norm_std)

            yb_cat  = to_categorical(yb_int, NUM_CLASS).astype(np.float32)
            yb_tf   = tf.constant(yb_int,  dtype=tf.int32)
            v1_tf   = tf.constant(v1,      dtype=tf.float32)
            v2_tf   = tf.constant(v2,      dtype=tf.float32)
            yb_cat_tf = tf.constant(yb_cat, dtype=tf.float32)

            with tf.GradientTape() as tape:
                # CE loss on view 1
                logits  = mdl(v1_tf, training=True)
                ce_loss = tf.reduce_mean(
                    tf.keras.losses.categorical_crossentropy(yb_cat_tf, logits)
                )

                # SupCon: concatenate embeddings from both views
                emb1    = emb_mdl(v1_tf, training=True)
                emb2    = emb_mdl(v2_tf, training=True)
                emb_cat = tf.concat([emb1, emb2], axis=0)
                y_cat_2 = tf.concat([yb_tf, yb_tf], axis=0)
                if cb_int is not None and CROSS_CAPTURE_POS:
                    cb_tf   = tf.constant(cb_int, dtype=tf.int32)
                    c_cat_2 = tf.concat([cb_tf, cb_tf], axis=0)
                else:
                    c_cat_2 = None
                sc_loss = supervised_contrastive_loss(emb_cat, y_cat_2, TEMPERATURE,
                                                      captures=c_cat_2)

                total_loss = ce_loss + SUPCON_LAMBDA * sc_loss

            grads = tape.gradient(total_loss, mdl.trainable_variables)
            # Gradient clipping for stability
            grads, _ = tf.clip_by_global_norm(grads, 5.0)
            opt.apply_gradients(zip(grads, mdl.trainable_variables))

            epoch_ce     += float(ce_loss)
            epoch_supcon += float(sc_loss)

        return epoch_ce / n_batch, epoch_supcon / n_batch, step

    def _val_accuracy(mdl):
        preds = mdl.predict(X_val_pp, batch_size=BATCH_SIZE, verbose=0)
        return float(np.mean(np.argmax(preds, axis=1) == y_val))

    # ── Init probe: best-of-N random restarts ─────────────────────────────
    # A bad random init can trap CE+SupCon in a poor optimum that the rest
    # of the pipeline (fine-tuning, trajectory, evaluation) never recovers
    # from. Train each candidate briefly, keep the best-validation weights.
    # Identical procedure for every seed; it improves the shared backbone
    # that all methods (baselines included) are evaluated on.
    if N_INIT_RESTARTS > 1:
        print(f'\n  Init probe: {N_INIT_RESTARTS} restarts × {PROBE_EPOCHS} epochs each ...')
        probe_lr_fn = lambda s: LR * min(1.0, (s + 1) / WARMUP_STEPS)
        best_probe_acc, best_probe_weights = -np.inf, None
        for r in range(N_INIT_RESTARTS):
            tf.random.set_seed(cfg.seed + 9973 * r)
            np.random.seed(cfg.seed + 9973 * r)
            cand     = model if r == 0 else _build_model()
            cand_emb = emb_model if r == 0 else _make_emb_model(cand)
            cand_opt = Adam(learning_rate=LR)
            pstep = 0
            for _pe in range(PROBE_EPOCHS):
                _, _, pstep = _train_one_epoch(cand, cand_emb, cand_opt,
                                               probe_lr_fn, pstep)
            acc = _val_accuracy(cand)
            print(f'    restart {r + 1}/{N_INIT_RESTARTS}: probe val_acc={acc:.4f}')
            if acc > best_probe_acc:
                best_probe_acc     = acc
                best_probe_weights = cand.get_weights()
        model.set_weights(best_probe_weights)
        print(f'  Init probe done — continuing from best init (val_acc={best_probe_acc:.4f})')
        # Restore the experiment seed for the main run
        np.random.seed(cfg.seed)
        tf.random.set_seed(cfg.seed)

    # ── Training loop ─────────────────────────────────────────────────────
    best_val_acc  = -np.inf
    no_improve    = 0
    global_step   = 0

    def _main_lr_fn(s):
        if s < WARMUP_STEPS:
            return LR * (s + 1) / WARMUP_STEPS
        return float(lr_schedule(s - WARMUP_STEPS))

    print(f'\n  Training for up to {EPOCHS} epochs (CE + SupCon) ...')
    t0 = time.time()

    for epoch in range(EPOCHS):
        epoch_ce, epoch_supcon, global_step = _train_one_epoch(
            model, emb_model, optimizer, _main_lr_fn, global_step
        )

        # Validation
        val_acc = _val_accuracy(model)

        if (epoch + 1) % 10 == 0 or epoch < 5:
            print(f'  Epoch {epoch+1:3d}/{EPOCHS}  '
                  f'ce={epoch_ce:.4f}  sc={epoch_supcon:.4f}  '
                  f'val_acc={val_acc:.4f}  '
                  f'best={best_val_acc:.4f}')

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            no_improve   = 0
            model.save(model_path)
        else:
            no_improve += 1

        if no_improve >= PATIENCE:
            print(f'  Early stop at epoch {epoch+1}  best_val_acc={best_val_acc:.4f}')
            break

    # Reload best
    model = tf.keras.models.load_model(
        model_path,
        custom_objects={'L2Normalize': rf_models.L2Normalize},
    )
    model.compile(
        optimizer = Adam(LR),
        loss      = 'categorical_crossentropy',
        metrics   = ['accuracy'],
    )
    _, val_acc = model.evaluate(X_val_pp, y_val_cat, batch_size=BATCH_SIZE, verbose=0)
    _, test_acc = model.evaluate(
        X_test_pp, y_test_cat, batch_size=BATCH_SIZE, verbose=0
    )
    model.save(val_model_path)
    model.save(test_model_path)
    duration = time.time() - t0

    print(
        f'\n  Phase 1 complete  val_acc={val_acc:.4f}  '
        f'test_acc={test_acc:.4f}  time={duration:.0f}s'
    )
    print(f'  Downstream Phase 1 model saved → {model_path}')
    print(f'  Validation model saved → {val_model_path}')
    print(f'  Test model saved → {test_model_path}')

    return {
        'model_path':      model_path,
        'norm_path':       norm_path,
        'val_model_path':  val_model_path,
        'test_model_path': test_model_path,
        'val_acc':         float(val_acc),
        'test_acc':        float(test_acc),
        'same_day_acc':    float(test_acc),
        'num_class':       NUM_CLASS,
    }
