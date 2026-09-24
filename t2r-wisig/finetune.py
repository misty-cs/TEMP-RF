#!/usr/bin/env python3
"""
finetune.py  —  Cross-day few-shot fine-tuning for RF fingerprinting.

KEY CHANGES FROM PREVIOUS VERSION
-----------------------------------
FIX 1: Removed cross-day contrastive loss from Stage 1.
        The contrastive loss paired Day N-1 embeddings (pretrained model)
        with Day N embeddings (fine-tuning model) which live in different
        embedding spaces — producing noisy, conflicting gradients that
        degraded accuracy from 76% to ~28%.  Stage 1 now uses plain CE
        with frozen backbone, which is simpler and more reliable.

FIX 2: Unfroze embedding layer in Stage 2.
        Previously the embedding and emb_l2norm layers were frozen in
        Stage 2 to "preserve Stage 1 alignment."  Since Stage 1 no longer
        does cross-day alignment, there is no reason to freeze them — the
        embedding must adapt to the new day's distribution.

FIX 3: Re-initialize from Phase 1 model each fine-tune step (controlled
        by always_reinit flag, default True for this non-chronological split).
        When always_reinit=True, each day's fine-tuning starts from the
        original strong Phase 1 baseline instead of the previous day's
        (potentially degraded) model.

FIX 4: Increased default ft_n_train to 1600/class and validation split
        to 15% for more reliable early stopping.

FIX 5: Prototype replay centroids now use the CURRENT day's fine-tuned
        model embeddings (not the pretrained model's Day N-1 embeddings),
        so the anchor is consistent with the model being trained.
"""

from __future__ import annotations

import os
import random
from collections import defaultdict

import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import CosineDecay

import rf_models
import signal_preprocessing as sp
import tta_bn
import embedding_whitening as ew


# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_RES_DIR     = os.path.join(_CURRENT_DIR, 'res_out')
_MODEL_DIR   = os.path.join(_RES_DIR, 'modelDir')
os.makedirs(_MODEL_DIR, exist_ok=True)

_RESULTS_TXT = os.path.join(_RES_DIR, 'results_finetune.txt')


# ------------------------------------------------------------------
# Results logging
# ------------------------------------------------------------------

def _write_results(line: str, also_print: bool = False) -> None:
    with open(_RESULTS_TXT, 'a') as f:
        print(line, file=f, flush=True)
    if also_print:
        print(line)


# ------------------------------------------------------------------
# Preprocessing helpers
# ------------------------------------------------------------------

# Multipath augmentation for the fine-tuning path. Set by run_experiment from
# the config so phase 2 augments with the same channel model as phase 1;
# leaving these at 0 reproduces the original flat-channel augmentation.
MULTIPATH_TAPS = 0
MULTIPATH_MAG  = 0.0


def preprocess(X: np.ndarray, augment: bool = False) -> np.ndarray:
    return sp.preprocess_batch(
        X,
        do_remove_cfo      = True,
        do_per_slice_norm  = True,
        augment            = augment,
        phase_rot_range    = np.pi,
        amp_jitter_db      = 3.0,
        noise_snr_db       = 25.0,
        augment_apply_prob = 0.7,
        multipath_taps     = MULTIPATH_TAPS,
        multipath_mag      = MULTIPATH_MAG,
    )


def compute_zscore(X: np.ndarray):
    mean_ = X.mean(axis=(0, 1), keepdims=True).astype(np.float32)
    std_  = (X.std(axis=(0, 1), keepdims=True) + 1e-8).astype(np.float32)
    return mean_, std_


def apply_zscore(X: np.ndarray, mean_: np.ndarray, std_: np.ndarray) -> np.ndarray:
    return ((X - mean_) / std_).astype(np.float32)


def _smooth_ce(
    y_cat: tf.Tensor,
    logits: tf.Tensor,
    smoothing: float = 0.05,
) -> tf.Tensor:
    """Categorical cross-entropy with label smoothing (epsilon=smoothing)."""
    y_cat = tf.cast(y_cat, tf.float32)
    logits = tf.cast(logits, tf.float32)
    n_cls  = tf.cast(tf.shape(logits)[-1], tf.float32)
    smooth = y_cat * (1.0 - smoothing) + smoothing / n_cls
    return tf.reduce_mean(tf.keras.losses.categorical_crossentropy(smooth, logits))


# ------------------------------------------------------------------
# Augmented Keras Sequence
# ------------------------------------------------------------------

class AugmentedSequence(tf.keras.utils.Sequence):
    def __init__(self, X, y, batch_size, augment, zscore_mean, zscore_std):
        self.X           = X
        self.y           = y
        self.batch_size  = batch_size
        self.augment     = augment
        self.zscore_mean = zscore_mean
        self.zscore_std  = zscore_std
        self.indices     = np.arange(len(X))

    def __len__(self):
        return int(np.ceil(len(self.X) / self.batch_size))

    def __getitem__(self, idx):
        batch_idx = self.indices[idx * self.batch_size: (idx + 1) * self.batch_size]
        Xb = preprocess(self.X[batch_idx].copy(), augment=self.augment)
        Xb = apply_zscore(Xb, self.zscore_mean, self.zscore_std)
        return Xb, self.y[batch_idx]

    def on_epoch_end(self):
        np.random.shuffle(self.indices)


# ------------------------------------------------------------------
# Weight transfer helper
# ------------------------------------------------------------------

def transfer_weights(
    src_model:         tf.keras.Model,
    dst_model:         tf.keras.Model,
    freeze_until_name: str  = 'fc1_drop',
    keep_bn_trainable: bool = True,
) -> int:
    src_by_name    = {l.name: l for l in src_model.layers}
    transferred    = 0
    frozen         = 0
    still_freezing = True

    for layer in dst_model.layers:
        if layer.name in src_by_name:
            src_w = src_by_name[layer.name].get_weights()
            if src_w:
                try:
                    layer.set_weights(src_w)
                    transferred += 1
                except ValueError as e:
                    print(f"  [skip] {layer.name}: {e}")

        if still_freezing:
            is_bn = isinstance(layer, tf.keras.layers.BatchNormalization)
            if keep_bn_trainable and is_bn:
                layer.trainable = True
            else:
                layer.trainable = False
                frozen += 1
            if layer.name == freeze_until_name:
                still_freezing = False

    print(f"  Weight transfer: {transferred} layers copied, {frozen} frozen "
          f"(BN {'trainable' if keep_bn_trainable else 'also frozen'}).")
    return transferred


# ------------------------------------------------------------------
# ------------------------------------------------------------------
# Cross-day alignment loss (Stage 2)
# Replaces the weak cosine-centroid replay with a SupCon-style loss
# that directly aligns target-day embeddings to source-day prototypes.
# ------------------------------------------------------------------

def _supcon_alignment_loss(
    emb_tgt:       tf.Tensor,
    y_tgt_int:     tf.Tensor,
    src_centroids: tf.Tensor,
    weight:        float,
    temperature:   float = 0.10,
) -> tf.Tensor:
    """
    Supervised contrastive alignment between target embeddings and
    source-day class centroids.

    For each target embedding, its source centroid is the positive and
    all other centroids are negatives.  This directly minimises the
    angular distance between a device's Day-N embedding and its Day-N+1
    embedding while keeping inter-class separation large.

    emb_tgt       : (N, D)  L2-normalised target embeddings (from model)
    y_tgt_int     : (N,)    class labels
    src_centroids : (K, D)  L2-normalised source centroids (frozen)
    weight        : float   loss multiplier
    temperature   : float   SupCon temperature (lower = sharper)
    """
    emb_norm  = tf.math.l2_normalize(emb_tgt, axis=1)        # (N, D)
    cent_norm = tf.math.l2_normalize(src_centroids, axis=1)  # (K, D)

    # Similarity of each embedding to every centroid: (N, K)
    sim = tf.matmul(emb_norm, cent_norm, transpose_b=True) / temperature

    # Positive index = the centroid of the correct class
    # Cross-entropy over centroids (treating correct centroid as "label")
    loss = tf.reduce_mean(
        tf.keras.losses.sparse_categorical_crossentropy(
            y_tgt_int, sim, from_logits=True
        )
    )
    return weight * loss



# ------------------------------------------------------------------
# CNN fine-tuning wrapper
# ------------------------------------------------------------------

class CNN:
    """
    Two-stage cross-day fine-tuning.

    Stage 1: Frozen backbone + BN trainable, plain CE loss (FIX 1: no contrastive)
    Stage 2: Full fine-tune with prototype replay; embedding layer unfrozen (FIX 2)
    """

    def __init__(self, opts, dataOpts):
        self.verbose          = opts.verbose
        self.train_model_path = opts.modelPath
        self.model_type       = opts.modelType
        self.input_shape      = None
        self.count            = 0

        self.batch_size       = 64
        self.stage1_epochs    = 50
        self.stage2_epochs    = 80
        self.stage1_lr        = 3e-4
        self.stage2_lr        = 5e-5
        self.patience         = 25
        self.replay_weight    = 0.2
        self.emb_size         = 128
        self.src_num_class    = 16
        self.activation       = getattr(opts, 'activation', 'elu')
        self.dropout_conv     = getattr(opts, 'dropout_conv', 0.3)
        self.dropout_fc       = getattr(opts, 'dropout_fc', 0.4)
        self.l2_reg           = getattr(opts, 'l2_reg', 1e-4)

        # FIX 3: when True, always re-initialise from the base Phase 1
        # model instead of the previously fine-tuned model.
        self.always_reinit        = False
        self.phase1_model_path    = None   # set by phase2 runner if always_reinit=True

    # ----------------------------------------------------------------
    # Load pretrained weights
    # ----------------------------------------------------------------

    def _load_pretrained(self, path: str = None) -> tf.keras.Model:
        path = path or self.train_model_path
        print(f"  Loading pretrained model from:\n  {path}")
        try:
            ref = tf.keras.models.load_model(
                path,
                custom_objects={'L2Normalize': rf_models.L2Normalize},
            )
            print(f"  Loaded via load_model  (output shape: {ref.output_shape})")
            return ref
        except Exception as e:
            print(f"  load_model failed: {e}\n  Rebuilding skeleton and loading weights ...")

        import h5py, zipfile, tempfile
        ref = rf_models.create_model(
            model_type     = self.model_type,
            inp_shape      = self.input_shape,
            num_class      = self.src_num_class,
            emb_size       = self.emb_size,
            classification = True,
            activation     = self.activation,
            dropout_conv   = self.dropout_conv,
            dropout_fc     = self.dropout_fc,
            l2_reg         = self.l2_reg,
        )
        ref.compile(loss='categorical_crossentropy',
                    optimizer='adam', metrics=['accuracy'])

        with tempfile.TemporaryDirectory() as tmp:
            with zipfile.ZipFile(path, 'r') as z:
                z.extract('model.weights.h5', tmp)
            weights_path = os.path.join(tmp, 'model.weights.h5')
            with h5py.File(weights_path, 'r') as f:
                dst_by_name = {l.name: l for l in ref.layers}
                def _copy(layer_name, h5_group):
                    if layer_name not in dst_by_name:
                        return
                    dst_layer  = dst_by_name[layer_name]
                    vars_in_file = []
                    def _collect(name, obj):
                        if isinstance(obj, h5py.Dataset):
                            vars_in_file.append(np.array(obj))
                    h5_group.visititems(_collect)
                    if not vars_in_file:
                        return
                    dst_vars = dst_layer.get_weights()
                    if len(vars_in_file) != len(dst_vars):
                        return
                    if any(a.shape != b.shape for a, b in zip(vars_in_file, dst_vars)):
                        return
                    dst_layer.set_weights(vars_in_file)
                for key in f.keys():
                    grp = f[key]
                    if isinstance(grp, h5py.Group):
                        _copy(key, grp)
        return ref

    # ----------------------------------------------------------------
    # Core two-stage fine-tuning
    # ----------------------------------------------------------------

    def tune_the_model(
        self,
        X_train:   np.ndarray,
        y_train:   np.ndarray,
        NUM_CLASS: int,
        X_src:     np.ndarray | None = None,
        y_src:     np.ndarray | None = None,
    ):
        os.makedirs(_MODEL_DIR, exist_ok=True)
        save_path = os.path.join(_MODEL_DIR, f'tune_model_{NUM_CLASS}.keras')

        # Stratified 15% validation split for stable per-device early stopping.
        y_all_int = (np.argmax(y_train, axis=1) if np.asarray(y_train).ndim == 2
                     else np.asarray(y_train, dtype=np.int32)).astype(np.int32)
        tr_idx, val_idx = [], []
        for cls in np.unique(y_all_int):
            cls_idx = np.where(y_all_int == cls)[0]
            cls_idx = cls_idx[np.random.permutation(len(cls_idx))]
            n_val_cls = max(1, int(np.ceil(len(cls_idx) * 0.15)))
            if n_val_cls >= len(cls_idx):
                raise ValueError(
                    f"Class {int(cls)} has only {len(cls_idx)} training samples; "
                    "cannot create a non-empty stratified train/val split."
                )
            val_idx.extend(cls_idx[:n_val_cls].tolist())
            tr_idx.extend(cls_idx[n_val_cls:].tolist())
        tr_idx = np.array(tr_idx, dtype=np.int64)
        val_idx = np.array(val_idx, dtype=np.int64)
        np.random.shuffle(tr_idx)
        np.random.shuffle(val_idx)
        X_val, y_val = X_train[val_idx], y_train[val_idx]
        X_tr,  y_tr  = X_train[tr_idx],  y_train[tr_idx]

        # Fit z-score on target-day train data
        print("\n[Z-score] Fitting normalisation on target-day train data...")
        X_tr_pp         = preprocess(X_tr, augment=False)
        zscore_mean, zscore_std = compute_zscore(X_tr_pp)
        X_tr_ppz = apply_zscore(X_tr_pp, zscore_mean, zscore_std)
        y_tr_int = (np.argmax(y_tr, axis=1) if np.asarray(y_tr).ndim == 2
                    else np.asarray(y_tr, dtype=np.int32)).astype(np.int32)

        # ── Stage 0: load pretrained ─────────────────────────────────
        # FIX 3: optionally always re-init from Phase 1 model
        pretrain_path = (self.phase1_model_path
                         if self.always_reinit and self.phase1_model_path
                         else self.train_model_path)
        print(f"\n[Stage 0] Loading pretrained weights from {pretrain_path}...")
        ref_model = self._load_pretrained(pretrain_path)

        # ── Stage 0.5: TTA-BN ────────────────────────────────────────
        print("\n[Stage 0.5] Adapting BN statistics to target domain...")
        adapt_model = rf_models.create_model(
            model_type     = self.model_type,
            inp_shape      = self.input_shape,
            num_class      = NUM_CLASS,
            emb_size       = self.emb_size,
            classification = True,
            activation     = self.activation,
            dropout_conv   = self.dropout_conv,
            dropout_fc     = self.dropout_fc,
            l2_reg         = self.l2_reg,
        )
        adapt_model.compile(loss='categorical_crossentropy',
                            optimizer='adam', metrics=['accuracy'])
        transfer_weights(ref_model, adapt_model,
                         freeze_until_name='fc1_drop',
                         keep_bn_trainable=True)
        tta_bn.adapt_bn_statistics(adapt_model, X_tr_ppz,
                                   batch_size=self.batch_size, n_passes=3)
        bn_stats = tta_bn.save_bn_statistics(adapt_model)
        tta_bn.reset_bn_statistics(ref_model, bn_stats)
        print("  BN stats transferred back to reference model.")

        # Preprocess source data (for replay only)
        X_src_ppz = None
        y_src_int = None
        if X_src is not None and y_src is not None:
            X_src_pp  = preprocess(X_src, augment=False)
            X_src_ppz = apply_zscore(X_src_pp, zscore_mean, zscore_std)
            y_src_int = (np.argmax(y_src, axis=1)
                         if np.asarray(y_src).ndim == 2
                         else np.asarray(y_src, dtype=np.int32)).astype(np.int32)

        # ── Stage 1: plain CE head warm-up (backbone frozen) ─────────
        # FIX 1: removed cross-day contrastive loss — plain CE only.
        print("\n[Stage 1] Head warm-up: frozen backbone, plain CE loss...")
        model = rf_models.create_model(
            model_type     = self.model_type,
            inp_shape      = self.input_shape,
            num_class      = NUM_CLASS,
            emb_size       = self.emb_size,
            classification = True,
            activation     = self.activation,
            dropout_conv   = self.dropout_conv,
            dropout_fc     = self.dropout_fc,
            l2_reg         = self.l2_reg,
        )
        transfer_weights(ref_model, model,
                         freeze_until_name='fc1_drop',
                         keep_bn_trainable=True)

        optimizer_s1 = Adam(self.stage1_lr)
        best_s1      = 0.0
        no_imp_s1    = 0

        N_s1     = len(X_tr)
        n_bat_s1 = int(np.ceil(N_s1 / self.batch_size))

        X_s1_ppz = apply_zscore(
            preprocess(X_tr.copy(), augment=True), zscore_mean, zscore_std
        )
        y_tr_cat = (y_tr if np.asarray(y_tr).ndim == 2
                    else to_categorical(y_tr, NUM_CLASS))
        X_val_ppz = apply_zscore(
            preprocess(X_val.copy(), augment=False), zscore_mean, zscore_std
        )
        y_val_cat = (y_val if np.asarray(y_val).ndim == 2
                     else to_categorical(y_val, NUM_CLASS))

        for epoch in range(self.stage1_epochs):
            perm       = np.random.permutation(N_s1)
            x_shuf     = X_s1_ppz[perm]
            y_shuf_cat = y_tr_cat[perm]

            epoch_ce = 0.0
            for b in range(n_bat_s1):
                sl     = slice(b * self.batch_size, (b + 1) * self.batch_size)
                xb     = tf.constant(x_shuf[sl])
                yb_cat = tf.constant(y_shuf_cat[sl])

                with tf.GradientTape() as tape:
                    logits  = model(xb, training=True)
                    ce_loss = _smooth_ce(yb_cat, logits, smoothing=0.05)
                grads = tape.gradient(ce_loss, model.trainable_variables)
                grads, _ = tf.clip_by_global_norm(grads, 5.0)
                optimizer_s1.apply_gradients(
                    zip(grads, model.trainable_variables)
                )
                epoch_ce += float(ce_loss)

            epoch_ce /= n_bat_s1
            val_logits = model.predict(
                X_val_ppz, batch_size=self.batch_size, verbose=0
            )
            val_acc = float(np.mean(
                np.argmax(val_logits, axis=1) == np.argmax(y_val_cat, axis=1)
            ))

            print(f"  Stage1 Epoch {epoch+1:3d}/{self.stage1_epochs}  "
                  f"ce={epoch_ce:.4f}  val_acc={val_acc:.4f}")

            if val_acc > best_s1:
                best_s1   = val_acc
                no_imp_s1 = 0
                model.save_weights(
                    save_path.replace('.keras', '_s1_best.weights.h5')
                )
            else:
                no_imp_s1 += 1

            if no_imp_s1 >= self.patience:
                print(f"  [Stage 1] Early stop  best_val_acc={best_s1:.4f}")
                break

        s1_best = save_path.replace('.keras', '_s1_best.weights.h5')
        if os.path.exists(s1_best):
            model.load_weights(s1_best)
            print(f"  [Stage 1] Best weights restored  val_acc={best_s1:.4f}")

        model.compile(
            loss      = 'categorical_crossentropy',
            optimizer = Adam(self.stage1_lr),
            metrics   = ['accuracy'],
        )

        # ── Stage 2: full fine-tune + SupCon cross-day alignment ─────
        # All layers trainable. Source-day centroids are frozen anchors;
        # the SupCon alignment loss pulls target embeddings toward the
        # correct source centroid while pushing away wrong ones.
        print("\n[Stage 2] Full fine-tuning (CE + SupCon alignment)...")
        for layer in model.layers:
            layer.trainable = True

        # Build embedding sub-model (used inside the training loop)
        _emb_name = ('emb_l2norm'
                     if any(l.name == 'emb_l2norm' for l in model.layers)
                     else 'embedding')
        emb_submodel = tf.keras.Model(
            inputs  = model.input,
            outputs = model.get_layer(_emb_name).output,
        )

        # Compute source-day centroids (frozen anchors for alignment)
        src_centroid_tensor = None
        if X_src_ppz is not None:
            print("  Computing source centroids for SupCon alignment...")
            src_emb = emb_submodel.predict(X_src_ppz, batch_size=128, verbose=0)
            replay_centroids = ew.compute_source_centroids(src_emb, y_src_int)

            emb_dim = src_emb.shape[1]
            centroid_matrix = np.zeros((NUM_CLASS, emb_dim), dtype=np.float32)
            for cls, vec in replay_centroids.items():
                if cls < NUM_CLASS:
                    centroid_matrix[cls] = vec
            src_centroid_tensor = tf.constant(centroid_matrix, dtype=tf.float32)
            print(f"  Source centroids computed for {len(replay_centroids)} classes.")


        N_s2      = len(X_tr)
        n_batches = int(np.ceil(N_s2 / self.batch_size))

        # Cosine LR schedule over the full Stage 2 budget
        s2_total_steps = self.stage2_epochs * n_batches
        s2_lr_schedule = CosineDecay(
            initial_learning_rate = self.stage2_lr,
            decay_steps           = s2_total_steps,
            alpha                 = 1e-7,
        )
        optimizer_s2   = Adam(self.stage2_lr)
        s2_global_step = 0

        best_val_acc = best_s1
        no_improve   = 0

        y_tr_cat2 = (y_tr if np.asarray(y_tr).ndim == 2
                     else to_categorical(y_tr, NUM_CLASS))
        X_val_ppz2 = apply_zscore(
            preprocess(X_val.copy(), augment=False), zscore_mean, zscore_std
        )
        y_val_cat2 = (y_val if np.asarray(y_val).ndim == 2
                      else to_categorical(y_val, NUM_CLASS))

        for epoch in range(self.stage2_epochs):
            # Re-augment each epoch so the model never memorises a fixed
            # augmented copy — each epoch sees fresh channel perturbations.
            X_s2_ppz   = apply_zscore(
                preprocess(X_tr.copy(), augment=True), zscore_mean, zscore_std
            )
            perm       = np.random.permutation(N_s2)
            x_shuf     = X_s2_ppz[perm]
            y_shuf_cat = y_tr_cat2[perm]
            y_shuf_int = y_tr_int[perm]

            epoch_loss = 0.0
            for b in range(n_batches):
                sl     = slice(b * self.batch_size, (b + 1) * self.batch_size)
                xb     = tf.constant(x_shuf[sl])
                yb_cat = tf.constant(y_shuf_cat[sl])
                yb_int = tf.constant(y_shuf_int[sl], dtype=tf.int32)

                current_lr = float(s2_lr_schedule(s2_global_step))
                optimizer_s2.learning_rate.assign(current_lr)

                with tf.GradientTape() as tape:
                    logits  = model(xb, training=True)
                    ce_loss = _smooth_ce(yb_cat, logits, smoothing=0.05)
                    if src_centroid_tensor is not None:
                        emb_tgt = emb_submodel(xb, training=True)
                        al_loss = _supcon_alignment_loss(
                            emb_tgt, yb_int,
                            src_centroid_tensor, self.replay_weight,
                        )
                        total = ce_loss + al_loss
                    else:
                        total = ce_loss

                grads = tape.gradient(total, model.trainable_variables)
                grads, _ = tf.clip_by_global_norm(grads, 5.0)
                optimizer_s2.apply_gradients(zip(grads, model.trainable_variables))
                epoch_loss += float(total)
                s2_global_step += 1

            epoch_loss /= n_batches
            val_logits  = model.predict(X_val_ppz2,
                                        batch_size=self.batch_size, verbose=0)
            val_acc     = float(np.mean(
                np.argmax(val_logits, axis=1) == np.argmax(y_val_cat2, axis=1)
            ))

            print(f"  Stage2 Epoch {epoch+1:3d}/{self.stage2_epochs}  "
                  f"loss={epoch_loss:.4f}  val_acc={val_acc:.4f}  "
                  f"lr={current_lr:.2e}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                no_improve   = 0
                model.save_weights(
                    save_path.replace('.keras', '_s2_best.weights.h5')
                )
            else:
                no_improve += 1

            if no_improve >= self.patience:
                print(f"  [Stage 2] Early stop  best_val_acc={best_val_acc:.4f}")
                break

        best_w2 = save_path.replace('.keras', '_s2_best.weights.h5')
        if os.path.exists(best_w2):
            model.load_weights(best_w2)
            print(f"  [Stage 2] Best weights restored  val_acc={best_val_acc:.4f}")

        model.save(save_path)
        print(f"  Saved fine-tuned model to {save_path}")

        return (model, ref_model, zscore_mean, zscore_std,
                X_tr_ppz, y_tr_int, X_src_ppz, y_src_int)

    # ----------------------------------------------------------------
    # Prototype classifier
    # ----------------------------------------------------------------

    def _prototype_classify(
        self,
        model:        tf.keras.Model,
        X_pp:         np.ndarray,
        y_int:        np.ndarray,
        X_tr_ppz:     np.ndarray,
        y_tr_int:     np.ndarray,
        n_slices_agg: int = 5,
    ) -> float:
        emb_name  = ('emb_l2norm'
                     if any(l.name == 'emb_l2norm' for l in model.layers)
                     else 'embedding')
        emb_model = tf.keras.Model(
            inputs  = model.input,
            outputs = model.get_layer(emb_name).output,
        )
        emb_tr    = emb_model.predict(X_tr_ppz, batch_size=128, verbose=0)
        classes   = np.unique(y_tr_int)
        centroids = np.stack([emb_tr[y_tr_int == c].mean(axis=0) for c in classes])
        centroids = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-8)

        emb_te = emb_model.predict(X_pp, batch_size=128, verbose=0)

        if n_slices_agg > 1:
            agg_embs, agg_labels = [], []
            for cls in np.unique(y_int):
                cls_embs = emb_te[y_int == cls]
                n_trim   = (len(cls_embs) // n_slices_agg) * n_slices_agg
                if n_trim == 0:
                    continue
                pooled = (cls_embs[:n_trim]
                          .reshape(-1, n_slices_agg, emb_te.shape[1])
                          .mean(axis=1))
                agg_embs.append(pooled)
                agg_labels.extend([cls] * len(pooled))
            emb_agg = np.vstack(agg_embs)
            y_agg   = np.array(agg_labels, dtype=np.int32)
        else:
            emb_agg, y_agg = emb_te, y_int

        emb_agg = emb_agg / (np.linalg.norm(emb_agg, axis=1, keepdims=True) + 1e-8)
        preds   = classes[np.argmax(emb_agg @ centroids.T, axis=1)]
        acc     = float(np.mean(preds == y_agg))
        print(f"  proto_agg{n_slices_agg}={acc:.4f}")
        return acc

    # ----------------------------------------------------------------
    # Evaluation
    # ----------------------------------------------------------------

    def test(
        self,
        model:       tf.keras.Model,
        X_test:      np.ndarray,
        y_test:      np.ndarray,
        zscore_mean: np.ndarray,
        zscore_std:  np.ndarray,
        X_tr_ppz:    np.ndarray | None = None,
        y_tr_int:    np.ndarray | None = None,
    ) -> tuple[dict, np.ndarray, np.ndarray]:
        self.count += 1
        print(f"\nTest run #{self.count}")

        if len(X_test) == 0:
            print("  [test] empty test set — returning NaN metrics.")
            nan_r = {k: float('nan') for k in (
                'softmax', 'bn_adapted', 'whiten_mean', 'whiten_pc',
                'whiten_affine', 'proto_agg1', 'proto_agg5', 'silhouette')}
            return nan_r, X_test, np.empty((0,), dtype=np.int32)

        X_pp  = preprocess(X_test, augment=False)
        X_pp  = apply_zscore(X_pp, zscore_mean, zscore_std)

        y_int = (np.argmax(y_test, axis=1) if np.asarray(y_test).ndim == 2
                 else np.asarray(y_test, dtype=np.int32))
        y_cat = (y_test if np.asarray(y_test).ndim == 2
                 else to_categorical(y_test, int(y_int.max()) + 1))

        results: dict = {}

        # Softmax
        score = model.evaluate(X_pp, y_cat, batch_size=128, verbose=1)
        acc   = float(score[1])
        results['softmax'] = acc
        print(f"  Softmax accuracy: {acc:.4f}")

        # TTA-BN
        original_bn = tta_bn.save_bn_statistics(model)
        tta_bn.adapt_bn_statistics(model, X_pp, batch_size=128, n_passes=2)
        score_bn = model.evaluate(X_pp, y_cat, batch_size=128, verbose=1)
        bn_acc   = float(score_bn[1])
        results['bn_adapted'] = bn_acc
        print(f"  BN-adapted accuracy: {bn_acc:.4f}  Δ={bn_acc - acc:+.4f}")
        tta_bn.reset_bn_statistics(model, original_bn)

        # Whitening + prototype
        if X_tr_ppz is not None and y_tr_int is not None:
            whiten_res = ew.evaluate_whitening(
                model      = model,
                x_src      = X_tr_ppz,
                y_src      = y_tr_int,
                x_tgt      = X_pp,
                y_tgt      = y_int,
                batch_size = 128,
            )
            results['whiten_mean']   = whiten_res.get('mean_shift', float('nan'))
            results['whiten_pc']     = whiten_res.get('per_class',  float('nan'))
            results['whiten_affine'] = whiten_res.get('affine',     float('nan'))

            results['proto_agg1'] = self._prototype_classify(
                model, X_pp, y_int, X_tr_ppz, y_tr_int, n_slices_agg=1,
            )
            results['proto_agg5'] = self._prototype_classify(
                model, X_pp, y_int, X_tr_ppz, y_tr_int, n_slices_agg=5,
            )
        else:
            for k in ('whiten_mean', 'whiten_pc', 'whiten_affine',
                      'proto_agg1', 'proto_agg5'):
                results[k] = float('nan')

        # Silhouette
        try:
            from sklearn.metrics import silhouette_score
            _en = ('emb_l2norm'
                   if any(l.name == 'emb_l2norm' for l in model.layers)
                   else 'embedding')
            _em = tf.keras.Model(inputs=model.input,
                                 outputs=model.get_layer(_en).output)
            emb_te = _em.predict(X_pp, batch_size=128, verbose=0)
            sil    = silhouette_score(emb_te, y_int, metric='cosine')
            results['silhouette'] = float(sil)
            print(f"  Silhouette: {sil:.4f}")
        except Exception as e:
            results['silhouette'] = float('nan')

        return results, X_pp, y_int

    # ----------------------------------------------------------------
    # Full pipeline
    # ----------------------------------------------------------------

    def tune(
        self,
        X_train:   np.ndarray,
        y_train:   np.ndarray,
        X_test:    np.ndarray,
        y_test:    np.ndarray,
        NUM_CLASS: int,
        X_src:     np.ndarray | None = None,
        y_src:     np.ndarray | None = None,
        run_tag:   str = '',
    ) -> dict:
        self.input_shape = (X_train.shape[1], X_train.shape[2])

        (model, ref_model,
         zscore_mean, zscore_std,
         X_tr_ppz, y_tr_int,
         X_src_ppz, y_src_int) = self.tune_the_model(
            X_train   = X_train,
            y_train   = y_train,
            NUM_CLASS = NUM_CLASS,
            X_src     = X_src,
            y_src     = y_src,
        )

        results, X_test_ppz, y_test_int = self.test(
            model       = model,
            X_test      = X_test,
            y_test      = y_test,
            zscore_mean = zscore_mean,
            zscore_std  = zscore_std,
            X_tr_ppz    = X_tr_ppz,
            y_tr_int    = y_tr_int,
        )

        # Save centroids
        ft_train_emb       = ew.extract_embeddings(model, X_tr_ppz, 128)
        ft_train_centroids = ew.compute_source_centroids(ft_train_emb, y_tr_int)
        ft_centroid_path   = os.path.join(
            _MODEL_DIR, f'ft_centroids_{run_tag or "latest"}.npz'
        )
        ew.save_centroids(ft_train_centroids, ft_centroid_path)

        results['zscore_mean'] = zscore_mean
        results['zscore_std']  = zscore_std
        return results


# ------------------------------------------------------------------
# Few-shot data preparation
# ------------------------------------------------------------------

def prepare_data(
    X: np.ndarray,
    y: np.ndarray,
    n_train: int = 100,
    n_test:  int = 500,
    groups:  np.ndarray | None = None,
    split_mode: str = 'random',
):
    """
    Split one day into train/test subsets.

    split_mode
    ----------
    'random'  : shuffle slices within each class (historical behaviour).
                Slices recorded microseconds apart land on both sides, so
                the model can match on channel/AGC state rather than on the
                device — in-day accuracy comes out inflated.
    'capture' : train and test drawn from DIFFERENT capture files, so no
                recording is shared across the split. Requires `groups`
                (N, 2) = [file_index, position] from
                data_utils.load_day_raw_with_groups.
    """
    if split_mode not in ('random', 'capture'):
        raise ValueError(f"split_mode must be 'random' or 'capture', "
                         f"got {split_mode!r}")
    if split_mode == 'capture' and groups is None:
        raise ValueError("split_mode='capture' requires groups")

    X_tr, X_te = [], []
    y_tr, y_te = [], []

    if split_mode == 'capture':
        for cls in sorted({int(v) for v in y}):
            idx = np.flatnonzero(y == cls)
            files = np.unique(groups[idx, 0])
            if len(files) < 2:
                raise ValueError(
                    f"Class {cls} has {len(files)} capture file(s); "
                    "split_mode='capture' needs at least 2."
                )
            n_tr_files = max(1, int(round(len(files) * 0.6)))
            tr_files   = set(files[:n_tr_files].tolist())
            tr_idx = idx[np.isin(groups[idx, 0], list(tr_files))]
            te_idx = idx[~np.isin(groups[idx, 0], list(tr_files))]
            rng = np.random.default_rng(0)
            tr_idx = tr_idx[rng.permutation(len(tr_idx))][:n_train]
            te_idx = te_idx[rng.permutation(len(te_idx))][:n_test]
            if len(tr_idx) < n_train or len(te_idx) < n_test:
                raise ValueError(
                    f"Class {cls}: capture split yields {len(tr_idx)} train / "
                    f"{len(te_idx)} test, need {n_train}/{n_test}. "
                    "Load more slices or lower ft_n_train/n_test."
                )
            X_tr.extend(X[tr_idx])
            X_te.extend(X[te_idx])
            y_tr.extend([cls] * len(tr_idx))
            y_te.extend([cls] * len(te_idx))
    else:
        data_by_class: dict = defaultdict(list)
        for xi, yi in zip(X, y):
            data_by_class[int(yi)].append(xi)

        for cls, samples in data_by_class.items():
            required = n_train + n_test
            if len(samples) < required:
                raise ValueError(
                    f"Class {cls} has {len(samples)} samples, but prepare_data "
                    f"requires {required} ({n_train} train + {n_test} test). "
                    "Lower ft_n_train/n_test or load more slices."
                )
            random.shuffle(samples)
            train_samples = samples[:n_train]
            test_samples  = samples[n_train: n_train + n_test]
            X_tr.extend(train_samples)
            X_te.extend(test_samples)
            y_tr.extend([cls] * len(train_samples))
            y_te.extend([cls] * len(test_samples))

    tr_perm = np.random.permutation(len(X_tr))
    X_tr    = np.array(X_tr, dtype=np.float32)[tr_perm]
    y_tr    = np.array(y_tr, dtype=np.int32)[tr_perm]

    if X_te:
        te_perm = np.random.permutation(len(X_te))
        X_te    = np.array(X_te, dtype=np.float32)[te_perm]
        y_te    = np.array(y_te, dtype=np.int32)[te_perm]
    else:
        X_te = np.empty((0, X_tr.shape[1], X_tr.shape[2]), dtype=np.float32)
        y_te = np.empty((0,), dtype=np.int32)

    NUM_CLASS = len(np.unique(np.concatenate([y_tr, y_te])
                              if len(y_te) else y_tr))
    return (
        X_tr, to_categorical(y_tr, NUM_CLASS),
        X_te, to_categorical(y_te, NUM_CLASS),
        NUM_CLASS,
    )
