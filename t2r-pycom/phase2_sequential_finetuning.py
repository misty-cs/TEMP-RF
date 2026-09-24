#!/usr/bin/env python3
"""
phase2_sequential_finetuning.py  —  Phase 2: Sequential fine-tuning.

Current protocol
----------------
Each fine-tune day starts from the Phase 1 model by default. Validation is
an internal stratified split of the current fine-tune day inside CNN.tune().
"""

from __future__ import annotations

import os
import time
from typing import Optional

import numpy as np
import tensorflow as tf

from experiment_config import ExperimentConfig
from data_utils import load_day_raw
import rf_models
import finetune as _ft
from finetune import CNN, prepare_data


def run_phase2(cfg: ExperimentConfig, phase1_result: dict) -> dict:

    print('\n' + '=' * 62)
    print('  PHASE 2 — Sequential fine-tuning')
    print(f'  Days      : {cfg.finetune_days}')
    print(f'  Devices   : all_known_ids={cfg.all_known_ids}  ({len(cfg.all_known_ids)} classes)')
    print(f'  always_reinit : {getattr(cfg, "ft_always_reinit", False)}')
    print('=' * 62)

    np.random.seed(cfg.seed)
    tf.random.set_seed(cfg.seed)

    _ft._MODEL_DIR = cfg.model_dir

    results_path = os.path.join(cfg.results_dir, 'results_phase2.txt')

    def _log(line):
        print(line)
        with open(results_path, 'a') as f:
            print(line, file=f, flush=True)

    _log(f'\n### Phase 2  started at {time.ctime()}')
    _log(f'### finetune_days={cfg.finetune_days}  ft_n_train={cfg.ft_n_train}/class')
    _log(f'### always_reinit={getattr(cfg, "ft_always_reinit", False)}')

    ft_devices = cfg.all_known_ids
    NUM_CLASS  = len(ft_devices)

    current_model_path = phase1_result['model_path']
    current_norm_path  = phase1_result['norm_path']

    step_results: list = []
    all_models:   list = [(cfg.init_day, current_model_path, current_norm_path)]

    prev_day = cfg.init_day
    prev_X:  Optional[np.ndarray] = None
    prev_y:  Optional[np.ndarray] = None

    # Cross-session variant (T2R_XSESSION=1): every session seen so far is
    # pooled and the same device on different sessions forms a positive pair.
    XS = os.environ.get('T2R_XSESSION', '0') == '1'
    pool_X: list = []; pool_y: list = []; pool_s: list = []
    if XS:
        from xsession_finetune import xsession_tune
        _log('### cross-session SupCon enabled (T2R_XSESSION=1)')
        X0, y0 = load_day_raw(cfg, day_id=cfg.init_day, device_ids=ft_devices)
        X0_tr, y0_tr_cat, _, _, _ = prepare_data(
            X0, y0, n_train=cfg.ft_n_train,
            n_test=int(os.environ.get('T2R_FT_NTEST', '500')))
        pool_X.append(X0_tr); pool_y.append(np.argmax(y0_tr_cat, 1))
        pool_s.append(np.full(len(X0_tr), cfg.init_day, dtype=np.int32))
        del X0, y0

    import os as _os
    for day in cfg.finetune_days:
        if getattr(cfg, 'dataset_layout', '') == 'wisig_full' and day == cfg.traj_day:
            _os.environ['T2R_WISIG_ROLE'] = 'ft'
        else:
            _os.environ.pop('T2R_WISIG_ROLE', None)
        print(f'\n{"─"*62}')
        print(f'  Fine-tune step: model from Day {prev_day} → target Day {day}')
        print(f'{"─"*62}')

        t0 = time.time()

        # Load raw IQ. CNN.tune owns preprocessing and z-score fitting.
        ft_split = getattr(cfg, 'ft_split_mode', 'random')
        if ft_split == 'capture':
            from data_utils import load_day_raw_with_groups
            X_day, y_day, g_day = load_day_raw_with_groups(
                cfg, day_id=day, device_ids=ft_devices)
        else:
            X_day, y_day = load_day_raw(cfg, day_id=day, device_ids=ft_devices)
            g_day = None

        # NOTE: norm_path is saved AFTER cnn.tune() so we can use the
        # CNN's own z-score stats (computed on preprocessed data) rather
        # than load_day's stats (computed on raw data). They are different
        # pipelines and the model was trained with the CNN's stats.
        norm_path = os.path.join(cfg.model_dir, f'phase2_norm_d{day}.npz')

        # FIX 2: increased n_train to cfg.ft_n_train (default 1600)
        X_tr, y_tr_cat, X_te, y_te_cat, nc = prepare_data(
            X_day, y_day,
            n_train = cfg.ft_n_train,
            n_test  = int(os.environ.get('T2R_FT_NTEST', '500')),
            groups     = g_day,
            split_mode = ft_split,
        )
        assert nc == NUM_CLASS, (
            f"Day {day} returned {nc} classes, expected {NUM_CLASS}."
        )

        # Source replay is only valid when it moves forward in time. With
        # configs like init_day=5 and finetune_days=[1,2,3,4,6], the first
        # target day is earlier than the init day, so skip replay for that
        # step instead of anchoring Day 1 to future Day 5 embeddings.
        if prev_X is not None and prev_day < day:
            X_src_ft = prev_X
            y_src_ft = prev_y
        elif prev_day < day:
            X_src_ft, y_src_ft = load_day_raw(
                cfg, day_id=prev_day, device_ids=ft_devices
            )
        else:
            X_src_ft = None
            y_src_ft = None
            print(f'  [replay] skipped: source Day {prev_day} is not before target Day {day}.')

        # Build CNN wrapper
        class _Opts:
            modelType = cfg.model_type
            verbose   = cfg.verbose
            activation = cfg.activation
            dropout_conv = cfg.dropout_conv
            dropout_fc = cfg.dropout_fc
            l2_reg = cfg.l2_reg
            # FIX 1: always_reinit uses Phase 1 model path
            modelPath = (phase1_result['model_path']
                         if getattr(cfg, 'ft_always_reinit', False)
                         else current_model_path)

        class _DataOpts:
            file_key  = cfg.file_key
            start_idx = cfg.start_idx
            slice_len = cfg.slice_len
            stride    = cfg.stride
            mul_trans = cfg.mul_trans
            window    = cfg.window
            data_type = cfg.data_type

        _ft._MODEL_DIR = cfg.model_dir

        cnn = CNN(_Opts(), _DataOpts())
        cnn.stage1_epochs  = 50    # head warmup — more data needs more steps
        cnn.stage2_epochs  = 80    # full fine-tune needs more time
        cnn.patience       = 25    # scaled with longer training
        cnn.batch_size     = 64
        cnn.stage1_lr      = 5e-4  # safer for head-only training (was 1e-3, too high)
        cnn.stage2_lr      = 5e-5  # full fine-tune LR
        # 0.2: CE dominates, alignment is a light regulariser not a hard constraint
        cnn.replay_weight  = 0.2
        cnn.src_num_class  = NUM_CLASS
        cnn.emb_size       = cfg.emb_size
        cnn.activation     = cfg.activation
        cnn.dropout_conv   = cfg.dropout_conv
        cnn.dropout_fc     = cfg.dropout_fc
        cnn.l2_reg         = cfg.l2_reg
        cnn.input_shape    = (X_tr.shape[1], X_tr.shape[2])

        # FIX 1: wire up always_reinit and Phase 1 model path
        cnn.always_reinit     = getattr(cfg, 'ft_always_reinit', False)
        cnn.phase1_model_path = phase1_result['model_path']

        if XS:
            y_tr_int = np.argmax(y_tr_cat, 1).astype(np.int32)
            results = xsession_tune(
                cnn,
                np.concatenate(pool_X), np.concatenate(pool_y), np.concatenate(pool_s),
                X_tr, y_tr_int, np.full(len(X_tr), day, dtype=np.int32),
                X_te, y_te_cat, nc, cfg.model_dir, log=_log)
            pool_X.append(X_tr); pool_y.append(y_tr_int)
            pool_s.append(np.full(len(X_tr), day, dtype=np.int32))
        else:
            results = cnn.tune(
                X_train   = X_tr,
                y_train   = y_tr_cat,
                X_test    = X_te,
                y_test    = y_te_cat,
                NUM_CLASS = nc,
                X_src     = X_src_ft,
                y_src     = y_src_ft,
                run_tag   = f'phase2_d{day}',
            )

        # Save the CNN's own z-score stats (fitted on preprocessed target-day
        # data inside CNN.tune_the_model). These are the stats the model was
        # actually trained with — they must be used for all inference.
        cnn_zscore_mean = results.pop('zscore_mean', None)
        cnn_zscore_std  = results.pop('zscore_std',  None)
        if cnn_zscore_mean is not None and cnn_zscore_std is not None:
            np.savez(norm_path, mean=cnn_zscore_mean, std=cnn_zscore_std)
            print(f'  CNN z-score stats saved → {norm_path}')
        else:
            raise RuntimeError(
                "CNN.tune() did not return z-score stats; refusing to save "
                "invalid normalization stats."
            )

        # Resolve saved model path
        model_save_path = os.path.join(cfg.model_dir, f'phase2_ft_d{day}.keras')
        ft_saved        = os.path.join(cfg.model_dir, f'tune_model_{nc}.keras')

        if os.path.exists(ft_saved):
            os.replace(ft_saved, model_save_path)
            print(f'  Model saved → {model_save_path}')
        else:
            # Fallback: rebuild from best weights
            saved = False
            for suffix in ['_s2_best.weights.h5', '_s1_best.weights.h5']:
                candidate = os.path.join(
                    cfg.model_dir, f'tune_model_{nc}{suffix}'
                )
                if os.path.exists(candidate):
                    print(f'  [fallback] Rebuilding model from weights: {candidate}')
                    m = rf_models.create_model(
                        model_type     = cfg.model_type,
                        inp_shape      = (X_tr.shape[1], X_tr.shape[2]),
                        num_class      = nc,
                        emb_size       = cfg.emb_size,
                        classification = True,
                        activation     = cfg.activation,
                        dropout_conv   = cfg.dropout_conv,
                        dropout_fc     = cfg.dropout_fc,
                        l2_reg         = cfg.l2_reg,
                    )
                    m.load_weights(candidate)
                    m.save(model_save_path)
                    saved = True
                    print(f'  Model saved → {model_save_path}')
                    break
            if not saved:
                raise RuntimeError(
                    f"Fine-tune for Day {day} did not produce {ft_saved} or "
                    "best-weight checkpoints; refusing to continue with a copied model."
                )

        # FIX 1: if always_reinit, keep current_model_path pointing at
        # Phase 1 so every step re-initializes from the same baseline.
        if not getattr(cfg, 'ft_always_reinit', False):
            current_model_path = model_save_path
        current_norm_path = norm_path

        duration = time.time() - t0
        results['day']      = day
        results['duration'] = duration
        step_results.append(results)
        all_models.append((day, model_save_path, norm_path))

        _log(
            f'\n  day={day}  '
            f'softmax={results.get("softmax",    float("nan")):.4f}  '
            f'bn_adapted={results.get("bn_adapted", float("nan")):.4f}  '
            f'proto_agg1={results.get("proto_agg1", float("nan")):.4f}  '
            f'xs_centroid_val={results.get("xs_centroid_val", float("nan")):.4f}  '
            f'time={duration:.0f}s'
        )

        prev_day = day
        prev_X   = X_day
        prev_y   = y_day

    # Final model is always the last fine-tuned model (sequential or reinit)
    final_model_path = os.path.join(
        cfg.model_dir, f'phase2_ft_d{cfg.finetune_days[-1]}.keras'
    )
    final_norm_path  = os.path.join(
        cfg.model_dir, f'phase2_norm_d{cfg.finetune_days[-1]}.npz'
    )

    _log(f'\n### Phase 2 complete.  Final model → {final_model_path}')

    return {
        'final_model_path': final_model_path,
        'final_norm_path':  final_norm_path,
        'step_results':     step_results,
        'all_models':       all_models,
        'num_class':        NUM_CLASS,
    }
