#!/usr/bin/env python3
"""
run_experiment.py — Main pipeline entry point.

Usage examples
--------------
# Standard run:
python run_experiment.py --data_root /path/to/neu2

# Chain fine-tune weights across days instead of reinitialising from Phase 1:
python run_experiment.py --data_root /path/to/neu2 --no_ft_always_reinit

# Resume from Phase 3:
python run_experiment.py --data_root /path/to/neu2 --resume_from 3
"""

import argparse
import os
import time

import numpy as np
import tensorflow as tf

from experiment_config import ExperimentConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root',        default=os.environ.get('T2R_DATA_ROOT', './data/neu'))
    parser.add_argument('--output_root',      default='res_out')
    parser.add_argument('--wisig_full_receiver', default='node24-16')
    parser.add_argument('--wisig_rx_index', type=int, default=0)
    parser.add_argument('--wisig_equalized', type=int, default=0)
    parser.add_argument('--dataset_layout',   default='auto',
                        choices=['folder_iq','neu','pycom_indoor','wisig_manysig','wisig_full'],
                        help='Dataset folder layout. auto detects Day_1/... as pycom_indoor.')
    parser.add_argument('--file_key',         default='*.bin',
                        help='IQ capture glob inside each device folder. auto changes *.bin to *.dat for pycom_indoor.')
    parser.add_argument('--location',         default='',
                        help='Optional subfolder inside each device folder.')
    parser.add_argument('--num_slice',        type=int,   default=3000)
    # Sensitivity-analysis knobs. Defaults match the inherited NEU settings,
    # so the main results are unchanged; these exist so the Pycom sensitivity
    # sweep can vary one factor at a time from a pre-declared grid.
    parser.add_argument('--slice_len',        type=int,   default=288,
                        help='Samples per slice (sensitivity grid: 256/288/512)')
    parser.add_argument('--aug_snr_db',       type=float, default=20.0,
                        help='AWGN SNR for augmentation; lower = noisier')
    parser.add_argument('--aug_amp_db',       type=float, default=2.0,
                        help='Amplitude jitter (dB)')
    parser.add_argument('--aug_prob',         type=float, default=0.7,
                        help='Probability an augmentation is applied')
    parser.add_argument('--n_known',          type=int,   default=16)
    parser.add_argument('--n_val_dev',        type=int,   default=0)
    parser.add_argument('--n_unknown',        type=int,   default=4)
    parser.add_argument('--init_day',         type=int,   default=1,
                        choices=range(1, 10))
    parser.add_argument('--finetune_days',    type=str,   default='2,3,4,5,6',
                        help='Comma-separated days for sequential fine-tuning')
    parser.add_argument('--traj_day',         type=int,   default=7,
                        choices=range(1, 10))
    parser.add_argument('--test_day',         type=int,   default=8,
                        choices=range(1, 10))
    parser.add_argument('--ft_n_train',       type=int,   default=1600,
                        help='Labelled samples per class per fine-tune step')
    parser.add_argument('--ft_always_reinit', dest='ft_always_reinit',
                        action='store_true', default=True,
                        help='Re-initialise from Phase 1 model every fine-tune step')
    parser.add_argument('--no_ft_always_reinit', dest='ft_always_reinit',
                        action='store_false',
                        help='Chain each fine-tune step from the previous fine-tuned model')
    parser.add_argument('--ft_split_mode',    default='random',
                        choices=['random', 'capture'],
                        help="Phase-2 in-day train/test split. 'random' is the "
                             "historical shuffle (train/test slices may share a "
                             "recording, inflating in-day accuracy); 'capture' "
                             "splits across different capture files.")
    parser.add_argument('--traj_ewma',        type=float, default=0.8)
    parser.add_argument('--traj_threshold',   type=float, default=None)
    parser.add_argument('--aug_multipath_taps', type=int, default=0,
                        help='random FIR echo taps added to training-time '
                             'channel augmentation (0 = original flat channel)')
    parser.add_argument('--aug_multipath_mag', type=float, default=0.0,
                        help='max echo magnitude relative to the direct path')
    parser.add_argument('--seed',             type=int,   default=42)
    parser.add_argument('--device_shift',     type=int,   default=0,
                        help='Rotate which devices are unknown (e.g. 0, 4, 8). '
                             'Requires a full pipeline run per shift.')
    parser.add_argument('--resume_from',      type=int,   default=1,
                        choices=[1, 2, 3, 4],
                        help='Resume from this phase. Default 1 = run all phases. '
                             'Use 2 to skip Phase 1 (requires existing phase1_df_d{init_day}.keras). '
                             'Use 3 to skip Phases 1+2. Use 4 to re-run Phase 4 only.')
    parser.add_argument('--skip_plots',       action='store_true',
                        help='Skip paper plot regeneration after Phase 4.')
    args = parser.parse_args()
    finetune_days = [int(d.strip()) for d in args.finetune_days.split(',')
                     if d.strip()]
    unsupported_days = [d for d in finetune_days if d not in range(1, 10)]
    if unsupported_days:
        parser.error(f'finetune_days must use days 1-9 only; got {unsupported_days}')

    cfg = ExperimentConfig(
        data_root        = args.data_root,
        output_root      = args.output_root,
        dataset_layout   = args.dataset_layout,
        wisig_full_receiver = args.wisig_full_receiver,
        wisig_rx_index   = args.wisig_rx_index,
        wisig_equalized  = args.wisig_equalized,
        file_key         = args.file_key,
        location         = args.location,
        n_known          = args.n_known,
        n_val_dev        = args.n_val_dev,
        n_unknown        = args.n_unknown,
        init_day         = args.init_day,
        finetune_days    = finetune_days,
        traj_day         = args.traj_day,
        test_day         = args.test_day,
        num_slice        = args.num_slice,
        slice_len        = args.slice_len,
        aug_snr_db       = args.aug_snr_db,
        aug_amp_db       = args.aug_amp_db,
        aug_prob         = args.aug_prob,
        ft_n_train       = args.ft_n_train,
        ft_always_reinit = args.ft_always_reinit,
        ft_split_mode    = args.ft_split_mode,
        traj_ewma        = args.traj_ewma,
        traj_threshold   = args.traj_threshold,
        seed             = args.seed,
        device_shift     = args.device_shift,
        aug_multipath_taps = args.aug_multipath_taps,
        aug_multipath_mag  = args.aug_multipath_mag,
    )

    # Phase 2 fine-tunes through a separate preprocessing entry point, so the
    # same channel model has to be pushed onto it explicitly.
    import finetune as _ft_mod
    _ft_mod.MULTIPATH_TAPS = cfg.aug_multipath_taps
    _ft_mod.MULTIPATH_MAG  = cfg.aug_multipath_mag

    np.random.seed(cfg.seed)
    tf.random.set_seed(cfg.seed)

    print(cfg.summary())

    t_total = time.time()

    # ── Phase 1 ───────────────────────────────────────────────────────────
    if args.resume_from <= 1:
        from phase1_initial_training import run_phase1
        phase1_result = run_phase1(cfg)
    else:
        phase1_result = _resume_phase1(cfg)

    # ── Phase 2 ───────────────────────────────────────────────────────────
    if args.resume_from <= 2:
        from phase2_sequential_finetuning import run_phase2
        phase2_result = run_phase2(cfg, phase1_result)
    else:
        phase2_result = _resume_phase2(cfg)

    # ── Phase 3 ───────────────────────────────────────────────────────────
    if args.resume_from <= 3:
        from phase3_trajectory_building import run_phase3
        phase3_result = run_phase3(cfg, phase2_result)
    else:
        phase3_result = _resume_phase3(cfg)

    # ── Phase 4 ───────────────────────────────────────────────────────────
    from phase4_openset_evaluation import run_phase4
    final_results = run_phase4(cfg, phase3_result)

    # ── Paper plots ───────────────────────────────────────────────────────
    if args.skip_plots:
        print('[paper_plots] skipped by --skip_plots')
    else:
        try:
            from paper_plots import generate_all_paper_plots
            generate_all_paper_plots(cfg, phase1_result, phase2_result,
                                     phase3_result, final_results)
        except Exception as e:
            print(f'[paper_plots] skipped: {e}')

        try:
            import json
            multiseed_summary = None
            ms_path = os.path.join('res_out_multiseed', 'aggregate_results.json')
            if os.path.exists(ms_path):
                with open(ms_path) as f:
                    multiseed_summary = json.load(f)
            from paper_plots_extended import generate_all_extended_plots
            generate_all_extended_plots(cfg, final_results, multiseed_summary)
        except Exception as e:
            print(f'[paper_plots_extended] skipped: {e}')

    total_time = time.time() - t_total
    print(f'\nExperiment complete!  Total time: {total_time:.0f}s')
    return final_results


# ── Resume helpers ────────────────────────────────────────────────────────

def _resume_phase1(cfg: ExperimentConfig) -> dict:
    model_path = os.path.join(cfg.model_dir, f'phase1_df_d{cfg.init_day}.keras')
    norm_path  = os.path.join(cfg.model_dir, f'phase1_norm_d{cfg.init_day}.npz')
    val_model_path = os.path.join(
        cfg.model_dir, 'phase1_validation_models',
        f'phase1_val_d{cfg.init_day}.keras',
    )
    test_model_path = os.path.join(
        cfg.model_dir, 'phase1_test_models',
        f'phase1_test_d{cfg.init_day}.keras',
    )
    for p in [model_path, norm_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f'Resume file missing: {p}')
    return {
        'model_path':      model_path,
        'norm_path':       norm_path,
        'val_model_path':  val_model_path if os.path.exists(val_model_path) else None,
        'test_model_path': test_model_path if os.path.exists(test_model_path) else None,
        'num_class':       len(cfg.all_known_ids),
    }


def _resume_phase2(cfg: ExperimentConfig) -> dict:
    last_day   = cfg.finetune_days[-1]
    model_path = os.path.join(cfg.model_dir, f'phase2_ft_d{last_day}.keras')
    norm_path  = os.path.join(cfg.model_dir, f'phase2_norm_d{last_day}.npz')
    for p in [model_path, norm_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f'Resume file missing: {p}')

    all_models = [(cfg.init_day,
                   os.path.join(cfg.model_dir, f'phase1_df_d{cfg.init_day}.keras'),
                   os.path.join(cfg.model_dir, f'phase1_norm_d{cfg.init_day}.npz'))]
    for d in cfg.finetune_days:
        all_models.append((
            d,
            os.path.join(cfg.model_dir, f'phase2_ft_d{d}.keras'),
            os.path.join(cfg.model_dir, f'phase2_norm_d{d}.npz'),
        ))

    return {
        'final_model_path': model_path,
        'final_norm_path':  norm_path,
        'all_models':       all_models,
        'num_class':        len(cfg.all_known_ids),
    }


def _resume_phase3(cfg: ExperimentConfig) -> dict:
    from temporal_trajectory import TemporalTrajectory

    model_path = os.path.join(cfg.model_dir,
                              f'phase2_ft_d{cfg.finetune_days[-1]}.keras')
    traj_path  = os.path.join(cfg.model_dir,
                              f'phase3_trajectory_d{cfg.traj_day}.npz')
    norm_path  = os.path.join(cfg.model_dir, 'phase3_norm_stats.npz')
    calib_path = os.path.join(cfg.model_dir, f'phase3_calibration_d{cfg.traj_day}.npz')

    for p in [model_path, traj_path, norm_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f'Resume file missing: {p}')

    traj   = TemporalTrajectory.load(traj_path)
    nstats = np.load(norm_path)

    threshold = cfg.traj_threshold or 2.0
    history_days_keep = None
    history_day_decay = 1.0
    trajectory_classifier = None
    sweep_path = os.path.join(cfg.results_dir, 'results_phase3.txt')
    if os.path.exists(sweep_path):
        with open(sweep_path) as f:
            for line in f:
                if 'using threshold:' in line:
                    try:
                        threshold = float(line.split('using threshold:')[1].strip())
                    except ValueError:
                        pass
                elif line.strip().startswith('best_days_keep='):
                    raw = line.split('=', 1)[1].strip()
                    history_days_keep = None if raw == 'None' else int(raw)
                    trajectory_classifier = trajectory_classifier or 'history_mahalanobis'
                elif line.strip().startswith('best_day_decay='):
                    history_day_decay = float(line.split('=', 1)[1].strip())
                    trajectory_classifier = trajectory_classifier or 'history_mahalanobis'
                elif line.strip().startswith('best_n_cosine_prototypes='):
                    trajectory_classifier = 'cosine_prototypes'

    return {
        'model_path':      model_path,
        'norm_mean':       nstats['mean'],
        'norm_std':        nstats['std'],
        'norm_path':       norm_path,
        'trajectory':      traj,
        'trajectory_path': traj_path,
        'calibration_path': calib_path if os.path.exists(calib_path) else None,
        'threshold':       threshold,
        'trajectory_classifier': trajectory_classifier,
        'history_days_keep': history_days_keep,
        'history_day_decay': history_day_decay,
        'num_class':       len(cfg.all_known_ids),
    }


if __name__ == '__main__':
    main()
