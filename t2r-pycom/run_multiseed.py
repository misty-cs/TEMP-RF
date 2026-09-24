#!/usr/bin/env python3
"""
run_multiseed.py — Run the full pipeline across multiple seeds and aggregate results.

Each seed gets its own output directory. Seeds already completed are skipped
unless --force is passed.

Usage:
    python3 run_multiseed.py
    python3 run_multiseed.py --seeds 42,7,13
    python3 run_multiseed.py --force --seeds 7,13,21,100
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from pathlib import Path

VENV_PYTHON = os.environ.get('T2R_PYTHON', sys.executable)
DEFAULT_SEEDS = [42, 7, 13, 21, 100]


def _build_cmd(
    python: str,
    seed: int,
    out_dir: Path,
    data_root: str,
    dataset_layout: str,
    file_key: str,
    location: str,
    finetune_days: str,
    traj_day: int,
    test_day: int,
    ft_n_train: int,
) -> list[str]:
    return [
        python, '-u', 'run_experiment.py',
        '--data_root',     data_root,
        '--dataset_layout', dataset_layout,
        '--file_key',      file_key,
        '--location',      location,
        '--output_root',   str(out_dir),
        '--seed',          str(seed),
        '--finetune_days', finetune_days,
        '--traj_day',      str(traj_day),
        '--test_day',      str(test_day),
        '--no_ft_always_reinit',
        '--ft_n_train',    str(ft_n_train),
        '--skip_plots',
    ]


def _parse_result(result_file: Path) -> dict | None:
    """Extract key metrics from results_phase4.txt (takes last occurrence)."""
    try:
        text = result_file.read_text()
    except FileNotFoundError:
        return None

    out = {}
    # Take the LAST match — file may accumulate results from multiple runs
    matches = re.findall(
        r'Multi-proto cosine traj \(Day \d+\)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)',
        text,
    )
    if matches:
        last = matches[-1]
        out['closed_acc'] = float(last[0])
        out['open_acc']   = float(last[1])
        out['unk_det']    = float(last[2])
        out['auroc']      = float(last[3])
    # Softmax baseline (last occurrence)
    sm = re.findall(r'Softmax \(no rejection\)\s+([\d.]+)', text)
    if sm:
        out['softmax_acc'] = float(sm[-1])
    return out if out else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root',   default=os.environ.get('T2R_DATA_ROOT', './data/neu'))
    parser.add_argument('--output_base', default='res_out_multiseed')
    parser.add_argument('--dataset_layout', default='auto',
                        choices=['auto', 'neu', 'pycom_indoor'])
    parser.add_argument('--file_key',    default='*.bin')
    parser.add_argument('--location',    default='')
    parser.add_argument('--finetune_days', default='2,3,4,5,6,7')
    parser.add_argument('--traj_day',    type=int, default=7)
    parser.add_argument('--test_day',    type=int, default=8)
    parser.add_argument('--ft_n_train',  type=int, default=2000)
    parser.add_argument('--python',      default=VENV_PYTHON)
    parser.add_argument('--seeds',       default=','.join(map(str, DEFAULT_SEEDS)),
                        help='Comma-separated seed list')
    parser.add_argument('--force',       action='store_true',
                        help='Re-run even if already complete')
    parser.add_argument('--dry_run',     action='store_true')
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(',') if s.strip()]
    output_base = Path(args.output_base)
    output_base.mkdir(parents=True, exist_ok=True)

    print('Multi-seed run')
    print('--------------')
    print(f'seeds        : {seeds}')
    print(f'output_base  : {output_base}')
    print(f'dataset      : {args.dataset_layout}  {args.file_key}  {args.location or "<none>"}')
    print(f'finetune_days: {args.finetune_days}  traj_day={args.traj_day}  test_day={args.test_day}')
    print(f'ft_n_train   : {args.ft_n_train}/class')
    print()

    all_results: dict[int, dict] = {}

    for seed in seeds:
        out_dir     = output_base / f'seed_{seed}'
        result_file = out_dir / 'results_phase4.txt'
        log_file    = output_base / f'seed_{seed}.log'

        # Check if already done
        if result_file.exists() and not args.force:
            r = _parse_result(result_file)
            if r:
                print(f'[seed {seed}] already done — closed_acc={r.get("closed_acc","?")}  '
                      f'AUROC={r.get("auroc","?")}')
                all_results[seed] = r
                continue
            else:
                print(f'[seed {seed}] result file exists but parse failed — re-running')

        out_dir.mkdir(parents=True, exist_ok=True)
        cmd = _build_cmd(
            args.python, seed, out_dir, args.data_root, args.dataset_layout,
            args.file_key, args.location, args.finetune_days, args.traj_day,
            args.test_day, args.ft_n_train
        )
        print(f'[seed {seed}] starting ...')
        print('  ' + ' '.join(cmd))

        if args.dry_run:
            continue

        t0 = time.time()
        with log_file.open('w') as lf:
            proc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, text=True)
        elapsed = time.time() - t0

        if proc.returncode != 0:
            print(f'[seed {seed}] FAILED after {elapsed/3600:.2f}h  (log: {log_file})')
            return 1

        r = _parse_result(result_file)
        if r:
            all_results[seed] = r
            print(f'[seed {seed}] done in {elapsed/3600:.2f}h — '
                  f'closed_acc={r["closed_acc"]:.4f}  AUROC={r["auroc"]:.4f}')
        else:
            print(f'[seed {seed}] done but could not parse results')

    if len(all_results) < 2:
        print('\nNot enough completed seeds to aggregate.')
        return 0

    # ── Aggregate ──────────────────────────────────────────────────────────
    import statistics
    metrics = ['closed_acc', 'open_acc', 'unk_det', 'auroc', 'softmax_acc']
    print('\n' + '=' * 55)
    print('  AGGREGATE RESULTS  (Multi-proto cosine trajectory)')
    print('=' * 55)
    print(f'  Seeds: {sorted(all_results.keys())}')
    print()
    for metric in metrics:
        vals = [r[metric] for r in all_results.values() if metric in r]
        if not vals:
            continue
        mean = statistics.mean(vals)
        std  = statistics.stdev(vals) if len(vals) > 1 else 0.0
        print(f'  {metric:<15}: {mean:.4f} ± {std:.4f}   '
              f'[{min(vals):.4f} – {max(vals):.4f}]  n={len(vals)}')

    # Save summary JSON
    summary = {
        'seeds':   sorted(all_results.keys()),
        'per_seed': {str(k): v for k, v in all_results.items()},
        'mean':    {m: statistics.mean([r[m] for r in all_results.values() if m in r])
                    for m in metrics if any(m in r for r in all_results.values())},
        'std':     {m: (statistics.stdev([r[m] for r in all_results.values() if m in r])
                        if len([r for r in all_results.values() if m in r]) > 1 else 0.0)
                    for m in metrics if any(m in r for r in all_results.values())},
    }
    summary_path = output_base / 'aggregate_results.json'
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f'\n  Summary saved → {summary_path}')
    print('=' * 55)

    # ── Generate plots from the first completed seed ───────────────────────
    # Main paper plots use one seed's full results; extended plots use the
    # aggregate_results.json for error bars across all seeds.
    best_seed = sorted(all_results.keys())[0]
    best_out  = output_base / f'seed_{best_seed}'
    print(f'\n  Generating plots from seed {best_seed} → {best_out}/paper_plots/')

    import sys
    sys.path.insert(0, str(Path(__file__).parent))

    try:
        from experiment_config import ExperimentConfig
        cfg_plot = ExperimentConfig(
            data_root      = args.data_root,
            output_root    = str(best_out),
            dataset_layout = args.dataset_layout,
            file_key       = args.file_key,
            location       = args.location,
            ft_n_train     = args.ft_n_train,
            finetune_days  = [int(d.strip()) for d in args.finetune_days.split(',')
                              if d.strip()],
            traj_day       = args.traj_day,
            test_day       = args.test_day,
        )

        from paper_plots import generate_all_paper_plots
        import numpy as np, tensorflow as tf
        import rf_models
        from temporal_trajectory import TemporalTrajectory

        def _load_p1():
            mp  = best_out / 'modelDir' / f'phase1_df_d{cfg_plot.init_day}.keras'
            np_ = best_out / 'modelDir' / f'phase1_norm_d{cfg_plot.init_day}.npz'
            return {'model_path': str(mp), 'norm_path': str(np_)}

        def _load_p2():
            sr_path = best_out / 'results_phase2.txt'
            step_results = []
            if sr_path.exists():
                for line in sr_path.read_text().splitlines():
                    if line.strip().startswith('day='):
                        parts = dict(p.split('=') for p in line.split('  ') if '=' in p)
                        row = {}
                        for k, v in parts.items():
                            if k == 'time' and v.endswith('s'):
                                v = v[:-1]
                            try:
                                row[k] = float(v)
                            except ValueError:
                                pass
                        step_results.append(row)
            return {'step_results': step_results}

        def _load_p3():
            traj_path = best_out / 'modelDir' / f'phase3_trajectory_d{cfg_plot.traj_day}.npz'
            norm_path = best_out / 'modelDir' / 'phase3_norm_stats.npz'
            last_day  = cfg_plot.finetune_days[-1]
            model_path = best_out / 'modelDir' / f'phase3_bn_adapted_d{cfg_plot.traj_day}.keras'
            if not model_path.exists():
                model_path = best_out / 'modelDir' / f'phase2_ft_d{last_day}.keras'
            traj  = TemporalTrajectory.load(str(traj_path))
            nstat = np.load(str(norm_path))
            return {
                'trajectory': traj,
                'model_path': str(model_path),
                'norm_mean':  nstat['mean'],
                'norm_std':   nstat['std'],
            }

        def _load_p4():
            res   = {}
            rpath = best_out / 'results_phase4.txt'
            if rpath.exists():
                for line in rpath.read_text().splitlines():
                    if '=' in line:
                        try:
                            k, v = line.split('=', 1)
                            res[k.strip()] = float(v.strip())
                        except Exception:
                            pass
            return res

        p4 = _load_p4()
        generate_all_paper_plots(cfg_plot, _load_p1(), _load_p2(), _load_p3(), p4)

        from paper_plots_extended import generate_all_extended_plots
        multiseed_summary = json.loads(summary_path.read_text())
        generate_all_extended_plots(cfg_plot, p4, multiseed_summary)

        print('  All plots generated.')
    except Exception as e:
        print(f'  Plot generation failed: {e}')
        import traceback; traceback.print_exc()

    # ── Paper-ready tables aggregated across seeds ────────────────────────
    # Reads only the LAST Phase-4 pass per seed, so an appended re-run cannot
    # silently mix configurations across tables.
    try:
        import subprocess
        agg = Path(__file__).parent / 'aggregate_tables.py'
        if agg.is_file():
            seed_dirs = sorted(str(p) for p in output_base.glob('seed_*')
                               if p.is_dir())
            if seed_dirs:
                print(f'\n  Aggregating tables across {len(seed_dirs)} seed(s) '
                      f'→ {output_base}/tables/')
                subprocess.run(
                    [sys.executable, str(agg), *seed_dirs,
                     '--out', str(output_base / 'tables')],
                    check=False,
                )
    except Exception as e:
        print(f'  Table aggregation failed: {e}')

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
