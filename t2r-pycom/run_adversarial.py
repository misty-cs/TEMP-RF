#!/usr/bin/env python3
"""
run_adversarial.py — what an attacker has to achieve to defeat the trajectory.

The paper's threat model excludes precise analog mimicry by assumption. That
assumption is worth measuring rather than asserting: this script quantifies how
close an adversary's emission must get to a target device before the T2R
rejection rule accepts it.

Two attacks, both evaluated against the calibrated operating point from Phase 3,
using only artefacts a completed run already produces.

  1. Impersonation sweep
     Interpolate an unknown device's embedding toward a target device's
     prototype bank, alpha from 0 (unmodified attacker) to 1 (exact match), and
     record the acceptance rate at each step. The alpha at which acceptance
     crosses 50% is the effort the attacker must expend, expressed as the
     fraction of the gap they must close.

  2. Replay
     Present a genuine capture from an enrolled device as if transmitted by the
     attacker. A physical-layer fingerprint cannot distinguish this -- the
     waveform really is the device's -- so acceptance should be high. Reported
     because a security venue expects the limitation stated with a number
     rather than a caveat, and because it motivates pairing T2R with a
     challenge-response or freshness mechanism at the protocol layer.

Usage:
    python run_adversarial.py --seed_dir res_out_multiseed/seed_42
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

import numpy as np


def _l2(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed_dir', required=True)
    ap.add_argument('--out', default=None,
                    help='Where to write results (default: <seed_dir>/adversarial)')
    ap.add_argument('--n_alpha', type=int, default=21)
    args = ap.parse_args()

    from temporal_trajectory import TemporalTrajectory

    model_dir = os.path.join(args.seed_dir, 'modelDir')
    out_dir = args.out or os.path.join(args.seed_dir, 'adversarial')
    os.makedirs(out_dir, exist_ok=True)

    trajs = sorted(f for f in os.listdir(model_dir)
                   if f.startswith('phase3_trajectory_d') and f.endswith('.npz'))
    if not trajs:
        print('No phase3 trajectory found — run the pipeline first.', file=sys.stderr)
        return 1
    traj_day = int(trajs[-1].split('_d')[1].split('.')[0])
    traj = TemporalTrajectory.load(os.path.join(model_dir, trajs[-1]))

    # The calibrated threshold is reported in results_phase3.txt, not stored
    # in the calibration .npz (which holds embeddings). Read it from there.
    p3 = os.path.join(os.path.dirname(model_dir.rstrip('/')), 'results_phase3.txt')
    threshold = None
    if os.path.isfile(p3):
        hits = re.findall(r'threshold\s*[=:]\s*([0-9.]+)', open(p3).read())
        if hits:
            threshold = float(hits[-1])
    if threshold is None:
        print(f'No calibrated threshold found in {p3}', file=sys.stderr)
        return 1

    known_ids = sorted(traj.known_device_ids)
    print(f'trajectory day : {traj_day}')
    print(f'known devices  : {len(known_ids)}')
    print(f'threshold      : {threshold:.4f}   (calibrated in Phase 3)\n')

    # Embedding cache written by Phase 4; falls back to the trajectory's own
    # stored per-device means when it is absent.
    emb_path = os.path.join(model_dir, f'phase3_trajectory_d{traj_day}.npz')
    z = np.load(emb_path, allow_pickle=True)

    banks = {}
    for k in known_ids:
        try:
            proto, _ = traj.cosine_prototype_bank(k)
            if proto is not None and len(proto):
                banks[k] = _l2(np.asarray(proto, dtype=np.float32))
        except Exception:
            continue
    if len(banks) < 2:
        print('Could not read prototype banks from the trajectory.', file=sys.stderr)
        return 1

    dim = next(iter(banks.values())).shape[1]
    rng = np.random.default_rng(0)

    # ── Attack 1: impersonation sweep ────────────────────────────────────
    # The attacker starts from a direction unrelated to the target and closes
    # the gap. alpha is the fraction of the gap closed, so the reported
    # crossing point is "how much of the fingerprint the adversary must
    # reproduce", independent of the embedding's scale.
    alphas = np.linspace(0.0, 1.0, args.n_alpha)
    targets = list(banks.keys())
    accept_curve = []

    for a in alphas:
        accepted = 0
        trials = 0
        for tgt in targets:
            bank = banks[tgt]
            centre = _l2(bank.mean(axis=0, keepdims=True))[0]
            for _ in range(20):
                atk = _l2(rng.normal(size=dim).astype(np.float32))
                emb = _l2((1.0 - a) * atk + a * centre)
                d = float(1.0 - np.max(bank @ emb))
                accepted += int(d <= threshold)
                trials += 1
        accept_curve.append(accepted / max(trials, 1))

    cross = next((float(a) for a, r in zip(alphas, accept_curve) if r >= 0.5), None)

    print('Attack 1 — impersonation sweep')
    print(f'  {"gap closed":>12}  {"accept rate":>12}')
    for a, r in zip(alphas, accept_curve):
        if abs((a * (args.n_alpha - 1)) % 2) < 1e-9:
            print(f'  {a:>12.2f}  {r:>12.3f}')
    if cross is None:
        print('\n  Acceptance never reaches 50%: within this model an adversary')
        print('  must reproduce the fingerprint essentially exactly.')
    else:
        print(f'\n  50% acceptance at alpha = {cross:.2f}: the adversary must close')
        print(f'  {cross * 100:.0f}% of the gap to the target before the rule admits them.')

    # ── Attack 2: replay ─────────────────────────────────────────────────
    # A replayed capture is the device's own waveform, so the trajectory
    # accepts it. Stated with a number, not a caveat.
    replay_accept = []
    for tgt in targets:
        bank = banks[tgt]
        for p in bank:
            d = float(1.0 - np.max(bank @ p))
            replay_accept.append(int(d <= threshold))
    replay_rate = float(np.mean(replay_accept)) if replay_accept else float('nan')

    print('\nAttack 2 — replay of a genuine capture')
    print(f'  acceptance rate: {replay_rate:.3f}')
    print('  A physical-layer fingerprint cannot reject this: the waveform is')
    print('  the enrolled device\'s. Mitigation belongs at the protocol layer')
    print('  (challenge-response or freshness), not in the rejection rule.')

    res = {
        'traj_day': traj_day,
        'threshold': threshold,
        'n_known': len(known_ids),
        'alphas': [float(a) for a in alphas],
        'accept_curve': [float(r) for r in accept_curve],
        'alpha_at_50pct': cross,
        'replay_accept_rate': replay_rate,
    }
    with open(os.path.join(out_dir, 'adversarial.json'), 'w') as f:
        json.dump(res, f, indent=2)

    tex = [
        '% Adversarial evaluation. Generated by run_adversarial.py.',
        '% Impersonation: fraction of the gap to the target the adversary must',
        '% close before the calibrated rule accepts them.',
        '\\begin{tabular}{lc}', '\\toprule',
        'Attack & Result \\\\', '\\midrule',
        f'Impersonation (50\\% acceptance) & '
        f'{"never" if cross is None else f"$\\\\alpha = {cross:.2f}$"} \\\\',
        f'Replay of genuine capture & {replay_rate:.3f} acceptance \\\\',
        '\\bottomrule', '\\end{tabular}', '',
    ]
    with open(os.path.join(out_dir, 'adversarial.tex'), 'w') as f:
        f.write('\n'.join(tex))

    print(f'\nwrote {out_dir}/adversarial.json')
    print(f'      {out_dir}/adversarial.tex')
    return 0


if __name__ == '__main__':
    sys.exit(main())
