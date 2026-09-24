# Temporal Multi-Prototype Enrollment for Open-Set RF Fingerprinting

This repository contains the experiment code for the ICC 2027 submission
**"Temporal Multi-Prototype Enrollment for Open-Set RF Fingerprinting"**.

The code evaluates open-set RF fingerprinting under cross-session shift using
two datasets:

- **WiSig** equalized WiFi captures, receiver `node24-16`
- **Pycom WiFi indoor** raw time-domain I/Q captures

The implementation uses a shared four-phase pipeline:

1. **Initial embedding training** on the first enrollment session.
2. **Sequential fine-tuning** on later enrolled-device sessions.
3. **Prototype construction and threshold calibration** using enrolled devices
   only.
4. **Open-set evaluation** on a held-out deployment session containing enrolled
   and unknown devices.

All baseline decision rules share Phases 1-3. They differ only in Phase 4.

## Repository Layout

```text
t2r-wisig/       WiSig experiment code
t2r-pycom/       Pycom experiment code
figures/         Figure-generation scripts
run_campaign.sh  Convenience runner for WiSig and Pycom
```

The two experiment folders intentionally share the same four-phase learning and
evaluation pipeline while keeping dataset-specific loaders and preprocessing
separate. WiSig and Pycom have different file formats, capture structures, and
signal conditioning, so separating them avoids hiding dataset-specific
assumptions inside one large loader.

Datasets and trained model outputs are intentionally not included. Put each
dataset under the corresponding `data/` directory or pass a dataset path with
the command-line options described below.

## Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r t2r-wisig/requirements.txt
```

The WiSig and Pycom folders use the same dependency set.

## WiSig Protocol

The default WiSig setup is:

- Phase 1: session 1
- Phase 2: session 2
- Phase 3: session 3
- Phase 4: session 4
- 12 enrolled devices and 4 unknown devices
- 256-sample I/Q inputs

Run one seed:

```bash
cd t2r-wisig
python run_experiment.py \
  --dataset_layout wisig_full \
  --data_root data \
  --wisig_full_receiver node24-16 \
  --output_root res/seed42 \
  --seed 42 \
  --device_shift 0 \
  --n_known 12 \
  --n_unknown 4 \
  --init_day 1 \
  --finetune_days 2 \
  --traj_day 3 \
  --test_day 4 \
  --num_slice 780 \
  --slice_len 256 \
  --ft_n_train 150 \
  --no_ft_always_reinit \
  --skip_plots
```

Run the WiSig reference-window sweep:

```bash
cd t2r-wisig
bash run_window_sweep.sh
```

## Pycom Protocol

The default Pycom setup is:

- Phase 1: day 1
- Phase 2: days 2 and 3
- Phase 3: day 4
- Phase 4: day 5
- 16 enrolled devices and 4 unknown devices
- capture-disjoint splitting

Run one seed:

```bash
cd t2r-pycom
python run_experiment.py \
  --dataset_layout pycom_indoor \
  --data_root data/pycom_indoor \
  --output_root res/seed42 \
  --seed 42 \
  --device_shift 0 \
  --n_known 16 \
  --n_unknown 4 \
  --ft_split_mode capture \
  --init_day 1 \
  --finetune_days 2,3 \
  --traj_day 4 \
  --test_day 5 \
  --num_slice 780 \
  --slice_len 256 \
  --ft_n_train 150 \
  --no_ft_always_reinit \
  --skip_plots
```

Run the Pycom reference-window sweep:

```bash
cd t2r-pycom
bash run_window_sweep.sh
```

## Full Convenience Run

From the repository root:

```bash
bash run_campaign.sh
```

This runs WiSig and Pycom with five seeds. Finished seeds are skipped if their
`results_phase4.txt` file already exists.

## Notes

- Unknown-device samples and deployment-session labels are used only in final
  Phase 4 scoring.
- Rejection thresholds are calibrated using enrolled-device data only.
- Generated result folders are ignored by Git. Commit source code and scripts,
  not model checkpoints or dataset files.
