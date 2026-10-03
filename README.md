# TEMP-RFF: Session-by-Session Enrollment for Open-Set RF Fingerprinting

Code, protocol and results for TEMP-RFF, a framework that enrolls radio
transmitters session by session (initial training, sequential fine-tuning with
replay, reference-session adaptation with prototype banks) and evaluates
open-set decision rules on one shared backbone: identify enrolled devices and
reject devices that were never enrolled, across capture days.

## Datasets (not included; download from the sources)

| Name in code | Dataset | Days | Devices used | Input |
|---|---|---|---|---|
| `wisig` | WiSig ManySig/Full-WiSig, receiver node24-16 (Hanna et al.) | 4 dates | 16 (12 known + 4 unknown per split) | equalized preamble, 256 samples |
| `pycom` | Oregon State WiFi (Pycom), Indoor (Elmaghbub & Hamdaoui) | 5 | 20 (16 + 4) | packet-aligned raw IQ, 256 samples |
| `lora` | Oregon State LoRa, Diff_Days_Indoor (Elmaghbub & Hamdaoui, IEEE Access 2021) | 5 | 20 (16 + 4) | one SF7 preamble chirp, 1024 samples |

Place (or link) the data as `data/wisig/`, `data/pycom_indoor/` (Day_1 ...
Day_5) and `data/lora_indoor/` (Day1 ... Day5). The Pycom source was verified
file by file (`explore/pycom_verify/`); see `docs/PROTOCOL.md` section 14.
The LoRa source stores 11 recordings twice; `data/lora_indoor` is the
de-duplicated tree made by `python data_prep/lora_dedup.py <download> <tree>`
(both copies excluded; `docs/PROTOCOL.md` section 16.1).

## Install

    pip install -r requirements.txt        # Python 3.13, TensorFlow 2.20 (CPU)

## Run the complete study

    ./run_all.sh

Per-split Phase-1/2 tuning (validation data only) for all datasets, tuned
pilots with an audit gate, then the final runs (every device unknown once, five
seeds, joint-training control) and the aggregation. Each step is repeated
automatically after interruptions; failed attempts are kept as `*.failedN`.
Single steps: `python main.py --dataset wisig` prints the plan; `--go` runs.

## Where to look

| Path | Content |
|---|---|
| `docs/PROTOCOL.md` | the protocol (authoritative), all versions and changes with dates and reasons |
| `docs/FAIRNESS.md` | what every decision rule uses and how it rejects |
| `docs/METHODS_HISTORY.md` | decision log: how every value was found |
| `docs/review/` | external reviews of the code and protocol |
| `explore/` | exploratory analyses (disclosed; not confirmatory) |
| `res_tune/<dataset>/` | tuning: every candidate, its score and the selection per split |
| `res_pilot_tuned/`, `res/`, `res_joint/` | pilots and final runs: `common_eval.json`, `audit.json`, manifests |
| `res/<dataset>/SUMMARY.md`, `res/PRIMARY_*.md` | aggregated results and the primary statistical test |

Every run stores the SHA-256 of all code files, settings, software versions and
a data inventory (`manifest_*.json`); the audit (`audit.json`) checks group
disjointness, provenance, the test set and protocol invariance. The git history
shows which committed code produced which results.

## Code map

Active: `main.py`, `run_experiment.py`, `phase1_initial_training.py`,
`phase2_sequential_finetuning.py`, `phase3_trajectory_building.py`,
`finetune.py`, `common_eval.py`, `openmax.py`, `selection.py`, `audit_run.py`,
`provenance_checks.py`, `manifest.py`, `repro.py`, `aggregate.py`,
`joint_eval.py`, `tune_phase1.py`, `tune_phase2.py`, `datasets.py`,
`experiment_config.py`, `data_utils.py`, `load_slice_IQ.py`,
`signal_preprocessing.py`, `rf_models.py`, `temporal_trajectory.py`,
`tta_bn.py`, `pool_check.py`, `build_packet_cache.py`.
Present but switched off by the protocol's settings: `phase4_openset_evaluation.py`,
`paper_plots*.py`, `xsession_finetune.py`, `proto_improve.py`,
`embedding_whitening.py`.
