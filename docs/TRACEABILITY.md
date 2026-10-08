# Traceability: where every number of the paper comes from

Every reported number is computed by committed code from committed per-run
results. This file maps each part of the paper to its source. Paths are
relative to the repository root; `<ds>` is `wisig`, `pycom` or `lora`,
`<S>` a device split (0, 4, 8, 12[, 16]) and `<s>` a seed (42, 7, 13, 21, 100).

## 1. How one run is made

| Step | Code | Record in the run folder `res/<ds>/shift<S>/seed<s>/` |
|---|---|---|
| Phases 1-3 (training, fine-tuning, BN, banks) | `main.py` -> `run_experiment.py` -> `phase1_initial_training.py`, `phase2_sequential_finetuning.py`, `phase3_trajectory_building.py` | `run.log` (full training output), `results_phase2.txt`, `results_phase3.txt`, `manifest_train.json` |
| Which data went where (group-disjoint parts) | same | `modelDir/phase1_train_groups_d1.npz`, `phase2_train_groups_d*.npz`, `phase3_bank_groups_d*.npz` |
| Normalization statistics, Phase-2 centroids | same | `modelDir/*norm*.npz`, `modelDir/ft_centroids_phase2_d*.npz` |
| Phase-3/4 selection (P, W, decay, every rule's parameters) | `common_eval.py`, `selection.py` | `modelDir/phase3_selection.json` (SEL of every candidate on part B), `common_eval.json` (`selection`) |
| Phase 4 (all decision rules, all threshold protocols) | `common_eval.py`, `openmax.py` | `common_eval.json`, `common_eval.log`, `manifest_eval.json` |
| Audit (all checks; a run is used only if all pass) | `audit_run.py`, `provenance_checks.py` | `audit.json`, `audit.log` |
| Large artifacts (models, banks, calibration scores) | same | not in git; listed with size and SHA-256 in `docs/ARTIFACTS.tsv` |

Each manifest stores the SHA-256 of all code files, all settings, the software
versions and an inventory of the data files; `main.py` reuses a run only if all
of them match (`manifest.run_status`).

## 2. Paper elements

| Paper element | Source data | Produced by |
|---|---|---|
| Dataset table, device pools | `datasets.py`, `docs/pool_check_lora.txt`, `pool_check.py` | `pool_check.py --dataset <ds>` |
| LoRa duplicate exclusion | `data_prep/lora_duplicates.json` | `data_prep/lora_dedup.py` |
| Pycom copy verification | `explore/pycom_verify/` | scripts in that folder |
| Tuning grids, selection rule | `tune_phase1.py`, `tune_phase2.py` | -- |
| Tuning outcomes table | `res_tune/<ds>/selected_phase1.json`, `selected_phase2.json` | `tune_phase1.py`, `tune_phase2.py` |
| Every tuning candidate | `res_tune/<ds>/shift<S>/lam*_T*_seed42/phase1_acc.json`, `res_tune/<ds>/shift<S>/phase2/replay*_lr*/`, `res_tune/<ds>/shift<S>/tune.json` | same; logs in `docs/logs/tune1_*.txt`, `tune2_*.txt` |
| Exp. 1 main table, per-split table, tests | `res/<ds>/shift<S>/seed<s>/common_eval.json` (`rules`) | `aggregate.py` -> `docs/results_3seeds/`, `docs/results_5seeds/` (`SUMMARY_<ds>.md`, `PRIMARY_*.md`) |
| Exp. 2 phase contributions | `common_eval.json` (`stages`) | `aggregate.py` (section "Pipeline stages") |
| Exp. 2 joint-training control | `res_joint/<ds>/shift0/seed<s>/joint_eval.json`, `audit.json` | `main.py --joint`, `joint_eval.py`; summary in `aggregate.py` |
| Exp. 3 window W | `common_eval.json` (`window_rules`, rules `temp_rff_W1..W3`) | `aggregate.py` (section "Forced prototype window") |
| Exp. 3 defaults vs. selected | `common_eval.json` (rules `*_default`) | `aggregate.py` |
| Exp. 3 bank size P, Exp. 4 trajectories / t-SNE, Exp. 6 ROC and threshold sweep, Exp. 7 confusion / per device | saved models and data (hashes in `docs/ARTIFACTS.tsv`) | `analysis/make_figures.py` (added after the final runs; read-only, makes no choices) |
| Exp. 5 few-shot | `common_eval.json` (`fewshot`) | `aggregate.py` (section "Few-shot") |
| Exp. 6 threshold protocols | `common_eval.json` (`rules.*.fixed90`, `objective`, `tuned`) | `aggregate.py` |
| Appendix D (history, exploratory analyses, deviations) | `docs/PROTOCOL.md` (all versions and changes, dated), `docs/METHODS_HISTORY.md`, `explore/README.md`, `docs/superseded_runs/` | -- |
| Run order, interruptions, scheduler | `docs/logs/run_all.txt`, `docs/logs/governor.txt`, failed attempts kept as `*.failedN` | `run_all.sh`, `run_lora_parallel.sh`, `tools/governor.py` |

## 3. Which code produced which result

The git history records every code change; each stage of the study was
committed when it finished. Code was frozen during the study: tuning
identities and run manifests contain the SHA-256 of the code they ran, so a
result can be matched to the exact commit (`git log`, then compare the hashes
in the manifest with `manifest.code_sha256()` at that commit).

## 4. Data

The raw datasets are public and are not redistributed here (see `README.md`
for the sources). `data_prep/lora_duplicates.json` and the pool-check output
define exactly which LoRa files were used; every run's manifest contains the
data inventory (file names, sizes, modification times and a sampled content
hash).
