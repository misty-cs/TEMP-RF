# TEMP-RFF evaluation protocol — current version (v2)

This file is the ONE current description of the protocol. It replaces the
descriptions in METHODS_HISTORY.md (the dated decision log: how each value was
found and what changed). FAIRNESS.md lists per rule what it uses and how it
rejects. If code and this file disagree, the code is wrong or this file must be
updated before any run.

Version v2 was written on 2026-09-27/28, before any final run, after three
external reviews (review/PROJECT_AUDIT.md, PROJECT_RECHECK.md,
PROJECT_RECHECK_2026-09-28.md). The primary comparison family was fixed on
2026-09-28 (section 7). v2 is frozen: any later change is a new version with
its date and reason.

## 1. Data and cohorts

| | WiSig ManySig (equalized) | Pycom indoor (raw 802.11b) |
|---|---|---|
| Input | first 256 samples of each released packet (preamble), 25 MS/s | packet-aligned 256-sample slices, 25 MS/s (DC removed, gap detection, template alignment, quality \|corr\| >= 0.8 to the file's own template) |
| CFO | kept (as released) | kept |
| Devices | 16 transmitters, receiver node24-16 | 20 devices (folders 1-4, 6-13, 16-19, 21, 24, 25, 26) |
| Sessions | 03-01 Phase 1, 03-08 Phase 2, 03-15 reference, 03-23 test | Day 1 Phase 1, Days 2-3 Phase 2, Day 4 reference, Day 5 test |
| Groups | bursts | capture files |
| Slices | 780 per device and session (round-robin over bursts) | 780 per device and session (water-filling over files) |

Cohort rule (label-free, identical for every device):
- WiSig: transmitter present on all four dates (availability); ranked by the
  worst **pre-test** date burst count, minimum 780 signals on pre-test dates.
  The test date is not used for ranking. Top 16.
- Pycom (`pool_check.py`): no duplicate recordings with another device (hash
  scan over all days: a data-integrity check of identical files); quality on
  the **pre-test** days only (>= 780 good packets, >= 3 files with >= 50 good
  packets per day); availability on the test day (all 5 files).
- Both cohorts were first built with a rule that also looked at the test day;
  the pre-test-only rule gives the same devices (WiSig: same 16, same split
  membership; Pycom: device_5 and device_14 fail on pre-test days). Disclose.

NEU is excluded (equalized 48-value blocks without preamble; every slicing
was at chance).

Data identity (`manifest._data_inventory`, recorded in every manifest and in
every tuning identity): WiSig: name, size, mtime and full SHA-256 of each
session file. Pycom (466 GB raw): per session, every file of the device pool
with size, mtime and a sampled content hash (three 64 KiB blocks: start,
middle, end) — a documented weaker identity than a full hash; plus the SHA-256
of the packet-detector source (`load_slice_IQ.packet_code_id`), which is also
part of every packet-cache key, so cached packets are never reused after a
detector change. A missing or empty session or device folder makes the
identity fail.

## 2. Device splits and seeds

Every device is unknown exactly once: WiSig shifts 0, 4, 8, 12 (12 known, 4
unknown); Pycom shifts 0, 4, 8, 12, 16 (16 known, 4 unknown). Seeds 42, 7, 13,
21, 100 per split. Seeds of one split share devices and recordings; splits
overlap in enrolled devices.

## 3. Hyperparameter selection (validation data only, per split)

One score and one tie rule for every selection (`selection.py`):
SEL = 0.5 x closed accuracy + 0.5 x leave-one-device-out (LOCO) AUROC;
candidates within 0.001 of the best are tied; a tied prior default is kept,
otherwise the simplest candidate.

| Stage | What | Grid | Data | Default / simplicity |
|---|---|---|---|---|
| Phase 1 (`tune_phase1.py`) | SupCon lambda, T | lambda {0, .25, .5, 1} x T {.05, .07, .1} (10) | Phase-1 validation groups of the init session; nearest centroid of training embeddings | (0.5, 0.07) / smaller lambda, T |
| Phase 2 (`tune_phase2.py`) | replay weight, stage-2 lr | {0, .1, .2, .5} x {1e-5, 5e-5, 1e-4} (12) | Phase-2 validation groups, mean over fine-tuning sessions | (0.2, 5e-5) / smaller replay, lr |
| Phase 3 (every run) | EWMA alpha | {0.5, 0.8, 1.0} | reference Part B | config default / closest to 1 |
| Phase 3 (every run) | bank P, window W, decay | P {1, 5, 20, 50, 100, all} x W 1..N x {1, .8, .6} | Part B | (all, 1, 1) / W, 1-decay, P |
| common_eval (every run) | baselines | see FAIRNESS.md | Part B | per rule |

Phase 1 and 2 are tuned per split on that split's known devices only, seed 42,
same epochs as the final runs. Unknown devices, the reference session and the
test session are never used.

Provenance of a selection: every tuning result stores an identity hash of the
code it runs (with all imported local modules), all T2R_* settings, software
versions, the data inventory, seed, epochs and split; Phase 2 also the Phase-1
selection identity and the Phase-1 checkpoint SHA-256. A split is selected
only when its whole grid has current results. main.py recomputes both
identities before a run and refuses a split whose selection is stale.

## 4. Pipeline per run (`main.py` -> `run_experiment.py`)

- Phase 1: CNN (128-d L2-normalised embedding), CE + lambda x SupCon(T). Split
  by groups: WiSig bursts 70/15/15, Pycom 3/1/1 files (train/val/test).
- Phase 2: per fine-tuning session 150 labels per device, group-disjoint
  127 train / 23 validation / 60 held-out. Stage 1 head only; stage 2 all
  layers with CE + replay x alignment to the previous session's centroids.
  **Replay uses only training slices**: first step the Phase-1 training groups,
  later steps the previous session's 127 training slices per device.
- Phase 3: reference session split by whole groups into A 50% / B 25% / C 25%
  (split depends only on labels and groups). **Batch-norm statistics are
  adapted on Part A only** (unlabelled). Banks of earlier sessions use only
  groups that were never trained on (Phase 1 val/test groups, Phase 2 non-training
  groups); Phase 3 stops if a provenance file is missing. Saved: A/B/C indices
  and groups, bank (device, group) identities, selection table.
- Phase 4 (legacy) is skipped. All numbers come from `common_eval.py`.

## 5. Decision rules and thresholds (`common_eval.py`)

Every rule returns (prediction, unknown score); a slice is rejected if its
score is above the threshold. The **same** prediction and score are used by
every threshold protocol (checked: the tuned protocol reproduces each rule's
closed accuracy and AUROC; the run stops otherwise).

| Protocol | Pseudo-unknowns | Threshold |
|---|---|---|
| fixed90 | none | 90th percentile of the rule's scores on **all** of Part C |
| objective | hardest 25% of Part C per device by static-centroid distance (a stress subset fixed once for every rule) | max 0.5 x (accepted accuracy + pseudo-unknown rejection), >= 50% acceptance of the other 75% |
| tuned (primary) | LOCO: slices of device c scored by the same rule **with device c removed** — refitted (probe, k-NN), recalibrated (OpenMax), renormalised over the other logits (MSP), or with its references removed (centroid / prototype rules) | candidates on C (accept 80/90/95 %, Youden's J); method chosen on B by 0.5 x (accepted and correct) + 0.5 x (pseudo-unknown rejected), ties -> q90 |

Test data only scores. The shared network has seen every known device; this
holds equally for every rule.

Also reported: forced windows (temp_rff_W*), untuned defaults (*_default),
prototype-weight ablation (temp_rff_nowt, fixed settings of temp_rff),
temp_rff_p1 (own W/decay selection), few-shot (temp_rff, static, knn and the
equal-history controls static_window, knn_window; same support), pipeline
stages (Part-A centroids), raw-input centroid, joint-training control,
selection-weight sensitivity (0.3/0.5/0.7 with the same tie rule).

## 6. Run validity (`audit_run.py`, `manifest.py`)

Each process of a run gets PYTHONHASHSEED = seed and TF_DETERMINISTIC_OPS = 1
from the launcher. The audit fails closed on missing provenance and checks:
sessions; devices; Phase-1/2 parts present, non-empty, all devices, group-disjoint;
reference parts reproduce the saved indices, fresh-loaded groups and labels,
all devices, group-disjoint, BN on Part A; bank identities share no group with
training groups; test set recomputed (input hash, per-device counts = 780);
protocol invariance; selection sensitivity; all rules and selection tables;
Part-A reload exact; same code, same settings, determinism; artifacts present.
The audit stores SHA-256 of the result, both manifests, the Phase-3 model and
calibration file.

A run counts as complete (main.py skip; aggregate.py use) only if the audit
passed, every audited artifact is unchanged, and code, T2R_* settings, the
effective configuration (without output path, seed and split), the software
versions of training and evaluation, and the data identity match the current
ones. The audit also checks that training (run_experiment.py CLI) and
evaluation (datasets.make_cfg) used the same configuration and versions.
A folder with other code/settings/data or without a training manifest is
STALE: it is neither reused nor overwritten. aggregate.py refuses a tag whose
runs come from more than one study (code, settings except per-split tuned
values, data inventory, training and evaluation versions, configuration).
Joint runs have their own audit (Phase-1 groups, joint sessions, devices, test
budget, same code/settings/versions, determinism) and artifact hashes; a joint
run is paired only if its audit passed, its files are unchanged, its test input
equals the pipeline run's, and it belongs to the same study with the same
Phase-1 settings (only T2R_JOINT_DAYS differs).

## 7. Statistics (`aggregate.py --tags wisig pycom`)

Primary outcomes: closed accuracy, AUROC, tuned known-ok, tuned unknown
detection. Primary family (fixed 2026-09-28, before any confirmatory run, by
the author's decision): TEMP-RFF vs msp, probe, openmax, static, knn,
static_window, knn_window x 4 outcomes x both datasets = 56 tests. Test: split
level (seeds averaged per split), paired t-test and Wilcoxon, each
Holm-corrected over these 56 (`aggregate.py --tags wisig pycom`). All other
rules (untuned defaults, temp_rff_p1, temp_rff_nowt, temp_rff_W*, raw
centroid) are secondary: split-level estimates with uncorrected p-values,
exploratory only. Per-run tests are secondary.

With 4-5 splits the smallest two-sided Wilcoxon p is 0.125 / 0.0625, so
Wilcoxon cannot reach significance after correction. Low power does not make
the t-test's assumptions hold: with 4-5 dependent splits no significance is
promised, and the paper reports effect estimates, their spread and split wins
first. Missing and excluded runs are listed at the top of every summary.

## 8. Limits to state in the paper

- Shared-backbone comparison: Phase 1/2 settings are selected with a
  centroid-based proxy and shared by all rules; not a comparison with
  independently trained methods.
- Unequal tuning budgets (TEMP-RFF 54/72 bank candidates + alpha; knn_window
  24/32; static_window 3/4; temp_rff_p1 9/12; OpenMax 30; knn 8; probe 4); defaults rows show every rule untuned.
- Part B is reused for method settings and the tuned protocol's method choice.
- Historical test feedback: one tie-rule change after an old-model diagnostic
  (FAIRNESS.md section 0).
- Cohorts are availability/quality-conditioned (section 1).
- Splits overlap in devices; seeds do not add capture sessions.

## 9. Order of work

tune_phase1 (per split) -> tune_phase2 (per split) -> pilot (one split x one
seed per dataset, full audit) -> full runs (WiSig 20, Pycom 25, joint) ->
`aggregate.py --tags wisig pycom` -> paper.

## 10. Pilot and exploratory steps after v2 (disclosed in the paper)

- 2026-09-28 pilots (`res_pilot/`, split 0, seed 42, Phase-1/2 code defaults):
  WiSig and Pycom passed the audit. The pilot numbers are not used in the paper.
- The Pycom pilot was near chance on its test session. Exploratory analyses on
  pre-test sessions only (Days 1-3) followed: `../explore/README.md` (E1 input
  diagnostics, E2 input length for Phase 1). Any change they cause is a new
  protocol version (v3) with date and reason, and the paper states that the
  Pycom test session had been seen before the change.
- Folders: `res_pilot/` pilots; `../explore/` exploratory; `res_tune/` tuning;
  `res/` final runs. Only `res/` gives reported numbers.

## 11. Two run sets from 2026-09-29 (decided by the author before any final run)

**Final (paper) runs = v2 + Pycom L = 256** (`run_full_plan.sh`):
per-split Phase-1/2 tuning on validation data (`res_tune/`) -> tuned pilot
(split 0, seed 42, `res_pilot_tuned/`) -> only if both audits pass: all splits
x 5 seeds + joint (`res/`, `res_joint/`) -> `aggregate.py --root res`.
Only these runs give the paper's final numbers.

**Draft runs = v3** (`run_draft.sh`), for a draft paper to the advisors by
Oct 2, because the full plan needs ~9 days of compute:
- Phase-1/2 hyperparameters NOT tuned: the defaults fixed before any data was
  seen (lambda 0.5 hand-set, T 0.07 Khosla et al. 2020, replay 0.2, stage-2
  lr 5e-5, 100 Phase-1 epochs); nothing selected on test data. Phase-3,
  baseline and threshold selection unchanged (validation data, every run).
- Not replaced by tuning on split 0: its known devices are unknown in other
  splits (leakage); defaults were preferred.
- All splits x 5 seeds + joint, seed by seed; `res_draft/`,
  `res_joint_draft/`; `aggregate.py --root res_draft`.
- Draft numbers are labelled "draft, untuned Phase 1/2" and are replaced by
  the final runs.

**Both:** Pycom input stays 256 samples (the v2 value; author's decision
2026-09-29). The exploratory length comparison (../explore/README.md, E2; one
seed, Day-1 validation: val score 0.578 / 0.593 for 256 / 1024; 4096 pending at
the decision) showed small differences, and longer inputs cannot finish in
time. The paper discloses it.

The draft runs have normal CPU priority; the full plan runs at lower priority
(`nice`) in parallel, so its dates move by the time the draft runs take.

## 12. Split fixes found in the first draft runs (2026-09-29; all runs restarted)

The pilot tested split 0 only. The first draft runs showed problems in WiSig
splits 4, 8, 12, where four low-diversity transmitters (devices 12-15: a few
very large bursts, many tiny ones; one burst holds up to 57% of a device's
slices) are enrolled. Three group splits could leave a part (almost) empty:
1. **Phase 2 failed** (`Class 8: 127 train / 14 val / 14 test, need
   127/23/60`). Fix `finetune.capture_group_parts` (used by prepare_data and by
   the saved provenance): if the default rule cannot supply the validation (23)
   and held-out (60) slices, whole groups are taken from the end until held-out
   and then validation are full; the rest trains.
2. **Reference Part B or C with 1 slice** for one device. Fix
   `_group_three_split`: if B or C would get < half of its share of slices,
   groups are balanced greedily (largest first, to the part furthest below its
   share; every part keeps >= 1 group).
3. **Phase-1 validation/test with 4 / 7 slices** for one device. Fix
   `phase1_initial_training.wisig_burst_split`: the same greedy balance if
   validation or test would get < half of its 15% share. Worst device after the
   fix: 25 validation / 24 test slices (one burst holds most of its slices; this
   is the best a burst-disjoint split allows).
All fixes keep the default rule wherever it worked: Pycom (every split) and
WiSig split 0 give identical splits (checked: identical prepare_data output
with the same global seed; Phase-1 split 6526/1496/1338 and reference parts
4773/2329/2258 as in the pilot). The runs made before the fixes came from other
code and were moved, not deleted, to
../archive_local/first_attempt_preSplitFix/; all draft and tuning runs were
restarted with the fixed code.

## 13. Augmentation seed (found 2026-09-29 during the draft runs)

`signal_preprocessing.preprocess_batch` (and two helper augmenters) created
their random generator without a seed, so Phase-1/2 data augmentation used
fresh OS randomness: the same run seed did not give bit-identical models
(TensorFlow itself was checked deterministic). Results remain valid (random
augmentation; the seeds measure run-to-run variation), but repeated runs are
not bit-identical.
- Draft runs (`res_draft/`): made with the unseeded augmentation; the draft
  paper states "not bit-reproducible; seeds give run-to-run variation".
- Final runs: the fix (generator seed drawn from NumPy's global generator,
  which repro.set_all seeds; `../pending_fix/`) is applied automatically when
  the draft runs finish, and the full plan restarts with the fixed code
  (`resume_full_plan.sh`). Checked: two processes give identical augmentation;
  the Pycom packet-cache key is unchanged.

## 14. Pycom local copy was not bit-identical to the source (found 2026-09-30)

A byte-level check of the local Pycom Indoor copy against the Oregon State
server (`../explore/pycom_verify/`: exact size + three 64 KiB blocks per file;
first 64 KiB of all 1,250 server files matched against all local files):
1,207 files identical, 43 wrong: 29 files are copies of another device's file
placed in the wrong folder (manual download), 14 files missing. The server has
NO duplicate recordings, so the "duplicate devices" found earlier (Day 1:
12=15, 22=23; Days 2-3: 14=15; Day 2: 43=44; Day 5: 19=20) were download
errors, and the device pool built on them (pool_check.py) is not valid.
In the pool, device_19's Day-5 (test) files were device_20's recordings.
Consequence: all Pycom runs made so far (pilot, exploratory E1/E2, draft) used
partly wrong data and are void for Pycom. The Pycom draft was stopped.
Next: download the 43 correct files, re-verify, re-run pool_check.py, rebuild
the packet cache, then run Pycom again. WiSig is not affected.
- 2026-09-30, after the repair (all 1,250 files verified bit-identical in size and
  sampled content; 29 re-downloaded, 14 renamed back from browser "(1)" copies):
  pool_check.py now detects duplicates on the data (none exist) instead of a
  hard-coded list. New pool (first 20 passing, folder order): folders 1-4, 6-13,
  15-19, 21-23 (fail: 5, 14, 20, 28, 29). The augmentation-seed fix (section 13)
  is applied from now on, so all new Pycom runs and the final runs use it. New
  Pycom pilot: res_pilot2/.

## 15. Bank-size bug in TemporalTrajectory.load (found 2026-09-30)

`TemporalTrajectory.load()` saved but did not restore `n_cosine_prototypes`, so a
loaded trajectory used the default P = 5 for every later `update()`. Zero-shot
results are not affected (they use the saved banks). Affected: the TEMP-RFF
few-shot curve in common_eval.py, where the target-session support was always
reduced to 5 prototypes, even when P = all was selected (17 of 20 WiSig draft
runs): the curve stayed flat at 0.890 from k = 10 on, while k-NN kept rising.
Fixed 2026-09-30 (load restores P). The WiSig draft runs were made before the
fix: their TEMP-RFF few-shot numbers are invalid and are not reported; all their
zero-shot numbers stand. The Pycom pilot was restarted with the fix (res_pilot2/).

## 16. Version v4 (2026-10-01, before any tuning of this version) — LoRa, one repository

Decided by the author on 2026-10-01; all tuning and final runs of the paper are
made with this version, in this repository (https://github.com/misty-cs/TEMP-RF).
The earlier working folder (icc-rerun/t2r) and its runs (pilots, the WiSig
draft) are superseded; their code differs (see sections 12-15).

- **Third dataset: LoRa** (Oregon State, Diff_Days_Indoor; Elmaghbub & Hamdaoui,
  IEEE Access 2021): 25 Pycom LoRa devices (SF7, 125 kHz), USRP B210 at 1 MS/s,
  raw float32 IQ, 10 x 20 s transmissions per device and day, five days.
  Design as Pycom: Day 1 Phase 1, Days 2-3 Phase 2, Day 4 reference, Day 5 test;
  16 known + 4 unknown, five splits (every device unknown once), five seeds.
  Slice = one SF7 preamble chirp (1024 samples), found by dechirping and aligned
  by non-coherent correlation (robust to CFO), quality >= 0.5
  (`load_slice_IQ.read_lora_slices`, cached; code hash in the cache key); CFO kept;
  780 slices per device and day, water-filled over the 10 files.
  Pool: Device9 is missing on Day 2 at the source (HTTP 403), so the pool is the
  first 20 devices present on all five days (folders 1-8, 10-21), checked by
  `pool_check.py --dataset lora` before LoRa tuning (same rule as Pycom).
- **Disclosure (LoRa):** before this decision, an exploratory network-free check
  (`explore/lora_oregon/step2_crossday.py`) used the first 2 s of 2 of 10
  transmissions of devices 1-10 on Days 1-5, including the test day 5
  (raw-preamble centroid 0.70-0.81 across days, chance 0.10). It informed the
  choice of the dataset and of the slice (preamble chirps), not any setting
  that is tuned; it is reported as exploratory.
- **Device numbering by folder name** (`load_slice_IQ.folder_index`): labels come
  from the folder's number, not its position, so a missing folder on one day
  cannot shift labels. Checked: Pycom data identical to the previous code.
- **Primary family** extended to three datasets: 7 rules x 4 outcomes x 3
  datasets = 84 tests (section 7 otherwise unchanged).
- **Run control** (`run_all.sh`): tuning WiSig -> Pycom -> LoRa (LoRa after its
  download is complete and its pool rule passes), tuned pilots with an audit
  gate, final runs only after all tuning. Interrupted steps (e.g. drive
  disconnects) are repeated automatically; failed attempts are kept as
  `*.failedN`. Code is frozen during the study (tuning identities depend on it);
  results are committed to git after each stage, so the history shows which
  committed code produced which results.
- Includes the fixes of sections 12-15 (split fallbacks, verified Pycom copy and
  pool, seeded augmentation, bank-size restore).

### 16.1 LoRa duplicated recordings (2026-10-02, before any LoRa tuning)

- **Finding.** `pool_check.py --dataset lora` on the complete download failed
  only its duplicate test (quality passed for all 20 pool devices). A full
  scan (`data_prep/lora_dedup.py`, SHA-256 of whole files) found 11 recordings
  that are stored byte for byte twice: 10 under two neighbouring device IDs on
  the same day with the same file number (e.g. Day4 Device18 and Device19
  IQ_5), 1 twice under one device (Day2 Device7 IQ_1 = IQ_2). The server
  copies are identical as well (first 64 KiB of 3 pairs compared), so it is a
  source error, not a download error. Two independent 20 s captures cannot be
  equal; the metadata gives no capture time or checksum, so the true
  transmitter cannot be determined. List: `data_prep/lora_duplicates.json`.
- **Decision (author, 2026-10-02).** Both copies of every duplicated recording
  are excluded (22 of 1,240 files); all devices are kept. The device pool is
  unchanged (folders 1-8, 10-21). Pool devices lose 1-2 of 10 files on at most
  two days (Device5/6/7/8/12/14/18/19); 780 slices per device and day are
  still drawn (water-filled over the remaining files); the test day of the
  pool is not affected. Alternatives considered: keep only the 12 devices
  without duplicates (smaller design than WiSig/Pycom) or drop LoRa.
- **Implementation.** `data/lora_indoor` points to a tree of links to the
  download without the excluded files (`data_prep/lora_dedup.py SRC DST`;
  the download itself is unchanged). `pool_check.py` expects 10 files minus
  the excluded ones per device and day; `run_all.sh` v2 checks the tree with
  `lora_dedup.py --check`. Result of the pool rule on this tree:
  `docs/pool_check_lora.txt` (POOL PASS, no duplicates).
- **Effect on other results: none.** Tuning identities hash only the code the
  tuning runs (`manifest.PHASE1_CODE`, `PHASE2_CODE`), which does not include
  `pool_check.py`, so WiSig and Pycom tuning stay valid. Run manifests hash
  all top-level `*.py`, so the WiSig tuned pilot made before this change
  (audit passed) is kept in `docs/superseded_runs/wisig_pilot_v1code/` and
  the pilot is repeated with the current code.
- **Run control (author's decision, 2026-10-02).** To save time, LoRa tuning
  runs in parallel with Pycom tuning (separate process, same commands as
  `run_all.sh`, `tune()`); `run_all.sh` v1 keeps running Pycom tuning and then
  waits at its old LoRa completeness check, which the de-duplicated tree never
  satisfies, so it never runs LoRa. When the tuning of both datasets is
  complete, `run_all.sh` v2 is started; it skips every complete tuning
  candidate and continues with the tuned pilots, final runs and aggregation.
  Results do not depend on the order (seeded, deterministic, separate folders).
  The automatic v1 commit after Pycom tuning may contain partial LoRa tuning
  files; the LoRa commits after each LoRa phase hold the complete state.
- **WiSig final runs started early (author's decision, 2026-10-02).** WiSig
  tuning is complete and WiSig uses no Pycom/LoRa result, so its final runs
  start as soon as its tuned pilot audit passes (`run_wisig_early.sh`, same
  commands as `run_all.sh` `final()`), in parallel with Pycom/LoRa tuning.
  `run_all.sh` v2 waits for it and then skips every complete WiSig run. The
  WiSig tuned pilot was repeated with the current code (22:19).
- **LoRa tuning with one worker per split (author's decision, 2026-10-04).**
  To save time, `run_lora_parallel.sh` runs `tune_phase{1,2}.py --dataset lora
  --shifts S` for the five splits at once (disjoint splits, separate folders),
  then one run over all splits that skips every complete candidate and makes
  the selection. Same grid, seeds and code; results do not depend on the
  order. The candidate running when the single-process tuning was stopped
  (split 0, lambda 0.5, T 0.1) is kept as `*.failedN` and repeated.
- **Load governor (author's standing approval, 2026-10-05).** The Mac thrashed
  with seven training jobs (load 66, swap 12 GB; almost no progress for 8 h).
  `tools/governor.py` keeps at most four training jobs running and pauses the
  others with SIGSTOP/SIGCONT (no work lost); Pycom/WiSig jobs first, then the
  LoRa job whose current candidate has run longest. Run control only; results
  do not depend on it. Log: `tools/governor.log`.
- **Final stage in two parts for the ICC deadline (author's decision,
  2026-10-05).** Stage A: final runs with seeds 42, 7, 13 for every split of
  every dataset, and the joint-training control with the same seeds (the joint
  control runs on split 0 only, by design of `main.py`); then the 3-seed
  aggregation (`docs/results_3seeds/`, made from `res_3seeds/` and
  `res_joint_3seeds/`, relative links to exactly these runs, because
  `aggregate.py` uses every run under its root). Stage B: seeds 21 and 100
  (+ joint control), then the 5-seed aggregation (`docs/results_5seeds/`). The
  full protocol (five seeds) is unchanged; only the order is. Note: a WiSig
  joint job was stopped on 2026-10-05 while running split 0 seed 13; that
  incomplete run is kept as `*.failedN` and repeated. (An earlier version of
  this entry reduced the joint control to seed 42; that was withdrawn the same
  day, before any run used it, after checking that the joint control is only
  five runs per dataset.)
- **Run control v3 (2026-10-05).** `run_all.sh` v3 runs one pipeline per
  dataset (its tuning complete -> tuned pilot gate -> stage A), each with one
  `main.py` worker per split (and per seed for the joint control), so a dataset
  does not wait for the tuning of another. The governor gives LoRa (the
  critical path) up to three of its four slots.
- **Governor, 2026-10-05 20:30:** LoRa (critical path) may use all four slots
  (`tools/governor.py 4 4`); other jobs run whenever fewer than four LoRa jobs exist.
- **2026-10-06 (author's decision):** the Pycom and LoRa joint-training control
  moves from stage A to stage B (all five seeds there), to have the 3-seed
  results about 8 h earlier. The 3-seed summary therefore has the joint control
  for WiSig only. Order only; no result changes.
- **Governor, 2026-10-07 22:00:** back to `tools/governor.py 4 3` (Pycom Phase-2 tuning, a single serial process, would otherwise become the critical path).
- **Logs in the repository (2026-10-08):** top-level logs are excluded by
  `.gitignore` (`*.log`), so `tools/collect_logs.sh` copies the run-control,
  tuning, pool-check, final-run and governor logs to `docs/logs/*.txt`. It is
  run before every push (first snapshot 2026-10-08).
- **Repository completeness (2026-10-08):** per-run logs, the selection tables
  (`modelDir/phase3_selection.json`), the data-split group lists, normalization
  statistics and Phase-2 centroids are now tracked; only trained models and
  large arrays stay outside git, indexed with SHA-256 in `docs/ARTIFACTS.tsv`
  (`tools/artifact_index.py`). `docs/TRACEABILITY.md` maps every paper element
  to its source.
