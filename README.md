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

## Settings used for the paper

TEMP-RFF is the complete training, adaptation and reference-bank framework.
The learned final decision rules share one trained backbone within each run.
MP-NN is one of these rules; in saved results its key is `temp_rff`.
The raw-input centroid uses no network. Settings below describe the final
WiSig, Pycom and LoRa study, rather than generic defaults in ExperimentConfig
or exploratory datasets. The saved manifests are the exact record for each run.

### Sessions, inputs and repetitions

| Setting | WiSig | Pycom | LoRa |
|---|---|---|---|
| Known / unknown devices per split | 12 / 4 | 16 / 4 | 16 / 4 |
| Device splits | 4 | 5 | 5 |
| Seeds per split | 42, 7, 13, 21, 100 | 42, 7, 13, 21, 100 | 42, 7, 13, 21, 100 |
| Final runs | 20 | 25 | 25 |
| Initial training session | 1 | 1 | 1 |
| Fine-tuning sessions | 2 | 2, 3 | 2, 3 |
| Reference session | 3 | 4 | 4 |
| Test session | 4 | 5 | 5 |
| Input length (complex samples) | 256 | 256 | 1024 |
| Sampling rate | 25 MS/s | 25 MS/s | 1 MS/s |
| Slices per device per session | 780 | 780 | 780 |

Session numbers identify capture days, not necessarily consecutive calendar days.
WiSig uses receiver `node24-16`. Carrier frequency offset is retained for all
three datasets (`T2R_KEEP_CFO=1`). Pycom uses packet-aligned slices
(`T2R_PACKET=1`); LoRa uses aligned SF7 preamble chirps. LoRa's 11 duplicate
recording pairs are excluded on both sides (22 files).

### Exact device groups

IDs below are the zero-based IDs saved by the code, not one-based folder names.
Each row is repeated with all five seeds. A run means training and evaluation
for one device split and one seed: 4 x 5 = 20 WiSig runs and 5 x 5 = 25 runs
for each other dataset. Every device is unknown in exactly one split (five
runs), and known in the remaining splits.

**wisig**

| Split ID | Known device IDs | Unknown device IDs |
|---|---|---|
| 0 | 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 | 12, 13, 14, 15 |
| 4 | 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 | 0, 1, 2, 3 |
| 8 | 0, 1, 2, 3, 8, 9, 10, 11, 12, 13, 14, 15 | 4, 5, 6, 7 |
| 12 | 0, 1, 2, 3, 4, 5, 6, 7, 12, 13, 14, 15 | 8, 9, 10, 11 |

**pycom**

| Split ID | Known device IDs | Unknown device IDs |
|---|---|---|
| 0 | 0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17 | 18, 20, 21, 22 |
| 4 | 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 20, 21, 22 | 0, 1, 2, 3 |
| 8 | 0, 1, 2, 3, 9, 10, 11, 12, 14, 15, 16, 17, 18, 20, 21, 22 | 5, 6, 7, 8 |
| 12 | 0, 1, 2, 3, 5, 6, 7, 8, 14, 15, 16, 17, 18, 20, 21, 22 | 9, 10, 11, 12 |
| 16 | 0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 18, 20, 21, 22 | 14, 15, 16, 17 |

**lora**

| Split ID | Known device IDs | Unknown device IDs |
|---|---|---|
| 0 | 0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16 | 17, 18, 19, 20 |
| 4 | 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20 | 0, 1, 2, 3 |
| 8 | 0, 1, 2, 3, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20 | 4, 5, 6, 7 |
| 12 | 0, 1, 2, 3, 4, 5, 6, 7, 13, 14, 15, 16, 17, 18, 19, 20 | 9, 10, 11, 12 |
| 16 | 0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 17, 18, 19, 20 | 13, 14, 15, 16 |

### Backbone, augmentation and training

| Setting | Value |
|---|---|
| CNN | Four blocks, each with two Conv1D layers, batch normalization, ELU, max pooling and dropout |
| Block filter counts | 32, 64, 128, 256 |
| Convolution kernel size | 8 |
| Dense bottleneck / embedding | 128 / 128; embedding L2-normalized |
| Convolution / dense dropout | 0.3 / 0.4 |
| L2 regularization | 0.0001 |
| Phase-1 loss | Cross-entropy + lambda x supervised contrastive loss |
| Phase-1 optimizer / batch size | Adam / 128 |
| Phase-1 learning rate | 0.0005; 200 warm-up steps, then cosine decay |
| Phase-1 training | Up to 100 epochs; patience 40; best validation-accuracy checkpoint |
| Initialization probes | Three initializations, 12 probe epochs each |
| Phase-1 phase rotation / amplitude jitter | Approximately +/- pi / 2 dB |
| Phase-1 noise / augmentation probability | 20 dB SNR / 0.7 |
| Multipath augmentation | Disabled (0 taps, magnitude 0) |
| Phase-2 adaptation labels | 150 per device per fine-tuning day: 127 train + 23 validation |
| Phase-2 held-out evaluation | An additional 60 slices per device, outside the 150-label adaptation budget |
| Phase-2 stages | Head only, then all layers; maximum 50 / 80 epochs |
| Phase-2 batch size / patience | 64 / 25 |
| Phase-2 head learning rate | 0.0005 |
| Phase-2 full-model loss | Cross-entropy + selected replay weight x previous-session centroid alignment |
| Replay samples | Only earlier training slices; no earlier validation or held-out groups |
| Sequential initialization | Continue from the preceding model (`ft_always_reinit=False`) |

Train, validation and held-out samples are separated by capture groups
(WiSig bursts; Pycom/LoRa capture files). Reference parts also use whole groups;
50/25/25 are group allocations, not guaranteed exact slice percentages.

### Tuning grids and selected training values

Phase-1 and Phase-2 tuning is performed separately for each device split,
on its known devices only, using seed 42. The resulting values are reused
for that split's five final seeds.

| Parameter | Candidates |
|---|---|
| Phase-1 contrastive weight lambda | 0, 0.25, 0.5, 1 |
| Phase-1 temperature T | 0.05, 0.07, 0.1; lambda=0 uses T=0.07 only (10 combinations) |
| Phase-2 replay weight | 0, 0.1, 0.2, 0.5 |
| Phase-2 full-model learning rate | 0.00001, 0.00005, 0.0001 (12 combinations with replay) |
| Reference EWMA alpha | 0.5, 0.8, 1 |
| Prototypes per device P | 1, 5, 20, 50, 100, all |
| Enrollment window W | 1 through 3 for WiSig; 1 through 4 for Pycom/LoRa |
| Recency decay gamma | 1, 0.8, 0.6 |

The selection score is 0.5 x closed-set accuracy + 0.5 x pseudo-unknown AUROC.
For pseudo-unknown evaluation, each enrolled device is removed from the
references in turn. Candidates within 0.001 of the best are tied: retain a
tied prior default, otherwise choose the simplest candidate. The bank prior
default is P=all, W=1, gamma=1. Reference/baseline parameters are selected
within each final run, so they can differ across seeds.

| Dataset | Split ID | lambda | T | Replay weight | Full-model fine-tuning learning rate |
|---|---|---|---|---|---|
| wisig | 0 | 0.5 | 0.07 | 0.2 | 5e-05 |
| wisig | 4 | 0.5 | 0.07 | 0.1 | 0.0001 |
| wisig | 8 | 0.25 | 0.07 | 0.0 | 0.0001 |
| wisig | 12 | 0.0 | 0.07 | 0.5 | 0.0001 |
| pycom | 0 | 0.5 | 0.1 | 0.2 | 0.0001 |
| pycom | 4 | 0.0 | 0.07 | 0.5 | 0.0001 |
| pycom | 8 | 0.25 | 0.1 | 0.0 | 0.0001 |
| pycom | 12 | 0.25 | 0.1 | 0.0 | 0.0001 |
| pycom | 16 | 0.25 | 0.05 | 0.1 | 0.0001 |
| lora | 0 | 1.0 | 0.1 | 0.1 | 1e-05 |
| lora | 4 | 1.0 | 0.1 | 0.0 | 5e-05 |
| lora | 8 | 0.25 | 0.1 | 0.2 | 0.0001 |
| lora | 12 | 1.0 | 0.07 | 0.0 | 0.0001 |
| lora | 16 | 1.0 | 0.1 | 0.0 | 5e-05 |

### Reference construction and decision rules

Reference-day groups are split into A (50%), B (25%) and C (25%). Batch-norm
statistics adapt on A only, without labels. A supplies reference embeddings;
B selects settings; C supplies threshold candidates. Earlier-session banks
use groups excluded from training. All learned rules use the same Phase-3
network; access to earlier reference history differs by rule.

| Rule | Reference data and selected settings |
|---|---|
| MSP | Classifier head; 1 - maximum softmax probability |
| Linear probe | Part-A embeddings; logistic-regression C in {0.01, 0.1, 1, 10} |
| OpenMax | Part-A logits; euclidean/eucos distance, tail {10, 20, 50}, alpha {1, 2, 5, min(10,K), K} |
| Static centroid | One mean per device from Part A |
| k-NN | Part A; k in {1, 5, 10, 20}, uniform or inverse-distance votes |
| MP-NN (`temp_rff`) | Prototype banks; selected P, W and gamma |
| Static/k-NN window controls | Access to earlier history plus Part A; independently selected window (and k/voting for k-NN) |
| Per-session centroids (`temp_rff_p1`) | One mean per device/session; selected window and decay |
| Raw centroid | Part-A input slices; phase-invariant cosine; no network |

K is the number of known devices. Untuned controls use MP-NN P=all/W=1/gamma=1,
k-NN k=5/uniform, probe C=1, and OpenMax euclidean/tail=20/alpha=min(10,K).
Additional ablations force the bank window or remove prototype weights.
These are explicitly named configurations, not different trained backbones.

#### MP-NN scoring details

For unit-length query embedding q, device k has distance
`dist(q,k) = 1 - max_i [(0.9 + 0.1*v[k,i]) * dot(q,p[k,i])]`.
The prototypes p[k,i] come from its last W enrollment sessions. Cluster
weights are normalized cluster counts (ones when every embedding is retained),
multiplied by gamma raised to the number of newer retained sessions, then
normalized by the maximum weight in that device's retained bank to obtain
v[k,i] in [0,1]. Assign the device with minimum distance; reject as unknown
when that minimum exceeds the rule's calibrated threshold. Part B selects
P, W and gamma. See `temporal_trajectory.py::cosine_prototype_bank` and
`common_eval.py::mp_distance` for the implementation.

### Thresholds and metrics

| Protocol | Procedure |
|---|---|
| Tuned (primary) | Threshold candidates on C: 80%, 90%, 95% acceptance and Youden's J; choose the method on B using equal weights for known accepted-and-correct and pseudo-unknown rejection; tied methods prefer q90 |
| Fixed90 | 90th percentile of scores on all Part-C samples |
| Objective stress test | Treat the hardest 25% of C per device (by static-centroid distance) as a fixed stress subset; optimize equal-weight acceptance accuracy and stress-subset rejection, with at least 50% acceptance of the remainder |

Higher unknown score means more likely unknown; reject scores above the
threshold. Primary metrics are closed-set accuracy, AUROC, and tuned-threshold
known accepted-and-correct and unknown detection. Closed-set accuracy and
AUROC do not depend on the rejection threshold.

The main evaluation selects settings without test-day samples or actual
unknown-device data. Cohort eligibility checks test-day availability only.
Few-shot adaptation is a separate experiment that explicitly uses labeled
test-day support.

### Few-shot, controls and statistics

- Requested support per known device: 1, 5, 10, 20, 50, 100, 200 and up to
  500 samples; three draws per run. Support and query use separate capture
  groups. Available support can be smaller than requested: `k_eff` in
  `common_eval.json` records the actual average count.
- MP-NN, static, k-NN and the history controls receive the same support.
  MP-NN adds a new session; static/k-NN add support to their references.
  Threshold calibration remains on the reference day.
- The joint-training control is saved under `res_joint/<dataset>/shift0/`;
  it covers split 0, rather than every device split.
- Overall means use 20 WiSig runs and 25 runs per other dataset. Primary
  statistical comparisons average seeds within each device split first;
  seeds sharing a split are not independent device partitions.
- The final three-dataset family has 84 comparisons: seven competitors x
  four outcomes x three datasets. Paired t-tests and Wilcoxon tests use
  Holm correction within each test family. See
  `docs/results_5seeds/PRIMARY_wisig_pycom_lora.md` for the saved report.

### Exact settings and software records

For any final run, inspect:

- `res/<dataset>/shift<id>/seed<seed>/manifest_train.json`: effective config,
  environment overrides, command, known/unknown IDs, software versions and hashes.
- `manifest_eval.json`: evaluation provenance and environment.
- `modelDir/phase3_selection.json`: bank/trajectory candidates and selections.
- `common_eval.json`: selected decision-rule settings, baseline selection
  tables, metrics and effective few-shot counts.
- `audit.json`: saved checks and artifact hashes. All 70 final runs have
  `all_pass: true`; this records the completed audit, not a new audit on checkout.
- `res_tune/<dataset>/selected_phase1.json` and `selected_phase2.json`: all
  selected per-split training settings and tuning candidates.

The inspected final manifests record Python 3.13.5, TensorFlow 2.20.0,
Keras 3.13.2, NumPy 2.2.6, SciPy 1.17.0 and scikit-learn 1.8.0.
Exact numerical thresholds are associated with each rule/run and may require
its saved calibration artifacts; the README does not imply one shared cutoff.
Generic defaults in `experiment_config.py` can differ from paper settings:
`datasets.py`, launch overrides and the final manifests determine the study.

## Where to look

| Path | Content |
|---|---|
| `docs/PROTOCOL.md` | the protocol (authoritative), all versions and changes with dates and reasons |
| `docs/FAIRNESS.md` | what every decision rule uses and how it rejects |
| `docs/METHODS_HISTORY.md` | decision log: how every value was found |
| `docs/review/` | external reviews of the code and protocol |
| `docs/TRACEABILITY.md` | **where every number of the paper comes from** (start here) |
| `docs/ARTIFACTS.tsv` | size and SHA-256 of the large files not stored in git (models, banks, calibration) |
| `docs/logs/` | run-control, tuning, pool-check, final-run and scheduler logs (`tools/collect_logs.sh`) |
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
