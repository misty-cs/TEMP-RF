# Exploratory work (not confirmatory)

Everything in this folder was done AFTER the pilot, whose test sessions had
already been seen. It informs design decisions; it is not evidence for the
paper's claims. Each step lists what data it used, what it decided and why.
The paper discloses these steps (see DISCLOSURE text at the end).

Folder layout of the project:

| Folder | Content | Role |
|---|---|---|
| `t2r/res_pilot/` | pilots, WiSig and Pycom, split 0, seed 42, code defaults for Phase 1/2 | code check (audit) only; numbers not used |
| `explore/` | this folder | exploratory design decisions |
| `t2r/res_tune/` | Phase-1 and Phase-2 tuning per split (validation data only) | parameter selection |
| `t2r/res/` | final runs | all reported numbers |

## E1. Pycom input diagnostics (2026-09-28)

- Why: the Pycom pilot (2026-09-28) passed the audit, but every learned rule
  was near chance on the Day-5 test (closed accuracy 0.06-0.10, 16 devices),
  below a raw-input centroid (0.15). Phase-1 validation accuracy was 0.38.
- Data: Pycom Days 1-3 only, all 20 pool devices; Days 4 (reference) and 5
  (test) were not read. No network.
- Script and output: `pycom_input_length/diagnostic/`.
- Findings (with the pipeline's quality filter):
  labels consistent across days; packets last >= 4096 samples (164 us);
  aligned preambles are almost identical across devices (template |corr|
  0.96 between devices vs 0.97-0.98 within a device); the 256-sample window
  position does not matter; longer inputs raise raw-centroid accuracy
  (new files same day: 256 -> 0.175, 1024 -> 0.235, 4096 -> 0.339; chance 0.05).
  The CFO estimator used was unreliable; no conclusion about CFO.

## E2. Pycom input length for the learned model (2026-09-28/29)

- Question: does a longer input help the learned Phase-1 model?
- Data: Days 1-3, the 16 known devices of split 0; Days 4-5 not read.
- Packets: the pipeline's detection, 256-sample alignment, quality filter and
  selection (780 per device and day); 4096 samples from each aligned start;
  only starts with 4096 samples available (4 of all starts dropped). The first
  256 samples were checked equal to the pipeline's own packets for every file.
- Training: the pipeline's Phase 1 unchanged (lambda 0.5, T 0.07 defaults,
  seed 42, at most 100 epochs, validation checkpoint and early stopping), input
  = first L samples, L in {256, 1024, 4096}; same packets, split and budget.
- Selection rule (fixed before the runs): Day-1 validation only (val_sel;
  val_acc reported). Days 2-3 are reported, not used for selection.
- Results (`pycom_input_length/runs/L*/result.json`; chance 0.0625):

  | L | Day-1 val score / acc | Day-1 test file | Day 2 softmax / centroid / re-enrolled | Day 3 softmax / centroid / re-enrolled | time |
  |---|---|---|---|---|---|
  | 256 | 0.578 / 0.403 | 0.505 | 0.157 / 0.156 / 0.315 | 0.097 / 0.097 / 0.332 | 42 min |
  | 1024 | 0.593 / 0.425 | 0.519 | 0.163 / 0.161 / 0.381 | 0.140 / 0.137 / 0.286 | 76 min |
  | 4096 | 0.626 / 0.464 | 0.580 | 0.168 / 0.168 / 0.388 | 0.142 / 0.141 / 0.359 | 391 min |

- Decision (author, 2026-09-29, made while 4096 was still running): L = 256
  for all runs, because 4096 is ~9x slower per run and cannot finish by the
  deadline. The pre-set validation rule prefers L = 4096 (val score +0.048);
  gains are small, from one seed, and cross-day accuracy stays low. The paper
  states this.

## DISCLOSURE (text for the paper, to be completed)

"All hyperparameters were selected on validation data only (Section X):
Phase-1 contrastive weight and temperature, Phase-2 replay weight and learning
rate per device split on that split's enrolled devices; Phase-3 and baseline
settings on a held-out part of the reference session. Before the final runs, a
pilot on one split showed near-chance Pycom performance; an exploratory
analysis on pre-test sessions (Days 1-3, no test data) led us to [choice of
input length]. Because the pilot's test session had been seen before this
change, the Pycom results should be read as [exploratory / confirmed on
untouched splits]. The WiSig protocol was not changed after the pilot."
