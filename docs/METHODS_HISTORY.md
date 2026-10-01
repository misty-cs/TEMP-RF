# Methods history (decision log)

HISTORY ONLY. The current protocol is PROTOCOL.md; where a section below
disagrees with it, PROTOCOL.md is correct. This log keeps how each value was
found and what was changed, in order, so every number can be traced.
Superseded by v2 (2026-09-27): section 1 and 18 (split 0 / accuracy-only
tuning -> per split, SEL score, identity-checked); 6 and 20 (thresholds:
fixed90 on all of C; tuned protocol uses each rule's own score with refitted /
renormalised LOCO pseudo-unknowns); 7 (P = 500 -> P selected); 13 (replay from
the whole previous session -> training slices only); 14 and 17 (BN on the full
reference session -> Part A only); 21 (completion by PHASE3_DONE -> audit +
artifact hashes); 22 (OpenMax alpha grid now includes min(10, K), 30 settings;
temp_rff_p1 now selects its own W/decay); 23 (Pycom quality rule on pre-test
days only).

## 1. SupCon weight (lambda) and temperature (T) — [PENDING]

- Phase-1 loss: L = L_CE + lambda * L_SupCon(T).
- Selected on **validation data only**, never on the test session:
  - grid lambda in {0, 0.25, 0.5, 1}, T in {0.05, 0.07, 0.1}; lambda = 0 ignores T,
    so 10 settings;
  - Phase 1 only, init session only, known devices of device split 0; unknown
    devices are never loaded;
  - metric: Phase-1 validation accuracy
    - WiSig: held-out bursts (burst-disjoint from training),
    - Pycom: one held-out capture file per device (capture-disjoint from training);
  - 100 epochs (same as the final runs), seed 42; ties keep the prior default (0.5, 0.07);
  - the selected pair is then fixed for all device splits, seeds and phases.
- Script: `tune_phase1.py`; results: `res_tune/<dataset>/tune.json`.
- Prior defaults before tuning: lambda = 0.5 (hand-set), T = 0.07 (Khosla et al., 2020).
- Limitation to state: validation is within the init session, so the choice
  optimises within-session accuracy, not cross-session transfer.

Paper sentence (fill in values after the run):
> The contrastive weight and temperature were selected by grid search
> (lambda in {0, 0.25, 0.5, 1}, T in {0.05, 0.07, 0.1}) on Phase-1 validation
> accuracy of the first session only (held-out bursts for WiSig, a held-out
> capture file for Pycom), using the enrolled devices of one split; test
> sessions and unknown devices were never used. The selected values
> (lambda = __, T = __ for WiSig; lambda = __, T = __ for Pycom) were fixed for all
> splits and seeds.

## 2. Phase-1 data split — [SET]

- WiSig: burst-disjoint 70 / 15 / 15 (train / val / test) per device.
- Pycom: per device, 3 capture files train, 1 file val, 1 file test (previously
  2 test files and a 15% val drawn from the training files, which let model
  selection reward capture memorisation).

## 3. Input slices — [SET]

- WiSig: the released equalized packets (first 256 samples = preamble, 25 MS/s);
  verified to be the equalized Full-WiSig files by byte size (IdSig_info eq=1).
- Pycom (IEEE 802.11b, raw, 25 MS/s): packet-aligned slices, label-free
  (DC removed per recording; packet start = end of a >= 5 us gap 15 dB below the
  median 1-us power; start refined by correlation with a template built from the
  same file; 156 packets per file spread over the recording). Check with a raw
  nearest-centroid (no network): spread 0.074 / 0.073 vs packet 0.189 / 0.158
  (same-day new files / cross-day), 16 devices.

## 4. CFO — [SET]

- CFO is kept (T2R_KEEP_CFO=1) for WiSig and Pycom. WiSig re-applies CFO after
  equalization on purpose. Raw nearest-centroid, WiSig day 3 -> day 4, 12 devices:
  CFO kept 0.791 vs CFO removed 0.193. CFO-removed runs remain available as an
  ablation (`main.py --cfo remove`).

## 5. Device splits — [SET]

- Every device is unknown exactly once: WiSig (16 devices) shifts 0, 4, 8, 12;
  Pycom (20 devices) shifts 0, 4, 8, 12, 16. Five seeds per split.
- Fixed a WiSig bug where the shift was applied twice (split 0 unaffected).

## 6. Open-set thresholds — [SET, in common_eval.py]

- All rules share the Phase-3 model and reference-session parts:
  A (50%) prototypes / support / OpenMax fit, B (25%) selection, C (25%) thresholds;
  pseudo-unknowns = hardest 25% of C per device (by static-centroid distance).
- Two protocols for every rule: fixed90 (accept 90% of calibration knowns) and
  objective (max 0.5 x (accepted accuracy + pseudo-unknown rejection) with >= 50%
  calibration acceptance).

## 7. Other fixed settings (not tuned)

- Model: 1-D CNN, 128-d L2-normalised embedding; Adam 5e-4, warmup + cosine,
  batch 128, up to 100 epochs, early stop on val (patience 40), best of 3 inits.
- Augmentation (training only): phase rotation +-pi, amplitude +-2 dB, 20 dB SNR
  noise, probability 0.7. No frequency-shift augmentation.
- Prototypes: P = 500 per device and session (k-means).

## 8. Reproducibility — [SET]

- `repro.set_all(seed)` sets Python, NumPy and TensorFlow seeds and enables
  TensorFlow op determinism; called by run_experiment.py, joint_eval.py and
  tune_phase1.py. Test: two trainings with the same seed give identical weights.

## 9. Pycom device pool — [SET]

- device_15 and device_20 have recordings identical to other devices
  (Day 1: 12 = 15; Days 2-3: 14 = 15; Day 5: 19 = 20). They are replaced by
  device_21 and device_24 (no duplicates in the full hash scan). Pool (folder
  numbers): 1-14, 16-19, 21, 24 (`T2R_DEVICE_POOL`). No split contains a
  duplicate pair.

## 10. WiSig packet selection — [SET]

- 780 packets per device and session are taken round-robin over bursts (one
  packet per burst per round, random burst order, seeded), so every available
  burst is used. Low-diversity transmitters (ranks 12-15: 41-69 bursts on
  2021-03-08) keep all their bursts instead of mostly repeated packets.

## 11. Pycom packet counts — [SET]

- No spread fallback: a file with fewer packets than requested gives all it has,
  and the missing count moves to the device's other files (water filling), so
  every device keeps 780 packet-aligned slices.
- The packet reader uses seek+read with retries instead of a memory map, so a
  short USB disconnect does not kill a run.

## 12. Checks before tuning — [DONE / RUNNING]

- WiSig burst split (Phase 1, split 0, day 1): after removing each device's
  common preamble, only 1.1% of val slices and 0.8% of test slices have a
  near-copy (residual cosine > 0.9) in train (max 9.9% for one device). The
  burst-disjoint val is therefore not inflated by repeated packets.
- Pycom packet cache (`build_packet_cache.py`, label-free, keyed on file size,
  mtime and detector settings): per-file packet count and alignment quality in
  `cache_packets/report.json`. [RUNNING]

## 13. Phase 2 (sequential fine-tuning) — [SET]

- Per fine-tuning session (WiSig 03-08; Pycom Day_2 then Day_3), label budget
  150 slices per known device, split group-disjointly (groups = bursts for WiSig,
  capture files for Pycom):
  - train 127 slices from 60% of the groups,
  - validation 23 slices from half of the remaining groups (model selection only),
  - held-out 60 slices from the other remaining groups (report only).
  Unit tests: 0 shared bursts / files / slices between the three parts.
  (Before: validation was a random 15% of the training slices, i.e. same bursts
  or same capture files, and WiSig's held-out split was random.)
- Stage 1: classifier head only (backbone frozen, BN trainable), CE with label
  smoothing 0.05, up to 50 epochs, lr 5e-4, batch 64.
- Stage 2: all layers, CE + 0.2 x alignment to the previous session's device
  centroids (replay), up to 80 epochs, lr 5e-5 with cosine decay.
- Best weights by validation accuracy, early stop patience 25.
- Fixed, not tuned (same for all datasets): 150 labels per session, epochs
  50 / 80, learning rates 5e-4 / 5e-5, replay weight 0.2, smoothing 0.05.
- Joint-training control uses the same 127 training slices per session.

## 14. Phase 3 (reference session, prototypes, selection) — [SET]

- Batch-norm statistics are adapted to the reference session (no labels; no
  test-session data).
- Reference session split per device into A 50% (prototypes), B 25% (selection),
  C 25% (thresholds) by whole groups (WiSig bursts; Pycom capture files: 3/1/1).
  Before: random slices, so A and C shared bursts/files, which made calibration
  distances too small and thresholds too strict at test.
- EWMA alpha in {0.5, 0.8, 1.0} is selected on B with the single-centroid rule's
  own score (the only rule that uses the EWMA mean).
- Bank size P in {1, 5, 20, 50, 100, all} (all = every reference embedding is a
  prototype), window W in 1..N and recency decay in {1, 0.8, 0.6} are selected on
  B with 0.5 x closed accuracy + 0.5 x leave-one-device-out AUROC; ties keep the
  smaller P. Before: P was fixed to 500 from a test-day diagnostic, and with
  390-780 points per device and session it made almost every slice a prototype.
- Test on a copy of an old WiSig run (split 0, seed 42): parts 4773 / 2329 / 2258,
  selected alpha 1.0, P = 5 (all: 0.9601 vs 5: 0.9603), W = 1; common_eval Part-A
  consistency check passes (difference 0.0).

## 15. Label bug fix (WiSig, device splits other than 0) — [SET]

- `load_day_raw` / `load_day_raw_with_groups` returned raw WiSig device ids,
  not 0..K-1 (Pycom was correct). Invisible for split 0 (known ids 0-11 = labels),
  but splits 4/8/12 would mislabel Phase 1/2 and the old Phase-4 test mapping.
  Fixed at the loader; checked for split 4: all four loaders give labels 0-11,
  same order, same slices. Old split-0 results are unaffected.

## 16. Earlier-session prototypes from non-training data — [SET, verify in pilot]

- Phase 1 saves its (device, group) training pairs (`phase1_train_groups_d*.npz`),
  Phase 2 its training groups per session (`phase2_train_groups_d*.npz`).
  Phase 3 builds the banks of earlier sessions only from groups that were not
  trained on (Phase 1: its val + test groups; Phase 2: the 40% non-training
  groups). The reference session was never trained on. Phase 3 stops if a
  device has no held-out slice.

## 17. Phase 3 points to state in the paper

- Selection score fixed a priori: 0.5 x closed accuracy + 0.5 x leave-one-
  device-out AUROC on Part B.
- The score can be nearly flat across P (e.g. 0.9603 vs 0.9601); report the
  distribution of the selected P and W over all runs.
- Batch-norm adaptation to the reference session is unlabelled and uses no test
  data.

## 18. Phase 2 replay weight and stage-2 learning rate — [PENDING]

- Selected on validation data only, after the lambda/T choice: `tune_phase2.py`
  reuses the Phase-1 model of the selected (lambda, T) and repeats Phase 2 for
  replay weight in {0, 0.1, 0.2, 0.5} x stage-2 lr in {1e-5, 5e-5, 1e-4}
  (12 settings), split 0, seed 42. Metric: mean over fine-tuning sessions of the
  Phase-2 best validation accuracy (group-disjoint: other bursts / another capture
  file). Reference and test sessions and unknown devices are never used. Ties
  keep the prior default (0.2, 5e-5). Settings: T2R_FT_REPLAY, T2R_FT_LR2.

## 19. Parameter inventory (Phases 1-3)

Selected on validation data: lambda, T (Phase 1, pending); replay weight,
stage-2 lr (Phase 2, pending); EWMA alpha, bank size P, window W, decay (Phase 3,
Part B, every run).
Fixed a priori, identical for all datasets and methods:
- Phase 1: lr 5e-4, warmup 200 steps + cosine, batch 128, up to 100 epochs,
  patience 40, best of 3 starts x 12 epochs, gradient clip 5; augmentation phase
  +-pi, amplitude +-2 dB, noise 20 dB SNR, probability 0.7; CNN with 128-d
  L2-normalised embedding, dropout 0.3 / 0.4, L2 1e-4, ELU.
- Phase 2: stage-1 lr 5e-4, epochs 50 / 80, patience 25, batch 64, label
  smoothing 0.05.
- Phase 3: selection score 0.5 / 0.5; prototype similarity weight 0.9 + 0.1 w;
  MiniBatchKMeans (n_init 3, random_state 0); 3 batch-norm adaptation passes.
Protocol constants: 256-sample slices, 780 slices per session, 150 labels per
fine-tuning session, reference parts 50/25/25.

## 20. Calibration tuning and Phase-3 ablations (common_eval.py) — [SET]

- Tuned calibration, identical for every rule (openmax, static, knn, centroid,
  temp_rff, temp_rff_nowt, raw_centroid_cfo):
  - pseudo-unknowns = leave-one-device-out: a known slice is scored against the
    other devices only (distance to its own device removed);
  - candidate thresholds computed on Part C: known-accept quantiles 80 / 90 / 95 %
    and Youden's J (accept rate + pseudo-unknown rejection - 1);
  - the method is chosen on Part B by open-set balanced accuracy
    0.5 x (accepted and correct) + 0.5 x (pseudo-unknown rejected); ties keep q90;
  - Part C sets the threshold, Part B chooses the method, test data only scores.
  - k-NN uses its class-wise form here (mean distance to the k nearest reference
    slices of each device), OpenMax uses 1 - revised class probability.
- The earlier protocols (fixed90; objective with hardest-25% pseudo-unknowns and
  >= 50% acceptance) are kept for comparison.
- Prototype-weight ablation: `temp_rff_nowt` = TEMP-RFF without the
  0.9 + 0.1 w weight.
- Selection-weight sensitivity: Phase 3 saves its selection table
  (`phase3_selection.json`); common_eval records the setting chosen with weights
  0.3 / 0.5 / 0.7 on closed accuracy. Code test (old model): 0.3 and 0.5 choose
  P = 5, W = 1; 0.7 chooses P = all, W = 3.

## 21. Evaluation (common_eval.py) and aggregation — [SET]

- The legacy Phase-4 script is skipped (T2R_SKIP_PHASE4=1); all reported numbers
  come from common_eval.py. main.py treats a run as trained when PHASE3_DONE exists.
- Stage ablation and joint-training control now build their centroids from Part A
  of the reference session (same group split and seed as Phase 3), like every
  other rule (before: the full reference session).
- Few-shot: support and query come from different groups of the target session;
  3 draws; the bank adds the support as a new session; thresholds stay calibrated
  on Part C of the reference session (no target labels beyond the support). The
  effective support per device is recorded (k_eff), because half of a device's
  groups can hold fewer slices than the requested k (WiSig: about 390).
- aggregate.py builds res/<tag>/SUMMARY.md: all rules x all threshold protocols
  (mean +- std over splits x seeds), per-split means, paired TEMP-RFF-vs-rule tests
  (t-test and Wilcoxon; correct for multiple comparisons, e.g. Holm), window
  ablation, few-shot, stages, joint vs Phase 2, and selected-setting counts.

## 22. Baseline fairness update — [SET]

- New baselines: MSP (maximum softmax probability) and a linear probe
  (logistic regression on Part-A embeddings, C selected on B, MSP rejection).
- OpenMax follows the reference implementation: rank-weighted revision
  ((alpha + 1 - i)/alpha x Weibull CDF), distance euclidean or eucos, fitted on
  Part-A logits; distance, tail and alpha selected on B (24 settings) with the
  Phase-3 score, refitting without the held-out device for the LOCO part.
  Vectorised scoring (identical results to the loop, max difference 1e-7).
- k-NN search: k x uniform / distance-weighted vote (8 settings), same score.
- "Single centroid" (EWMA mean) replaced by temp_rff_p1: one plain mean per
  device and session over the selected window (equals static when W = 1).
- Untuned defaults for every tunable rule (*_default rows).
- Few-shot for temp_rff, static and knn with the same support.
- Every candidate score saved (selection tables); Holm correction in aggregate.py.
- Run manifests (code SHA-256, settings, data, versions) and an automatic audit
  per run (audit_run.py); see FAIRNESS.md.
- Observed on a code test (old model, not a result): the untuned temp_rff default
  (P = all) scored higher on test than the selected P = 5, whose Part-B score was
  only 0.0002 higher. The Phase-3 tie rule is under review (see section 23).

## 23. Pycom packet quality and device pool rule — [SET]

- Per file, packets are kept only if they match the file's own preamble template
  (|corr| >= 0.8; template rebuilt twice from the best-matching half; label-free,
  no other device used). After filtering, per-file alignment quality is
  0.926-0.99 (median 0.978); before, 23 files had 0.15-0.45 because the detector
  found false packet starts.
- Device rule, identical for every device and day (`pool_check.py`): no duplicate
  recordings; >= 780 good packets per day; >= 3 files with >= 50 good packets per
  day. device_5 (Day 4: 496 packets, 3 good files) and device_14 (Day 1: 583, 2
  good files) fail and are replaced by the first candidates that pass.
- Packet cache keyed on the resolved file path (links and direct paths give one
  key) and written through a file handle.

## 24. Equal data, per-split tuning, split-level statistics — [SET]

- static_window and knn_window: the same reference embeddings as TEMP-RFF (all
  embeddings of the window sessions), each selecting its own window W (and k,
  vote) on B with the Phase-3 score.
- lambda/T and the Phase-2 replay weight / stage-2 lr are tuned per device split
  on that split's known devices (tune_phase1.py / tune_phase2.py -> 
  res_tune/<ds>/selected_phase{1,2}.json); main.py uses each split's values and
  warns if a split has none.
- Primary statistics: seeds averaged per split, paired tests over splits,
  Holm-corrected; per-run tests secondary.
- Pycom pool after the quality rule (pool_check.py PASS): folders 1-4, 6-13,
  16-19, 21, 24, 25, 26 (device_5, device_14 fail the rule; device_25, 26 are the
  first candidates that pass; 27 and 30 also pass, 28 and 29 fail).

## 25. One selection score and one tie rule everywhere — [SET]

- `selection.py`: SEL = 0.5 x closed accuracy + 0.5 x leave-one-device-out AUROC;
  ties within 0.001 keep the prior default, else the simplest candidate.
- Used by: tune_phase1.py (Phase-1 val: nearest-centroid on train-embedding
  centroids), tune_phase2.py (Phase-2 val, mean over sessions), Phase 3 (alpha;
  P / W / decay on Part B), and every baseline selection in common_eval.py.
  (Before: Phase 1/2 used val accuracy only; tie rules differed.)
- Model selection inside training (best epoch / best start) still uses validation
  accuracy, the standard early-stopping signal.
- Pre-registered primary outcomes and test: FAIRNESS.md section 0 (with the
  disclosure about the Phase-3 tie rule).
