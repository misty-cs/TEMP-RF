# Fairness of the comparison

Every number in the paper comes from `common_eval.py` on runs produced by
`main.py`, and every run is checked by `audit_run.py` (`audit.json`). This file
states, for every rule, what it uses and how it decides. The current protocol is
`PROTOCOL.md`; how each value was found is in `METHODS_HISTORY.md`.

## 0. Pre-registered analysis (fixed on 2026-09-27, before any final run)

Primary outcomes (reported first, for every rule, both datasets):
1. closed-set accuracy on the known test devices (threshold-free);
2. AUROC known vs unknown (threshold-free);
3. at the operating point of the **tuned** threshold protocol: known accepted and
   correct (known ok) and unknown detection.

Primary comparison (family fixed 2026-09-28): TEMP-RFF vs msp, probe, openmax,
static, knn, static_window, knn_window, at the **split level** (seeds averaged
within each device split), paired t-test and Wilcoxon, each Holm-corrected over
these 7 rules x 4 outcomes x both datasets = 56 tests. Other rules: secondary; effect sizes and split wins reported with the p-values.

Secondary (reported, not used for the main claims): fixed90 and objective
protocols, per-run tests, forced windows, few-shot, pipeline stages, joint
control, untuned defaults, prototype-weight ablation, selection sensitivity.

Selection rule for every tuned choice (Phase-1 lambda/T, Phase-2 replay/lr,
Phase-3 alpha/P/W/decay, all baseline settings): SEL = 0.5 x closed accuracy +
0.5 x leave-one-device-out AUROC on that stage's selection data; ties within
0.001 keep the prior default, else the simplest candidate (`selection.py`).

Disclosure: before this rule was unified, a code test on an old model (not a
result; CFO-mismatched) showed that the Phase-3 choice P = 5 (B score 0.9603)
did worse on its test session than the prior default P = all (0.9601). The
unified tie rule was adopted for consistency with the other selections; it
changed that test choice to P = all. No other setting was changed after seeing
test numbers.

## 1. What every rule shares (identical inputs)

- the same trained network per run (Phase-3 model, batch norm adapted without
  labels on Part A of the reference session only);
- the same enrolled (known) devices, the same device split and seed;
- the same reference session, split once into group-disjoint parts
  A (50%, fit / prototypes / support), B (25%, selection), C (25%, thresholds);
- the same test session with the same known and unknown slices (test-set SHA-1
  stored in `common_eval.json`; the audit reloads the test set and recomputes it);
- the same three threshold protocols (section 3) and the same metrics.

No model, setting or threshold uses unknown devices or the test session; they
are used only for final scoring. Exception, disclosed: cohort eligibility needs
each device to be present on the test day (availability only; quality is
judged on pre-test days; PROTOCOL.md section 1).

## 2. The rules

A slice is **rejected as unknown if its unknown score is above the threshold**;
otherwise it gets the predicted device.

| Rule | Fitted on | Prediction | Unknown score | Tuned on Part B (search space) |
|---|---|---|---|---|
| softmax | classifier head (Phase 1/2 labels) | argmax logit | – (closed-set only) | – |
| msp | classifier head | argmax softmax | 1 − max softmax probability | – |
| probe | logistic regression on Part-A embeddings | argmax probability | 1 − max probability | C ∈ {0.01, 0.1, 1, 10} |
| openmax | Weibull tails on Part-A logits | argmax revised probability | OpenMax unknown probability | distance {euclidean, eucos} × tail {10, 20, 50} × α {1, 2, 5, min(10, K), K} (30) |
| static | one Part-A mean per device | nearest centroid (cosine) | 1 − best cosine similarity | – |
| knn | Part-A embeddings | k-nearest vote (uniform or 1/d weighted) | mean cosine distance to the k nearest | k {1, 5, 10, 20} × vote {uniform, weighted} |
| static_window | the same reference embeddings as temp_rff (all embeddings of the W window sessions) | nearest pooled mean | 1 − best cosine similarity | W (own choice) |
| knn_window | the same reference embeddings as temp_rff | k-nearest vote | mean cosine distance to the k nearest | W × k × vote (own choice) |
| temp_rff_p1 | one plain mean per device and session (W sessions) | nearest session mean | 1 − best cosine similarity | W × decay {1, 0.8, 0.6} (own choice) |
| temp_rff | prototype banks (earlier sessions from non-training groups + Part A) | nearest prototype | 1 − best weighted cosine similarity | α, P {1, 5, 20, 50, 100, all}, W, decay (Phase 3) |
| temp_rff_nowt | as temp_rff | as temp_rff | 1 − best cosine similarity (no prototype weight) | fixed to temp_rff's settings (ablation of the weight only) |
| temp_rff_W* | as temp_rff, window forced to W, decay 1 | – | as temp_rff | – (ablation) |
| raw_centroid(_cfo) | one Part-A mean of the input slices, no network | nearest centroid | 1 − phase-invariant cosine | – (reference) |
| *_default | untuned defaults: knn k 5 uniform; openmax euclidean, tail 20, α min(10, K); probe C 1; temp_rff P all, W 1, decay 1 | | | none (shows the effect of tuning) |

**Selection score and tie rule (identical for every tuned choice, see section 0):**
0.5 × closed accuracy on B + 0.5 × leave-one-device-out AUROC on B, where each
known device in turn is held out (removed from the references / refitted without
it) and its B slices act as unknowns. Every candidate and its score is saved
(`common_eval.json → selection`; Phase 3 in `phase3_selection.json`).

**Equal data.** temp_rff uses earlier sessions plus Part A; static, knn, probe and
openmax use Part A only. `static_window` and `knn_window` use exactly the same
reference embeddings as temp_rff, and choose their own window, so "temp_rff vs
knn_window" compares methods on equal data.

**Tuning budget differs** (temp_rff 6 × N × 3 = 54 WiSig / 72 Pycom, plus α;
knn_window N × 8 = 24 / 32; temp_rff_p1 N × 3; static_window N; openmax 30;
knn 8; probe 4; static 0). This is reported, and the `*_default` rows
show every rule without tuning.

**Hyperparameters of the network** (lambda, T in Phase 1; replay weight and
stage-2 learning rate in Phase 2) are tuned **per device split, on that split's
known devices only** (`tune_phase1.py`, `tune_phase2.py`), so a device that is
unknown in a split never influences that split's settings. main.py applies each
split's own values; they are recorded in the run manifest.

## 3. Thresholds (identical procedure for every rule)

| Protocol | Pseudo-unknowns | Threshold | Chosen on |
|---|---|---|---|
| fixed90 | none | accepts 90% of all Part-C slices | C |
| objective | hardest 25% of Part C per device (by static distance; one stress subset, defined once for every rule) | max 0.5 × (accepted accuracy + pseudo-unknown rejection), ≥ 50% acceptance of the other 75% | C |
| tuned | leave-one-device-out: slices of device c scored by the same rule with c removed (probe and k-NN refitted, OpenMax recalibrated, MSP renormalised over the other logits, centroid / prototype rules without c's references) | candidates on C: accept 80 / 90 / 95 %, Youden's J | method chosen on B by 0.5 × (accepted and correct) + 0.5 × (pseudo-unknown rejected); threshold from C |

Every protocol uses the rule's OWN prediction and unknown score; thresholds
only move the operating point. The tuned protocol reproduces each rule's closed
accuracy and AUROC exactly (checked in common_eval.py and by the audit).

The weight 0.5 in all scores is fixed a priori (equal weight to legitimate
devices and to rejection).

## 4. Statistics

The seeds of a device split share its devices, so they are not independent.
The **primary test** averages the seeds within each split and tests TEMP-RFF
against the 7 primary competitors over the splits for the four primary
outcomes (paired t-test and Wilcoxon), each Holm-corrected over these 56 tests
(`aggregate.py --tags wisig pycom`); other rules are exploratory (uncorrected). Effect sizes and
split wins are reported with every p-value; with 4-5 splits the smallest
Wilcoxon p is 0.125 / 0.0625. Per-run tests are secondary. Only audited runs
with unchanged artifacts from one study enter; missing runs are listed.

## 5. Few-shot

The same target-session support (group-disjoint from the query) is given to
temp_rff (as a new session), static and knn (added to Part A), and to the
equal-history controls static_window and knn_window (added to their own window
references: the same earlier sessions as temp_rff + Part A). Thresholds are
calibrated on the reference session as in zero-shot. The window controls pick
their own window, so they have equal access to history, not necessarily the
same selected samples.

## 6. What the audit checks for every run (`audit.json`)

sessions distinct; known and unknown device sets disjoint; Phase-1 and Phase-2
train / val / held-out parts present, non-empty, covering every device,
group-disjoint; reference parts A/B/C match the saved indices, the freshly
loaded groups and labels, cover every device, are group-disjoint, and BN used
Part A only; the saved bank identities of every earlier session share no group
with its training groups; test set reloaded (input hash, per-device counts =
budget); tuned protocol keeps each rule's closed accuracy and AUROC; selection
sensitivity at weight 0.5 reproduces Phase 3; every rule has all protocols and
selection tables; Part-A reload exact; training and evaluation used identical
code, T2R_* settings, configuration and software versions; deterministic
TensorFlow ops; artifacts present, with
their SHA-256 stored. Missing provenance makes the check fail.

## 7. Provenance

`manifest_train.json` and `manifest_eval.json` in every run: command, full
configuration, all T2R_* settings, PYTHONHASHSEED, SHA-256 of every code file,
software versions, data inventory, device lists, test-set hash. Tuning results
and selections carry identity hashes (PROTOCOL.md section 3).
