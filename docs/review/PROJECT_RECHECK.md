# Recheck of revised project — 2026-09-27

Verdict: substantially improved, but not yet ready for final runs. This supersedes the original audit where fixes are listed below. No experiment code was changed during this review.

## Fixed in the inspected code

- Phase-1 output now retains val_sel; reproduced the actual serializer.
- Tuned calibration uses native rule predictions/scores; synthetic k-NN test confirms identical closed accuracy and AUROC. Probe and OpenMax refit/recalibrate for held-out classes, and MSP renormalizes remaining logits.
- Replay uses Phase-1 training groups and subsequent Phase-2 training samples, not whole previous sessions.
- BN adaptation uses Part A only; missing training-group provenance stops Phase 3.
- OpenMax includes default alpha=10; fixed90 uses all Part C.
- P=1 selects its own window/decay; sensitivity uses the actual tie rule; few-shot includes window-based controls.
- Aggregation excludes absent/failed audits, reports missing runs, includes all four primary outcomes, and supports both datasets with Holm correction for each test family.
- Tuning requires complete candidate grids; CFO variants have separate folders. Saved group/index provenance is more extensive.
- WiSig quality ranking no longer uses test-day quality; Pycom's current checker uses pre-test quality. Test availability and historical cohort choices still need disclosure.

## Remaining actionable findings

### P1: Normal training/evaluation will disagree on the audit's environment check

`repro.py:6–7` sets PYTHONHASHSEED and TF_DETERMINISTIC_OPS in the training subprocess. `common_eval.py` never calls repro.set_all, and main.py launches it as a separate subprocess with the original parent environment. `audit_run.py:142–143` compares the entire stored environment dictionaries, including those two fields. With the current parent environment, the exact assignments yield differences PYTHONHASHSEED=42 and TF_DETERMINISTIC_OPS=1. Therefore an otherwise valid normal run will fail same_settings unless these values were already exported by the caller.

Set a consistent reproducibility environment before launching each subprocess, and initialize evaluation determinism explicitly. PYTHONHASHSEED should be set before interpreter startup to control Python hash randomization.

### P1: Phase-2 selected settings are not validated against the current Phase-2 code

`tune_phase2.py:74` hashes Phase-2 code for individual candidates, but its selected record at lines 112–114 does not retain that identity. `main.py:62` only checks that the stored Phase-1 identity agrees with Phase 1. Editing finetune.py, phase2_sequential_finetuning.py or tta_bn.py therefore leaves the Phase-1 identity unchanged and allows stale replay/lr choices into new runs. The stored checkpoint hash is not checked by main.py either. The Phase-2 tuner itself accepts an old Phase-1 selection with any identity string without recomputing its validity.

Persist and verify the selected Phase-2 identity and source-checkpoint hash, and validate the Phase-1 selection before Phase-2 tuning.

### P2: Completion and aggregation are not bound to the artifacts that passed audit

`manifest.py:93` returns complete based on a passing audit plus current training code/environment without requiring common_eval.json or manifest_eval.json. Reproduced: a temporary directory with only a matching training manifest and all_pass audit was classified complete. Missing training manifests are classified new even when artifacts already exist. `aggregate.py:59–72` trusts the all_pass boolean without checking result/manifest hashes or compatibility among included runs. Replacing an evaluated JSON or mixing independently audited configurations is not detected.

Tie the audit to hashes of the evaluated result, model and manifests; require those artifacts for completion and verify compatible study identities in aggregation. Include data identity and effective configuration in reuse checks, not just code/environment.

### P2: Joint runs never produce the manifest required by the new reuse logic

`main.py:94–98` uses manifest.run_status and only skips joint output when status is not new. `joint_eval.py` does not write manifest_train.json, so its successful output is always new and is retrained on subsequent invocations. It also lacks equivalent audit provenance. Add a joint-specific manifest/completion check and validate joint controls before aggregation.

### P2: Audit still permits incomplete evidence

The revised disjoint_by_class requires three named parts, but accepts all three when empty. Reproduced with an empty NPZ: PASS. It does not require every known device in every split. Test counts are checked against the stored counts, not independently recomputed per-device counts and expected budgets. Saved Phase-3 groups are checked for mutual disjointness, but are not compared directly to the freshly loaded groups at each saved index. Require nonempty per-device coverage and independent equality checks.

### P2: Tuning identities still omit relevant inputs

`manifest.identity` contains code/environment/extra fields but no dataset identity or software versions. PHASE1_CODE also omits modules executed by Phase 1, including temporal_trajectory.py (embedding extraction) and, for joint mode, finetune.py. Changes to data/cache contents or omitted code can leave a cached selection current. Include relevant transitive dependencies and a stable input-data/cache identity.

## Remaining interpretation/documentation limits

FAIRNESS.md, METHODS_NOTES.md and README_RUN.md still describe old settings: full-reference BN, inherited P=1 tuning, the old OpenMax grid, and conflicting tuning protocols. Consolidate them before freezing the protocol. Shared-backbone selection, unequal tuning budgets, overlap across device splits, validation reuse, historical test feedback and cohort conditioning still limit broad fairness claims. Window controls choose their own histories, so they have equal access to history rather than necessarily identical selected samples. No results yet establish empirical tuning adequacy.

## Checks run

- All 32 top-level Python files parsed successfully.
- Exact Phase-1 serialization retained val_sel.
- Extracted current k-NN and tuned calibration functions preserved closed accuracy and AUROC on synthetic data.
- Reproduced missing-artifact completion and empty-provenance acceptance.
- Main and Phase-2 tuner plan-only commands ran successfully.
- No completed common_eval.json files exist under res. No full training or end-to-end dataset evaluation was performed.
- The synthetic k-NN check completed its assertions but emitted NumPy/scikit-learn matrix-operation runtime warnings; their numerical/runtime cause was not investigated in this recheck.
