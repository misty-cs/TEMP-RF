# Third review — 2026-09-28

Verdict: the previous major implementation blockers have been addressed in the inspected source. The next useful validation is an end-to-end pilot after the remaining provenance checks below are tightened. This is not a certification of experimental fairness or tuning adequacy: no completed common evaluations exist here.

## Verified improvements

- main.py supplies reproducibility variables before every subprocess; common_eval.py initializes reproducibility. The prior default train/eval environment mismatch is resolved.
- Main now recomputes Phase-2 selection identity and verifies its source checkpoint; Phase-2 tuning validates Phase 1. Selected Phase-2 records retain their identity.
- Code identities include transitive local imports and software versions.
- Completion requires audited artifacts and their hashes. Synthetic tests: missing result artifacts are partial, complete artifacts are complete, a modified result is partial. Nonempty folders without training manifests are stale.
- Joint evaluation now writes train/eval manifests and result hashes, allowing reuse.
- Empty split provenance is rejected; valid disjoint groups with every device are accepted. Fresh reference-group comparisons and recomputed per-device test budgets are implemented.
- Main aggregation checks hashes and rejects multiple study keys within a tag.
- PROTOCOL.md consolidates the current protocol; previous methodological decisions are separated into METHODS_HISTORY.md.

## Remaining findings

1. **P2 — Data identity does not reliably reject missing/changed data.** manifest.py:38 records only basename and byte size for WiSig, so a same-size replacement remains identical even when its modification time changes. Folder datasets with missing sessions yield n_files=0 dictionaries, but data_identity at line 101 rejects only the literal string 'missing'. Reproduced data_identity successfully returning a hash for an entirely absent mock folder dataset. Packet-cache content is also outside the identity. Reject empty/missing required sessions and use a stable content inventory (or explicitly documented weaker metadata identity); record the relevant cache version/content identity.

2. **P2 — Joint aggregation has weaker validity checks than main aggregation.** aggregate.py:253 checks artifact hashes but ignores audit.all_pass and AUDIT_FAILED. Lines 255–258 only require matching split/seed and test-input hash, so different training settings, software or code can still enter a paired joint comparison. joint_eval.py writes all_pass=True unconditionally after evaluation; it is a provenance record, not the full pipeline audit. Require passing status, no failure marker, and compatible training protocol/code/version identities, allowing only the intentional joint-versus-sequential differences. Retain the distinction between provenance and a substantive audit.

3. **P2 — Reuse and study compatibility omit effective configuration/version checks.** manifest.run_status compares code, environment and optional data but not current runtime versions or the requested effective configuration. A synthetic intact-artifact run with an obsolete recorded NumPy version still returned complete. aggregate.study_key compares training versions but omits effective config and evaluation versions. Runs created through run_experiment.py CLI options (such as model or label-budget settings) can share code/environment while differing in configuration. Add normalized config comparison (excluding seed, split and output path as appropriate) and both training/evaluation software identities. Ordinary tuned main.py runs have an additional version check through tuning identities, which reduces but does not eliminate this gap, particularly for untuned pilots or partial runs.

4. **Protocol decision still open.** PROTOCOL.md calls v2 frozen, but section 7 leaves the primary comparison family open. Resolve and timestamp the family before confirmatory runs. The document also says the t-test and effect sizes carry the evidence because Wilcoxon cannot reach significance with 4–5 splits. Low power is not evidence that a different test's assumptions hold; retain the stated dependence limitations and emphasize effect estimates without promising significance.

Other disclosed limits remain: unequal search budgets, a shared centroid-selected backbone, overlapping enrolled devices across splits, reuse of Part B, and historical test feedback. These do not establish intentional bias; they constrain what the study can claim. Actual under-tuning still requires validation curves and completed runs.

## Checks performed

All 32 top-level Python files parsed. Synthetic tests verified missing/changed-artifact rejection, intact-artifact acceptance, empty-group rejection, valid-group acceptance, and the remaining missing-data inventory failure. Confirmed transitive dependencies include temporal_trajectory.py and finetune.py. Main and Phase-2 tuner plan-only commands succeeded. No training was launched and no source/data/results were changed. This review adds only this report.
