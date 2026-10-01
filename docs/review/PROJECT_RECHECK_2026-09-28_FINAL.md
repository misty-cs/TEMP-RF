# Fourth review — 2026-09-28

Verdict: no remaining blocker from the previous review was found in the revised code. Ready for an end-to-end pilot, not yet validated for final experimental claims. No completed common_eval.json files exist under res.

Verified with targeted checks:
- All 33 top-level Python files parse.
- Same-size changed data produces a different full content hash (temporary-file test).
- Missing folder datasets raise an error instead of receiving a valid data identity.
- Actual run_experiment CLI configuration construction matches datasets.make_cfg for WiSig and Pycom, without running training.
- Intact audited artifacts with matching configuration/software are accepted as complete.
- Changed effective configuration and obsolete software versions are classified stale. Runtime-version lookup was stubbed to test comparator branches in isolation.
- Changed result artifacts are rejected as partial.

Verified by source inspection:
- Joint aggregation requires passing joint status, no AUDIT_FAILED marker, intact artifacts, matching test input and compatible study identity/Phase-1 settings.
- Joint evaluation performs substantive pass/fail checks rather than writing unconditional success.
- The primary comparison family is fixed in PROTOCOL.md and implemented as seven competitors x four outcomes x two datasets (56 comparisons per statistical test family).
- Packet detector source identity is part of cache keys.

Minor documentation cleanup: aggregate.py's joint-table caption still says 'joint runs have no audit', despite the implemented joint checks. This does not alter computed results.

Limits: large raw files use a documented sampled hash plus metadata, not full content verification. The packet cache key still uses integer-second modification time; this is not a cryptographic guarantee against arbitrary same-size, same-timestamp data edits. These limitations do not invalidate the fixes tested here. Unequal tuning budgets, shared-backbone selection, dependent device splits and historical test feedback remain disclosed study limitations. Performance, empirical tuning adequacy, numerical stability and full audit success require actual pilots.

Recommended next step: one full split/seed pilot per dataset, including common evaluation and audit, then inspect selected settings and baseline validation curves before the final study. No training was launched or experiment source/data/results modified during this review.
