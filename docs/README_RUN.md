> **Superseded.** The current way to run the study is `./run_all.sh` (see the
> top-level README.md and docs/PROTOCOL.md section 16). This file is kept as the
> run guide of protocol v2/v3.

# TEMP-RFF runs (WiSig, Pycom)

Current protocol: `PROTOCOL.md` (v3 = section 11). Fairness per rule: `FAIRNESS.md`. Decision
log (how each value was found): `METHODS_HISTORY.md`. Dataset settings live
only in `datasets.py`. Nothing runs without `--go`.

Two run sets (PROTOCOL.md section 11):
- final: `./run_full_plan.sh` (tuning -> tuned pilot -> gate -> res/)
- draft: `./run_draft.sh` (Phase-1/2 defaults -> res_draft/)

Full order with tuning, step by step (what run_full_plan.sh does):

    python pool_check.py                                  # Pycom cohort rule (PASS expected)
    python tune_phase1.py --dataset wisig --go            # lambda, T per split
    python tune_phase1.py --dataset pycom --go
    python tune_phase2.py --dataset wisig --go            # replay, lr2 per split
    python tune_phase2.py --dataset pycom --go
    python main.py --dataset wisig --shifts 0 --seeds 42 --go    # pilot, then check audit.json
    python main.py --dataset pycom --shifts 0 --seeds 42 --go
    python main.py --dataset wisig --go                   # all splits x 5 seeds
    python main.py --dataset pycom --go
    python main.py --dataset wisig --joint --go           # joint-training control (split 0)
    python main.py --dataset pycom --joint --go
    python aggregate.py --tags wisig pycom                # summaries + combined primary test

Per run: `run_experiment.py` (Phases 1-3) -> `common_eval.py` -> `audit_run.py`
in `res/<tag>/shift<S>/seed<s>/`. A run is skipped only if its audit passed,
its audited artifacts are unchanged and code, settings and data match; a
folder made with other code/settings/data is reported STALE and left alone.
`--cfo remove` gives a separate variant (tag `<dataset>_nocfo`, own tuning).

Data links in `data/`: `wisig` (local pkl), `pycom_indoor` (TOSHIBA EXT drive).

Code map. Active in this protocol: main.py, run_experiment.py, phase1/2/3_*.py,
finetune.py, common_eval.py, openmax.py, selection.py, audit_run.py,
provenance_checks.py, manifest.py, repro.py, aggregate.py, joint_eval.py,
datasets.py, experiment_config.py, data_utils.py, load_slice_IQ.py,
signal_preprocessing.py, rf_models.py, temporal_trajectory.py, tta_bn.py,
pool_check.py, build_packet_cache.py. Present but switched off by the
protocol's settings (kept so that the code hash of the runs matches this
repository): phase4_openset_evaluation.py (T2R_SKIP_PHASE4=1), paper_plots*.py
(--skip_plots), xsession_finetune.py (T2R_XSESSION unset), proto_improve.py,
embedding_whitening.py (labelled whitening skipped: no adaptation set),
tune_phase1/2.py (not used in v3).
