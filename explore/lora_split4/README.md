# LoRa split 4: why is the AUROC of every learned decision close to 0? (2026-10-11)

Read-only analysis of finished final runs (no run, setting or result was changed).

Observation: in LoRa split 4 (unknown devices 0-3 = folders Device1-4) the AUROC of
all learned decisions is 0.01-0.14 over 3 seeds, while the network-free raw centroid
reaches 0.98. Other splits: 0.37-0.76.

Findings
1. Labels and device mapping are correct (loader remaps to sorted IDs; common_eval
   maps back the same way).
2. Raw slices of devices 0-3 are normal LoRa chirps (alignment quality ~0.99,
   peak/RMS ~1.16 like every device) - `raw_check.txt`, `spikes.txt`.
3. With the trained split-4 model, EVERY slice of devices 0-3 (reference day and test
   day) has an embedding identical to the centroid of known device 9 (similarity
   1.000; 100% of slices nearest to device 9). Device 9 itself: 0.998-0.999.
   The pre-normalization embedding of these devices is very long (||z|| ~600 vs
   86-390 for most others), always in the same direction - `report_shift4_seed42.txt`,
   `prenorm.txt`.
4. About 0.3% of LoRa slices (254 of 78,000; all devices and days) have low power;
   per-slice RMS normalization turns them into slices with large spikes (up to 15).
   This affects all devices about equally and is not the cause of finding 3 (the
   median slice of devices 0-3 is normal) - `empty_slices.txt`, `pipeline_input.txt`.

Interpretation: in split 4 the network learned features that make devices 0-3
indistinguishable from known device 9 (a collapse to one embedding direction), so
the unknown devices are accepted with high confidence. The raw input still separates
them, i.e. the learned embedding discards information that distinguishes these
devices (candidates: invariances from training augmentation, the chirp alignment).
This is a genuine failure mode of the learned representation on this split, not a
label or leakage error. Not changed, because any change now would be post hoc.
