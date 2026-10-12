# Summary: lora — COMPLETE

Audited runs used: 15 (splits [0, 4, 8, 12, 16], seeds [7, 13, 42]); expected splits [0, 4, 8, 12, 16] x seeds [42, 7, 13]

## All rules (mean ± std over runs)

| Rule | Closed acc. | AUROC | fixed90 accept | fixed90 open | fixed90 unk. det. | fixed90 known ok | objective accept | objective open | objective unk. det. | objective known ok | tuned accept | tuned open | tuned unk. det. | tuned known ok |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| softmax | 0.622 ± 0.044 | – | – | – | – | – | – | – | – | – | – | – | – | – |
| msp | 0.622 ± 0.044 | 0.339 ± 0.225 | 0.900 ± 0.019 | 0.654 ± 0.045 | 0.103 ± 0.115 | 0.589 ± 0.042 | 0.472 ± 0.040 | 0.849 ± 0.082 | 0.308 ± 0.253 | 0.398 ± 0.023 | 0.445 ± 0.386 | 0.855 ± 0.183 | 0.415 ± 0.483 | 0.316 ± 0.223 |
| probe | 0.614 ± 0.050 | 0.400 ± 0.221 | 0.896 ± 0.011 | 0.645 ± 0.050 | 0.112 ± 0.107 | 0.578 ± 0.047 | 0.490 ± 0.035 | 0.798 ± 0.073 | 0.397 ± 0.278 | 0.390 ± 0.034 | 0.816 ± 0.190 | 0.685 ± 0.103 | 0.129 ± 0.204 | 0.544 ± 0.098 |
| openmax | 0.612 ± 0.038 | 0.545 ± 0.278 | 0.904 ± 0.013 | 0.633 ± 0.048 | 0.225 ± 0.139 | 0.572 ± 0.041 | 0.633 ± 0.116 | 0.724 ± 0.086 | 0.459 ± 0.258 | 0.454 ± 0.074 | 0.774 ± 0.059 | 0.660 ± 0.056 | 0.362 ± 0.200 | 0.512 ± 0.065 |
| static | 0.627 ± 0.046 | 0.468 ± 0.235 | 0.893 ± 0.021 | 0.640 ± 0.048 | 0.141 ± 0.143 | 0.571 ± 0.038 | 0.443 ± 0.056 | 0.699 ± 0.036 | 0.470 ± 0.330 | 0.309 ± 0.034 | 0.716 ± 0.104 | 0.663 ± 0.034 | 0.262 ± 0.211 | 0.475 ± 0.077 |
| knn | 0.627 ± 0.046 | 0.535 ± 0.219 | 0.897 ± 0.014 | 0.630 ± 0.051 | 0.182 ± 0.132 | 0.565 ± 0.043 | 0.959 ± 0.035 | 0.629 ± 0.047 | 0.123 ± 0.102 | 0.603 ± 0.043 | 0.719 ± 0.064 | 0.638 ± 0.051 | 0.349 ± 0.220 | 0.458 ± 0.048 |
| temp_rff_p1 | 0.629 ± 0.048 | 0.475 ± 0.225 | 0.895 ± 0.022 | 0.640 ± 0.049 | 0.139 ± 0.143 | 0.573 ± 0.041 | 0.456 ± 0.061 | 0.701 ± 0.037 | 0.470 ± 0.330 | 0.319 ± 0.041 | 0.727 ± 0.116 | 0.664 ± 0.034 | 0.248 ± 0.211 | 0.484 ± 0.088 |
| static_window | 0.627 ± 0.046 | 0.472 ± 0.229 | 0.895 ± 0.023 | 0.640 ± 0.049 | 0.141 ± 0.143 | 0.572 ± 0.038 | 0.453 ± 0.056 | 0.700 ± 0.036 | 0.470 ± 0.330 | 0.316 ± 0.035 | 0.717 ± 0.106 | 0.662 ± 0.035 | 0.262 ± 0.211 | 0.476 ± 0.078 |
| knn_window | 0.629 ± 0.048 | 0.535 ± 0.219 | 0.899 ± 0.014 | 0.631 ± 0.052 | 0.179 ± 0.128 | 0.567 ± 0.045 | 0.961 ± 0.036 | 0.631 ± 0.048 | 0.118 ± 0.097 | 0.605 ± 0.046 | 0.720 ± 0.066 | 0.640 ± 0.051 | 0.347 ± 0.219 | 0.460 ± 0.051 |
| temp_rff | 0.629 ± 0.048 | 0.475 ± 0.225 | 0.895 ± 0.022 | 0.640 ± 0.049 | 0.139 ± 0.143 | 0.573 ± 0.041 | 0.456 ± 0.061 | 0.701 ± 0.037 | 0.470 ± 0.330 | 0.319 ± 0.041 | 0.727 ± 0.116 | 0.664 ± 0.034 | 0.248 ± 0.211 | 0.484 ± 0.088 |
| temp_rff_nowt | 0.630 ± 0.048 | 0.476 ± 0.228 | 0.896 ± 0.022 | 0.641 ± 0.049 | 0.139 ± 0.143 | 0.573 ± 0.041 | 0.459 ± 0.063 | 0.704 ± 0.038 | 0.471 ± 0.332 | 0.323 ± 0.044 | 0.723 ± 0.116 | 0.665 ± 0.034 | 0.251 ± 0.211 | 0.482 ± 0.089 |
| probe_default | 0.633 ± 0.042 | 0.333 ± 0.168 | 0.901 ± 0.011 | 0.665 ± 0.038 | 0.068 ± 0.062 | 0.599 ± 0.038 | 0.496 ± 0.031 | 0.821 ± 0.056 | 0.277 ± 0.218 | 0.406 ± 0.027 | 0.404 ± 0.425 | 0.867 ± 0.171 | 0.517 ± 0.443 | 0.284 ± 0.272 |
| knn_default | 0.613 ± 0.049 | 0.569 ± 0.184 | 0.899 ± 0.015 | 0.618 ± 0.052 | 0.206 ± 0.131 | 0.556 ± 0.046 | 0.979 ± 0.023 | 0.616 ± 0.050 | 0.116 ± 0.098 | 0.602 ± 0.045 | 0.763 ± 0.071 | 0.629 ± 0.048 | 0.316 ± 0.191 | 0.481 ± 0.075 |
| openmax_default | 0.617 ± 0.041 | 0.515 ± 0.278 | 0.902 ± 0.014 | 0.644 ± 0.041 | 0.163 ± 0.134 | 0.581 ± 0.035 | 0.470 ± 0.020 | 0.747 ± 0.057 | 0.581 ± 0.348 | 0.351 ± 0.028 | 0.602 ± 0.251 | 0.712 ± 0.095 | 0.429 ± 0.384 | 0.421 ± 0.152 |
| temp_rff_default | 0.598 ± 0.045 | 0.615 ± 0.125 | 0.902 ± 0.014 | 0.604 ± 0.046 | 0.227 ± 0.140 | 0.544 ± 0.044 | 0.973 ± 0.035 | 0.601 ± 0.045 | 0.131 ± 0.120 | 0.584 ± 0.047 | 0.784 ± 0.062 | 0.615 ± 0.043 | 0.340 ± 0.169 | 0.484 ± 0.062 |
| raw_centroid | 0.557 ± 0.047 | 0.711 ± 0.176 | 0.875 ± 0.013 | 0.602 ± 0.051 | 0.431 ± 0.322 | 0.526 ± 0.045 | 0.662 ± 0.025 | 0.605 ± 0.057 | 0.644 ± 0.245 | 0.399 ± 0.028 | 0.696 ± 0.120 | 0.603 ± 0.055 | 0.624 ± 0.283 | 0.420 ± 0.083 |
| raw_centroid_cfo | 0.560 ± 0.047 | 0.712 ± 0.175 | 0.872 ± 0.013 | 0.604 ± 0.052 | 0.433 ± 0.315 | 0.526 ± 0.041 | 0.657 ± 0.025 | 0.607 ± 0.054 | 0.647 ± 0.245 | 0.398 ± 0.033 | 0.731 ± 0.054 | 0.608 ± 0.053 | 0.588 ± 0.282 | 0.444 ± 0.055 |

Tuned calibration: method chosen on Part B, count over runs: msp: q80×1/q95×2/youden×12, probe: q80×2/q90×2/q95×3/youden×8, openmax: q80×4/youden×11, static: q80×5/youden×10, knn: q80×3/youden×12, temp_rff_p1: q80×4/q90×1/youden×10, static_window: q80×5/youden×10, knn_window: q80×3/youden×12, temp_rff: q80×4/q90×1/youden×10, temp_rff_nowt: q80×4/q90×1/youden×10, probe_default: q80×1/q90×1/q95×1/youden×12, knn_default: q80×7/q95×1/youden×7, openmax_default: q80×2/q90×1/youden×12, temp_rff_default: q80×11/q90×1/youden×3, raw_centroid: q80×7/youden×8, raw_centroid_cfo: q80×4/youden×11

## Per device split (mean over seeds)

| Rule | shift 0 closed / AUROC | shift 4 closed / AUROC | shift 8 closed / AUROC | shift 12 closed / AUROC | shift 16 closed / AUROC |
|---|---|---|---|---|---|
| softmax | 0.642 / – | 0.616 / – | 0.608 / – | 0.561 / – | 0.681 / – |
| msp | 0.642 / 0.512 | 0.616 / 0.010 | 0.608 / 0.266 | 0.561 / 0.272 | 0.681 / 0.637 |
| probe | 0.637 / 0.597 | 0.618 / 0.108 | 0.593 / 0.270 | 0.542 / 0.344 | 0.681 / 0.680 |
| openmax | 0.631 / 0.692 | 0.611 / 0.019 | 0.588 / 0.605 | 0.564 / 0.651 | 0.669 / 0.757 |
| static | 0.658 / 0.714 | 0.613 / 0.086 | 0.598 / 0.534 | 0.576 / 0.366 | 0.692 / 0.643 |
| static_window | 0.658 / 0.714 | 0.613 / 0.103 | 0.598 / 0.534 | 0.576 / 0.366 | 0.692 / 0.643 |
| knn | 0.651 / 0.742 | 0.600 / 0.136 | 0.598 / 0.623 | 0.585 / 0.535 | 0.699 / 0.640 |
| knn_window | 0.660 / 0.741 | 0.600 / 0.136 | 0.597 / 0.623 | 0.586 / 0.535 | 0.699 / 0.640 |
| temp_rff_p1 | 0.658 / 0.714 | 0.611 / 0.116 | 0.599 / 0.534 | 0.576 / 0.366 | 0.701 / 0.645 |
| temp_rff | 0.658 / 0.714 | 0.611 / 0.116 | 0.599 / 0.534 | 0.576 / 0.366 | 0.701 / 0.645 |
| raw_centroid_cfo | 0.594 / 0.591 | 0.524 / 0.979 | 0.519 / 0.781 | 0.532 / 0.728 | 0.628 / 0.484 |

## PRIMARY test: TEMP-RFF minus rule at the split level

Seeds are averaged within each device split first (the seeds of a split share its devices, so they are not independent); tests are over the splits. Outcomes: closed accuracy, AUROC, tuned-protocol known ok and unknown detection. Holm correction over one family: 84 tests: 7 rules ['msp', 'probe', 'openmax', 'static', 'knn', 'static_window', 'knn_window'] x 4 outcomes x tags ['wisig', 'pycom', 'lora'].

| Rule | outcome | mean diff | sd diff | splits won | t p | t p (Holm) | Wilcoxon p | Wilcoxon p (Holm) |
|---|---|---|---|---|---|---|---|---|
| knn_window | auroc | -0.060 | 0.070 | 1/5 | 0.128 | 1 | 0.125 | 1 |
| knn_window | closed | +0.000 | 0.007 | 3/5 | 0.942 | 1 | 1 | 1 |
| knn_window | tuned_known_ok | +0.024 | 0.053 | 3/5 | 0.373 | 1 | 0.625 | 1 |
| knn_window | tuned_unk_det | -0.099 | 0.080 | 0/5 | 0.0503 | 1 | 0.0625 | 1 |
| knn | auroc | -0.060 | 0.070 | 1/5 | 0.127 | 1 | 0.125 | 1 |
| knn | closed | +0.002 | 0.008 | 4/5 | 0.553 | 1 | 0.438 | 1 |
| knn | tuned_known_ok | +0.026 | 0.054 | 3/5 | 0.346 | 1 | 0.438 | 1 |
| knn | tuned_unk_det | -0.101 | 0.080 | 0/5 | 0.0479 | 1 | 0.0625 | 1 |
| msp | auroc | +0.136 | 0.101 | 5/5 | 0.0398 | 1 | 0.0625 | 1 |
| msp | closed | +0.007 | 0.013 | 3/5 | 0.291 | 1 | 0.312 | 1 |
| msp | tuned_known_ok | +0.167 | 0.284 | 3/5 | 0.258 | 1 | 0.438 | 1 |
| msp | tuned_unk_det | -0.167 | 0.404 | 3/5 | 0.407 | 1 | 0.812 | 1 |
| openmax | auroc | -0.070 | 0.145 | 2/5 | 0.345 | 1 | 0.438 | 1 |
| openmax | closed | +0.016 | 0.013 | 4/5 | 0.0498 | 1 | 0.125 | 1 |
| openmax | tuned_known_ok | -0.028 | 0.020 | 1/5 | 0.0339 | 1 | 0.125 | 1 |
| openmax | tuned_unk_det | -0.114 | 0.189 | 2/5 | 0.249 | 1 | 0.312 | 1 |
| probe | auroc | +0.075 | 0.119 | 4/5 | 0.23 | 1 | 0.312 | 1 |
| probe | closed | +0.015 | 0.015 | 4/5 | 0.099 | 1 | 0.188 | 1 |
| probe | tuned_known_ok | -0.060 | 0.073 | 1/5 | 0.139 | 1 | 0.188 | 1 |
| probe | tuned_unk_det | +0.119 | 0.156 | 4/5 | 0.164 | 1 | 0.188 | 1 |
| static_window | auroc | +0.003 | 0.006 | 3/5 | 0.305 | 1 | 0.625 | 1 |
| static_window | closed | +0.001 | 0.004 | 3/5 | 0.477 | 1 | 0.812 | 1 |
| static_window | tuned_known_ok | +0.008 | 0.020 | 2/5 | 0.426 | 1 | 1 | 1 |
| static_window | tuned_unk_det | -0.014 | 0.031 | 0/5 | 0.374 | 1 | 1 | 1 |
| static | auroc | +0.007 | 0.014 | 3/5 | 0.34 | 1 | 0.625 | 1 |
| static | closed | +0.001 | 0.004 | 3/5 | 0.481 | 1 | 0.625 | 1 |
| static | tuned_known_ok | +0.009 | 0.020 | 2/5 | 0.378 | 1 | 1 | 1 |
| static | tuned_unk_det | -0.014 | 0.031 | 1/5 | 0.378 | 1 | 1 | 1 |

## Secondary (exploratory): TEMP-RFF minus other rules at the split level, uncorrected

| Rule | outcome | mean diff | sd diff | splits won | t p | Wilcoxon p |
|---|---|---|---|---|---|---|
| knn_default | auroc | -0.094 | 0.079 | 0/5 | 0.057 | 0.0625 |
| knn_default | closed | +0.015 | 0.010 | 5/5 | 0.0224 | 0.0625 |
| knn_default | tuned_known_ok | +0.002 | 0.039 | 2/5 | 0.899 | 0.812 |
| knn_default | tuned_unk_det | -0.068 | 0.123 | 2/5 | 0.283 | 0.625 |
| openmax_default | auroc | -0.040 | 0.087 | 1/5 | 0.363 | 0.312 |
| openmax_default | closed | +0.012 | 0.012 | 4/5 | 0.0897 | 0.188 |
| openmax_default | tuned_known_ok | +0.062 | 0.053 | 4/5 | 0.0596 | 0.125 |
| openmax_default | tuned_unk_det | -0.181 | 0.253 | 1/5 | 0.184 | 0.188 |
| probe_default | auroc | +0.142 | 0.108 | 5/5 | 0.0426 | 0.0625 |
| probe_default | closed | -0.004 | 0.007 | 1/5 | 0.311 | 0.312 |
| probe_default | tuned_known_ok | +0.199 | 0.135 | 4/5 | 0.0301 | 0.125 |
| probe_default | tuned_unk_det | -0.269 | 0.355 | 1/5 | 0.165 | 0.188 |
| raw_centroid_cfo | auroc | -0.237 | 0.417 | 2/5 | 0.272 | 0.312 |
| raw_centroid_cfo | closed | +0.069 | 0.017 | 5/5 | 0.000735 | 0.0625 |
| raw_centroid_cfo | tuned_known_ok | +0.039 | 0.049 | 4/5 | 0.146 | 0.188 |
| raw_centroid_cfo | tuned_unk_det | -0.340 | 0.457 | 2/5 | 0.171 | 0.312 |
| raw_centroid | auroc | -0.236 | 0.417 | 2/5 | 0.274 | 0.312 |
| raw_centroid | closed | +0.072 | 0.019 | 5/5 | 0.00101 | 0.0625 |
| raw_centroid | tuned_known_ok | +0.064 | 0.034 | 5/5 | 0.0133 | 0.0625 |
| raw_centroid | tuned_unk_det | -0.376 | 0.436 | 2/5 | 0.126 | 0.312 |
| softmax | closed | +0.007 | 0.013 | 3/5 | 0.291 | 0.312 |
| temp_rff_default | auroc | -0.140 | 0.125 | 0/5 | 0.0665 | 0.0625 |
| temp_rff_default | closed | +0.031 | 0.010 | 5/5 | 0.00206 | 0.0625 |
| temp_rff_default | tuned_known_ok | +0.000 | 0.035 | 2/5 | 0.987 | 1 |
| temp_rff_default | tuned_unk_det | -0.092 | 0.130 | 2/5 | 0.189 | 0.312 |
| temp_rff_nowt | auroc | -0.001 | 0.010 | 1/5 | 0.81 | 1 |
| temp_rff_nowt | closed | -0.001 | 0.002 | 0/5 | 0.262 | 0.5 |
| temp_rff_nowt | tuned_known_ok | +0.002 | 0.005 | 1/5 | 0.434 | 1 |
| temp_rff_nowt | tuned_unk_det | -0.004 | 0.008 | 0/5 | 0.345 | 0.5 |
| temp_rff_p1 | auroc | +0.000 | 0.000 | 0/5 | nan | 1 |
| temp_rff_p1 | closed | +0.000 | 0.000 | 0/5 | nan | 1 |
| temp_rff_p1 | tuned_known_ok | +0.000 | 0.000 | 0/5 | nan | 1 |
| temp_rff_p1 | tuned_unk_det | +0.000 | 0.000 | 0/5 | nan | 1 |

With 5 splits the smallest possible two-sided Wilcoxon p is 0.0625; report effect sizes and wins, not only p-values.


## Secondary test: TEMP-RFF minus rule per run (same split and seed; not independent)

| Rule | outcome | mean diff | wins | paired t p | Wilcoxon p |
|---|---|---|---|---|---|
| softmax | closed | +0.007 | 9/15 | 0.261 | 0.3 |
| msp | closed | +0.007 | 9/15 | 0.261 | 0.3 |
| msp | auroc | +0.136 | 15/15 | 0.000136 | 6.1e-05 |
| msp | tuned_known_ok | +0.167 | 10/15 | 0.0431 | 0.0833 |
| msp | tuned_unk_det | -0.167 | 7/15 | 0.131 | 0.245 |
| probe | closed | +0.015 | 11/15 | 0.00467 | 0.00537 |
| probe | auroc | +0.075 | 11/15 | 0.0282 | 0.0302 |
| probe | tuned_known_ok | -0.060 | 2/15 | 0.0407 | 0.0219 |
| probe | tuned_unk_det | +0.119 | 12/15 | 0.0684 | 0.0353 |
| openmax | closed | +0.016 | 12/15 | 0.00292 | 0.00336 |
| openmax | auroc | -0.070 | 6/15 | 0.074 | 0.0946 |
| openmax | tuned_known_ok | -0.028 | 5/15 | 0.0253 | 0.0256 |
| openmax | tuned_unk_det | -0.114 | 4/15 | 0.0414 | 0.064 |
| static | closed | +0.001 | 4/15 | 0.437 | 0.553 |
| static | auroc | +0.007 | 9/15 | 0.303 | 0.229 |
| static | tuned_known_ok | +0.009 | 2/15 | 0.338 | 0.893 |
| static | tuned_unk_det | -0.014 | 1/15 | 0.338 | 0.655 |
| knn | closed | +0.002 | 9/15 | 0.63 | 0.679 |
| knn | auroc | -0.060 | 4/15 | 0.0105 | 0.0125 |
| knn | tuned_known_ok | +0.026 | 10/15 | 0.211 | 0.33 |
| knn | tuned_unk_det | -0.101 | 3/15 | 0.0101 | 0.0238 |
| temp_rff_p1 | closed | +0.000 | 0/15 | nan | nan |
| temp_rff_p1 | auroc | +0.000 | 0/15 | nan | nan |
| temp_rff_p1 | tuned_known_ok | +0.000 | 0/15 | nan | nan |
| temp_rff_p1 | tuned_unk_det | +0.000 | 0/15 | nan | nan |
| static_window | closed | +0.001 | 4/15 | 0.434 | 0.672 |
| static_window | auroc | +0.003 | 6/15 | 0.271 | 0.712 |
| static_window | tuned_known_ok | +0.008 | 2/15 | 0.384 | 0.893 |
| static_window | tuned_unk_det | -0.014 | 0/15 | 0.334 | 0.317 |
| knn_window | closed | +0.000 | 8/15 | 0.958 | 0.978 |
| knn_window | auroc | -0.060 | 4/15 | 0.0107 | 0.0151 |
| knn_window | tuned_known_ok | +0.024 | 10/15 | 0.26 | 0.359 |
| knn_window | tuned_unk_det | -0.099 | 3/15 | 0.0123 | 0.0238 |
| temp_rff_nowt | closed | -0.001 | 0/15 | 0.234 | 0.18 |
| temp_rff_nowt | auroc | -0.001 | 1/15 | 0.787 | 0.655 |
| temp_rff_nowt | tuned_known_ok | +0.002 | 1/15 | 0.392 | 0.655 |
| temp_rff_nowt | tuned_unk_det | -0.004 | 0/15 | 0.307 | 0.18 |
| probe_default | closed | -0.004 | 5/15 | 0.17 | 0.169 |
| probe_default | auroc | +0.142 | 14/15 | 0.000196 | 0.000305 |
| probe_default | tuned_known_ok | +0.199 | 9/15 | 0.0163 | 0.0302 |
| probe_default | tuned_unk_det | -0.269 | 7/15 | 0.0643 | 0.121 |
| knn_default | closed | +0.015 | 12/15 | 0.00596 | 0.00836 |
| knn_default | auroc | -0.094 | 1/15 | 0.00107 | 0.000427 |
| knn_default | tuned_known_ok | +0.002 | 8/15 | 0.895 | 0.755 |
| knn_default | tuned_unk_det | -0.068 | 6/15 | 0.118 | 0.221 |
| openmax_default | closed | +0.012 | 10/15 | 0.0161 | 0.0181 |
| openmax_default | auroc | -0.040 | 6/15 | 0.134 | 0.229 |
| openmax_default | tuned_known_ok | +0.062 | 11/15 | 0.0863 | 0.107 |
| openmax_default | tuned_unk_det | -0.181 | 2/15 | 0.0579 | 0.0277 |
| temp_rff_default | closed | +0.031 | 14/15 | 2.43e-06 | 0.000122 |
| temp_rff_default | auroc | -0.140 | 2/15 | 0.000748 | 0.000427 |
| temp_rff_default | tuned_known_ok | +0.000 | 9/15 | 0.989 | 0.804 |
| temp_rff_default | tuned_unk_det | -0.092 | 6/15 | 0.0559 | 0.151 |
| raw_centroid | closed | +0.072 | 15/15 | 6.48e-09 | 6.1e-05 |
| raw_centroid | auroc | -0.236 | 6/15 | 0.0336 | 0.0479 |
| raw_centroid | tuned_known_ok | +0.064 | 12/15 | 0.0204 | 0.0215 |
| raw_centroid | tuned_unk_det | -0.376 | 4/15 | 0.00452 | 0.0146 |
| raw_centroid_cfo | closed | +0.069 | 15/15 | 4.39e-09 | 6.1e-05 |
| raw_centroid_cfo | auroc | -0.237 | 6/15 | 0.0326 | 0.0413 |
| raw_centroid_cfo | tuned_known_ok | +0.039 | 12/15 | 0.0666 | 0.0946 |
| raw_centroid_cfo | tuned_unk_det | -0.340 | 5/15 | 0.00941 | 0.0467 |
## Forced prototype window (decay 1)

| W | Closed acc. | AUROC | tuned accept | tuned unk. det. |
|---|---|---|---|---|
| 1 | 0.627 ± 0.046 | 0.468 ± 0.235 | 0.716 ± 0.104 | 0.262 ± 0.211 |
| 2 | 0.626 ± 0.048 | 0.480 ± 0.223 | 0.819 ± 0.067 | 0.183 ± 0.157 |
| 3 | 0.620 ± 0.047 | 0.498 ± 0.215 | 0.839 ± 0.059 | 0.169 ± 0.134 |
| 4 | 0.621 ± 0.046 | 0.507 ± 0.211 | 0.843 ± 0.054 | 0.193 ± 0.159 |

## Few-shot target-session enrollment (same support for every rule)

| k requested | k effective | temp_rff closed / AUROC / tuned known ok | static closed / AUROC / tuned known ok | knn closed / AUROC / tuned known ok | static_window closed / AUROC / tuned known ok | knn_window closed / AUROC / tuned known ok |
|---|---|---|---|---|---|---|
| 1 | 1 | 0.614 / 0.481 / 0.503 | 0.631 / 0.468 / 0.470 | 0.629 / 0.536 / 0.459 | 0.631 / 0.472 / 0.471 | 0.631 / 0.536 / 0.461 |
| 5 | 5 | 0.630 / 0.484 / 0.518 | 0.631 / 0.469 / 0.470 | 0.630 / 0.536 / 0.462 | 0.631 / 0.472 / 0.471 | 0.632 / 0.536 / 0.464 |
| 10 | 10 | 0.634 / 0.481 / 0.520 | 0.632 / 0.469 / 0.473 | 0.630 / 0.536 / 0.464 | 0.631 / 0.472 / 0.474 | 0.632 / 0.536 / 0.467 |
| 20 | 20 | 0.637 / 0.478 / 0.519 | 0.632 / 0.470 / 0.477 | 0.631 / 0.537 / 0.470 | 0.632 / 0.473 / 0.477 | 0.633 / 0.536 / 0.472 |
| 50 | 50 | 0.638 / 0.479 / 0.523 | 0.634 / 0.471 / 0.492 | 0.634 / 0.538 / 0.480 | 0.633 / 0.475 / 0.493 | 0.636 / 0.538 / 0.482 |
| 100 | 100 | 0.639 / 0.480 / 0.530 | 0.635 / 0.474 / 0.509 | 0.637 / 0.541 / 0.486 | 0.634 / 0.477 / 0.509 | 0.639 / 0.541 / 0.487 |
| 200 | 200 | 0.639 / 0.480 / 0.527 | 0.637 / 0.477 / 0.515 | 0.642 / 0.545 / 0.495 | 0.636 / 0.478 / 0.513 | 0.644 / 0.545 / 0.496 |
| 500 | 390 | 0.640 / 0.479 / 0.529 | 0.637 / 0.480 / 0.528 | 0.648 / 0.553 / 0.498 | 0.637 / 0.481 / 0.526 | 0.649 / 0.553 / 0.499 |

## Pipeline stages (Part-A centroids)

| Stage | Softmax acc. | Centroid acc. | AUROC |
|---|---|---|---|
| phase1 | 0.613 ± 0.038 | 0.621 ± 0.045 | 0.485 ± 0.166 |
| phase2 | 0.618 ± 0.042 | 0.627 ± 0.045 | 0.471 ± 0.236 |
| phase3 | 0.622 ± 0.044 | 0.627 ± 0.046 | 0.468 ± 0.235 |

## Selected settings over runs

- W: 1×12, 2×2, 4×1
- decay: 0.6×2, 1.0×13
- P: 1×15
- knn (k, weighted): (10, False)×3, (20, False)×11, (5, False)×1
- openmax (dist, tail, alpha): ('euclidean', 10, 1)×15
- probe C: 0.01×11, 0.1×1, 1.0×1, 10.0×2
- static_window W: (1,)×14, (2,)×1
- knn_window W (k, weighted): (1, 10, False)×3, (1, 20, False)×11, (4, 20, False)×1
- temp_rff_p1 W decay: (1, 1.0)×12, (2, 0.6)×2, (4, 1.0)×1
- selection-weight sensitivity: choice changes with the weight (0.3/0.5/0.7) in 8 of 15 runs
