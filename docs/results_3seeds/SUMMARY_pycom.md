# Summary: pycom — COMPLETE

Audited runs used: 15 (splits [0, 4, 8, 12, 16], seeds [7, 13, 42]); expected splits [0, 4, 8, 12, 16] x seeds [42, 7, 13]

## All rules (mean ± std over runs)

| Rule | Closed acc. | AUROC | fixed90 accept | fixed90 open | fixed90 unk. det. | fixed90 known ok | objective accept | objective open | objective unk. det. | objective known ok | tuned accept | tuned open | tuned unk. det. | tuned known ok |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| softmax | 0.069 ± 0.020 | – | – | – | – | – | – | – | – | – | – | – | – | – |
| msp | 0.069 ± 0.020 | 0.504 ± 0.056 | 0.945 ± 0.027 | 0.070 ± 0.020 | 0.048 ± 0.025 | 0.066 ± 0.018 | 0.572 ± 0.099 | 0.074 ± 0.025 | 0.434 ± 0.101 | 0.042 ± 0.015 | 0.531 ± 0.265 | 0.092 ± 0.047 | 0.476 ± 0.268 | 0.039 ± 0.015 |
| probe | 0.102 ± 0.029 | 0.478 ± 0.045 | 0.992 ± 0.018 | 0.102 ± 0.029 | 0.004 ± 0.007 | 0.101 ± 0.029 | 0.583 ± 0.108 | 0.113 ± 0.030 | 0.384 ± 0.137 | 0.065 ± 0.018 | 0.317 ± 0.171 | 0.137 ± 0.047 | 0.676 ± 0.191 | 0.040 ± 0.017 |
| openmax | 0.083 ± 0.020 | 0.499 ± 0.059 | 0.729 ± 0.138 | 0.084 ± 0.024 | 0.275 ± 0.130 | 0.061 ± 0.020 | 0.244 ± 0.111 | 0.103 ± 0.053 | 0.759 ± 0.110 | 0.022 ± 0.008 | 0.263 ± 0.254 | 0.123 ± 0.086 | 0.737 ± 0.228 | 0.023 ± 0.014 |
| static | 0.101 ± 0.027 | 0.479 ± 0.074 | 0.648 ± 0.170 | 0.122 ± 0.039 | 0.317 ± 0.206 | 0.076 ± 0.024 | 0.153 ± 0.072 | 0.189 ± 0.095 | 0.840 ± 0.120 | 0.025 ± 0.011 | 0.369 ± 0.164 | 0.143 ± 0.049 | 0.628 ± 0.197 | 0.051 ± 0.027 |
| knn | 0.107 ± 0.024 | 0.504 ± 0.061 | 0.588 ± 0.165 | 0.126 ± 0.039 | 0.417 ± 0.208 | 0.070 ± 0.019 | 0.101 ± 0.056 | 0.248 ± 0.122 | 0.909 ± 0.059 | 0.021 ± 0.008 | 0.339 ± 0.150 | 0.155 ± 0.061 | 0.656 ± 0.185 | 0.047 ± 0.017 |
| temp_rff_p1 | 0.101 ± 0.027 | 0.479 ± 0.074 | 0.648 ± 0.170 | 0.122 ± 0.039 | 0.317 ± 0.206 | 0.076 ± 0.024 | 0.153 ± 0.072 | 0.189 ± 0.095 | 0.840 ± 0.120 | 0.025 ± 0.011 | 0.369 ± 0.164 | 0.143 ± 0.049 | 0.628 ± 0.197 | 0.051 ± 0.027 |
| static_window | 0.102 ± 0.028 | 0.475 ± 0.073 | 0.649 ± 0.171 | 0.123 ± 0.039 | 0.312 ± 0.210 | 0.076 ± 0.024 | 0.156 ± 0.071 | 0.184 ± 0.094 | 0.838 ± 0.119 | 0.025 ± 0.011 | 0.370 ± 0.163 | 0.143 ± 0.049 | 0.626 ± 0.195 | 0.051 ± 0.027 |
| knn_window | 0.107 ± 0.024 | 0.504 ± 0.061 | 0.588 ± 0.165 | 0.126 ± 0.039 | 0.417 ± 0.208 | 0.070 ± 0.019 | 0.101 ± 0.056 | 0.248 ± 0.122 | 0.909 ± 0.059 | 0.021 ± 0.008 | 0.339 ± 0.150 | 0.155 ± 0.061 | 0.656 ± 0.185 | 0.047 ± 0.017 |
| temp_rff | 0.108 ± 0.025 | 0.492 ± 0.068 | 0.599 ± 0.175 | 0.126 ± 0.039 | 0.394 ± 0.220 | 0.070 ± 0.019 | 0.111 ± 0.055 | 0.242 ± 0.138 | 0.888 ± 0.086 | 0.021 ± 0.006 | 0.358 ± 0.139 | 0.149 ± 0.066 | 0.632 ± 0.164 | 0.047 ± 0.016 |
| temp_rff_nowt | 0.108 ± 0.024 | 0.491 ± 0.066 | 0.599 ± 0.178 | 0.126 ± 0.038 | 0.395 ± 0.220 | 0.071 ± 0.019 | 0.116 ± 0.074 | 0.245 ± 0.138 | 0.882 ± 0.095 | 0.022 ± 0.008 | 0.348 ± 0.143 | 0.153 ± 0.064 | 0.638 ± 0.167 | 0.047 ± 0.015 |
| probe_default | 0.106 ± 0.036 | 0.482 ± 0.043 | 0.995 ± 0.006 | 0.106 ± 0.036 | 0.001 ± 0.002 | 0.106 ± 0.036 | 0.590 ± 0.108 | 0.122 ± 0.041 | 0.384 ± 0.109 | 0.071 ± 0.026 | 0.302 ± 0.169 | 0.172 ± 0.111 | 0.680 ± 0.176 | 0.041 ± 0.016 |
| knn_default | 0.108 ± 0.023 | 0.504 ± 0.063 | 0.590 ± 0.164 | 0.127 ± 0.041 | 0.417 ± 0.205 | 0.070 ± 0.019 | 0.098 ± 0.056 | 0.255 ± 0.129 | 0.911 ± 0.063 | 0.021 ± 0.007 | 0.342 ± 0.147 | 0.155 ± 0.062 | 0.651 ± 0.183 | 0.047 ± 0.016 |
| openmax_default | 0.084 ± 0.020 | 0.512 ± 0.063 | 0.726 ± 0.150 | 0.085 ± 0.027 | 0.260 ± 0.151 | 0.060 ± 0.019 | 0.242 ± 0.113 | 0.104 ± 0.052 | 0.786 ± 0.123 | 0.023 ± 0.010 | 0.167 ± 0.247 | 0.257 ± 0.314 | 0.847 ± 0.233 | 0.015 ± 0.015 |
| temp_rff_default | 0.108 ± 0.025 | 0.504 ± 0.062 | 0.584 ± 0.170 | 0.125 ± 0.039 | 0.425 ± 0.209 | 0.069 ± 0.019 | 0.104 ± 0.056 | 0.251 ± 0.132 | 0.905 ± 0.059 | 0.021 ± 0.008 | 0.338 ± 0.150 | 0.154 ± 0.062 | 0.659 ± 0.178 | 0.046 ± 0.017 |
| raw_centroid | 0.160 ± 0.023 | 0.507 ± 0.084 | 0.882 ± 0.038 | 0.153 ± 0.018 | 0.144 ± 0.103 | 0.135 ± 0.017 | 0.494 ± 0.027 | 0.106 ± 0.010 | 0.506 ± 0.108 | 0.053 ± 0.006 | 0.594 ± 0.126 | 0.119 ± 0.019 | 0.398 ± 0.204 | 0.072 ± 0.025 |
| raw_centroid_cfo | 0.191 ± 0.018 | 0.506 ± 0.082 | 0.883 ± 0.010 | 0.178 ± 0.015 | 0.138 ± 0.115 | 0.157 ± 0.012 | 0.506 ± 0.010 | 0.109 ± 0.010 | 0.492 ± 0.104 | 0.055 ± 0.005 | 0.566 ± 0.122 | 0.118 ± 0.015 | 0.428 ± 0.157 | 0.068 ± 0.024 |

Tuned calibration: method chosen on Part B, count over runs: msp: q80×4/youden×11, probe: youden×15, openmax: q80×3/youden×12, static: q80×5/youden×10, knn: q80×7/youden×8, temp_rff_p1: q80×5/youden×10, static_window: q80×5/youden×10, knn_window: q80×7/youden×8, temp_rff: q80×5/youden×10, temp_rff_nowt: q80×6/youden×9, probe_default: youden×15, knn_default: q80×7/youden×8, openmax_default: q80×2/youden×13, temp_rff_default: q80×6/youden×9, raw_centroid: q80×4/youden×11, raw_centroid_cfo: q80×2/youden×13

## Per device split (mean over seeds)

| Rule | shift 0 closed / AUROC | shift 4 closed / AUROC | shift 8 closed / AUROC | shift 12 closed / AUROC | shift 16 closed / AUROC |
|---|---|---|---|---|---|
| softmax | 0.068 / – | 0.064 / – | 0.075 / – | 0.051 / – | 0.087 / – |
| msp | 0.068 / 0.476 | 0.064 / 0.506 | 0.075 / 0.578 | 0.051 / 0.490 | 0.087 / 0.470 |
| probe | 0.117 / 0.465 | 0.069 / 0.464 | 0.116 / 0.522 | 0.088 / 0.461 | 0.117 / 0.479 |
| openmax | 0.097 / 0.502 | 0.065 / 0.538 | 0.076 / 0.495 | 0.075 / 0.481 | 0.103 / 0.478 |
| static | 0.116 / 0.543 | 0.070 / 0.419 | 0.109 / 0.528 | 0.082 / 0.458 | 0.130 / 0.448 |
| static_window | 0.116 / 0.543 | 0.070 / 0.419 | 0.109 / 0.528 | 0.086 / 0.441 | 0.130 / 0.448 |
| knn | 0.121 / 0.546 | 0.075 / 0.541 | 0.127 / 0.537 | 0.091 / 0.452 | 0.122 / 0.445 |
| knn_window | 0.121 / 0.546 | 0.075 / 0.541 | 0.127 / 0.537 | 0.091 / 0.452 | 0.122 / 0.445 |
| temp_rff_p1 | 0.116 / 0.543 | 0.070 / 0.419 | 0.109 / 0.528 | 0.082 / 0.458 | 0.130 / 0.448 |
| temp_rff | 0.121 / 0.543 | 0.081 / 0.539 | 0.133 / 0.521 | 0.087 / 0.418 | 0.121 / 0.440 |
| raw_centroid_cfo | 0.158 / 0.554 | 0.190 / 0.640 | 0.200 / 0.434 | 0.205 / 0.453 | 0.204 / 0.449 |

## PRIMARY test: TEMP-RFF minus rule at the split level

Seeds are averaged within each device split first (the seeds of a split share its devices, so they are not independent); tests are over the splits. Outcomes: closed accuracy, AUROC, tuned-protocol known ok and unknown detection. Holm correction over one family: 84 tests: 7 rules ['msp', 'probe', 'openmax', 'static', 'knn', 'static_window', 'knn_window'] x 4 outcomes x tags ['wisig', 'pycom', 'lora'].

| Rule | outcome | mean diff | sd diff | splits won | t p | t p (Holm) | Wilcoxon p | Wilcoxon p (Holm) |
|---|---|---|---|---|---|---|---|---|
| knn_window | auroc | -0.012 | 0.014 | 0/5 | 0.124 | 1 | 0.0625 | 1 |
| knn_window | closed | +0.001 | 0.005 | 3/5 | 0.597 | 1 | 0.625 | 1 |
| knn_window | tuned_known_ok | -0.000 | 0.006 | 1/5 | 0.954 | 1 | 0.875 | 1 |
| knn_window | tuned_unk_det | -0.024 | 0.042 | 1/5 | 0.264 | 1 | 0.375 | 1 |
| knn | auroc | -0.012 | 0.014 | 0/5 | 0.124 | 1 | 0.0625 | 1 |
| knn | closed | +0.001 | 0.005 | 3/5 | 0.597 | 1 | 0.625 | 1 |
| knn | tuned_known_ok | -0.000 | 0.006 | 1/5 | 0.954 | 1 | 0.875 | 1 |
| knn | tuned_unk_det | -0.024 | 0.042 | 1/5 | 0.264 | 1 | 0.375 | 1 |
| msp | auroc | -0.012 | 0.060 | 2/5 | 0.682 | 1 | 0.812 | 1 |
| msp | closed | +0.039 | 0.016 | 5/5 | 0.00532 | 0.447 | 0.0625 | 1 |
| msp | tuned_known_ok | +0.008 | 0.015 | 3/5 | 0.306 | 1 | 0.312 | 1 |
| msp | tuned_unk_det | +0.155 | 0.201 | 3/5 | 0.159 | 1 | 0.312 | 1 |
| openmax | auroc | -0.007 | 0.044 | 3/5 | 0.755 | 1 | 1 | 1 |
| openmax | closed | +0.025 | 0.018 | 5/5 | 0.0359 | 1 | 0.0625 | 1 |
| openmax | tuned_known_ok | +0.024 | 0.020 | 4/5 | 0.0541 | 1 | 0.125 | 1 |
| openmax | tuned_unk_det | -0.106 | 0.295 | 2/5 | 0.468 | 1 | 0.312 | 1 |
| probe | auroc | +0.014 | 0.059 | 2/5 | 0.625 | 1 | 0.812 | 1 |
| probe | closed | +0.007 | 0.007 | 4/5 | 0.112 | 1 | 0.125 | 1 |
| probe | tuned_known_ok | +0.007 | 0.020 | 3/5 | 0.517 | 1 | 0.812 | 1 |
| probe | tuned_unk_det | -0.044 | 0.210 | 2/5 | 0.665 | 1 | 0.625 | 1 |
| static_window | auroc | +0.017 | 0.058 | 2/5 | 0.559 | 1 | 0.812 | 1 |
| static_window | closed | +0.006 | 0.012 | 4/5 | 0.333 | 1 | 0.312 | 1 |
| static_window | tuned_known_ok | -0.004 | 0.015 | 2/5 | 0.552 | 1 | 0.625 | 1 |
| static_window | tuned_unk_det | +0.006 | 0.082 | 3/5 | 0.886 | 1 | 1 | 1 |
| static | auroc | +0.013 | 0.062 | 2/5 | 0.661 | 1 | 0.812 | 1 |
| static | closed | +0.007 | 0.012 | 4/5 | 0.27 | 1 | 0.312 | 1 |
| static | tuned_known_ok | -0.004 | 0.015 | 2/5 | 0.556 | 1 | 0.625 | 1 |
| static | tuned_unk_det | +0.003 | 0.087 | 3/5 | 0.934 | 1 | 1 | 1 |

## Secondary (exploratory): TEMP-RFF minus other rules at the split level, uncorrected

| Rule | outcome | mean diff | sd diff | splits won | t p | Wilcoxon p |
|---|---|---|---|---|---|---|
| knn_default | auroc | -0.011 | 0.015 | 0/5 | 0.169 | 0.0625 |
| knn_default | closed | -0.000 | 0.004 | 1/5 | 0.943 | 0.625 |
| knn_default | tuned_known_ok | -0.000 | 0.005 | 2/5 | 0.981 | 0.812 |
| knn_default | tuned_unk_det | -0.020 | 0.042 | 2/5 | 0.355 | 0.438 |
| openmax_default | auroc | -0.020 | 0.034 | 2/5 | 0.261 | 0.312 |
| openmax_default | closed | +0.024 | 0.014 | 5/5 | 0.0176 | 0.0625 |
| openmax_default | tuned_known_ok | +0.032 | 0.019 | 5/5 | 0.0192 | 0.0625 |
| openmax_default | tuned_unk_det | -0.215 | 0.286 | 2/5 | 0.167 | 0.312 |
| probe_default | auroc | +0.011 | 0.056 | 3/5 | 0.695 | 0.625 |
| probe_default | closed | +0.002 | 0.008 | 3/5 | 0.525 | 0.625 |
| probe_default | tuned_known_ok | +0.006 | 0.013 | 3/5 | 0.396 | 0.625 |
| probe_default | tuned_unk_det | -0.048 | 0.211 | 2/5 | 0.635 | 0.625 |
| raw_centroid_cfo | auroc | -0.014 | 0.068 | 1/5 | 0.668 | 0.438 |
| raw_centroid_cfo | closed | -0.083 | 0.033 | 0/5 | 0.00465 | 0.0625 |
| raw_centroid_cfo | tuned_known_ok | -0.021 | 0.017 | 0/5 | 0.0495 | 0.0625 |
| raw_centroid_cfo | tuned_unk_det | +0.204 | 0.229 | 4/5 | 0.118 | 0.125 |
| raw_centroid | auroc | -0.015 | 0.070 | 1/5 | 0.655 | 0.438 |
| raw_centroid | closed | -0.052 | 0.017 | 0/5 | 0.00244 | 0.0625 |
| raw_centroid | tuned_known_ok | -0.025 | 0.019 | 0/5 | 0.045 | 0.0625 |
| raw_centroid | tuned_unk_det | +0.234 | 0.255 | 4/5 | 0.11 | 0.125 |
| softmax | closed | +0.039 | 0.016 | 5/5 | 0.00532 | 0.0625 |
| temp_rff_default | auroc | -0.012 | 0.017 | 0/5 | 0.19 | 0.25 |
| temp_rff_default | closed | +0.001 | 0.005 | 2/5 | 0.8 | 0.75 |
| temp_rff_default | tuned_known_ok | +0.001 | 0.005 | 1/5 | 0.791 | 1 |
| temp_rff_default | tuned_unk_det | -0.028 | 0.042 | 0/5 | 0.215 | 0.5 |
| temp_rff_nowt | auroc | +0.001 | 0.006 | 2/5 | 0.631 | 0.75 |
| temp_rff_nowt | closed | +0.000 | 0.003 | 2/5 | 0.801 | 0.75 |
| temp_rff_nowt | tuned_known_ok | +0.000 | 0.001 | 1/5 | 0.937 | 1 |
| temp_rff_nowt | tuned_unk_det | -0.007 | 0.016 | 1/5 | 0.398 | 0.5 |
| temp_rff_p1 | auroc | +0.013 | 0.062 | 2/5 | 0.661 | 0.812 |
| temp_rff_p1 | closed | +0.007 | 0.012 | 4/5 | 0.27 | 0.312 |
| temp_rff_p1 | tuned_known_ok | -0.004 | 0.015 | 2/5 | 0.556 | 0.625 |
| temp_rff_p1 | tuned_unk_det | +0.003 | 0.087 | 3/5 | 0.934 | 1 |

With 5 splits the smallest possible two-sided Wilcoxon p is 0.0625; report effect sizes and wins, not only p-values.


## Secondary test: TEMP-RFF minus rule per run (same split and seed; not independent)

| Rule | outcome | mean diff | wins | paired t p | Wilcoxon p |
|---|---|---|---|---|---|
| softmax | closed | +0.039 | 15/15 | 9.89e-07 | 6.1e-05 |
| msp | closed | +0.039 | 15/15 | 9.89e-07 | 6.1e-05 |
| msp | auroc | -0.012 | 5/15 | 0.605 | 0.277 |
| msp | tuned_known_ok | +0.008 | 9/15 | 0.223 | 0.277 |
| msp | tuned_unk_det | +0.155 | 9/15 | 0.0521 | 0.0946 |
| probe | closed | +0.007 | 9/15 | 0.194 | 0.252 |
| probe | auroc | +0.014 | 8/15 | 0.523 | 0.524 |
| probe | tuned_known_ok | +0.007 | 8/15 | 0.273 | 0.421 |
| probe | tuned_unk_det | -0.044 | 6/15 | 0.454 | 0.524 |
| openmax | closed | +0.025 | 13/15 | 0.00277 | 0.00427 |
| openmax | auroc | -0.007 | 7/15 | 0.766 | 0.762 |
| openmax | tuned_known_ok | +0.024 | 14/15 | 0.00087 | 0.00061 |
| openmax | tuned_unk_det | -0.106 | 5/15 | 0.208 | 0.303 |
| static | closed | +0.007 | 10/15 | 0.117 | 0.0962 |
| static | auroc | +0.013 | 9/15 | 0.458 | 0.762 |
| static | tuned_known_ok | -0.004 | 7/15 | 0.429 | 0.638 |
| static | tuned_unk_det | +0.003 | 7/15 | 0.935 | 1 |
| knn | closed | +0.001 | 4/15 | 0.628 | 0.767 |
| knn | auroc | -0.012 | 5/15 | 0.0993 | 0.107 |
| knn | tuned_known_ok | -0.000 | 2/15 | 0.935 | 0.484 |
| knn | tuned_unk_det | -0.024 | 3/15 | 0.158 | 0.161 |
| temp_rff_p1 | closed | +0.007 | 10/15 | 0.117 | 0.0962 |
| temp_rff_p1 | auroc | +0.013 | 8/15 | 0.458 | 0.778 |
| temp_rff_p1 | tuned_known_ok | -0.004 | 7/15 | 0.429 | 0.638 |
| temp_rff_p1 | tuned_unk_det | +0.003 | 7/15 | 0.935 | 1 |
| static_window | closed | +0.006 | 10/15 | 0.16 | 0.124 |
| static_window | auroc | +0.017 | 9/15 | 0.33 | 0.561 |
| static_window | tuned_known_ok | -0.004 | 7/15 | 0.424 | 0.638 |
| static_window | tuned_unk_det | +0.006 | 7/15 | 0.891 | 0.802 |
| knn_window | closed | +0.001 | 4/15 | 0.628 | 0.767 |
| knn_window | auroc | -0.012 | 5/15 | 0.0993 | 0.107 |
| knn_window | tuned_known_ok | -0.000 | 2/15 | 0.935 | 0.484 |
| knn_window | tuned_unk_det | -0.024 | 3/15 | 0.158 | 0.161 |
| temp_rff_nowt | closed | +0.000 | 2/15 | 0.757 | 1 |
| temp_rff_nowt | auroc | +0.001 | 2/15 | 0.609 | 0.715 |
| temp_rff_nowt | tuned_known_ok | +0.000 | 2/15 | 0.954 | 1 |
| temp_rff_nowt | tuned_unk_det | -0.007 | 2/15 | 0.374 | 0.465 |
| probe_default | closed | +0.002 | 8/15 | 0.706 | 0.679 |
| probe_default | auroc | +0.011 | 8/15 | 0.63 | 0.524 |
| probe_default | tuned_known_ok | +0.006 | 11/15 | 0.228 | 0.151 |
| probe_default | tuned_unk_det | -0.048 | 7/15 | 0.416 | 0.639 |
| knn_default | closed | -0.000 | 7/15 | 0.964 | 0.804 |
| knn_default | auroc | -0.011 | 6/15 | 0.156 | 0.151 |
| knn_default | tuned_known_ok | -0.000 | 8/15 | 0.976 | 0.798 |
| knn_default | tuned_unk_det | -0.020 | 7/15 | 0.291 | 0.67 |
| openmax_default | closed | +0.024 | 14/15 | 0.00362 | 0.00427 |
| openmax_default | auroc | -0.020 | 7/15 | 0.282 | 0.389 |
| openmax_default | tuned_known_ok | +0.032 | 13/15 | 9.96e-05 | 0.00061 |
| openmax_default | tuned_unk_det | -0.215 | 4/15 | 0.0274 | 0.0103 |
| temp_rff_default | closed | +0.001 | 3/15 | 0.821 | 0.893 |
| temp_rff_default | auroc | -0.012 | 0/15 | 0.139 | 0.0431 |
| temp_rff_default | tuned_known_ok | +0.001 | 2/15 | 0.698 | 0.715 |
| temp_rff_default | tuned_unk_det | -0.028 | 0/15 | 0.107 | 0.0679 |
| raw_centroid | closed | -0.052 | 0/15 | 6.42e-07 | 6.1e-05 |
| raw_centroid | auroc | -0.015 | 8/15 | 0.469 | 0.599 |
| raw_centroid | tuned_known_ok | -0.025 | 2/15 | 0.00351 | 0.00427 |
| raw_centroid | tuned_unk_det | +0.234 | 12/15 | 0.00503 | 0.00537 |
| raw_centroid_cfo | closed | -0.083 | 0/15 | 1.48e-07 | 6.1e-05 |
| raw_centroid_cfo | auroc | -0.014 | 5/15 | 0.482 | 0.389 |
| raw_centroid_cfo | tuned_known_ok | -0.021 | 2/15 | 0.000804 | 0.00153 |
| raw_centroid_cfo | tuned_unk_det | +0.204 | 13/15 | 0.00315 | 0.00336 |
## Forced prototype window (decay 1)

| W | Closed acc. | AUROC | tuned accept | tuned unk. det. |
|---|---|---|---|---|
| 1 | 0.108 ± 0.025 | 0.493 ± 0.069 | 0.358 ± 0.139 | 0.632 ± 0.164 |
| 2 | 0.102 ± 0.019 | 0.523 ± 0.091 | 0.456 ± 0.099 | 0.563 ± 0.188 |
| 3 | 0.106 ± 0.017 | 0.519 ± 0.070 | 0.445 ± 0.122 | 0.584 ± 0.142 |
| 4 | 0.109 ± 0.025 | 0.518 ± 0.065 | 0.454 ± 0.126 | 0.575 ± 0.153 |

## Few-shot target-session enrollment (same support for every rule)

| k requested | k effective | temp_rff closed / AUROC / tuned known ok | static closed / AUROC / tuned known ok | knn closed / AUROC / tuned known ok | static_window closed / AUROC / tuned known ok | knn_window closed / AUROC / tuned known ok |
|---|---|---|---|---|---|---|
| 1 | 1 | 0.175 / 0.529 / 0.096 | 0.107 / 0.481 / 0.057 | 0.141 / 0.521 / 0.064 | 0.108 / 0.477 / 0.057 | 0.141 / 0.521 / 0.064 |
| 5 | 5 | 0.239 / 0.551 / 0.145 | 0.108 / 0.481 / 0.057 | 0.196 / 0.543 / 0.094 | 0.109 / 0.477 / 0.057 | 0.196 / 0.543 / 0.094 |
| 10 | 10 | 0.263 / 0.559 / 0.165 | 0.109 / 0.481 / 0.057 | 0.232 / 0.553 / 0.118 | 0.110 / 0.477 / 0.057 | 0.232 / 0.553 / 0.118 |
| 20 | 20 | 0.287 / 0.564 / 0.188 | 0.111 / 0.481 / 0.057 | 0.267 / 0.563 / 0.148 | 0.112 / 0.477 / 0.057 | 0.267 / 0.563 / 0.148 |
| 50 | 50 | 0.302 / 0.572 / 0.205 | 0.116 / 0.480 / 0.060 | 0.294 / 0.575 / 0.182 | 0.117 / 0.476 / 0.061 | 0.294 / 0.575 / 0.182 |
| 100 | 100 | 0.312 / 0.575 / 0.217 | 0.128 / 0.479 / 0.069 | 0.310 / 0.581 / 0.207 | 0.128 / 0.476 / 0.070 | 0.310 / 0.581 / 0.207 |
| 200 | 200 | 0.318 / 0.577 / 0.228 | 0.159 / 0.483 / 0.087 | 0.321 / 0.584 / 0.219 | 0.158 / 0.480 / 0.090 | 0.321 / 0.584 / 0.219 |
| 500 | 316 | 0.322 / 0.580 / 0.233 | 0.195 / 0.492 / 0.113 | 0.328 / 0.585 / 0.231 | 0.194 / 0.489 / 0.112 | 0.328 / 0.585 / 0.231 |

## Pipeline stages (Part-A centroids)

| Stage | Softmax acc. | Centroid acc. | AUROC |
|---|---|---|---|
| phase1 | 0.130 ± 0.036 | 0.116 ± 0.034 | 0.499 ± 0.046 |
| phase2 | 0.084 ± 0.025 | 0.118 ± 0.024 | 0.491 ± 0.071 |
| phase3 | 0.069 ± 0.020 | 0.101 ± 0.027 | 0.479 ± 0.074 |

## Selected settings over runs

- W: 1×14, 2×1
- decay: 0.6×1, 1.0×14
- P: 1×1, 5×2, 50×1, all×11
- knn (k, weighted): (1, False)×8, (10, False)×1, (10, True)×1, (20, True)×2, (5, False)×2, (5, True)×1
- openmax (dist, tail, alpha): ('euclidean', 10, 1)×2, ('euclidean', 10, 2)×2, ('euclidean', 20, 10)×1, ('euclidean', 50, 5)×2, ('eucos', 10, 1)×4, ('eucos', 10, 10)×1, ('eucos', 10, 2)×2, ('eucos', 10, 5)×1
- probe C: 0.01×3, 0.1×1, 1.0×1, 10.0×10
- static_window W: (1,)×14, (2,)×1
- knn_window W (k, weighted): (1, 1, False)×8, (1, 10, False)×1, (1, 10, True)×1, (1, 20, True)×2, (1, 5, False)×2, (1, 5, True)×1
- temp_rff_p1 W decay: (1, 1.0)×15
- selection-weight sensitivity: choice changes with the weight (0.3/0.5/0.7) in 8 of 15 runs
