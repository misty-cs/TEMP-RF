# Summary: lora — COMPLETE

Audited runs used: 25 (splits [0, 4, 8, 12, 16], seeds [7, 13, 21, 42, 100]); expected splits [0, 4, 8, 12, 16] x seeds [42, 7, 13, 21, 100]

## All rules (mean ± std over runs)

| Rule | Closed acc. | AUROC | fixed90 accept | fixed90 open | fixed90 unk. det. | fixed90 known ok | objective accept | objective open | objective unk. det. | objective known ok | tuned accept | tuned open | tuned unk. det. | tuned known ok |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| softmax | 0.624 ± 0.047 | – | – | – | – | – | – | – | – | – | – | – | – | – |
| msp | 0.624 ± 0.047 | 0.340 ± 0.222 | 0.898 ± 0.017 | 0.658 ± 0.048 | 0.102 ± 0.112 | 0.591 ± 0.046 | 0.463 ± 0.037 | 0.857 ± 0.084 | 0.312 ± 0.252 | 0.395 ± 0.027 | 0.345 ± 0.346 | 0.900 ± 0.160 | 0.469 ± 0.467 | 0.260 ± 0.209 |
| probe | 0.614 ± 0.051 | 0.383 ± 0.220 | 0.896 ± 0.012 | 0.646 ± 0.050 | 0.109 ± 0.102 | 0.579 ± 0.048 | 0.491 ± 0.042 | 0.800 ± 0.071 | 0.368 ± 0.272 | 0.392 ± 0.035 | 0.723 ± 0.320 | 0.729 ± 0.146 | 0.173 ± 0.255 | 0.485 ± 0.193 |
| openmax | 0.612 ± 0.041 | 0.553 ± 0.272 | 0.905 ± 0.015 | 0.633 ± 0.051 | 0.229 ± 0.141 | 0.573 ± 0.045 | 0.620 ± 0.123 | 0.732 ± 0.085 | 0.466 ± 0.264 | 0.450 ± 0.083 | 0.756 ± 0.078 | 0.668 ± 0.061 | 0.382 ± 0.215 | 0.505 ± 0.072 |
| static | 0.628 ± 0.045 | 0.484 ± 0.225 | 0.890 ± 0.025 | 0.640 ± 0.051 | 0.147 ± 0.143 | 0.569 ± 0.039 | 0.436 ± 0.056 | 0.698 ± 0.042 | 0.498 ± 0.327 | 0.303 ± 0.031 | 0.746 ± 0.095 | 0.658 ± 0.044 | 0.251 ± 0.200 | 0.490 ± 0.070 |
| knn | 0.627 ± 0.046 | 0.555 ± 0.209 | 0.899 ± 0.019 | 0.629 ± 0.051 | 0.196 ± 0.132 | 0.565 ± 0.042 | 0.958 ± 0.034 | 0.629 ± 0.047 | 0.133 ± 0.098 | 0.602 ± 0.044 | 0.723 ± 0.062 | 0.638 ± 0.049 | 0.355 ± 0.209 | 0.462 ± 0.054 |
| temp_rff_p1 | 0.630 ± 0.048 | 0.488 ± 0.219 | 0.894 ± 0.026 | 0.642 ± 0.052 | 0.144 ± 0.142 | 0.573 ± 0.044 | 0.444 ± 0.060 | 0.699 ± 0.043 | 0.498 ± 0.327 | 0.309 ± 0.037 | 0.759 ± 0.104 | 0.659 ± 0.044 | 0.238 ± 0.199 | 0.501 ± 0.081 |
| static_window | 0.629 ± 0.045 | 0.489 ± 0.218 | 0.894 ± 0.028 | 0.641 ± 0.052 | 0.147 ± 0.143 | 0.572 ± 0.039 | 0.448 ± 0.052 | 0.699 ± 0.042 | 0.498 ± 0.327 | 0.312 ± 0.032 | 0.751 ± 0.098 | 0.659 ± 0.045 | 0.251 ± 0.200 | 0.495 ± 0.071 |
| knn_window | 0.628 ± 0.047 | 0.555 ± 0.209 | 0.900 ± 0.019 | 0.630 ± 0.051 | 0.195 ± 0.130 | 0.566 ± 0.043 | 0.959 ± 0.035 | 0.630 ± 0.048 | 0.131 ± 0.095 | 0.603 ± 0.046 | 0.724 ± 0.063 | 0.639 ± 0.050 | 0.354 ± 0.209 | 0.463 ± 0.055 |
| temp_rff | 0.630 ± 0.048 | 0.488 ± 0.219 | 0.894 ± 0.026 | 0.642 ± 0.052 | 0.144 ± 0.142 | 0.573 ± 0.044 | 0.444 ± 0.060 | 0.699 ± 0.043 | 0.498 ± 0.327 | 0.309 ± 0.037 | 0.759 ± 0.104 | 0.659 ± 0.044 | 0.238 ± 0.199 | 0.501 ± 0.081 |
| temp_rff_nowt | 0.630 ± 0.047 | 0.502 ± 0.215 | 0.894 ± 0.026 | 0.641 ± 0.051 | 0.143 ± 0.142 | 0.573 ± 0.043 | 0.458 ± 0.066 | 0.698 ± 0.039 | 0.503 ± 0.321 | 0.319 ± 0.046 | 0.753 ± 0.103 | 0.659 ± 0.044 | 0.243 ± 0.199 | 0.497 ± 0.078 |
| probe_default | 0.633 ± 0.044 | 0.327 ± 0.171 | 0.901 ± 0.010 | 0.666 ± 0.041 | 0.074 ± 0.066 | 0.599 ± 0.038 | 0.491 ± 0.036 | 0.829 ± 0.060 | 0.279 ± 0.229 | 0.407 ± 0.030 | 0.393 ± 0.419 | 0.874 ± 0.174 | 0.505 ± 0.439 | 0.275 ± 0.259 |
| knn_default | 0.614 ± 0.049 | 0.587 ± 0.189 | 0.900 ± 0.019 | 0.617 ± 0.052 | 0.228 ± 0.136 | 0.555 ± 0.046 | 0.979 ± 0.022 | 0.616 ± 0.049 | 0.124 ± 0.091 | 0.602 ± 0.046 | 0.764 ± 0.063 | 0.628 ± 0.049 | 0.348 ± 0.193 | 0.480 ± 0.067 |
| openmax_default | 0.616 ± 0.044 | 0.519 ± 0.270 | 0.903 ± 0.015 | 0.645 ± 0.045 | 0.159 ± 0.123 | 0.582 ± 0.040 | 0.465 ± 0.022 | 0.751 ± 0.068 | 0.585 ± 0.345 | 0.349 ± 0.028 | 0.555 ± 0.288 | 0.722 ± 0.112 | 0.460 ± 0.390 | 0.390 ± 0.182 |
| temp_rff_default | 0.599 ± 0.047 | 0.624 ± 0.148 | 0.901 ± 0.016 | 0.603 ± 0.049 | 0.253 ± 0.147 | 0.543 ± 0.045 | 0.978 ± 0.030 | 0.601 ± 0.047 | 0.131 ± 0.103 | 0.587 ± 0.048 | 0.789 ± 0.061 | 0.613 ± 0.046 | 0.363 ± 0.187 | 0.485 ± 0.062 |
| raw_centroid | 0.559 ± 0.045 | 0.712 ± 0.173 | 0.875 ± 0.012 | 0.603 ± 0.049 | 0.430 ± 0.315 | 0.528 ± 0.041 | 0.663 ± 0.027 | 0.607 ± 0.053 | 0.646 ± 0.241 | 0.401 ± 0.027 | 0.705 ± 0.108 | 0.605 ± 0.052 | 0.610 ± 0.285 | 0.427 ± 0.078 |
| raw_centroid_cfo | 0.561 ± 0.047 | 0.712 ± 0.172 | 0.873 ± 0.013 | 0.605 ± 0.052 | 0.430 ± 0.313 | 0.528 ± 0.041 | 0.659 ± 0.023 | 0.609 ± 0.053 | 0.645 ± 0.243 | 0.400 ± 0.033 | 0.735 ± 0.046 | 0.609 ± 0.053 | 0.589 ± 0.276 | 0.448 ± 0.049 |

Tuned calibration: method chosen on Part B, count over runs: msp: q80×1/q95×2/youden×22, probe: q80×2/q90×4/q95×4/youden×15, openmax: q80×9/youden×16, static: q80×11/q90×1/youden×13, knn: q80×6/youden×19, temp_rff_p1: q80×9/q90×2/youden×14, static_window: q80×10/q90×1/youden×14, knn_window: q80×6/youden×19, temp_rff: q80×9/q90×2/youden×14, temp_rff_nowt: q80×9/q90×2/youden×14, probe_default: q80×1/q90×1/q95×1/youden×22, knn_default: q80×14/q95×1/youden×10, openmax_default: q80×4/q90×2/youden×19, temp_rff_default: q80×19/q90×1/youden×5, raw_centroid: q80×10/youden×15, raw_centroid_cfo: q80×8/youden×17

## Per device split (mean over seeds)

| Rule | shift 0 closed / AUROC | shift 4 closed / AUROC | shift 8 closed / AUROC | shift 12 closed / AUROC | shift 16 closed / AUROC |
|---|---|---|---|---|---|
| softmax | 0.655 / – | 0.618 / – | 0.600 / – | 0.564 / – | 0.685 / – |
| msp | 0.655 / 0.503 | 0.618 / 0.012 | 0.600 / 0.267 | 0.564 / 0.275 | 0.685 / 0.645 |
| probe | 0.646 / 0.563 | 0.614 / 0.101 | 0.587 / 0.262 | 0.545 / 0.307 | 0.679 / 0.681 |
| openmax | 0.635 / 0.677 | 0.614 / 0.030 | 0.586 / 0.629 | 0.555 / 0.674 | 0.669 / 0.755 |
| static | 0.662 / 0.718 | 0.614 / 0.113 | 0.596 / 0.544 | 0.575 / 0.399 | 0.690 / 0.645 |
| static_window | 0.662 / 0.718 | 0.620 / 0.137 | 0.596 / 0.544 | 0.575 / 0.399 | 0.690 / 0.645 |
| knn | 0.657 / 0.746 | 0.601 / 0.212 | 0.596 / 0.636 | 0.584 / 0.553 | 0.698 / 0.628 |
| knn_window | 0.662 / 0.746 | 0.601 / 0.212 | 0.596 / 0.636 | 0.584 / 0.553 | 0.698 / 0.628 |
| temp_rff_p1 | 0.662 / 0.718 | 0.617 / 0.131 | 0.597 / 0.544 | 0.575 / 0.399 | 0.699 / 0.648 |
| temp_rff | 0.662 / 0.718 | 0.617 / 0.131 | 0.597 / 0.544 | 0.575 / 0.399 | 0.699 / 0.648 |
| raw_centroid_cfo | 0.596 / 0.591 | 0.528 / 0.979 | 0.517 / 0.779 | 0.532 / 0.728 | 0.630 / 0.485 |

## PRIMARY test: TEMP-RFF minus rule at the split level

Seeds are averaged within each device split first (the seeds of a split share its devices, so they are not independent); tests are over the splits. Outcomes: closed accuracy, AUROC, tuned-protocol known ok and unknown detection. Holm correction over one family: 84 tests: 7 rules ['msp', 'probe', 'openmax', 'static', 'knn', 'static_window', 'knn_window'] x 4 outcomes x tags ['wisig', 'pycom', 'lora'].

| Rule | outcome | mean diff | sd diff | splits won | t p | t p (Holm) | Wilcoxon p | Wilcoxon p (Holm) |
|---|---|---|---|---|---|---|---|---|
| knn_window | auroc | -0.067 | 0.066 | 1/5 | 0.085 | 1 | 0.125 | 1 |
| knn_window | closed | +0.002 | 0.009 | 3/5 | 0.717 | 1 | 0.625 | 1 |
| knn_window | tuned_known_ok | +0.038 | 0.031 | 5/5 | 0.0526 | 1 | 0.0625 | 1 |
| knn_window | tuned_unk_det | -0.116 | 0.105 | 0/5 | 0.0689 | 1 | 0.0625 | 1 |
| knn | auroc | -0.067 | 0.066 | 1/5 | 0.0845 | 1 | 0.125 | 1 |
| knn | closed | +0.003 | 0.009 | 4/5 | 0.522 | 1 | 0.438 | 1 |
| knn | tuned_known_ok | +0.039 | 0.031 | 5/5 | 0.0496 | 1 | 0.0625 | 1 |
| knn | tuned_unk_det | -0.117 | 0.105 | 0/5 | 0.0671 | 1 | 0.0625 | 1 |
| msp | auroc | +0.148 | 0.104 | 5/5 | 0.034 | 1 | 0.0625 | 1 |
| msp | closed | +0.006 | 0.008 | 3/5 | 0.165 | 1 | 0.312 | 1 |
| msp | tuned_known_ok | +0.241 | 0.233 | 4/5 | 0.0816 | 1 | 0.125 | 1 |
| msp | tuned_unk_det | -0.231 | 0.355 | 2/5 | 0.219 | 1 | 0.438 | 1 |
| openmax | auroc | -0.065 | 0.146 | 2/5 | 0.375 | 1 | 0.438 | 1 |
| openmax | closed | +0.018 | 0.011 | 5/5 | 0.0225 | 1 | 0.0625 | 1 |
| openmax | tuned_known_ok | -0.005 | 0.015 | 2/5 | 0.513 | 1 | 1 | 1 |
| openmax | tuned_unk_det | -0.144 | 0.210 | 2/5 | 0.199 | 1 | 0.312 | 1 |
| probe | auroc | +0.105 | 0.121 | 4/5 | 0.125 | 1 | 0.188 | 1 |
| probe | closed | +0.016 | 0.010 | 5/5 | 0.0262 | 1 | 0.0625 | 1 |
| probe | tuned_known_ok | +0.015 | 0.108 | 3/5 | 0.767 | 1 | 1 | 1 |
| probe | tuned_unk_det | +0.065 | 0.158 | 3/5 | 0.41 | 1 | 0.625 | 1 |
| static_window | auroc | -0.001 | 0.003 | 2/5 | 0.608 | 1 | 0.625 | 1 |
| static_window | closed | +0.001 | 0.005 | 3/5 | 0.61 | 1 | 0.812 | 1 |
| static_window | tuned_known_ok | +0.006 | 0.023 | 1/5 | 0.588 | 1 | 0.875 | 1 |
| static_window | tuned_unk_det | -0.013 | 0.029 | 0/5 | 0.374 | 1 | 1 | 1 |
| static | auroc | +0.004 | 0.008 | 3/5 | 0.315 | 1 | 0.625 | 1 |
| static | closed | +0.002 | 0.004 | 4/5 | 0.244 | 1 | 0.125 | 1 |
| static | tuned_known_ok | +0.010 | 0.020 | 3/5 | 0.311 | 1 | 0.312 | 1 |
| static | tuned_unk_det | -0.013 | 0.029 | 1/5 | 0.377 | 1 | 1 | 1 |

## Secondary (exploratory): TEMP-RFF minus other rules at the split level, uncorrected

| Rule | outcome | mean diff | sd diff | splits won | t p | Wilcoxon p |
|---|---|---|---|---|---|---|
| knn_default | auroc | -0.099 | 0.086 | 1/5 | 0.0612 | 0.125 |
| knn_default | closed | +0.016 | 0.011 | 5/5 | 0.0345 | 0.0625 |
| knn_default | tuned_known_ok | +0.020 | 0.031 | 4/5 | 0.221 | 0.312 |
| knn_default | tuned_unk_det | -0.110 | 0.132 | 1/5 | 0.137 | 0.188 |
| openmax_default | auroc | -0.031 | 0.090 | 2/5 | 0.487 | 0.438 |
| openmax_default | closed | +0.014 | 0.010 | 5/5 | 0.0308 | 0.0625 |
| openmax_default | tuned_known_ok | +0.110 | 0.092 | 5/5 | 0.0547 | 0.0625 |
| openmax_default | tuned_unk_det | -0.223 | 0.211 | 0/5 | 0.0777 | 0.0625 |
| probe_default | auroc | +0.161 | 0.105 | 5/5 | 0.0263 | 0.0625 |
| probe_default | closed | -0.003 | 0.005 | 1/5 | 0.22 | 0.312 |
| probe_default | tuned_known_ok | +0.226 | 0.147 | 5/5 | 0.0263 | 0.0625 |
| probe_default | tuned_unk_det | -0.268 | 0.329 | 1/5 | 0.143 | 0.188 |
| raw_centroid_cfo | auroc | -0.225 | 0.410 | 2/5 | 0.288 | 0.312 |
| raw_centroid_cfo | closed | +0.069 | 0.017 | 5/5 | 0.000785 | 0.0625 |
| raw_centroid_cfo | tuned_known_ok | +0.053 | 0.034 | 5/5 | 0.0245 | 0.0625 |
| raw_centroid_cfo | tuned_unk_det | -0.351 | 0.450 | 2/5 | 0.156 | 0.312 |
| raw_centroid | auroc | -0.224 | 0.410 | 2/5 | 0.29 | 0.312 |
| raw_centroid | closed | +0.071 | 0.015 | 5/5 | 0.000485 | 0.0625 |
| raw_centroid | tuned_known_ok | +0.073 | 0.028 | 5/5 | 0.0044 | 0.0625 |
| raw_centroid | tuned_unk_det | -0.372 | 0.441 | 2/5 | 0.133 | 0.312 |
| softmax | closed | +0.006 | 0.008 | 3/5 | 0.165 | 0.312 |
| temp_rff_default | auroc | -0.136 | 0.127 | 1/5 | 0.0754 | 0.125 |
| temp_rff_default | closed | +0.031 | 0.010 | 5/5 | 0.00237 | 0.0625 |
| temp_rff_default | tuned_known_ok | +0.016 | 0.023 | 4/5 | 0.2 | 0.188 |
| temp_rff_default | tuned_unk_det | -0.125 | 0.145 | 1/5 | 0.127 | 0.188 |
| temp_rff_nowt | auroc | -0.014 | 0.028 | 1/5 | 0.313 | 0.5 |
| temp_rff_nowt | closed | +0.000 | 0.002 | 2/5 | 0.978 | 1 |
| temp_rff_nowt | tuned_known_ok | +0.004 | 0.005 | 3/5 | 0.148 | 0.25 |
| temp_rff_nowt | tuned_unk_det | -0.005 | 0.007 | 0/5 | 0.185 | 0.25 |
| temp_rff_p1 | auroc | +0.000 | 0.000 | 0/5 | nan | 1 |
| temp_rff_p1 | closed | +0.000 | 0.000 | 0/5 | nan | 1 |
| temp_rff_p1 | tuned_known_ok | +0.000 | 0.000 | 0/5 | nan | 1 |
| temp_rff_p1 | tuned_unk_det | +0.000 | 0.000 | 0/5 | nan | 1 |

With 5 splits the smallest possible two-sided Wilcoxon p is 0.0625; report effect sizes and wins, not only p-values.


## Secondary test: TEMP-RFF minus rule per run (same split and seed; not independent)

| Rule | outcome | mean diff | wins | paired t p | Wilcoxon p |
|---|---|---|---|---|---|
| softmax | closed | +0.006 | 15/25 | 0.165 | 0.253 |
| msp | closed | +0.006 | 15/25 | 0.165 | 0.253 |
| msp | auroc | +0.148 | 24/25 | 5.02e-07 | 5.96e-07 |
| msp | tuned_known_ok | +0.241 | 19/25 | 0.000152 | 0.000376 |
| msp | tuned_unk_det | -0.231 | 10/25 | 0.00632 | 0.0308 |
| probe | closed | +0.016 | 21/25 | 4.71e-05 | 5.39e-05 |
| probe | auroc | +0.105 | 18/25 | 0.000913 | 0.00145 |
| probe | tuned_known_ok | +0.015 | 6/25 | 0.706 | 0.304 |
| probe | tuned_unk_det | +0.065 | 18/25 | 0.298 | 0.107 |
| openmax | closed | +0.018 | 21/25 | 5.54e-06 | 1.23e-05 |
| openmax | auroc | -0.065 | 11/25 | 0.0343 | 0.0588 |
| openmax | tuned_known_ok | -0.005 | 12/25 | 0.682 | 0.56 |
| openmax | tuned_unk_det | -0.144 | 7/25 | 0.00271 | 0.00739 |
| static | closed | +0.002 | 6/25 | 0.117 | 0.201 |
| static | auroc | +0.004 | 16/25 | 0.271 | 0.071 |
| static | tuned_known_ok | +0.010 | 4/25 | 0.124 | 0.326 |
| static | tuned_unk_det | -0.013 | 1/25 | 0.183 | 0.285 |
| knn | closed | +0.003 | 16/25 | 0.383 | 0.353 |
| knn | auroc | -0.067 | 7/25 | 0.00219 | 0.00163 |
| knn | tuned_known_ok | +0.039 | 18/25 | 0.00968 | 0.0136 |
| knn | tuned_unk_det | -0.117 | 5/25 | 0.000779 | 0.00274 |
| temp_rff_p1 | closed | +0.000 | 0/25 | nan | nan |
| temp_rff_p1 | auroc | +0.000 | 0/25 | nan | nan |
| temp_rff_p1 | tuned_known_ok | +0.000 | 0/25 | nan | nan |
| temp_rff_p1 | tuned_unk_det | +0.000 | 0/25 | nan | nan |
| static_window | closed | +0.001 | 5/25 | 0.406 | 1 |
| static_window | auroc | -0.001 | 11/25 | 0.783 | 0.786 |
| static_window | tuned_known_ok | +0.006 | 3/25 | 0.385 | 0.61 |
| static_window | tuned_unk_det | -0.013 | 0/25 | 0.181 | 0.18 |
| knn_window | closed | +0.002 | 15/25 | 0.635 | 0.647 |
| knn_window | auroc | -0.067 | 7/25 | 0.00222 | 0.00203 |
| knn_window | tuned_known_ok | +0.038 | 18/25 | 0.0137 | 0.0173 |
| knn_window | tuned_unk_det | -0.116 | 5/25 | 0.000915 | 0.00302 |
| temp_rff_nowt | closed | +0.000 | 2/25 | 0.976 | 1 |
| temp_rff_nowt | auroc | -0.014 | 1/25 | 0.264 | 0.144 |
| temp_rff_nowt | tuned_known_ok | +0.004 | 3/25 | 0.16 | 0.144 |
| temp_rff_nowt | tuned_unk_det | -0.005 | 0/25 | 0.156 | 0.0679 |
| probe_default | closed | -0.003 | 10/25 | 0.0932 | 0.101 |
| probe_default | auroc | +0.161 | 23/25 | 4.78e-07 | 1.13e-06 |
| probe_default | tuned_known_ok | +0.226 | 16/25 | 0.000377 | 0.00103 |
| probe_default | tuned_unk_det | -0.268 | 11/25 | 0.0106 | 0.0324 |
| knn_default | closed | +0.016 | 20/25 | 8.93e-05 | 0.00014 |
| knn_default | auroc | -0.099 | 4/25 | 0.000316 | 6.37e-05 |
| knn_default | tuned_known_ok | +0.020 | 15/25 | 0.128 | 0.106 |
| knn_default | tuned_unk_det | -0.110 | 9/25 | 0.00596 | 0.0208 |
| openmax_default | closed | +0.014 | 20/25 | 0.000103 | 0.000281 |
| openmax_default | auroc | -0.031 | 10/25 | 0.178 | 0.2 |
| openmax_default | tuned_known_ok | +0.110 | 17/25 | 0.00479 | 0.00964 |
| openmax_default | tuned_unk_det | -0.223 | 3/25 | 0.00301 | 0.00193 |
| temp_rff_default | closed | +0.031 | 24/25 | 2.16e-10 | 1.79e-07 |
| temp_rff_default | auroc | -0.136 | 4/25 | 7.36e-05 | 1.23e-05 |
| temp_rff_default | tuned_known_ok | +0.016 | 16/25 | 0.28 | 0.22 |
| temp_rff_default | tuned_unk_det | -0.125 | 8/25 | 0.00496 | 0.0177 |
| raw_centroid | closed | +0.071 | 25/25 | 1.64e-15 | 5.96e-08 |
| raw_centroid | auroc | -0.224 | 10/25 | 0.00688 | 0.0115 |
| raw_centroid | tuned_known_ok | +0.073 | 22/25 | 0.000238 | 0.00025 |
| raw_centroid | tuned_unk_det | -0.372 | 7/25 | 0.000213 | 0.00142 |
| raw_centroid_cfo | closed | +0.069 | 25/25 | 2.33e-14 | 5.96e-08 |
| raw_centroid_cfo | auroc | -0.225 | 10/25 | 0.00668 | 0.00964 |
| raw_centroid_cfo | tuned_known_ok | +0.053 | 22/25 | 0.000716 | 0.00145 |
| raw_centroid_cfo | tuned_unk_det | -0.351 | 8/25 | 0.000376 | 0.00306 |
## Forced prototype window (decay 1)

| W | Closed acc. | AUROC | tuned accept | tuned unk. det. |
|---|---|---|---|---|
| 1 | 0.628 ± 0.045 | 0.484 ± 0.225 | 0.746 ± 0.095 | 0.251 ± 0.200 |
| 2 | 0.629 ± 0.049 | 0.500 ± 0.204 | 0.805 ± 0.071 | 0.199 ± 0.163 |
| 3 | 0.624 ± 0.048 | 0.523 ± 0.193 | 0.826 ± 0.052 | 0.198 ± 0.145 |
| 4 | 0.626 ± 0.047 | 0.514 ± 0.212 | 0.832 ± 0.049 | 0.218 ± 0.159 |

## Few-shot target-session enrollment (same support for every rule)

| k requested | k effective | temp_rff closed / AUROC / tuned known ok | static closed / AUROC / tuned known ok | knn closed / AUROC / tuned known ok | static_window closed / AUROC / tuned known ok | knn_window closed / AUROC / tuned known ok |
|---|---|---|---|---|---|---|
| 1 | 1 | 0.612 / 0.489 / 0.503 | 0.628 / 0.485 / 0.487 | 0.628 / 0.558 / 0.462 | 0.630 / 0.490 / 0.491 | 0.629 / 0.557 / 0.463 |
| 5 | 5 | 0.630 / 0.500 / 0.518 | 0.629 / 0.485 / 0.487 | 0.628 / 0.558 / 0.464 | 0.630 / 0.490 / 0.491 | 0.629 / 0.557 / 0.465 |
| 10 | 10 | 0.633 / 0.500 / 0.520 | 0.629 / 0.486 / 0.489 | 0.629 / 0.557 / 0.466 | 0.630 / 0.490 / 0.493 | 0.630 / 0.557 / 0.467 |
| 20 | 20 | 0.635 / 0.500 / 0.522 | 0.629 / 0.486 / 0.491 | 0.630 / 0.558 / 0.470 | 0.630 / 0.491 / 0.495 | 0.631 / 0.558 / 0.471 |
| 50 | 50 | 0.635 / 0.496 / 0.522 | 0.630 / 0.488 / 0.499 | 0.632 / 0.559 / 0.478 | 0.631 / 0.492 / 0.502 | 0.633 / 0.559 / 0.479 |
| 100 | 100 | 0.636 / 0.496 / 0.528 | 0.632 / 0.490 / 0.513 | 0.636 / 0.561 / 0.486 | 0.632 / 0.493 / 0.513 | 0.637 / 0.561 / 0.487 |
| 200 | 200 | 0.636 / 0.496 / 0.525 | 0.633 / 0.492 / 0.520 | 0.641 / 0.565 / 0.498 | 0.634 / 0.495 / 0.519 | 0.642 / 0.565 / 0.499 |
| 500 | 390 | 0.637 / 0.495 / 0.530 | 0.634 / 0.495 / 0.528 | 0.647 / 0.572 / 0.499 | 0.635 / 0.497 / 0.527 | 0.647 / 0.572 / 0.499 |

## Pipeline stages (Part-A centroids)

| Stage | Softmax acc. | Centroid acc. | AUROC |
|---|---|---|---|
| phase1 | 0.617 ± 0.042 | 0.624 ± 0.046 | 0.486 ± 0.177 |
| phase2 | 0.621 ± 0.046 | 0.627 ± 0.044 | 0.485 ± 0.227 |
| phase3 | 0.624 ± 0.047 | 0.628 ± 0.045 | 0.484 ± 0.225 |

Joint-training control vs sequential Phase 2 (paired, split 0; joint runs have no audit):

| metric | joint | Phase 2 | mean diff | wins |
|---|---|---|---|---|
| softmax | 0.630 ± 0.024 | 0.648 ± 0.018 | -0.018 | 2/5 |
| centroid | 0.656 ± 0.012 | 0.660 ± 0.008 | -0.005 | 2/5 |
| auroc | 0.689 ± 0.030 | 0.719 ± 0.015 | -0.030 | 0/5 |

## Selected settings over runs

- W: 1×20, 2×3, 3×1, 4×1
- decay: 0.6×3, 0.8×1, 1.0×21
- P: 1×25
- knn (k, weighted): (10, False)×3, (20, False)×21, (5, False)×1
- openmax (dist, tail, alpha): ('euclidean', 10, 1)×25
- probe C: 0.01×18, 0.1×2, 1.0×1, 10.0×4
- static_window W: (1,)×22, (2,)×3
- knn_window W (k, weighted): (1, 10, False)×3, (1, 20, False)×21, (4, 20, False)×1
- temp_rff_p1 W decay: (1, 1.0)×20, (2, 0.6)×3, (3, 0.8)×1, (4, 1.0)×1
- selection-weight sensitivity: choice changes with the weight (0.3/0.5/0.7) in 12 of 25 runs
