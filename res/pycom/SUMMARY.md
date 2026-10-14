# Summary: pycom — COMPLETE

Audited runs used: 25 (splits [0, 4, 8, 12, 16], seeds [7, 13, 21, 42, 100]); expected splits [0, 4, 8, 12, 16] x seeds [42, 7, 13, 21, 100]

## All rules (mean ± std over runs)

| Rule | Closed acc. | AUROC | fixed90 accept | fixed90 open | fixed90 unk. det. | fixed90 known ok | objective accept | objective open | objective unk. det. | objective known ok | tuned accept | tuned open | tuned unk. det. | tuned known ok |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| softmax | 0.070 ± 0.022 | – | – | – | – | – | – | – | – | – | – | – | – | – |
| msp | 0.070 ± 0.022 | 0.500 ± 0.050 | 0.945 ± 0.024 | 0.071 ± 0.023 | 0.050 ± 0.024 | 0.067 ± 0.021 | 0.551 ± 0.102 | 0.079 ± 0.031 | 0.449 ± 0.102 | 0.043 ± 0.016 | 0.575 ± 0.279 | 0.100 ± 0.070 | 0.427 ± 0.277 | 0.043 ± 0.019 |
| probe | 0.098 ± 0.026 | 0.478 ± 0.045 | 0.991 ± 0.016 | 0.099 ± 0.027 | 0.005 ± 0.009 | 0.098 ± 0.026 | 0.598 ± 0.118 | 0.109 ± 0.028 | 0.370 ± 0.131 | 0.065 ± 0.018 | 0.336 ± 0.201 | 0.142 ± 0.068 | 0.654 ± 0.218 | 0.040 ± 0.017 |
| openmax | 0.080 ± 0.023 | 0.500 ± 0.054 | 0.742 ± 0.141 | 0.079 ± 0.027 | 0.261 ± 0.132 | 0.059 ± 0.023 | 0.255 ± 0.119 | 0.096 ± 0.053 | 0.746 ± 0.121 | 0.023 ± 0.012 | 0.298 ± 0.257 | 0.110 ± 0.079 | 0.704 ± 0.242 | 0.026 ± 0.017 |
| static | 0.094 ± 0.025 | 0.493 ± 0.070 | 0.631 ± 0.161 | 0.114 ± 0.038 | 0.348 ± 0.186 | 0.069 ± 0.023 | 0.137 ± 0.070 | 0.189 ± 0.088 | 0.875 ± 0.107 | 0.024 ± 0.012 | 0.352 ± 0.168 | 0.138 ± 0.052 | 0.663 ± 0.190 | 0.047 ± 0.026 |
| knn | 0.105 ± 0.022 | 0.501 ± 0.056 | 0.598 ± 0.163 | 0.124 ± 0.038 | 0.400 ± 0.195 | 0.070 ± 0.021 | 0.098 ± 0.064 | 0.250 ± 0.129 | 0.912 ± 0.060 | 0.020 ± 0.008 | 0.340 ± 0.138 | 0.148 ± 0.061 | 0.656 ± 0.161 | 0.046 ± 0.017 |
| temp_rff_p1 | 0.094 ± 0.025 | 0.493 ± 0.070 | 0.631 ± 0.161 | 0.114 ± 0.038 | 0.348 ± 0.186 | 0.069 ± 0.023 | 0.137 ± 0.070 | 0.189 ± 0.088 | 0.875 ± 0.107 | 0.024 ± 0.012 | 0.352 ± 0.168 | 0.138 ± 0.052 | 0.663 ± 0.190 | 0.047 ± 0.026 |
| static_window | 0.096 ± 0.024 | 0.489 ± 0.072 | 0.641 ± 0.168 | 0.116 ± 0.035 | 0.334 ± 0.199 | 0.071 ± 0.022 | 0.144 ± 0.067 | 0.185 ± 0.088 | 0.865 ± 0.106 | 0.024 ± 0.012 | 0.347 ± 0.175 | 0.140 ± 0.049 | 0.668 ± 0.195 | 0.047 ± 0.026 |
| knn_window | 0.105 ± 0.022 | 0.501 ± 0.056 | 0.598 ± 0.163 | 0.124 ± 0.038 | 0.400 ± 0.195 | 0.070 ± 0.021 | 0.098 ± 0.064 | 0.250 ± 0.129 | 0.912 ± 0.060 | 0.020 ± 0.008 | 0.340 ± 0.138 | 0.148 ± 0.061 | 0.656 ± 0.161 | 0.046 ± 0.017 |
| temp_rff | 0.106 ± 0.022 | 0.492 ± 0.061 | 0.601 ± 0.161 | 0.124 ± 0.037 | 0.389 ± 0.199 | 0.070 ± 0.018 | 0.114 ± 0.075 | 0.237 ± 0.131 | 0.893 ± 0.085 | 0.021 ± 0.007 | 0.358 ± 0.126 | 0.142 ± 0.058 | 0.633 ± 0.146 | 0.046 ± 0.015 |
| temp_rff_nowt | 0.106 ± 0.021 | 0.492 ± 0.059 | 0.601 ± 0.164 | 0.123 ± 0.034 | 0.389 ± 0.199 | 0.070 ± 0.019 | 0.108 ± 0.071 | 0.244 ± 0.135 | 0.896 ± 0.083 | 0.020 ± 0.008 | 0.338 ± 0.131 | 0.147 ± 0.058 | 0.652 ± 0.150 | 0.045 ± 0.014 |
| probe_default | 0.100 ± 0.031 | 0.477 ± 0.044 | 0.995 ± 0.006 | 0.100 ± 0.031 | 0.002 ± 0.003 | 0.100 ± 0.031 | 0.606 ± 0.132 | 0.114 ± 0.037 | 0.359 ± 0.126 | 0.068 ± 0.022 | 0.336 ± 0.193 | 0.152 ± 0.096 | 0.642 ± 0.198 | 0.041 ± 0.017 |
| knn_default | 0.106 ± 0.021 | 0.501 ± 0.057 | 0.599 ± 0.164 | 0.124 ± 0.039 | 0.400 ± 0.194 | 0.070 ± 0.021 | 0.096 ± 0.065 | 0.255 ± 0.133 | 0.914 ± 0.062 | 0.019 ± 0.008 | 0.345 ± 0.137 | 0.146 ± 0.061 | 0.650 ± 0.160 | 0.046 ± 0.016 |
| openmax_default | 0.079 ± 0.022 | 0.511 ± 0.060 | 0.721 ± 0.157 | 0.078 ± 0.028 | 0.272 ± 0.162 | 0.056 ± 0.022 | 0.253 ± 0.143 | 0.093 ± 0.052 | 0.780 ± 0.142 | 0.022 ± 0.015 | 0.200 ± 0.261 | 0.188 ± 0.259 | 0.807 ± 0.264 | 0.016 ± 0.015 |
| temp_rff_default | 0.106 ± 0.022 | 0.503 ± 0.057 | 0.591 ± 0.158 | 0.123 ± 0.036 | 0.410 ± 0.193 | 0.069 ± 0.020 | 0.102 ± 0.066 | 0.248 ± 0.131 | 0.908 ± 0.060 | 0.020 ± 0.008 | 0.332 ± 0.135 | 0.148 ± 0.058 | 0.667 ± 0.155 | 0.045 ± 0.015 |
| raw_centroid | 0.158 ± 0.021 | 0.506 ± 0.080 | 0.869 ± 0.038 | 0.151 ± 0.017 | 0.154 ± 0.107 | 0.132 ± 0.017 | 0.496 ± 0.032 | 0.107 ± 0.010 | 0.505 ± 0.102 | 0.053 ± 0.006 | 0.577 ± 0.109 | 0.116 ± 0.016 | 0.422 ± 0.173 | 0.068 ± 0.021 |
| raw_centroid_cfo | 0.191 ± 0.018 | 0.506 ± 0.081 | 0.884 ± 0.011 | 0.177 ± 0.014 | 0.138 ± 0.113 | 0.156 ± 0.012 | 0.505 ± 0.009 | 0.109 ± 0.010 | 0.492 ± 0.103 | 0.055 ± 0.005 | 0.546 ± 0.100 | 0.115 ± 0.013 | 0.446 ± 0.151 | 0.064 ± 0.019 |

Tuned calibration: method chosen on Part B, count over runs: msp: q80×9/youden×16, probe: youden×25, openmax: q80×5/youden×20, static: q80×8/youden×17, knn: q80×11/youden×14, temp_rff_p1: q80×8/youden×17, static_window: q80×8/youden×17, knn_window: q80×11/youden×14, temp_rff: q80×8/youden×17, temp_rff_nowt: q80×9/youden×16, probe_default: youden×25, knn_default: q80×12/youden×13, openmax_default: q80×4/youden×21, temp_rff_default: q80×8/youden×17, raw_centroid: q80×4/youden×21, raw_centroid_cfo: q80×2/youden×23

## Per device split (mean over seeds)

| Rule | shift 0 closed / AUROC | shift 4 closed / AUROC | shift 8 closed / AUROC | shift 12 closed / AUROC | shift 16 closed / AUROC |
|---|---|---|---|---|---|
| softmax | 0.075 / – | 0.069 / – | 0.065 / – | 0.049 / – | 0.091 / – |
| msp | 0.075 / 0.478 | 0.069 / 0.505 | 0.065 / 0.549 | 0.049 / 0.505 | 0.091 / 0.466 |
| probe | 0.113 / 0.460 | 0.071 / 0.461 | 0.105 / 0.515 | 0.087 / 0.460 | 0.115 / 0.495 |
| openmax | 0.099 / 0.519 | 0.068 / 0.523 | 0.064 / 0.490 | 0.066 / 0.480 | 0.102 / 0.486 |
| static | 0.104 / 0.562 | 0.073 / 0.442 | 0.102 / 0.516 | 0.075 / 0.473 | 0.118 / 0.471 |
| static_window | 0.104 / 0.562 | 0.073 / 0.442 | 0.102 / 0.516 | 0.085 / 0.453 | 0.118 / 0.471 |
| knn | 0.121 / 0.544 | 0.077 / 0.526 | 0.118 / 0.533 | 0.092 / 0.456 | 0.118 / 0.446 |
| knn_window | 0.121 / 0.544 | 0.077 / 0.526 | 0.118 / 0.533 | 0.092 / 0.456 | 0.118 / 0.446 |
| temp_rff_p1 | 0.104 / 0.562 | 0.073 / 0.442 | 0.102 / 0.516 | 0.075 / 0.473 | 0.118 / 0.471 |
| temp_rff | 0.118 / 0.543 | 0.081 / 0.516 | 0.122 / 0.524 | 0.090 / 0.436 | 0.118 / 0.440 |
| raw_centroid_cfo | 0.159 / 0.555 | 0.187 / 0.640 | 0.200 / 0.433 | 0.204 / 0.453 | 0.204 / 0.450 |

## PRIMARY test: TEMP-RFF minus rule at the split level

Seeds are averaged within each device split first (the seeds of a split share its devices, so they are not independent); tests are over the splits. Outcomes: closed accuracy, AUROC, tuned-protocol known ok and unknown detection. Holm correction over one family: 84 tests: 7 rules ['msp', 'probe', 'openmax', 'static', 'knn', 'static_window', 'knn_window'] x 4 outcomes x tags ['wisig', 'pycom', 'lora'].

| Rule | outcome | mean diff | sd diff | splits won | t p | t p (Holm) | Wilcoxon p | Wilcoxon p (Holm) |
|---|---|---|---|---|---|---|---|---|
| knn_window | auroc | -0.009 | 0.007 | 0/5 | 0.0351 | 1 | 0.0625 | 1 |
| knn_window | closed | +0.000 | 0.003 | 3/5 | 0.822 | 1 | 0.812 | 1 |
| knn_window | tuned_known_ok | +0.001 | 0.003 | 2/5 | 0.663 | 1 | 0.812 | 1 |
| knn_window | tuned_unk_det | -0.022 | 0.028 | 0/5 | 0.146 | 1 | 0.0625 | 1 |
| knn | auroc | -0.009 | 0.007 | 0/5 | 0.0351 | 1 | 0.0625 | 1 |
| knn | closed | +0.000 | 0.003 | 3/5 | 0.822 | 1 | 0.812 | 1 |
| knn | tuned_known_ok | +0.001 | 0.003 | 2/5 | 0.663 | 1 | 0.812 | 1 |
| knn | tuned_unk_det | -0.022 | 0.028 | 0/5 | 0.146 | 1 | 0.0625 | 1 |
| msp | auroc | -0.009 | 0.050 | 2/5 | 0.719 | 1 | 0.625 | 1 |
| msp | closed | +0.036 | 0.018 | 5/5 | 0.0101 | 0.776 | 0.0625 | 1 |
| msp | tuned_known_ok | +0.003 | 0.016 | 3/5 | 0.66 | 1 | 1 | 1 |
| msp | tuned_unk_det | +0.207 | 0.233 | 4/5 | 0.119 | 1 | 0.188 | 1 |
| openmax | auroc | -0.008 | 0.037 | 2/5 | 0.659 | 1 | 0.625 | 1 |
| openmax | closed | +0.026 | 0.018 | 5/5 | 0.0346 | 1 | 0.0625 | 1 |
| openmax | tuned_known_ok | +0.021 | 0.016 | 4/5 | 0.0456 | 1 | 0.125 | 1 |
| openmax | tuned_unk_det | -0.071 | 0.226 | 2/5 | 0.522 | 1 | 0.625 | 1 |
| probe | auroc | +0.014 | 0.056 | 3/5 | 0.613 | 1 | 0.812 | 1 |
| probe | closed | +0.008 | 0.006 | 5/5 | 0.0548 | 1 | 0.0625 | 1 |
| probe | tuned_known_ok | +0.007 | 0.015 | 3/5 | 0.374 | 1 | 0.438 | 1 |
| probe | tuned_unk_det | -0.021 | 0.192 | 2/5 | 0.818 | 1 | 1 | 1 |
| static_window | auroc | +0.003 | 0.042 | 2/5 | 0.879 | 1 | 0.812 | 1 |
| static_window | closed | +0.009 | 0.008 | 5/5 | 0.0517 | 1 | 0.0625 | 1 |
| static_window | tuned_known_ok | -0.001 | 0.013 | 2/5 | 0.937 | 1 | 1 | 1 |
| static_window | tuned_unk_det | -0.035 | 0.095 | 2/5 | 0.455 | 1 | 0.625 | 1 |
| static | auroc | -0.001 | 0.046 | 2/5 | 0.966 | 1 | 0.812 | 1 |
| static | closed | +0.012 | 0.008 | 5/5 | 0.0293 | 1 | 0.0625 | 1 |
| static | tuned_known_ok | -0.001 | 0.013 | 2/5 | 0.929 | 1 | 1 | 1 |
| static | tuned_unk_det | -0.030 | 0.084 | 2/5 | 0.473 | 1 | 0.625 | 1 |

## Secondary (exploratory): TEMP-RFF minus other rules at the split level, uncorrected

| Rule | outcome | mean diff | sd diff | splits won | t p | Wilcoxon p |
|---|---|---|---|---|---|---|
| knn_default | auroc | -0.009 | 0.008 | 0/5 | 0.0648 | 0.0625 |
| knn_default | closed | +0.000 | 0.002 | 2/5 | 0.896 | 0.875 |
| knn_default | tuned_known_ok | +0.001 | 0.003 | 2/5 | 0.643 | 1 |
| knn_default | tuned_unk_det | -0.016 | 0.030 | 2/5 | 0.288 | 0.312 |
| openmax_default | auroc | -0.020 | 0.023 | 1/5 | 0.136 | 0.188 |
| openmax_default | closed | +0.027 | 0.014 | 5/5 | 0.0135 | 0.0625 |
| openmax_default | tuned_known_ok | +0.031 | 0.015 | 5/5 | 0.0106 | 0.0625 |
| openmax_default | tuned_unk_det | -0.174 | 0.212 | 2/5 | 0.14 | 0.312 |
| probe_default | auroc | +0.015 | 0.054 | 3/5 | 0.566 | 0.625 |
| probe_default | closed | +0.006 | 0.004 | 5/5 | 0.0401 | 0.0625 |
| probe_default | tuned_known_ok | +0.006 | 0.010 | 3/5 | 0.285 | 0.312 |
| probe_default | tuned_unk_det | -0.008 | 0.184 | 3/5 | 0.923 | 1 |
| raw_centroid_cfo | auroc | -0.014 | 0.076 | 1/5 | 0.697 | 0.438 |
| raw_centroid_cfo | closed | -0.085 | 0.028 | 0/5 | 0.00261 | 0.0625 |
| raw_centroid_cfo | tuned_known_ok | -0.017 | 0.010 | 0/5 | 0.0186 | 0.0625 |
| raw_centroid_cfo | tuned_unk_det | +0.188 | 0.208 | 4/5 | 0.113 | 0.125 |
| raw_centroid | auroc | -0.014 | 0.076 | 1/5 | 0.691 | 0.438 |
| raw_centroid | closed | -0.052 | 0.013 | 0/5 | 0.000792 | 0.0625 |
| raw_centroid | tuned_known_ok | -0.022 | 0.016 | 0/5 | 0.0403 | 0.0625 |
| raw_centroid | tuned_unk_det | +0.212 | 0.230 | 4/5 | 0.109 | 0.125 |
| softmax | closed | +0.036 | 0.018 | 5/5 | 0.0101 | 0.0625 |
| temp_rff_default | auroc | -0.011 | 0.013 | 0/5 | 0.122 | 0.0625 |
| temp_rff_default | closed | +0.000 | 0.004 | 3/5 | 0.989 | 1 |
| temp_rff_default | tuned_known_ok | +0.002 | 0.004 | 3/5 | 0.335 | 0.438 |
| temp_rff_default | tuned_unk_det | -0.033 | 0.031 | 0/5 | 0.0751 | 0.0625 |
| temp_rff_nowt | auroc | +0.000 | 0.003 | 2/5 | 0.953 | 1 |
| temp_rff_nowt | closed | +0.000 | 0.002 | 4/5 | 0.785 | 0.625 |
| temp_rff_nowt | tuned_known_ok | +0.001 | 0.002 | 4/5 | 0.161 | 0.125 |
| temp_rff_nowt | tuned_unk_det | -0.018 | 0.018 | 1/5 | 0.0872 | 0.125 |
| temp_rff_p1 | auroc | -0.001 | 0.046 | 2/5 | 0.966 | 0.812 |
| temp_rff_p1 | closed | +0.012 | 0.008 | 5/5 | 0.0293 | 0.0625 |
| temp_rff_p1 | tuned_known_ok | -0.001 | 0.013 | 2/5 | 0.929 | 1 |
| temp_rff_p1 | tuned_unk_det | -0.030 | 0.084 | 2/5 | 0.473 | 0.625 |

With 5 splits the smallest possible two-sided Wilcoxon p is 0.0625; report effect sizes and wins, not only p-values.


## Secondary test: TEMP-RFF minus rule per run (same split and seed; not independent)

| Rule | outcome | mean diff | wins | paired t p | Wilcoxon p |
|---|---|---|---|---|---|
| softmax | closed | +0.036 | 24/25 | 1.58e-08 | 5.96e-07 |
| msp | closed | +0.036 | 24/25 | 1.58e-08 | 5.96e-07 |
| msp | auroc | -0.009 | 9/25 | 0.588 | 0.458 |
| msp | tuned_known_ok | +0.003 | 13/25 | 0.486 | 0.493 |
| msp | tuned_unk_det | +0.207 | 17/25 | 0.00182 | 0.00278 |
| probe | closed | +0.008 | 16/25 | 0.0306 | 0.0318 |
| probe | auroc | +0.014 | 13/25 | 0.399 | 0.339 |
| probe | tuned_known_ok | +0.007 | 14/25 | 0.116 | 0.183 |
| probe | tuned_unk_det | -0.021 | 13/25 | 0.667 | 0.791 |
| openmax | closed | +0.026 | 21/25 | 4.06e-05 | 8.8e-05 |
| openmax | auroc | -0.008 | 10/25 | 0.585 | 0.458 |
| openmax | tuned_known_ok | +0.021 | 23/25 | 3.86e-05 | 3.19e-05 |
| openmax | tuned_unk_det | -0.071 | 9/25 | 0.213 | 0.241 |
| static | closed | +0.012 | 19/25 | 0.00125 | 0.00124 |
| static | auroc | -0.001 | 13/25 | 0.94 | 0.615 |
| static | tuned_known_ok | -0.001 | 13/25 | 0.894 | 0.819 |
| static | tuned_unk_det | -0.030 | 11/25 | 0.405 | 0.484 |
| knn | closed | +0.000 | 9/25 | 0.835 | 0.906 |
| knn | auroc | -0.009 | 9/25 | 0.0414 | 0.0422 |
| knn | tuned_known_ok | +0.001 | 7/25 | 0.574 | 0.796 |
| knn | tuned_unk_det | -0.022 | 5/25 | 0.0486 | 0.0583 |
| temp_rff_p1 | closed | +0.012 | 19/25 | 0.00125 | 0.00124 |
| temp_rff_p1 | auroc | -0.001 | 12/25 | 0.94 | 0.568 |
| temp_rff_p1 | tuned_known_ok | -0.001 | 13/25 | 0.894 | 0.819 |
| temp_rff_p1 | tuned_unk_det | -0.030 | 11/25 | 0.405 | 0.484 |
| static_window | closed | +0.009 | 18/25 | 0.00453 | 0.00296 |
| static_window | auroc | +0.003 | 14/25 | 0.798 | 1 |
| static_window | tuned_known_ok | -0.001 | 13/25 | 0.906 | 0.797 |
| static_window | tuned_unk_det | -0.035 | 11/25 | 0.335 | 0.415 |
| knn_window | closed | +0.000 | 9/25 | 0.835 | 0.906 |
| knn_window | auroc | -0.009 | 9/25 | 0.0414 | 0.0422 |
| knn_window | tuned_known_ok | +0.001 | 7/25 | 0.574 | 0.796 |
| knn_window | tuned_unk_det | -0.022 | 5/25 | 0.0486 | 0.0583 |
| temp_rff_nowt | closed | +0.000 | 4/25 | 0.728 | 0.953 |
| temp_rff_nowt | auroc | +0.000 | 3/25 | 0.954 | 0.594 |
| temp_rff_nowt | tuned_known_ok | +0.001 | 6/25 | 0.176 | 0.173 |
| temp_rff_nowt | tuned_unk_det | -0.018 | 3/25 | 0.0795 | 0.11 |
| probe_default | closed | +0.006 | 17/25 | 0.157 | 0.0667 |
| probe_default | auroc | +0.015 | 13/25 | 0.359 | 0.241 |
| probe_default | tuned_known_ok | +0.006 | 18/25 | 0.125 | 0.101 |
| probe_default | tuned_unk_det | -0.009 | 14/25 | 0.859 | 0.937 |
| knn_default | closed | +0.000 | 13/25 | 0.945 | 0.895 |
| knn_default | auroc | -0.009 | 9/25 | 0.0649 | 0.0318 |
| knn_default | tuned_known_ok | +0.001 | 14/25 | 0.591 | 0.677 |
| knn_default | tuned_unk_det | -0.016 | 11/25 | 0.177 | 0.412 |
| openmax_default | closed | +0.027 | 23/25 | 2.83e-05 | 3.81e-05 |
| openmax_default | auroc | -0.020 | 11/25 | 0.123 | 0.191 |
| openmax_default | tuned_known_ok | +0.031 | 22/25 | 3.22e-07 | 6.56e-06 |
| openmax_default | tuned_unk_det | -0.174 | 7/25 | 0.0142 | 0.0115 |
| temp_rff_default | closed | +0.000 | 5/25 | 0.991 | 0.878 |
| temp_rff_default | auroc | -0.011 | 1/25 | 0.0359 | 0.00691 |
| temp_rff_default | tuned_known_ok | +0.002 | 6/25 | 0.166 | 0.173 |
| temp_rff_default | tuned_unk_det | -0.033 | 0/25 | 0.0103 | 0.00763 |
| raw_centroid | closed | -0.052 | 0/25 | 1.59e-11 | 5.96e-08 |
| raw_centroid | auroc | -0.014 | 12/25 | 0.378 | 0.458 |
| raw_centroid | tuned_known_ok | -0.022 | 4/25 | 0.000206 | 0.000217 |
| raw_centroid | tuned_unk_det | +0.212 | 21/25 | 0.00024 | 0.00014 |
| raw_centroid_cfo | closed | -0.085 | 0/25 | 3.34e-13 | 5.96e-08 |
| raw_centroid_cfo | auroc | -0.014 | 9/25 | 0.382 | 0.339 |
| raw_centroid_cfo | tuned_known_ok | -0.017 | 3/25 | 3.9e-05 | 2.66e-05 |
| raw_centroid_cfo | tuned_unk_det | +0.188 | 20/25 | 0.000172 | 0.000217 |
## Forced prototype window (decay 1)

| W | Closed acc. | AUROC | tuned accept | tuned unk. det. |
|---|---|---|---|---|
| 1 | 0.106 ± 0.022 | 0.492 ± 0.061 | 0.358 ± 0.126 | 0.633 ± 0.146 |
| 2 | 0.103 ± 0.018 | 0.526 ± 0.080 | 0.464 ± 0.094 | 0.563 ± 0.162 |
| 3 | 0.110 ± 0.019 | 0.523 ± 0.058 | 0.456 ± 0.104 | 0.577 ± 0.124 |
| 4 | 0.109 ± 0.021 | 0.523 ± 0.058 | 0.477 ± 0.119 | 0.554 ± 0.145 |

## Few-shot target-session enrollment (same support for every rule)

| k requested | k effective | temp_rff closed / AUROC / tuned known ok | static closed / AUROC / tuned known ok | knn closed / AUROC / tuned known ok | static_window closed / AUROC / tuned known ok | knn_window closed / AUROC / tuned known ok |
|---|---|---|---|---|---|---|
| 1 | 1 | 0.169 / 0.529 / 0.096 | 0.097 / 0.494 / 0.050 | 0.130 / 0.515 / 0.058 | 0.100 / 0.489 / 0.050 | 0.130 / 0.515 / 0.058 |
| 5 | 5 | 0.240 / 0.551 / 0.156 | 0.098 / 0.493 / 0.051 | 0.181 / 0.535 / 0.085 | 0.100 / 0.489 / 0.051 | 0.181 / 0.535 / 0.085 |
| 10 | 10 | 0.262 / 0.560 / 0.169 | 0.099 / 0.493 / 0.051 | 0.218 / 0.546 / 0.109 | 0.101 / 0.489 / 0.051 | 0.218 / 0.546 / 0.109 |
| 20 | 20 | 0.282 / 0.563 / 0.189 | 0.101 / 0.493 / 0.051 | 0.252 / 0.556 / 0.138 | 0.103 / 0.488 / 0.051 | 0.252 / 0.556 / 0.138 |
| 50 | 50 | 0.296 / 0.570 / 0.198 | 0.106 / 0.492 / 0.054 | 0.283 / 0.569 / 0.174 | 0.108 / 0.488 / 0.055 | 0.283 / 0.569 / 0.174 |
| 100 | 100 | 0.303 / 0.571 / 0.207 | 0.117 / 0.490 / 0.060 | 0.299 / 0.574 / 0.197 | 0.118 / 0.486 / 0.061 | 0.299 / 0.574 / 0.197 |
| 200 | 200 | 0.311 / 0.573 / 0.219 | 0.148 / 0.490 / 0.078 | 0.313 / 0.578 / 0.212 | 0.148 / 0.488 / 0.081 | 0.313 / 0.578 / 0.212 |
| 500 | 316 | 0.314 / 0.576 / 0.222 | 0.187 / 0.499 / 0.106 | 0.320 / 0.580 / 0.222 | 0.186 / 0.497 / 0.106 | 0.320 / 0.580 / 0.222 |

## Pipeline stages (Part-A centroids)

| Stage | Softmax acc. | Centroid acc. | AUROC |
|---|---|---|---|
| phase1 | 0.130 ± 0.037 | 0.114 ± 0.038 | 0.490 ± 0.048 |
| phase2 | 0.087 ± 0.029 | 0.112 ± 0.022 | 0.502 ± 0.067 |
| phase3 | 0.070 ± 0.022 | 0.094 ± 0.025 | 0.493 ± 0.070 |

Joint-training control vs sequential Phase 2 (paired, split 0; joint runs have no audit):

| metric | joint | Phase 2 | mean diff | wins |
|---|---|---|---|---|
| softmax | 0.153 ± 0.018 | 0.103 ± 0.020 | +0.050 | 5/5 |
| centroid | 0.142 ± 0.025 | 0.129 ± 0.023 | +0.013 | 3/5 |
| auroc | 0.515 ± 0.042 | 0.570 ± 0.069 | -0.054 | 1/5 |

## Selected settings over runs

- W: 1×24, 2×1
- decay: 0.6×1, 1.0×24
- P: 1×1, 20×2, 5×4, 50×2, all×16
- knn (k, weighted): (1, False)×11, (10, False)×1, (10, True)×1, (20, True)×2, (5, False)×9, (5, True)×1
- openmax (dist, tail, alpha): ('euclidean', 10, 1)×5, ('euclidean', 10, 2)×2, ('euclidean', 20, 10)×1, ('euclidean', 50, 5)×2, ('eucos', 10, 1)×7, ('eucos', 10, 10)×1, ('eucos', 10, 16)×1, ('eucos', 10, 2)×3, ('eucos', 10, 5)×1, ('eucos', 50, 1)×1, ('eucos', 50, 10)×1
- probe C: 0.01×4, 0.1×4, 1.0×2, 10.0×15
- static_window W: (1,)×23, (2,)×2
- knn_window W (k, weighted): (1, 1, False)×11, (1, 10, False)×1, (1, 10, True)×1, (1, 20, True)×2, (1, 5, False)×9, (1, 5, True)×1
- temp_rff_p1 W decay: (1, 1.0)×25
- selection-weight sensitivity: choice changes with the weight (0.3/0.5/0.7) in 13 of 25 runs
