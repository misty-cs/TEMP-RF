# Summary: wisig — COMPLETE

Audited runs used: 20 (splits [0, 4, 8, 12], seeds [7, 13, 21, 42, 100]); expected splits [0, 4, 8, 12] x seeds [42, 7, 13, 21, 100]

## All rules (mean ± std over runs)

| Rule | Closed acc. | AUROC | fixed90 accept | fixed90 open | fixed90 unk. det. | fixed90 known ok | objective accept | objective open | objective unk. det. | objective known ok | tuned accept | tuned open | tuned unk. det. | tuned known ok |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| softmax | 0.808 ± 0.050 | – | – | – | – | – | – | – | – | – | – | – | – | – |
| msp | 0.808 ± 0.050 | 0.671 ± 0.134 | 0.878 ± 0.029 | 0.852 ± 0.050 | 0.295 ± 0.164 | 0.748 ± 0.050 | 0.506 ± 0.061 | 0.909 ± 0.063 | 0.719 ± 0.180 | 0.458 ± 0.049 | 0.825 ± 0.081 | 0.869 ± 0.062 | 0.383 ± 0.179 | 0.713 ± 0.055 |
| probe | 0.859 ± 0.054 | 0.644 ± 0.160 | 0.795 ± 0.054 | 0.926 ± 0.055 | 0.457 ± 0.236 | 0.735 ± 0.045 | 0.431 ± 0.058 | 0.955 ± 0.045 | 0.729 ± 0.192 | 0.412 ± 0.059 | 0.760 ± 0.148 | 0.929 ± 0.063 | 0.493 ± 0.260 | 0.702 ± 0.123 |
| openmax | 0.770 ± 0.064 | 0.676 ± 0.133 | 0.744 ± 0.082 | 0.887 ± 0.056 | 0.491 ± 0.170 | 0.658 ± 0.073 | 0.429 ± 0.091 | 0.931 ± 0.053 | 0.793 ± 0.193 | 0.400 ± 0.086 | 0.690 ± 0.086 | 0.909 ± 0.045 | 0.609 ± 0.161 | 0.628 ± 0.089 |
| static | 0.835 ± 0.049 | 0.725 ± 0.108 | 0.753 ± 0.066 | 0.912 ± 0.053 | 0.520 ± 0.195 | 0.685 ± 0.050 | 0.345 ± 0.081 | 0.982 ± 0.032 | 0.914 ± 0.128 | 0.339 ± 0.081 | 0.839 ± 0.080 | 0.891 ± 0.046 | 0.390 ± 0.169 | 0.748 ± 0.085 |
| knn | 0.877 ± 0.043 | 0.786 ± 0.085 | 0.630 ± 0.073 | 0.965 ± 0.032 | 0.800 ± 0.129 | 0.608 ± 0.067 | 0.542 ± 0.112 | 0.981 ± 0.022 | 0.852 ± 0.148 | 0.531 ± 0.109 | 0.679 ± 0.066 | 0.961 ± 0.025 | 0.750 ± 0.133 | 0.652 ± 0.063 |
| temp_rff_p1 | 0.837 ± 0.048 | 0.730 ± 0.104 | 0.768 ± 0.061 | 0.913 ± 0.050 | 0.532 ± 0.177 | 0.700 ± 0.038 | 0.353 ± 0.083 | 0.983 ± 0.026 | 0.897 ± 0.138 | 0.347 ± 0.082 | 0.864 ± 0.054 | 0.889 ± 0.045 | 0.388 ± 0.161 | 0.768 ± 0.060 |
| static_window | 0.835 ± 0.049 | 0.725 ± 0.108 | 0.753 ± 0.066 | 0.912 ± 0.053 | 0.520 ± 0.195 | 0.685 ± 0.050 | 0.345 ± 0.081 | 0.982 ± 0.032 | 0.914 ± 0.128 | 0.339 ± 0.081 | 0.839 ± 0.080 | 0.891 ± 0.046 | 0.390 ± 0.169 | 0.748 ± 0.085 |
| knn_window | 0.878 ± 0.043 | 0.785 ± 0.085 | 0.631 ± 0.073 | 0.964 ± 0.032 | 0.798 ± 0.128 | 0.608 ± 0.068 | 0.540 ± 0.111 | 0.981 ± 0.022 | 0.853 ± 0.149 | 0.529 ± 0.109 | 0.679 ± 0.066 | 0.961 ± 0.025 | 0.749 ± 0.132 | 0.652 ± 0.063 |
| temp_rff | 0.869 ± 0.043 | 0.791 ± 0.087 | 0.639 ± 0.071 | 0.964 ± 0.024 | 0.794 ± 0.139 | 0.615 ± 0.064 | 0.555 ± 0.165 | 0.976 ± 0.021 | 0.847 ± 0.151 | 0.541 ± 0.161 | 0.649 ± 0.101 | 0.961 ± 0.027 | 0.769 ± 0.154 | 0.623 ± 0.091 |
| temp_rff_nowt | 0.870 ± 0.043 | 0.795 ± 0.085 | 0.634 ± 0.068 | 0.964 ± 0.024 | 0.805 ± 0.134 | 0.610 ± 0.062 | 0.562 ± 0.151 | 0.977 ± 0.021 | 0.846 ± 0.151 | 0.548 ± 0.147 | 0.640 ± 0.094 | 0.963 ± 0.025 | 0.784 ± 0.144 | 0.616 ± 0.086 |
| probe_default | 0.866 ± 0.043 | 0.654 ± 0.174 | 0.793 ± 0.066 | 0.927 ± 0.052 | 0.460 ± 0.225 | 0.733 ± 0.055 | 0.431 ± 0.056 | 0.969 ± 0.034 | 0.764 ± 0.236 | 0.418 ± 0.055 | 0.763 ± 0.098 | 0.938 ± 0.049 | 0.456 ± 0.206 | 0.713 ± 0.081 |
| knn_default | 0.878 ± 0.043 | 0.784 ± 0.083 | 0.634 ± 0.077 | 0.963 ± 0.033 | 0.791 ± 0.129 | 0.610 ± 0.070 | 0.536 ± 0.103 | 0.982 ± 0.022 | 0.854 ± 0.150 | 0.526 ± 0.101 | 0.665 ± 0.073 | 0.964 ± 0.024 | 0.762 ± 0.142 | 0.640 ± 0.071 |
| openmax_default | 0.782 ± 0.060 | 0.685 ± 0.143 | 0.747 ± 0.083 | 0.902 ± 0.057 | 0.520 ± 0.170 | 0.672 ± 0.068 | 0.339 ± 0.066 | 0.936 ± 0.050 | 0.866 ± 0.201 | 0.316 ± 0.060 | 0.489 ± 0.090 | 0.936 ± 0.046 | 0.783 ± 0.181 | 0.456 ± 0.077 |
| temp_rff_default | 0.870 ± 0.045 | 0.799 ± 0.083 | 0.615 ± 0.069 | 0.969 ± 0.023 | 0.826 ± 0.124 | 0.596 ± 0.063 | 0.583 ± 0.115 | 0.978 ± 0.021 | 0.844 ± 0.150 | 0.569 ± 0.111 | 0.612 ± 0.096 | 0.970 ± 0.024 | 0.809 ± 0.137 | 0.593 ± 0.089 |
| raw_centroid | 0.851 ± 0.063 | 0.649 ± 0.104 | 0.663 ± 0.068 | 0.957 ± 0.032 | 0.518 ± 0.203 | 0.635 ± 0.067 | 0.405 ± 0.066 | 0.982 ± 0.010 | 0.779 ± 0.157 | 0.398 ± 0.065 | 0.461 ± 0.075 | 0.982 ± 0.010 | 0.733 ± 0.200 | 0.452 ± 0.075 |
| raw_centroid_cfo | 0.851 ± 0.063 | 0.649 ± 0.104 | 0.663 ± 0.068 | 0.957 ± 0.032 | 0.518 ± 0.203 | 0.635 ± 0.067 | 0.405 ± 0.066 | 0.982 ± 0.010 | 0.779 ± 0.157 | 0.398 ± 0.065 | 0.461 ± 0.075 | 0.982 ± 0.010 | 0.733 ± 0.200 | 0.452 ± 0.075 |

Tuned calibration: method chosen on Part B, count over runs: msp: q80×4/q90×5/youden×11, probe: q80×1/q90×4/q95×2/youden×13, openmax: q80×5/q90×3/youden×12, static: q90×4/q95×8/youden×8, knn: q90×3/q95×4/youden×13, temp_rff_p1: q90×4/q95×7/youden×9, static_window: q90×4/q95×8/youden×8, knn_window: q90×2/q95×4/youden×14, temp_rff: q90×4/q95×3/youden×13, temp_rff_nowt: q90×5/q95×3/youden×12, probe_default: q90×2/q95×2/youden×16, knn_default: q90×3/q95×3/youden×14, openmax_default: q80×1/youden×19, temp_rff_default: q90×5/q95×3/youden×12, raw_centroid: q80×6/youden×14, raw_centroid_cfo: q80×6/youden×14

## Per device split (mean over seeds)

| Rule | shift 0 closed / AUROC | shift 4 closed / AUROC | shift 8 closed / AUROC | shift 12 closed / AUROC |
|---|---|---|---|---|
| softmax | 0.795 / – | 0.748 / – | 0.849 / – | 0.840 / – |
| msp | 0.795 / 0.684 | 0.748 / 0.686 | 0.849 / 0.515 | 0.840 / 0.800 |
| probe | 0.876 / 0.598 | 0.802 / 0.712 | 0.881 / 0.467 | 0.875 / 0.800 |
| openmax | 0.711 / 0.663 | 0.759 / 0.717 | 0.792 / 0.507 | 0.819 / 0.818 |
| static | 0.842 / 0.672 | 0.798 / 0.674 | 0.866 / 0.775 | 0.836 / 0.780 |
| static_window | 0.842 / 0.672 | 0.798 / 0.674 | 0.866 / 0.775 | 0.836 / 0.780 |
| knn | 0.878 / 0.766 | 0.843 / 0.788 | 0.894 / 0.723 | 0.894 / 0.867 |
| knn_window | 0.878 / 0.766 | 0.845 / 0.786 | 0.894 / 0.723 | 0.894 / 0.867 |
| temp_rff_p1 | 0.844 / 0.687 | 0.800 / 0.724 | 0.866 / 0.720 | 0.836 / 0.791 |
| temp_rff | 0.883 / 0.763 | 0.817 / 0.804 | 0.883 / 0.723 | 0.894 / 0.872 |
| raw_centroid_cfo | 0.810 / 0.737 | 0.792 / 0.484 | 0.854 / 0.729 | 0.949 / 0.647 |

## PRIMARY test: TEMP-RFF minus rule at the split level

Seeds are averaged within each device split first (the seeds of a split share its devices, so they are not independent); tests are over the splits. Outcomes: closed accuracy, AUROC, tuned-protocol known ok and unknown detection. Holm correction over one family: 28 tests: 7 rules ['msp', 'probe', 'openmax', 'static', 'knn', 'static_window', 'knn_window'] x 4 outcomes x tags ['wisig'] (NOT the pre-registered family: that needs --tags wisig pycom).

| Rule | outcome | mean diff | sd diff | splits won | t p | t p (Holm) | Wilcoxon p | Wilcoxon p (Holm) |
|---|---|---|---|---|---|---|---|---|
| knn_window | auroc | +0.005 | 0.009 | 2/4 | 0.345 | 1 | 0.625 | 1 |
| knn_window | closed | -0.009 | 0.014 | 1/4 | 0.309 | 1 | 0.375 | 1 |
| knn_window | tuned_known_ok | -0.030 | 0.029 | 1/4 | 0.133 | 1 | 0.25 | 1 |
| knn_window | tuned_unk_det | +0.020 | 0.053 | 2/4 | 0.508 | 1 | 0.875 | 1 |
| knn | auroc | +0.005 | 0.008 | 2/4 | 0.343 | 1 | 0.625 | 1 |
| knn | closed | -0.008 | 0.013 | 1/4 | 0.314 | 1 | 0.375 | 1 |
| knn | tuned_known_ok | -0.030 | 0.029 | 1/4 | 0.132 | 1 | 0.25 | 1 |
| knn | tuned_unk_det | +0.019 | 0.051 | 2/4 | 0.514 | 1 | 0.875 | 1 |
| msp | auroc | +0.119 | 0.063 | 4/4 | 0.0317 | 0.602 | 0.125 | 1 |
| msp | closed | +0.061 | 0.023 | 4/4 | 0.0127 | 0.33 | 0.125 | 1 |
| msp | tuned_known_ok | -0.091 | 0.030 | 0/4 | 0.00898 | 0.251 | 0.125 | 1 |
| msp | tuned_unk_det | +0.386 | 0.134 | 4/4 | 0.0103 | 0.279 | 0.125 | 1 |
| openmax | auroc | +0.114 | 0.071 | 4/4 | 0.0478 | 0.706 | 0.125 | 1 |
| openmax | closed | +0.099 | 0.050 | 4/4 | 0.0286 | 0.572 | 0.125 | 1 |
| openmax | tuned_known_ok | -0.005 | 0.042 | 2/4 | 0.818 | 1 | 1 | 1 |
| openmax | tuned_unk_det | +0.160 | 0.067 | 4/4 | 0.0177 | 0.371 | 0.125 | 1 |
| probe | auroc | +0.146 | 0.084 | 4/4 | 0.0396 | 0.697 | 0.125 | 1 |
| probe | closed | +0.011 | 0.008 | 4/4 | 0.0767 | 0.997 | 0.125 | 1 |
| probe | tuned_known_ok | -0.079 | 0.062 | 0/4 | 0.0849 | 1 | 0.125 | 1 |
| probe | tuned_unk_det | +0.276 | 0.169 | 4/4 | 0.0471 | 0.706 | 0.125 | 1 |
| static_window | auroc | +0.065 | 0.080 | 3/4 | 0.201 | 1 | 0.25 | 1 |
| static_window | closed | +0.034 | 0.019 | 4/4 | 0.0387 | 0.697 | 0.125 | 1 |
| static_window | tuned_known_ok | -0.125 | 0.048 | 0/4 | 0.0135 | 0.339 | 0.125 | 1 |
| static_window | tuned_unk_det | +0.378 | 0.145 | 4/4 | 0.0137 | 0.339 | 0.125 | 1 |
| static | auroc | +0.065 | 0.080 | 3/4 | 0.201 | 1 | 0.25 | 1 |
| static | closed | +0.034 | 0.019 | 4/4 | 0.0387 | 0.697 | 0.125 | 1 |
| static | tuned_known_ok | -0.125 | 0.048 | 0/4 | 0.0135 | 0.339 | 0.125 | 1 |
| static | tuned_unk_det | +0.378 | 0.145 | 4/4 | 0.0137 | 0.339 | 0.125 | 1 |

## Secondary (exploratory): TEMP-RFF minus other rules at the split level, uncorrected

| Rule | outcome | mean diff | sd diff | splits won | t p | Wilcoxon p |
|---|---|---|---|---|---|---|
| knn_default | auroc | +0.007 | 0.012 | 2/4 | 0.336 | 0.625 |
| knn_default | closed | -0.008 | 0.014 | 2/4 | 0.315 | 0.625 |
| knn_default | tuned_known_ok | -0.018 | 0.028 | 1/4 | 0.303 | 0.375 |
| knn_default | tuned_unk_det | +0.007 | 0.055 | 2/4 | 0.829 | 0.875 |
| openmax_default | auroc | +0.106 | 0.099 | 4/4 | 0.123 | 0.125 |
| openmax_default | closed | +0.087 | 0.039 | 4/4 | 0.0204 | 0.125 |
| openmax_default | tuned_known_ok | +0.167 | 0.049 | 4/4 | 0.00651 | 0.125 |
| openmax_default | tuned_unk_det | -0.015 | 0.097 | 2/4 | 0.782 | 1 |
| probe_default | auroc | +0.137 | 0.113 | 4/4 | 0.0934 | 0.125 |
| probe_default | closed | +0.003 | 0.010 | 3/4 | 0.606 | 0.875 |
| probe_default | tuned_known_ok | -0.091 | 0.046 | 0/4 | 0.029 | 0.125 |
| probe_default | tuned_unk_det | +0.313 | 0.164 | 4/4 | 0.0314 | 0.125 |
| raw_centroid_cfo | auroc | +0.141 | 0.157 | 3/4 | 0.169 | 0.25 |
| raw_centroid_cfo | closed | +0.018 | 0.054 | 3/4 | 0.549 | 0.625 |
| raw_centroid_cfo | tuned_known_ok | +0.170 | 0.085 | 4/4 | 0.028 | 0.125 |
| raw_centroid_cfo | tuned_unk_det | +0.036 | 0.316 | 2/4 | 0.835 | 0.875 |
| raw_centroid | auroc | +0.141 | 0.157 | 3/4 | 0.169 | 0.25 |
| raw_centroid | closed | +0.018 | 0.054 | 3/4 | 0.549 | 0.625 |
| raw_centroid | tuned_known_ok | +0.170 | 0.085 | 4/4 | 0.028 | 0.125 |
| raw_centroid | tuned_unk_det | +0.036 | 0.316 | 2/4 | 0.835 | 0.875 |
| softmax | closed | +0.061 | 0.023 | 4/4 | 0.0127 | 0.125 |
| temp_rff_default | auroc | -0.008 | 0.013 | 2/4 | 0.295 | 0.625 |
| temp_rff_default | closed | -0.001 | 0.009 | 1/4 | 0.817 | 0.875 |
| temp_rff_default | tuned_known_ok | +0.030 | 0.041 | 4/4 | 0.24 | 0.125 |
| temp_rff_default | tuned_unk_det | -0.040 | 0.050 | 0/4 | 0.208 | 0.125 |
| temp_rff_nowt | auroc | -0.005 | 0.006 | 0/4 | 0.235 | 0.5 |
| temp_rff_nowt | closed | -0.001 | 0.001 | 0/4 | 0.194 | 0.5 |
| temp_rff_nowt | tuned_known_ok | +0.007 | 0.012 | 2/4 | 0.315 | 0.5 |
| temp_rff_nowt | tuned_unk_det | -0.016 | 0.024 | 0/4 | 0.286 | 0.5 |
| temp_rff_p1 | auroc | +0.060 | 0.038 | 4/4 | 0.0495 | 0.125 |
| temp_rff_p1 | closed | +0.033 | 0.020 | 4/4 | 0.0452 | 0.125 |
| temp_rff_p1 | tuned_known_ok | -0.145 | 0.047 | 0/4 | 0.00856 | 0.125 |
| temp_rff_p1 | tuned_unk_det | +0.381 | 0.103 | 4/4 | 0.00509 | 0.125 |

With 4 splits the smallest possible two-sided Wilcoxon p is 0.125; report effect sizes and wins, not only p-values.


## Secondary test: TEMP-RFF minus rule per run (same split and seed; not independent)

| Rule | outcome | mean diff | wins | paired t p | Wilcoxon p |
|---|---|---|---|---|---|
| softmax | closed | +0.061 | 19/20 | 5.99e-06 | 9.54e-06 |
| msp | closed | +0.061 | 19/20 | 5.99e-06 | 9.54e-06 |
| msp | auroc | +0.119 | 18/20 | 2.21e-05 | 9.54e-06 |
| msp | tuned_known_ok | -0.091 | 2/20 | 0.000352 | 0.000105 |
| msp | tuned_unk_det | +0.386 | 18/20 | 1.88e-07 | 9.54e-06 |
| probe | closed | +0.011 | 11/20 | 0.225 | 0.212 |
| probe | auroc | +0.146 | 17/20 | 1.17e-05 | 2.67e-05 |
| probe | tuned_known_ok | -0.079 | 4/20 | 0.00423 | 0.00365 |
| probe | tuned_unk_det | +0.276 | 18/20 | 0.000211 | 0.000261 |
| openmax | closed | +0.099 | 20/20 | 4.3e-07 | 1.91e-06 |
| openmax | auroc | +0.114 | 19/20 | 2.38e-05 | 2.67e-05 |
| openmax | tuned_known_ok | -0.005 | 9/20 | 0.808 | 0.596 |
| openmax | tuned_unk_det | +0.160 | 17/20 | 0.00018 | 0.000134 |
| static | closed | +0.034 | 16/20 | 0.00135 | 0.00121 |
| static | auroc | +0.065 | 14/20 | 0.00459 | 0.00558 |
| static | tuned_known_ok | -0.125 | 2/20 | 7.74e-07 | 9.54e-06 |
| static | tuned_unk_det | +0.378 | 19/20 | 2.24e-07 | 3.81e-06 |
| knn | closed | -0.008 | 6/20 | 0.215 | 0.211 |
| knn | auroc | +0.005 | 14/20 | 0.454 | 0.177 |
| knn | tuned_known_ok | -0.030 | 5/20 | 0.116 | 0.0995 |
| knn | tuned_unk_det | +0.019 | 9/20 | 0.457 | 0.363 |
| temp_rff_p1 | closed | +0.033 | 16/20 | 0.00172 | 0.00143 |
| temp_rff_p1 | auroc | +0.060 | 15/20 | 0.00207 | 0.00486 |
| temp_rff_p1 | tuned_known_ok | -0.145 | 0/20 | 9.56e-08 | 1.91e-06 |
| temp_rff_p1 | tuned_unk_det | +0.381 | 20/20 | 4.19e-08 | 1.91e-06 |
| static_window | closed | +0.034 | 16/20 | 0.00135 | 0.00121 |
| static_window | auroc | +0.065 | 14/20 | 0.00459 | 0.00558 |
| static_window | tuned_known_ok | -0.125 | 2/20 | 7.74e-07 | 9.54e-06 |
| static_window | tuned_unk_det | +0.378 | 19/20 | 2.24e-07 | 3.81e-06 |
| knn_window | closed | -0.009 | 6/20 | 0.189 | 0.211 |
| knn_window | auroc | +0.005 | 14/20 | 0.407 | 0.165 |
| knn_window | tuned_known_ok | -0.030 | 4/20 | 0.114 | 0.0884 |
| knn_window | tuned_unk_det | +0.020 | 10/20 | 0.432 | 0.334 |
| temp_rff_nowt | closed | -0.001 | 0/20 | 0.114 | 0.109 |
| temp_rff_nowt | auroc | -0.005 | 0/20 | 0.103 | 0.109 |
| temp_rff_nowt | tuned_known_ok | +0.007 | 3/20 | 0.116 | 0.109 |
| temp_rff_nowt | tuned_unk_det | -0.016 | 0/20 | 0.0991 | 0.109 |
| probe_default | closed | +0.003 | 9/20 | 0.716 | 0.841 |
| probe_default | auroc | +0.137 | 18/20 | 7.96e-05 | 4.77e-05 |
| probe_default | tuned_known_ok | -0.091 | 2/20 | 0.000252 | 0.00102 |
| probe_default | tuned_unk_det | +0.313 | 19/20 | 3.94e-06 | 8.2e-05 |
| knn_default | closed | -0.008 | 10/20 | 0.191 | 0.332 |
| knn_default | auroc | +0.007 | 16/20 | 0.247 | 0.123 |
| knn_default | tuned_known_ok | -0.018 | 7/20 | 0.402 | 0.202 |
| knn_default | tuned_unk_det | +0.007 | 12/20 | 0.818 | 0.368 |
| openmax_default | closed | +0.087 | 20/20 | 3.04e-07 | 1.91e-06 |
| openmax_default | auroc | +0.106 | 16/20 | 0.000261 | 0.00021 |
| openmax_default | tuned_known_ok | +0.167 | 19/20 | 2.08e-06 | 9.54e-06 |
| openmax_default | tuned_unk_det | -0.015 | 10/20 | 0.688 | 0.756 |
| temp_rff_default | closed | -0.001 | 1/20 | 0.765 | 0.5 |
| temp_rff_default | auroc | -0.008 | 2/20 | 0.137 | 0.225 |
| temp_rff_default | tuned_known_ok | +0.030 | 5/20 | 0.0972 | 0.0431 |
| temp_rff_default | tuned_unk_det | -0.040 | 0/20 | 0.0841 | 0.0431 |
| raw_centroid | closed | +0.018 | 13/20 | 0.183 | 0.165 |
| raw_centroid | auroc | +0.141 | 17/20 | 0.000639 | 0.00121 |
| raw_centroid | tuned_known_ok | +0.170 | 19/20 | 1.98e-06 | 1.91e-05 |
| raw_centroid | tuned_unk_det | +0.036 | 10/20 | 0.607 | 0.596 |
| raw_centroid_cfo | closed | +0.018 | 13/20 | 0.183 | 0.165 |
| raw_centroid_cfo | auroc | +0.141 | 17/20 | 0.000639 | 0.00121 |
| raw_centroid_cfo | tuned_known_ok | +0.170 | 19/20 | 1.98e-06 | 1.91e-05 |
| raw_centroid_cfo | tuned_unk_det | +0.036 | 10/20 | 0.607 | 0.596 |
## Forced prototype window (decay 1)

| W | Closed acc. | AUROC | tuned accept | tuned unk. det. |
|---|---|---|---|---|
| 1 | 0.871 ± 0.042 | 0.789 ± 0.086 | 0.646 ± 0.100 | 0.770 ± 0.155 |
| 2 | 0.871 ± 0.044 | 0.791 ± 0.094 | 0.674 ± 0.086 | 0.752 ± 0.162 |
| 3 | 0.872 ± 0.041 | 0.785 ± 0.100 | 0.675 ± 0.080 | 0.738 ± 0.162 |

## Few-shot target-session enrollment (same support for every rule)

| k requested | k effective | temp_rff closed / AUROC / tuned known ok | static closed / AUROC / tuned known ok | knn closed / AUROC / tuned known ok | static_window closed / AUROC / tuned known ok | knn_window closed / AUROC / tuned known ok |
|---|---|---|---|---|---|---|
| 1 | 1 | 0.902 / 0.806 / 0.641 | 0.832 / 0.719 / 0.743 | 0.890 / 0.791 / 0.652 | 0.832 / 0.719 / 0.743 | 0.889 / 0.789 / 0.651 |
| 5 | 5 | 0.930 / 0.832 / 0.686 | 0.833 / 0.720 / 0.744 | 0.917 / 0.808 / 0.663 | 0.833 / 0.720 / 0.744 | 0.916 / 0.806 / 0.662 |
| 10 | 10 | 0.888 / 0.798 / 0.629 | 0.833 / 0.721 / 0.747 | 0.929 / 0.821 / 0.685 | 0.833 / 0.721 / 0.747 | 0.929 / 0.819 / 0.683 |
| 20 | 20 | 0.889 / 0.799 / 0.632 | 0.835 / 0.723 / 0.747 | 0.944 / 0.834 / 0.718 | 0.835 / 0.723 / 0.747 | 0.944 / 0.833 / 0.716 |
| 50 | 49 | 0.890 / 0.799 / 0.635 | 0.838 / 0.729 / 0.755 | 0.958 / 0.853 / 0.764 | 0.838 / 0.729 / 0.755 | 0.958 / 0.852 / 0.762 |
| 100 | 97 | 0.890 / 0.800 / 0.633 | 0.843 / 0.737 / 0.762 | 0.966 / 0.868 / 0.790 | 0.843 / 0.737 / 0.762 | 0.966 / 0.867 / 0.789 |
| 200 | 192 | 0.890 / 0.800 / 0.633 | 0.851 / 0.749 / 0.781 | 0.972 / 0.881 / 0.809 | 0.851 / 0.749 / 0.781 | 0.972 / 0.880 / 0.809 |
| 500 | 379 | 0.890 / 0.799 / 0.632 | 0.859 / 0.761 / 0.805 | 0.976 / 0.892 / 0.834 | 0.859 / 0.761 / 0.805 | 0.977 / 0.892 / 0.835 |

## Pipeline stages (Part-A centroids)

| Stage | Softmax acc. | Centroid acc. | AUROC |
|---|---|---|---|
| phase1 | 0.652 ± 0.100 | 0.705 ± 0.085 | 0.623 ± 0.156 |
| phase2 | 0.807 ± 0.047 | 0.835 ± 0.048 | 0.724 ± 0.108 |
| phase3 | 0.808 ± 0.050 | 0.835 ± 0.049 | 0.725 ± 0.108 |

Joint-training control vs sequential Phase 2 (paired, split 0; joint runs have no audit):

| metric | joint | Phase 2 | mean diff | wins |
|---|---|---|---|---|
| softmax | 0.884 ± 0.044 | 0.807 ± 0.043 | +0.077 | 4/5 |
| centroid | 0.912 ± 0.048 | 0.852 ± 0.042 | +0.060 | 4/5 |
| auroc | 0.681 ± 0.175 | 0.667 ± 0.152 | +0.014 | 3/5 |

## Selected settings over runs

- W: 1×18, 2×2
- decay: 1.0×20
- P: 5×3, all×17
- knn (k, weighted): (1, False)×6, (10, False)×1, (10, True)×2, (5, False)×11
- openmax (dist, tail, alpha): ('euclidean', 10, 1)×15, ('eucos', 10, 1)×4, ('eucos', 10, 2)×1
- probe C: 0.01×7, 0.1×1, 1.0×3, 10.0×9
- static_window W: (1,)×20
- knn_window W (k, weighted): (1, 1, False)×5, (1, 10, False)×1, (1, 10, True)×2, (1, 5, False)×10, (1, 5, True)×2
- temp_rff_p1 W decay: (1, 1.0)×10, (2, 0.6)×2, (2, 0.8)×1, (2, 1.0)×4, (3, 0.8)×1, (3, 1.0)×2
- selection-weight sensitivity: choice changes with the weight (0.3/0.5/0.7) in 11 of 20 runs
