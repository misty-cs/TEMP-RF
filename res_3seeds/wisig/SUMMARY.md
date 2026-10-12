# Summary: wisig — COMPLETE

Audited runs used: 12 (splits [0, 4, 8, 12], seeds [7, 13, 42]); expected splits [0, 4, 8, 12] x seeds [42, 7, 13]

## All rules (mean ± std over runs)

| Rule | Closed acc. | AUROC | fixed90 accept | fixed90 open | fixed90 unk. det. | fixed90 known ok | objective accept | objective open | objective unk. det. | objective known ok | tuned accept | tuned open | tuned unk. det. | tuned known ok |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| softmax | 0.843 ± 0.073 | – | – | – | – | – | – | – | – | – | – | – | – | – |
| msp | 0.843 ± 0.073 | 0.684 ± 0.111 | 0.852 ± 0.043 | 0.889 ± 0.074 | 0.362 ± 0.137 | 0.757 ± 0.070 | 0.502 ± 0.064 | 0.937 ± 0.076 | 0.738 ± 0.173 | 0.471 ± 0.072 | 0.854 ± 0.104 | 0.895 ± 0.067 | 0.364 ± 0.185 | 0.764 ± 0.109 |
| probe | 0.877 ± 0.060 | 0.645 ± 0.139 | 0.791 ± 0.094 | 0.949 ± 0.045 | 0.452 ± 0.178 | 0.750 ± 0.091 | 0.406 ± 0.072 | 0.994 ± 0.006 | 0.761 ± 0.168 | 0.404 ± 0.072 | 0.850 ± 0.098 | 0.934 ± 0.051 | 0.371 ± 0.170 | 0.793 ± 0.092 |
| openmax | 0.799 ± 0.081 | 0.724 ± 0.114 | 0.700 ± 0.096 | 0.930 ± 0.065 | 0.610 ± 0.227 | 0.648 ± 0.074 | 0.440 ± 0.109 | 0.967 ± 0.033 | 0.808 ± 0.159 | 0.425 ± 0.104 | 0.728 ± 0.094 | 0.937 ± 0.040 | 0.616 ± 0.172 | 0.683 ± 0.099 |
| static | 0.860 ± 0.068 | 0.755 ± 0.084 | 0.752 ± 0.079 | 0.948 ± 0.043 | 0.568 ± 0.149 | 0.711 ± 0.072 | 0.356 ± 0.125 | 0.999 ± 0.003 | 0.946 ± 0.080 | 0.355 ± 0.124 | 0.898 ± 0.109 | 0.895 ± 0.065 | 0.284 ± 0.271 | 0.801 ± 0.095 |
| knn | 0.897 ± 0.059 | 0.814 ± 0.087 | 0.625 ± 0.091 | 0.987 ± 0.022 | 0.857 ± 0.095 | 0.617 ± 0.088 | 0.592 ± 0.129 | 0.991 ± 0.011 | 0.831 ± 0.163 | 0.586 ± 0.126 | 0.781 ± 0.095 | 0.965 ± 0.033 | 0.634 ± 0.201 | 0.754 ± 0.091 |
| temp_rff_p1 | 0.859 ± 0.064 | 0.768 ± 0.085 | 0.773 ± 0.057 | 0.935 ± 0.053 | 0.573 ± 0.153 | 0.722 ± 0.059 | 0.364 ± 0.110 | 0.998 ± 0.004 | 0.942 ± 0.079 | 0.363 ± 0.110 | 0.908 ± 0.086 | 0.893 ± 0.059 | 0.282 ± 0.235 | 0.810 ± 0.086 |
| static_window | 0.860 ± 0.068 | 0.755 ± 0.084 | 0.752 ± 0.079 | 0.948 ± 0.043 | 0.568 ± 0.149 | 0.711 ± 0.072 | 0.356 ± 0.125 | 0.999 ± 0.003 | 0.946 ± 0.080 | 0.355 ± 0.124 | 0.898 ± 0.109 | 0.895 ± 0.065 | 0.283 ± 0.271 | 0.801 ± 0.095 |
| knn_window | 0.897 ± 0.059 | 0.814 ± 0.087 | 0.626 ± 0.091 | 0.987 ± 0.022 | 0.858 ± 0.096 | 0.618 ± 0.087 | 0.591 ± 0.129 | 0.991 ± 0.011 | 0.832 ± 0.164 | 0.585 ± 0.126 | 0.782 ± 0.094 | 0.965 ± 0.032 | 0.634 ± 0.201 | 0.754 ± 0.091 |
| temp_rff | 0.890 ± 0.064 | 0.806 ± 0.089 | 0.677 ± 0.107 | 0.978 ± 0.024 | 0.756 ± 0.148 | 0.661 ± 0.104 | 0.494 ± 0.184 | 0.993 ± 0.010 | 0.888 ± 0.153 | 0.490 ± 0.182 | 0.798 ± 0.125 | 0.947 ± 0.054 | 0.551 ± 0.305 | 0.753 ± 0.110 |
| temp_rff_nowt | 0.893 ± 0.060 | 0.805 ± 0.091 | 0.656 ± 0.095 | 0.985 ± 0.018 | 0.818 ± 0.133 | 0.645 ± 0.091 | 0.546 ± 0.132 | 0.993 ± 0.009 | 0.860 ± 0.162 | 0.541 ± 0.130 | 0.775 ± 0.118 | 0.966 ± 0.028 | 0.608 ± 0.273 | 0.747 ± 0.108 |
| probe_default | 0.880 ± 0.063 | 0.645 ± 0.145 | 0.789 ± 0.091 | 0.954 ± 0.044 | 0.466 ± 0.177 | 0.752 ± 0.091 | 0.411 ± 0.088 | 0.992 ± 0.010 | 0.756 ± 0.169 | 0.408 ± 0.087 | 0.834 ± 0.104 | 0.939 ± 0.052 | 0.406 ± 0.174 | 0.781 ± 0.096 |
| knn_default | 0.896 ± 0.059 | 0.816 ± 0.088 | 0.622 ± 0.083 | 0.988 ± 0.021 | 0.863 ± 0.092 | 0.614 ± 0.080 | 0.599 ± 0.133 | 0.991 ± 0.011 | 0.829 ± 0.165 | 0.593 ± 0.130 | 0.750 ± 0.092 | 0.971 ± 0.030 | 0.690 ± 0.212 | 0.727 ± 0.087 |
| openmax_default | 0.812 ± 0.079 | 0.707 ± 0.162 | 0.690 ± 0.073 | 0.948 ± 0.043 | 0.661 ± 0.167 | 0.654 ± 0.069 | 0.308 ± 0.042 | 0.959 ± 0.045 | 0.872 ± 0.153 | 0.295 ± 0.042 | 0.504 ± 0.111 | 0.965 ± 0.029 | 0.799 ± 0.167 | 0.485 ± 0.103 |
| temp_rff_default | 0.906 ± 0.054 | 0.825 ± 0.086 | 0.599 ± 0.075 | 0.989 ± 0.017 | 0.881 ± 0.083 | 0.592 ± 0.070 | 0.618 ± 0.134 | 0.991 ± 0.010 | 0.817 ± 0.173 | 0.613 ± 0.132 | 0.693 ± 0.110 | 0.983 ± 0.016 | 0.789 ± 0.165 | 0.681 ± 0.110 |
| raw_centroid | 0.853 ± 0.065 | 0.648 ± 0.108 | 0.672 ± 0.073 | 0.954 ± 0.037 | 0.507 ± 0.209 | 0.640 ± 0.068 | 0.398 ± 0.062 | 0.980 ± 0.011 | 0.793 ± 0.154 | 0.390 ± 0.061 | 0.466 ± 0.073 | 0.980 ± 0.011 | 0.720 ± 0.212 | 0.457 ± 0.074 |
| raw_centroid_cfo | 0.853 ± 0.065 | 0.648 ± 0.108 | 0.672 ± 0.073 | 0.954 ± 0.037 | 0.507 ± 0.209 | 0.640 ± 0.068 | 0.398 ± 0.062 | 0.980 ± 0.011 | 0.793 ± 0.154 | 0.390 ± 0.061 | 0.466 ± 0.073 | 0.980 ± 0.011 | 0.720 ± 0.212 | 0.457 ± 0.074 |

Tuned calibration: method chosen on Part B, count over runs: msp: q80×1/q90×1/youden×10, probe: q95×4/youden×8, openmax: q90×1/q95×1/youden×10, static: youden×12, knn: youden×12, temp_rff_p1: q95×2/youden×10, static_window: youden×12, knn_window: youden×12, temp_rff: q90×1/q95×2/youden×9, temp_rff_nowt: q90×1/q95×1/youden×10, probe_default: q80×1/q90×1/q95×1/youden×9, knn_default: q95×1/youden×11, openmax_default: q80×2/youden×10, temp_rff_default: q90×1/q95×1/youden×10, raw_centroid: q80×5/youden×7, raw_centroid_cfo: q80×5/youden×7

## Per device split (mean over seeds)

| Rule | shift 0 closed / AUROC | shift 4 closed / AUROC | shift 8 closed / AUROC | shift 12 closed / AUROC |
|---|---|---|---|---|
| softmax | 0.787 / – | 0.795 / – | 0.862 / – | 0.929 / – |
| msp | 0.787 / 0.695 | 0.795 / 0.700 | 0.862 / 0.554 | 0.929 / 0.788 |
| probe | 0.867 / 0.694 | 0.812 / 0.663 | 0.891 / 0.479 | 0.937 / 0.746 |
| openmax | 0.744 / 0.760 | 0.779 / 0.709 | 0.764 / 0.617 | 0.908 / 0.809 |
| static | 0.837 / 0.779 | 0.801 / 0.718 | 0.871 / 0.692 | 0.930 / 0.830 |
| static_window | 0.837 / 0.779 | 0.801 / 0.718 | 0.871 / 0.692 | 0.930 / 0.830 |
| knn | 0.893 / 0.869 | 0.829 / 0.772 | 0.914 / 0.738 | 0.953 / 0.877 |
| knn_window | 0.894 / 0.870 | 0.829 / 0.772 | 0.914 / 0.738 | 0.953 / 0.877 |
| temp_rff_p1 | 0.834 / 0.833 | 0.800 / 0.715 | 0.871 / 0.692 | 0.930 / 0.830 |
| temp_rff | 0.878 / 0.840 | 0.827 / 0.777 | 0.922 / 0.751 | 0.934 / 0.854 |
| raw_centroid_cfo | 0.810 / 0.737 | 0.793 / 0.481 | 0.857 / 0.728 | 0.952 / 0.646 |

## PRIMARY test: TEMP-RFF minus rule at the split level

Seeds are averaged within each device split first (the seeds of a split share its devices, so they are not independent); tests are over the splits. Outcomes: closed accuracy, AUROC, tuned-protocol known ok and unknown detection. Holm correction over one family: 84 tests: 7 rules ['msp', 'probe', 'openmax', 'static', 'knn', 'static_window', 'knn_window'] x 4 outcomes x tags ['wisig', 'pycom', 'lora'].

| Rule | outcome | mean diff | sd diff | splits won | t p | t p (Holm) | Wilcoxon p | Wilcoxon p (Holm) |
|---|---|---|---|---|---|---|---|---|
| knn_window | auroc | -0.009 | 0.021 | 2/4 | 0.464 | 1 | 0.625 | 1 |
| knn_window | closed | -0.007 | 0.012 | 1/4 | 0.35 | 1 | 0.375 | 1 |
| knn_window | tuned_known_ok | -0.000 | 0.025 | 1/4 | 0.987 | 1 | 0.875 | 1 |
| knn_window | tuned_unk_det | -0.083 | 0.144 | 1/4 | 0.331 | 1 | 0.375 | 1 |
| knn | auroc | -0.009 | 0.021 | 2/4 | 0.471 | 1 | 0.625 | 1 |
| knn | closed | -0.007 | 0.012 | 1/4 | 0.353 | 1 | 0.375 | 1 |
| knn | tuned_known_ok | -0.000 | 0.025 | 1/4 | 0.998 | 1 | 0.875 | 1 |
| knn | tuned_unk_det | -0.083 | 0.144 | 1/4 | 0.332 | 1 | 0.375 | 1 |
| msp | auroc | +0.122 | 0.062 | 4/4 | 0.0292 | 1 | 0.125 | 1 |
| msp | closed | +0.047 | 0.037 | 4/4 | 0.0824 | 1 | 0.125 | 1 |
| msp | tuned_known_ok | -0.011 | 0.049 | 2/4 | 0.69 | 1 | 0.875 | 1 |
| msp | tuned_unk_det | +0.187 | 0.164 | 3/4 | 0.108 | 1 | 0.25 | 1 |
| openmax | auroc | +0.082 | 0.038 | 4/4 | 0.0232 | 1 | 0.125 | 1 |
| openmax | closed | +0.092 | 0.065 | 4/4 | 0.0655 | 1 | 0.125 | 1 |
| openmax | tuned_known_ok | +0.071 | 0.029 | 4/4 | 0.0165 | 1 | 0.125 | 1 |
| openmax | tuned_unk_det | -0.065 | 0.169 | 2/4 | 0.496 | 1 | 0.875 | 1 |
| probe | auroc | +0.160 | 0.077 | 4/4 | 0.025 | 1 | 0.125 | 1 |
| probe | closed | +0.014 | 0.014 | 3/4 | 0.138 | 1 | 0.25 | 1 |
| probe | tuned_known_ok | -0.039 | 0.045 | 1/4 | 0.175 | 1 | 0.25 | 1 |
| probe | tuned_unk_det | +0.180 | 0.187 | 3/4 | 0.151 | 1 | 0.25 | 1 |
| static_window | auroc | +0.051 | 0.018 | 4/4 | 0.0112 | 0.93 | 0.125 | 1 |
| static_window | closed | +0.031 | 0.021 | 4/4 | 0.0586 | 1 | 0.125 | 1 |
| static_window | tuned_known_ok | -0.048 | 0.044 | 0/4 | 0.118 | 1 | 0.125 | 1 |
| static_window | tuned_unk_det | +0.267 | 0.146 | 4/4 | 0.0355 | 1 | 0.125 | 1 |
| static | auroc | +0.051 | 0.018 | 4/4 | 0.0112 | 0.93 | 0.125 | 1 |
| static | closed | +0.031 | 0.021 | 4/4 | 0.0586 | 1 | 0.125 | 1 |
| static | tuned_known_ok | -0.048 | 0.044 | 0/4 | 0.118 | 1 | 0.125 | 1 |
| static | tuned_unk_det | +0.267 | 0.146 | 4/4 | 0.0355 | 1 | 0.125 | 1 |

## Secondary (exploratory): TEMP-RFF minus other rules at the split level, uncorrected

| Rule | outcome | mean diff | sd diff | splits won | t p | Wilcoxon p |
|---|---|---|---|---|---|---|
| knn_default | auroc | -0.011 | 0.022 | 2/4 | 0.389 | 0.625 |
| knn_default | closed | -0.006 | 0.014 | 1/4 | 0.452 | 0.625 |
| knn_default | tuned_known_ok | +0.027 | 0.042 | 3/4 | 0.295 | 0.375 |
| knn_default | tuned_unk_det | -0.139 | 0.127 | 0/4 | 0.116 | 0.125 |
| openmax_default | auroc | +0.098 | 0.102 | 4/4 | 0.15 | 0.125 |
| openmax_default | closed | +0.078 | 0.063 | 4/4 | 0.0908 | 0.125 |
| openmax_default | tuned_known_ok | +0.268 | 0.069 | 4/4 | 0.00437 | 0.125 |
| openmax_default | tuned_unk_det | -0.248 | 0.219 | 1/4 | 0.108 | 0.25 |
| probe_default | auroc | +0.160 | 0.082 | 4/4 | 0.0295 | 0.125 |
| probe_default | closed | +0.010 | 0.013 | 3/4 | 0.214 | 0.25 |
| probe_default | tuned_known_ok | -0.028 | 0.029 | 1/4 | 0.151 | 0.25 |
| probe_default | tuned_unk_det | +0.144 | 0.176 | 3/4 | 0.199 | 0.25 |
| raw_centroid_cfo | auroc | +0.158 | 0.119 | 4/4 | 0.0775 | 0.125 |
| raw_centroid_cfo | closed | +0.038 | 0.040 | 3/4 | 0.159 | 0.25 |
| raw_centroid_cfo | tuned_known_ok | +0.297 | 0.138 | 4/4 | 0.0229 | 0.125 |
| raw_centroid_cfo | tuned_unk_det | -0.170 | 0.186 | 0/4 | 0.165 | 0.125 |
| raw_centroid | auroc | +0.158 | 0.119 | 4/4 | 0.0775 | 0.125 |
| raw_centroid | closed | +0.038 | 0.040 | 3/4 | 0.159 | 0.25 |
| raw_centroid | tuned_known_ok | +0.297 | 0.138 | 4/4 | 0.0229 | 0.125 |
| raw_centroid | tuned_unk_det | -0.170 | 0.186 | 0/4 | 0.165 | 0.125 |
| softmax | closed | +0.047 | 0.037 | 4/4 | 0.0824 | 0.125 |
| temp_rff_default | auroc | -0.019 | 0.020 | 0/4 | 0.156 | 0.25 |
| temp_rff_default | closed | -0.015 | 0.011 | 0/4 | 0.0651 | 0.25 |
| temp_rff_default | tuned_known_ok | +0.072 | 0.058 | 3/4 | 0.087 | 0.25 |
| temp_rff_default | tuned_unk_det | -0.239 | 0.193 | 0/4 | 0.0892 | 0.25 |
| temp_rff_nowt | auroc | +0.001 | 0.015 | 2/4 | 0.909 | 1 |
| temp_rff_nowt | closed | -0.003 | 0.003 | 0/4 | 0.127 | 0.25 |
| temp_rff_nowt | tuned_known_ok | +0.006 | 0.010 | 2/4 | 0.322 | 0.5 |
| temp_rff_nowt | tuned_unk_det | -0.057 | 0.054 | 0/4 | 0.126 | 0.25 |
| temp_rff_p1 | auroc | +0.038 | 0.027 | 4/4 | 0.0677 | 0.125 |
| temp_rff_p1 | closed | +0.032 | 0.021 | 4/4 | 0.0569 | 0.125 |
| temp_rff_p1 | tuned_known_ok | -0.057 | 0.043 | 0/4 | 0.0762 | 0.125 |
| temp_rff_p1 | tuned_unk_det | +0.268 | 0.114 | 4/4 | 0.0183 | 0.125 |

With 4 splits the smallest possible two-sided Wilcoxon p is 0.125; report effect sizes and wins, not only p-values.


## Secondary test: TEMP-RFF minus rule per run (same split and seed; not independent)

| Rule | outcome | mean diff | wins | paired t p | Wilcoxon p |
|---|---|---|---|---|---|
| softmax | closed | +0.047 | 11/12 | 0.00152 | 0.00146 |
| msp | closed | +0.047 | 11/12 | 0.00152 | 0.00146 |
| msp | auroc | +0.122 | 12/12 | 0.000158 | 0.000488 |
| msp | tuned_known_ok | -0.011 | 6/12 | 0.597 | 0.733 |
| msp | tuned_unk_det | +0.187 | 9/12 | 0.0421 | 0.0425 |
| probe | closed | +0.014 | 8/12 | 0.0237 | 0.0522 |
| probe | auroc | +0.160 | 12/12 | 9.7e-05 | 0.000488 |
| probe | tuned_known_ok | -0.039 | 6/12 | 0.161 | 0.339 |
| probe | tuned_unk_det | +0.180 | 8/12 | 0.0641 | 0.0771 |
| openmax | closed | +0.092 | 11/12 | 0.000881 | 0.000977 |
| openmax | auroc | +0.082 | 11/12 | 0.000512 | 0.000977 |
| openmax | tuned_known_ok | +0.071 | 10/12 | 0.00114 | 0.00244 |
| openmax | tuned_unk_det | -0.065 | 6/12 | 0.329 | 0.339 |
| static | closed | +0.031 | 11/12 | 0.00117 | 0.000977 |
| static | auroc | +0.051 | 12/12 | 0.000159 | 0.000488 |
| static | tuned_known_ok | -0.048 | 1/12 | 0.0349 | 0.00195 |
| static | tuned_unk_det | +0.267 | 11/12 | 0.00141 | 0.000977 |
| knn | closed | -0.007 | 4/12 | 0.24 | 0.365 |
| knn | auroc | -0.009 | 6/12 | 0.428 | 0.91 |
| knn | tuned_known_ok | -0.000 | 6/12 | 0.998 | 1 |
| knn | tuned_unk_det | -0.083 | 5/12 | 0.152 | 0.206 |
| temp_rff_p1 | closed | +0.032 | 11/12 | 0.00124 | 0.000977 |
| temp_rff_p1 | auroc | +0.038 | 10/12 | 0.02 | 0.0322 |
| temp_rff_p1 | tuned_known_ok | -0.057 | 1/12 | 0.0249 | 0.0186 |
| temp_rff_p1 | tuned_unk_det | +0.268 | 10/12 | 0.00208 | 0.00195 |
| static_window | closed | +0.031 | 11/12 | 0.00117 | 0.000977 |
| static_window | auroc | +0.051 | 12/12 | 0.000159 | 0.000488 |
| static_window | tuned_known_ok | -0.048 | 1/12 | 0.0349 | 0.00195 |
| static_window | tuned_unk_det | +0.267 | 11/12 | 0.00141 | 0.000977 |
| knn_window | closed | -0.007 | 3/12 | 0.232 | 0.375 |
| knn_window | auroc | -0.009 | 5/12 | 0.412 | 0.791 |
| knn_window | tuned_known_ok | -0.000 | 5/12 | 0.983 | 0.922 |
| knn_window | tuned_unk_det | -0.083 | 4/12 | 0.151 | 0.16 |
| temp_rff_nowt | closed | -0.003 | 1/12 | 0.116 | 0.125 |
| temp_rff_nowt | auroc | +0.001 | 3/12 | 0.885 | 0.812 |
| temp_rff_nowt | tuned_known_ok | +0.006 | 4/12 | 0.121 | 0.188 |
| temp_rff_nowt | tuned_unk_det | -0.057 | 0/12 | 0.0689 | 0.0625 |
| probe_default | closed | +0.010 | 8/12 | 0.0369 | 0.064 |
| probe_default | auroc | +0.160 | 12/12 | 9.86e-05 | 0.000488 |
| probe_default | tuned_known_ok | -0.028 | 7/12 | 0.321 | 0.569 |
| probe_default | tuned_unk_det | +0.144 | 8/12 | 0.104 | 0.129 |
| knn_default | closed | -0.006 | 6/12 | 0.315 | 0.636 |
| knn_default | auroc | -0.011 | 7/12 | 0.343 | 0.733 |
| knn_default | tuned_known_ok | +0.027 | 8/12 | 0.112 | 0.266 |
| knn_default | tuned_unk_det | -0.139 | 4/12 | 0.0185 | 0.0425 |
| openmax_default | closed | +0.078 | 11/12 | 0.00203 | 0.00146 |
| openmax_default | auroc | +0.098 | 9/12 | 0.00914 | 0.00928 |
| openmax_default | tuned_known_ok | +0.268 | 12/12 | 1.62e-06 | 0.000488 |
| openmax_default | tuned_unk_det | -0.248 | 3/12 | 0.0149 | 0.0522 |
| temp_rff_default | closed | -0.015 | 0/12 | 0.0359 | 0.0156 |
| temp_rff_default | auroc | -0.019 | 1/12 | 0.114 | 0.0469 |
| temp_rff_default | tuned_known_ok | +0.072 | 7/12 | 0.0208 | 0.0156 |
| temp_rff_default | tuned_unk_det | -0.239 | 0/12 | 0.0151 | 0.0156 |
| raw_centroid | closed | +0.038 | 9/12 | 0.0508 | 0.064 |
| raw_centroid | auroc | +0.158 | 10/12 | 0.00164 | 0.00244 |
| raw_centroid | tuned_known_ok | +0.297 | 12/12 | 2.13e-05 | 0.000488 |
| raw_centroid | tuned_unk_det | -0.170 | 3/12 | 0.0845 | 0.0923 |
| raw_centroid_cfo | closed | +0.038 | 9/12 | 0.0508 | 0.064 |
| raw_centroid_cfo | auroc | +0.158 | 10/12 | 0.00164 | 0.00244 |
| raw_centroid_cfo | tuned_known_ok | +0.297 | 12/12 | 2.13e-05 | 0.000488 |
| raw_centroid_cfo | tuned_unk_det | -0.170 | 3/12 | 0.0845 | 0.0923 |
## Forced prototype window (decay 1)

| W | Closed acc. | AUROC | tuned accept | tuned unk. det. |
|---|---|---|---|---|
| 1 | 0.890 ± 0.064 | 0.806 ± 0.090 | 0.796 ± 0.127 | 0.551 ± 0.305 |
| 2 | 0.882 ± 0.062 | 0.803 ± 0.087 | 0.810 ± 0.115 | 0.525 ± 0.290 |
| 3 | 0.878 ± 0.062 | 0.781 ± 0.098 | 0.772 ± 0.119 | 0.557 ± 0.295 |

## Few-shot target-session enrollment (same support for every rule)

| k requested | k effective | temp_rff closed / AUROC / tuned known ok | static closed / AUROC / tuned known ok | knn closed / AUROC / tuned known ok | static_window closed / AUROC / tuned known ok | knn_window closed / AUROC / tuned known ok |
|---|---|---|---|---|---|---|
| 1 | 1 | 0.928 / 0.825 / 0.788 | 0.857 / 0.752 / 0.799 | 0.902 / 0.818 / 0.752 | 0.857 / 0.752 / 0.799 | 0.902 / 0.819 / 0.752 |
| 5 | 5 | 0.949 / 0.859 / 0.824 | 0.857 / 0.752 / 0.800 | 0.937 / 0.834 / 0.773 | 0.857 / 0.752 / 0.800 | 0.937 / 0.834 / 0.775 |
| 10 | 10 | 0.953 / 0.855 / 0.823 | 0.858 / 0.753 / 0.802 | 0.950 / 0.846 / 0.790 | 0.858 / 0.753 / 0.802 | 0.950 / 0.846 / 0.791 |
| 20 | 20 | 0.959 / 0.863 / 0.840 | 0.859 / 0.755 / 0.804 | 0.961 / 0.858 / 0.799 | 0.859 / 0.755 / 0.804 | 0.962 / 0.858 / 0.801 |
| 50 | 49 | 0.960 / 0.864 / 0.845 | 0.862 / 0.761 / 0.808 | 0.973 / 0.874 / 0.821 | 0.862 / 0.761 / 0.808 | 0.973 / 0.875 / 0.825 |
| 100 | 97 | 0.963 / 0.873 / 0.847 | 0.865 / 0.770 / 0.809 | 0.979 / 0.887 / 0.850 | 0.865 / 0.770 / 0.809 | 0.979 / 0.888 / 0.854 |
| 200 | 191 | 0.965 / 0.873 / 0.859 | 0.871 / 0.781 / 0.822 | 0.981 / 0.899 / 0.876 | 0.871 / 0.781 / 0.822 | 0.981 / 0.900 / 0.878 |
| 500 | 377 | 0.965 / 0.876 / 0.866 | 0.882 / 0.791 / 0.833 | 0.983 / 0.909 / 0.894 | 0.882 / 0.791 / 0.833 | 0.983 / 0.909 / 0.896 |

## Pipeline stages (Part-A centroids)

| Stage | Softmax acc. | Centroid acc. | AUROC |
|---|---|---|---|
| phase1 | 0.678 ± 0.055 | 0.739 ± 0.065 | 0.654 ± 0.120 |
| phase2 | 0.847 ± 0.067 | 0.866 ± 0.061 | 0.760 ± 0.076 |
| phase3 | 0.843 ± 0.073 | 0.860 ± 0.068 | 0.755 ± 0.084 |

Joint-training control vs sequential Phase 2 (paired, split 0; joint runs have no audit):

| metric | joint | Phase 2 | mean diff | wins |
|---|---|---|---|---|
| softmax | 0.828 ± 0.048 | 0.809 ± 0.047 | +0.019 | 2/3 |
| centroid | 0.872 ± 0.056 | 0.852 ± 0.041 | +0.021 | 2/3 |
| auroc | 0.775 ± 0.065 | 0.790 ± 0.078 | -0.015 | 1/3 |

## Selected settings over runs

- W: 1×11, 3×1
- decay: 1.0×12
- P: 1×1, 20×1, 5×4, all×6
- knn (k, weighted): (1, False)×1, (10, False)×3, (10, True)×2, (5, False)×5, (5, True)×1
- openmax (dist, tail, alpha): ('euclidean', 10, 1)×10, ('euclidean', 10, 2)×1, ('eucos', 10, 1)×1
- probe C: 0.01×3, 1.0×8, 10.0×1
- static_window W: (1,)×12
- knn_window W (k, weighted): (1, 1, False)×1, (1, 10, False)×3, (1, 10, True)×2, (1, 5, False)×4, (1, 5, True)×1, (3, 1, False)×1
- temp_rff_p1 W decay: (1, 1.0)×8, (2, 0.8)×1, (2, 1.0)×2, (3, 1.0)×1
- selection-weight sensitivity: choice changes with the weight (0.3/0.5/0.7) in 7 of 12 runs
