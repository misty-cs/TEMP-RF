# Summary: wisig — COMPLETE

Audited runs used: 20 (splits [0, 4, 8, 12], seeds [7, 13, 21, 42, 100]); expected splits [0, 4, 8, 12] x seeds [42, 7, 13, 21, 100]

## All rules (mean ± std over runs)

| Rule | Closed acc. | AUROC | fixed90 accept | fixed90 open | fixed90 unk. det. | fixed90 known ok | objective accept | objective open | objective unk. det. | objective known ok | tuned accept | tuned open | tuned unk. det. | tuned known ok |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| softmax | 0.844 ± 0.066 | – | – | – | – | – | – | – | – | – | – | – | – | – |
| msp | 0.844 ± 0.066 | 0.676 ± 0.094 | 0.848 ± 0.054 | 0.897 ± 0.064 | 0.369 ± 0.164 | 0.760 ± 0.064 | 0.489 ± 0.083 | 0.944 ± 0.065 | 0.749 ± 0.147 | 0.461 ± 0.083 | 0.845 ± 0.089 | 0.902 ± 0.058 | 0.364 ± 0.155 | 0.762 ± 0.096 |
| probe | 0.886 ± 0.058 | 0.652 ± 0.125 | 0.799 ± 0.074 | 0.954 ± 0.041 | 0.443 ± 0.155 | 0.762 ± 0.076 | 0.403 ± 0.069 | 0.994 ± 0.006 | 0.778 ± 0.161 | 0.401 ± 0.070 | 0.832 ± 0.097 | 0.944 ± 0.046 | 0.397 ± 0.170 | 0.784 ± 0.090 |
| openmax | 0.800 ± 0.086 | 0.698 ± 0.099 | 0.694 ± 0.085 | 0.928 ± 0.067 | 0.580 ± 0.210 | 0.641 ± 0.061 | 0.443 ± 0.101 | 0.966 ± 0.035 | 0.793 ± 0.145 | 0.427 ± 0.098 | 0.717 ± 0.086 | 0.940 ± 0.041 | 0.591 ± 0.161 | 0.675 ± 0.095 |
| static | 0.869 ± 0.063 | 0.746 ± 0.084 | 0.739 ± 0.071 | 0.953 ± 0.037 | 0.593 ± 0.161 | 0.703 ± 0.064 | 0.356 ± 0.110 | 0.996 ± 0.010 | 0.926 ± 0.079 | 0.355 ± 0.110 | 0.880 ± 0.101 | 0.909 ± 0.056 | 0.328 ± 0.234 | 0.798 ± 0.089 |
| knn | 0.906 ± 0.057 | 0.802 ± 0.074 | 0.619 ± 0.081 | 0.987 ± 0.019 | 0.834 ± 0.086 | 0.610 ± 0.079 | 0.608 ± 0.112 | 0.990 ± 0.011 | 0.816 ± 0.135 | 0.602 ± 0.111 | 0.752 ± 0.109 | 0.972 ± 0.028 | 0.655 ± 0.174 | 0.730 ± 0.105 |
| temp_rff_p1 | 0.868 ± 0.061 | 0.755 ± 0.084 | 0.755 ± 0.058 | 0.946 ± 0.045 | 0.596 ± 0.163 | 0.713 ± 0.053 | 0.371 ± 0.108 | 0.994 ± 0.013 | 0.924 ± 0.077 | 0.368 ± 0.107 | 0.891 ± 0.083 | 0.908 ± 0.053 | 0.316 ± 0.207 | 0.808 ± 0.080 |
| static_window | 0.869 ± 0.063 | 0.746 ± 0.084 | 0.739 ± 0.071 | 0.953 ± 0.037 | 0.593 ± 0.161 | 0.703 ± 0.064 | 0.356 ± 0.110 | 0.996 ± 0.010 | 0.926 ± 0.079 | 0.355 ± 0.110 | 0.880 ± 0.101 | 0.909 ± 0.056 | 0.328 ± 0.234 | 0.798 ± 0.089 |
| knn_window | 0.906 ± 0.057 | 0.802 ± 0.075 | 0.619 ± 0.081 | 0.987 ± 0.019 | 0.834 ± 0.086 | 0.611 ± 0.079 | 0.608 ± 0.112 | 0.990 ± 0.011 | 0.816 ± 0.136 | 0.602 ± 0.111 | 0.752 ± 0.109 | 0.972 ± 0.028 | 0.655 ± 0.174 | 0.730 ± 0.105 |
| temp_rff | 0.899 ± 0.059 | 0.799 ± 0.079 | 0.651 ± 0.102 | 0.980 ± 0.021 | 0.768 ± 0.143 | 0.638 ± 0.098 | 0.547 ± 0.184 | 0.992 ± 0.010 | 0.850 ± 0.137 | 0.543 ± 0.182 | 0.764 ± 0.123 | 0.960 ± 0.046 | 0.602 ± 0.271 | 0.731 ± 0.108 |
| temp_rff_nowt | 0.901 ± 0.056 | 0.798 ± 0.081 | 0.633 ± 0.088 | 0.986 ± 0.016 | 0.818 ± 0.110 | 0.624 ± 0.085 | 0.583 ± 0.140 | 0.991 ± 0.010 | 0.833 ± 0.138 | 0.578 ± 0.139 | 0.745 ± 0.113 | 0.972 ± 0.025 | 0.651 ± 0.229 | 0.724 ± 0.105 |
| probe_default | 0.888 ± 0.059 | 0.652 ± 0.126 | 0.801 ± 0.076 | 0.956 ± 0.041 | 0.437 ± 0.162 | 0.766 ± 0.078 | 0.411 ± 0.082 | 0.993 ± 0.009 | 0.775 ± 0.165 | 0.409 ± 0.082 | 0.826 ± 0.093 | 0.948 ± 0.047 | 0.409 ± 0.192 | 0.782 ± 0.087 |
| knn_default | 0.905 ± 0.057 | 0.804 ± 0.076 | 0.621 ± 0.076 | 0.987 ± 0.019 | 0.835 ± 0.087 | 0.613 ± 0.074 | 0.605 ± 0.120 | 0.990 ± 0.010 | 0.817 ± 0.137 | 0.599 ± 0.119 | 0.736 ± 0.100 | 0.975 ± 0.025 | 0.694 ± 0.176 | 0.717 ± 0.097 |
| openmax_default | 0.818 ± 0.076 | 0.701 ± 0.133 | 0.685 ± 0.070 | 0.946 ± 0.050 | 0.648 ± 0.156 | 0.646 ± 0.061 | 0.313 ± 0.051 | 0.962 ± 0.044 | 0.875 ± 0.151 | 0.302 ± 0.053 | 0.512 ± 0.094 | 0.963 ± 0.032 | 0.782 ± 0.148 | 0.493 ± 0.090 |
| temp_rff_default | 0.906 ± 0.054 | 0.813 ± 0.076 | 0.597 ± 0.070 | 0.988 ± 0.015 | 0.861 ± 0.077 | 0.590 ± 0.068 | 0.638 ± 0.125 | 0.989 ± 0.011 | 0.802 ± 0.141 | 0.632 ± 0.124 | 0.689 ± 0.105 | 0.983 ± 0.015 | 0.775 ± 0.146 | 0.678 ± 0.105 |
| raw_centroid | 0.851 ± 0.063 | 0.649 ± 0.104 | 0.663 ± 0.068 | 0.957 ± 0.032 | 0.518 ± 0.203 | 0.635 ± 0.067 | 0.399 ± 0.064 | 0.982 ± 0.010 | 0.793 ± 0.149 | 0.392 ± 0.063 | 0.461 ± 0.075 | 0.982 ± 0.010 | 0.733 ± 0.200 | 0.452 ± 0.075 |
| raw_centroid_cfo | 0.851 ± 0.063 | 0.649 ± 0.104 | 0.663 ± 0.068 | 0.957 ± 0.032 | 0.518 ± 0.203 | 0.635 ± 0.067 | 0.399 ± 0.064 | 0.982 ± 0.010 | 0.793 ± 0.149 | 0.392 ± 0.063 | 0.461 ± 0.075 | 0.982 ± 0.010 | 0.733 ± 0.200 | 0.452 ± 0.075 |

Tuned calibration: method chosen on Part B, count over runs: msp: q80×2/q90×4/q95×1/youden×13, probe: q80×1/q90×1/q95×5/youden×13, openmax: q80×1/q90×1/q95×1/youden×17, static: q95×2/youden×18, knn: q90×1/q95×2/youden×17, temp_rff_p1: q95×5/youden×15, static_window: q95×2/youden×18, knn_window: q90×1/q95×2/youden×17, temp_rff: q90×2/q95×3/youden×15, temp_rff_nowt: q90×2/q95×3/youden×15, probe_default: q80×1/q90×2/q95×1/youden×16, knn_default: q90×1/q95×3/youden×16, openmax_default: q80×2/youden×18, temp_rff_default: q90×2/q95×2/youden×16, raw_centroid: q80×6/youden×14, raw_centroid_cfo: q80×6/youden×14

## Per device split (mean over seeds)

| Rule | shift 0 closed / AUROC | shift 4 closed / AUROC | shift 8 closed / AUROC | shift 12 closed / AUROC |
|---|---|---|---|---|
| softmax | 0.805 / – | 0.789 / – | 0.873 / – | 0.909 / – |
| msp | 0.805 / 0.670 | 0.789 / 0.679 | 0.873 / 0.584 | 0.909 / 0.773 |
| probe | 0.894 / 0.680 | 0.821 / 0.681 | 0.908 / 0.516 | 0.921 / 0.732 |
| openmax | 0.718 / 0.725 | 0.783 / 0.682 | 0.799 / 0.629 | 0.900 / 0.757 |
| static | 0.863 / 0.748 | 0.809 / 0.683 | 0.894 / 0.732 | 0.908 / 0.822 |
| static_window | 0.863 / 0.748 | 0.809 / 0.683 | 0.894 / 0.732 | 0.908 / 0.822 |
| knn | 0.911 / 0.842 | 0.838 / 0.747 | 0.922 / 0.757 | 0.953 / 0.860 |
| knn_window | 0.912 / 0.843 | 0.838 / 0.747 | 0.922 / 0.757 | 0.953 / 0.860 |
| temp_rff_p1 | 0.860 / 0.787 | 0.809 / 0.681 | 0.894 / 0.732 | 0.908 / 0.822 |
| temp_rff | 0.901 / 0.827 | 0.833 / 0.743 | 0.914 / 0.769 | 0.946 / 0.856 |
| raw_centroid_cfo | 0.810 / 0.737 | 0.792 / 0.484 | 0.854 / 0.729 | 0.949 / 0.647 |

## PRIMARY test: TEMP-RFF minus rule at the split level

Seeds are averaged within each device split first (the seeds of a split share its devices, so they are not independent); tests are over the splits. Outcomes: closed accuracy, AUROC, tuned-protocol known ok and unknown detection. Holm correction over one family: 84 tests: 7 rules ['msp', 'probe', 'openmax', 'static', 'knn', 'static_window', 'knn_window'] x 4 outcomes x tags ['wisig', 'pycom', 'lora'].

| Rule | outcome | mean diff | sd diff | splits won | t p | t p (Holm) | Wilcoxon p | Wilcoxon p (Holm) |
|---|---|---|---|---|---|---|---|---|
| knn_window | auroc | -0.003 | 0.011 | 1/4 | 0.619 | 1 | 0.625 | 1 |
| knn_window | closed | -0.008 | 0.002 | 0/4 | 0.00852 | 0.665 | 0.125 | 1 |
| knn_window | tuned_known_ok | +0.001 | 0.021 | 2/4 | 0.942 | 1 | 0.875 | 1 |
| knn_window | tuned_unk_det | -0.053 | 0.092 | 1/4 | 0.334 | 1 | 0.375 | 1 |
| knn | auroc | -0.003 | 0.011 | 1/4 | 0.63 | 1 | 0.625 | 1 |
| knn | closed | -0.008 | 0.002 | 0/4 | 0.00776 | 0.624 | 0.125 | 1 |
| knn | tuned_known_ok | +0.001 | 0.021 | 2/4 | 0.933 | 1 | 0.875 | 1 |
| knn | tuned_unk_det | -0.053 | 0.092 | 1/4 | 0.335 | 1 | 0.375 | 1 |
| msp | auroc | +0.122 | 0.058 | 4/4 | 0.0243 | 1 | 0.125 | 1 |
| msp | closed | +0.055 | 0.028 | 4/4 | 0.0296 | 1 | 0.125 | 1 |
| msp | tuned_known_ok | -0.031 | 0.020 | 0/4 | 0.0506 | 1 | 0.125 | 1 |
| msp | tuned_unk_det | +0.238 | 0.089 | 4/4 | 0.0126 | 0.96 | 0.125 | 1 |
| openmax | auroc | +0.101 | 0.032 | 4/4 | 0.00819 | 0.647 | 0.125 | 1 |
| openmax | closed | +0.098 | 0.064 | 4/4 | 0.0552 | 1 | 0.125 | 1 |
| openmax | tuned_known_ok | +0.056 | 0.022 | 4/4 | 0.0136 | 0.991 | 0.125 | 1 |
| openmax | tuned_unk_det | +0.011 | 0.064 | 3/4 | 0.748 | 1 | 0.625 | 1 |
| probe | auroc | +0.147 | 0.079 | 4/4 | 0.0342 | 1 | 0.125 | 1 |
| probe | closed | +0.013 | 0.009 | 4/4 | 0.0594 | 1 | 0.125 | 1 |
| probe | tuned_known_ok | -0.053 | 0.031 | 0/4 | 0.0424 | 1 | 0.125 | 1 |
| probe | tuned_unk_det | +0.205 | 0.120 | 4/4 | 0.0421 | 1 | 0.125 | 1 |
| static_window | auroc | +0.053 | 0.021 | 4/4 | 0.0163 | 1 | 0.125 | 1 |
| static_window | closed | +0.030 | 0.009 | 4/4 | 0.00743 | 0.624 | 0.125 | 1 |
| static_window | tuned_known_ok | -0.067 | 0.021 | 0/4 | 0.00745 | 0.624 | 0.125 | 1 |
| static_window | tuned_unk_det | +0.274 | 0.102 | 4/4 | 0.0126 | 0.96 | 0.125 | 1 |
| static | auroc | +0.053 | 0.021 | 4/4 | 0.0163 | 1 | 0.125 | 1 |
| static | closed | +0.030 | 0.009 | 4/4 | 0.00743 | 0.624 | 0.125 | 1 |
| static | tuned_known_ok | -0.067 | 0.021 | 0/4 | 0.00745 | 0.624 | 0.125 | 1 |
| static | tuned_unk_det | +0.274 | 0.102 | 4/4 | 0.0127 | 0.96 | 0.125 | 1 |

## Secondary (exploratory): TEMP-RFF minus other rules at the split level, uncorrected

| Rule | outcome | mean diff | sd diff | splits won | t p | Wilcoxon p |
|---|---|---|---|---|---|---|
| knn_default | auroc | -0.006 | 0.008 | 1/4 | 0.242 | 0.25 |
| knn_default | closed | -0.006 | 0.002 | 0/4 | 0.00334 | 0.125 |
| knn_default | tuned_known_ok | +0.014 | 0.025 | 3/4 | 0.347 | 0.375 |
| knn_default | tuned_unk_det | -0.091 | 0.065 | 0/4 | 0.0675 | 0.125 |
| openmax_default | auroc | +0.098 | 0.086 | 4/4 | 0.106 | 0.125 |
| openmax_default | closed | +0.080 | 0.051 | 4/4 | 0.0524 | 0.125 |
| openmax_default | tuned_known_ok | +0.238 | 0.057 | 4/4 | 0.00366 | 0.125 |
| openmax_default | tuned_unk_det | -0.180 | 0.155 | 1/4 | 0.103 | 0.25 |
| probe_default | auroc | +0.147 | 0.084 | 4/4 | 0.0399 | 0.125 |
| probe_default | closed | +0.010 | 0.007 | 4/4 | 0.07 | 0.125 |
| probe_default | tuned_known_ok | -0.050 | 0.033 | 0/4 | 0.0531 | 0.125 |
| probe_default | tuned_unk_det | +0.193 | 0.077 | 4/4 | 0.0155 | 0.125 |
| raw_centroid_cfo | auroc | +0.149 | 0.102 | 4/4 | 0.0607 | 0.125 |
| raw_centroid_cfo | closed | +0.047 | 0.040 | 3/4 | 0.0971 | 0.25 |
| raw_centroid_cfo | tuned_known_ok | +0.279 | 0.132 | 4/4 | 0.0244 | 0.125 |
| raw_centroid_cfo | tuned_unk_det | -0.131 | 0.181 | 0/4 | 0.245 | 0.125 |
| raw_centroid | auroc | +0.149 | 0.102 | 4/4 | 0.0607 | 0.125 |
| raw_centroid | closed | +0.047 | 0.040 | 3/4 | 0.0971 | 0.25 |
| raw_centroid | tuned_known_ok | +0.279 | 0.132 | 4/4 | 0.0244 | 0.125 |
| raw_centroid | tuned_unk_det | -0.131 | 0.181 | 0/4 | 0.245 | 0.125 |
| softmax | closed | +0.055 | 0.028 | 4/4 | 0.0296 | 0.125 |
| temp_rff_default | auroc | -0.014 | 0.011 | 0/4 | 0.0889 | 0.25 |
| temp_rff_default | closed | -0.007 | 0.007 | 0/4 | 0.132 | 0.25 |
| temp_rff_default | tuned_known_ok | +0.053 | 0.039 | 3/4 | 0.0713 | 0.25 |
| temp_rff_default | tuned_unk_det | -0.173 | 0.137 | 0/4 | 0.0859 | 0.25 |
| temp_rff_nowt | auroc | +0.001 | 0.010 | 2/4 | 0.852 | 0.75 |
| temp_rff_nowt | closed | -0.002 | 0.002 | 0/4 | 0.12 | 0.25 |
| temp_rff_nowt | tuned_known_ok | +0.007 | 0.013 | 2/4 | 0.35 | 0.5 |
| temp_rff_nowt | tuned_unk_det | -0.049 | 0.055 | 0/4 | 0.175 | 0.25 |
| temp_rff_p1 | auroc | +0.043 | 0.013 | 4/4 | 0.00655 | 0.125 |
| temp_rff_p1 | closed | +0.031 | 0.010 | 4/4 | 0.00851 | 0.125 |
| temp_rff_p1 | tuned_known_ok | -0.077 | 0.035 | 0/4 | 0.0218 | 0.125 |
| temp_rff_p1 | tuned_unk_det | +0.286 | 0.091 | 4/4 | 0.00817 | 0.125 |

With 4 splits the smallest possible two-sided Wilcoxon p is 0.125; report effect sizes and wins, not only p-values.


## Secondary test: TEMP-RFF minus rule per run (same split and seed; not independent)

| Rule | outcome | mean diff | wins | paired t p | Wilcoxon p |
|---|---|---|---|---|---|
| softmax | closed | +0.055 | 18/20 | 1.09e-05 | 1.34e-05 |
| msp | closed | +0.055 | 18/20 | 1.09e-05 | 1.34e-05 |
| msp | auroc | +0.122 | 20/20 | 3.19e-07 | 1.91e-06 |
| msp | tuned_known_ok | -0.031 | 8/20 | 0.0645 | 0.0929 |
| msp | tuned_unk_det | +0.238 | 16/20 | 0.000465 | 0.000851 |
| probe | closed | +0.013 | 12/20 | 0.0688 | 0.0532 |
| probe | auroc | +0.147 | 18/20 | 6.54e-06 | 1.34e-05 |
| probe | tuned_known_ok | -0.053 | 6/20 | 0.00839 | 0.0136 |
| probe | tuned_unk_det | +0.205 | 15/20 | 0.00233 | 0.00315 |
| openmax | closed | +0.098 | 18/20 | 2.75e-05 | 9.54e-06 |
| openmax | auroc | +0.101 | 18/20 | 1.61e-06 | 1.34e-05 |
| openmax | tuned_known_ok | +0.056 | 14/20 | 0.00242 | 0.00486 |
| openmax | tuned_unk_det | +0.011 | 12/20 | 0.854 | 0.956 |
| static | closed | +0.030 | 18/20 | 0.00124 | 0.00148 |
| static | auroc | +0.053 | 19/20 | 1.54e-05 | 0.000168 |
| static | tuned_known_ok | -0.067 | 2/20 | 0.000229 | 0.000214 |
| static | tuned_unk_det | +0.274 | 18/20 | 1.07e-05 | 0.000155 |
| knn | closed | -0.008 | 6/20 | 0.125 | 0.215 |
| knn | auroc | -0.003 | 10/20 | 0.676 | 0.784 |
| knn | tuned_known_ok | +0.001 | 9/20 | 0.921 | 0.717 |
| knn | tuned_unk_det | -0.053 | 9/20 | 0.215 | 0.407 |
| temp_rff_p1 | closed | +0.031 | 18/20 | 0.00109 | 0.00128 |
| temp_rff_p1 | auroc | +0.043 | 17/20 | 0.000739 | 0.00621 |
| temp_rff_p1 | tuned_known_ok | -0.077 | 2/20 | 0.000266 | 0.000625 |
| temp_rff_p1 | tuned_unk_det | +0.286 | 17/20 | 2.08e-05 | 0.000214 |
| static_window | closed | +0.030 | 18/20 | 0.00124 | 0.00148 |
| static_window | auroc | +0.053 | 19/20 | 1.54e-05 | 0.000168 |
| static_window | tuned_known_ok | -0.067 | 2/20 | 0.000229 | 0.000214 |
| static_window | tuned_unk_det | +0.274 | 18/20 | 1.07e-05 | 0.000155 |
| knn_window | closed | -0.008 | 5/20 | 0.122 | 0.211 |
| knn_window | auroc | -0.003 | 9/20 | 0.658 | 0.927 |
| knn_window | tuned_known_ok | +0.001 | 8/20 | 0.931 | 0.776 |
| knn_window | tuned_unk_det | -0.053 | 8/20 | 0.214 | 0.352 |
| temp_rff_nowt | closed | -0.002 | 1/20 | 0.0967 | 0.0747 |
| temp_rff_nowt | auroc | +0.001 | 4/20 | 0.797 | 0.463 |
| temp_rff_nowt | tuned_known_ok | +0.007 | 5/20 | 0.0933 | 0.0747 |
| temp_rff_nowt | tuned_unk_det | -0.049 | 0/20 | 0.0381 | 0.0277 |
| probe_default | closed | +0.010 | 13/20 | 0.0953 | 0.04 |
| probe_default | auroc | +0.147 | 18/20 | 4.41e-06 | 1.34e-05 |
| probe_default | tuned_known_ok | -0.050 | 8/20 | 0.0322 | 0.0441 |
| probe_default | tuned_unk_det | +0.193 | 13/20 | 0.00677 | 0.0215 |
| knn_default | closed | -0.006 | 11/20 | 0.203 | 0.49 |
| knn_default | auroc | -0.006 | 12/20 | 0.466 | 0.956 |
| knn_default | tuned_known_ok | +0.014 | 12/20 | 0.277 | 0.312 |
| knn_default | tuned_unk_det | -0.091 | 9/20 | 0.0437 | 0.154 |
| openmax_default | closed | +0.080 | 19/20 | 2.69e-05 | 9.54e-06 |
| openmax_default | auroc | +0.098 | 17/20 | 0.000159 | 3.62e-05 |
| openmax_default | tuned_known_ok | +0.238 | 20/20 | 5.74e-09 | 1.91e-06 |
| openmax_default | tuned_unk_det | -0.180 | 7/20 | 0.0101 | 0.0484 |
| temp_rff_default | closed | -0.007 | 1/20 | 0.142 | 0.0929 |
| temp_rff_default | auroc | -0.014 | 1/20 | 0.068 | 0.0251 |
| temp_rff_default | tuned_known_ok | +0.053 | 8/20 | 0.0121 | 0.0117 |
| temp_rff_default | tuned_unk_det | -0.173 | 0/20 | 0.00914 | 0.0117 |
| raw_centroid | closed | +0.047 | 16/20 | 0.00127 | 0.00169 |
| raw_centroid | auroc | +0.149 | 18/20 | 8.29e-06 | 9.54e-06 |
| raw_centroid | tuned_known_ok | +0.279 | 20/20 | 4.36e-08 | 1.91e-06 |
| raw_centroid | tuned_unk_det | -0.131 | 7/20 | 0.0597 | 0.0696 |
| raw_centroid_cfo | closed | +0.047 | 16/20 | 0.00127 | 0.00169 |
| raw_centroid_cfo | auroc | +0.149 | 18/20 | 8.29e-06 | 9.54e-06 |
| raw_centroid_cfo | tuned_known_ok | +0.279 | 20/20 | 4.36e-08 | 1.91e-06 |
| raw_centroid_cfo | tuned_unk_det | -0.131 | 7/20 | 0.0597 | 0.0696 |
## Forced prototype window (decay 1)

| W | Closed acc. | AUROC | tuned accept | tuned unk. det. |
|---|---|---|---|---|
| 1 | 0.899 ± 0.059 | 0.799 ± 0.080 | 0.763 ± 0.123 | 0.602 ± 0.271 |
| 2 | 0.891 ± 0.060 | 0.797 ± 0.077 | 0.772 ± 0.114 | 0.586 ± 0.262 |
| 3 | 0.887 ± 0.058 | 0.779 ± 0.085 | 0.752 ± 0.107 | 0.593 ± 0.251 |

## Few-shot target-session enrollment (same support for every rule)

| k requested | k effective | temp_rff closed / AUROC / tuned known ok | static closed / AUROC / tuned known ok | knn closed / AUROC / tuned known ok | static_window closed / AUROC / tuned known ok | knn_window closed / AUROC / tuned known ok |
|---|---|---|---|---|---|---|
| 1 | 1 | 0.931 / 0.813 / 0.758 | 0.865 / 0.743 / 0.794 | 0.915 / 0.805 / 0.730 | 0.865 / 0.743 / 0.794 | 0.915 / 0.805 / 0.730 |
| 5 | 5 | 0.951 / 0.846 / 0.796 | 0.865 / 0.744 / 0.795 | 0.943 / 0.822 / 0.755 | 0.865 / 0.744 / 0.795 | 0.943 / 0.822 / 0.755 |
| 10 | 10 | 0.956 / 0.846 / 0.799 | 0.866 / 0.744 / 0.797 | 0.953 / 0.835 / 0.776 | 0.866 / 0.744 / 0.797 | 0.953 / 0.835 / 0.777 |
| 20 | 20 | 0.962 / 0.854 / 0.820 | 0.867 / 0.746 / 0.799 | 0.964 / 0.848 / 0.795 | 0.867 / 0.746 / 0.799 | 0.964 / 0.848 / 0.796 |
| 50 | 49 | 0.965 / 0.861 / 0.834 | 0.869 / 0.752 / 0.805 | 0.974 / 0.866 / 0.825 | 0.869 / 0.752 / 0.805 | 0.974 / 0.866 / 0.827 |
| 100 | 97 | 0.968 / 0.870 / 0.844 | 0.872 / 0.760 / 0.810 | 0.979 / 0.879 / 0.849 | 0.872 / 0.760 / 0.810 | 0.979 / 0.879 / 0.851 |
| 200 | 192 | 0.971 / 0.874 / 0.861 | 0.877 / 0.770 / 0.823 | 0.981 / 0.891 / 0.873 | 0.877 / 0.770 / 0.823 | 0.982 / 0.892 / 0.874 |
| 500 | 379 | 0.971 / 0.879 / 0.868 | 0.888 / 0.781 / 0.836 | 0.984 / 0.901 / 0.891 | 0.888 / 0.781 / 0.836 | 0.984 / 0.902 / 0.892 |

## Pipeline stages (Part-A centroids)

| Stage | Softmax acc. | Centroid acc. | AUROC |
|---|---|---|---|
| phase1 | 0.694 ± 0.058 | 0.741 ± 0.065 | 0.645 ± 0.134 |
| phase2 | 0.846 ± 0.063 | 0.869 ± 0.060 | 0.755 ± 0.082 |
| phase3 | 0.844 ± 0.066 | 0.869 ± 0.063 | 0.746 ± 0.084 |

Joint-training control vs sequential Phase 2 (paired, split 0; joint runs have no audit):

| metric | joint | Phase 2 | mean diff | wins |
|---|---|---|---|---|
| softmax | 0.849 ± 0.044 | 0.822 ± 0.041 | +0.027 | 4/5 |
| centroid | 0.881 ± 0.041 | 0.866 ± 0.046 | +0.015 | 3/5 |
| auroc | 0.738 ± 0.070 | 0.769 ± 0.063 | -0.030 | 1/5 |

## Selected settings over runs

- W: 1×19, 3×1
- decay: 1.0×20
- P: 1×1, 20×1, 5×5, all×13
- knn (k, weighted): (1, False)×4, (10, False)×3, (10, True)×2, (5, False)×10, (5, True)×1
- openmax (dist, tail, alpha): ('euclidean', 10, 1)×17, ('euclidean', 10, 2)×1, ('euclidean', 20, 1)×1, ('eucos', 10, 1)×1
- probe C: 0.01×4, 0.1×2, 1.0×11, 10.0×3
- static_window W: (1,)×20
- knn_window W (k, weighted): (1, 1, False)×4, (1, 10, False)×3, (1, 10, True)×2, (1, 5, False)×9, (1, 5, True)×1, (3, 1, False)×1
- temp_rff_p1 W decay: (1, 1.0)×14, (2, 0.8)×1, (2, 1.0)×4, (3, 1.0)×1
- selection-weight sensitivity: choice changes with the weight (0.3/0.5/0.7) in 9 of 20 runs
