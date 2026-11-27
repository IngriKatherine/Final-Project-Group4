Follow-up designed after the first results were seen; 2026 is not an untouched holdout.

| period                            |   horizon | model                                |    RMSE |     MAE |   DirAcc |   R2_oos_vs_RW |   pred_std_ratio |   corr_pred_actual |   seed_RMSE_std |   coverage80 |   gate_mean |
|:----------------------------------|----------:|:-------------------------------------|--------:|--------:|---------:|---------------:|-----------------:|-------------------:|----------------:|-------------:|------------:|
| test 2022-2025                    |         1 | Original LSTM                        |  1.9642 |  1.4061 |   0.5406 |         0.0049 |           0.0561 |             0.056  |          0.0056 |     nan      |    nan      |
| test 2022-2025                    |         1 | Gated residual LSTM                  |  2.0285 |  1.4372 |   0.5035 |        -0.0614 |           0.189  |            -0.0372 |          0.0175 |       0.8144 |      0.2191 |
| test 2022-2025                    |         1 | Chronos-feature LSTM                 |  1.9691 |  1.4108 |   0.4925 |        -0.0002 |           0.0382 |             0.0027 |          0.0014 |       0.8034 |      0.4756 |
| test 2022-2025                    |         1 | Chronos-feature LSTM, Chronos masked |  1.9726 |  1.4097 |   0.5145 |        -0.0037 |           0.0333 |             0.0099 |          0.0054 |     nan      |    nan      |
| test 2022-2025                    |         1 | Random walk                          |  1.969  |  1.4096 | nan      |         0      |           0      |           nan      |        nan      |     nan      |    nan      |
| test 2022-2025                    |         1 | Chronos-2                            |  2.0807 |  1.4622 |   0.5226 |        -0.1168 |           0.2981 |            -0.0119 |        nan      |       0.7914 |    nan      |
| 2026 (seen before; not a holdout) |         1 | Original LSTM                        |  3.7239 |  2.5946 |   0.4751 |         0.0006 |           0.0489 |             0.0249 |          0.0065 |     nan      |    nan      |
| 2026 (seen before; not a holdout) |         1 | Gated residual LSTM                  |  3.7416 |  2.5713 |   0.5339 |        -0.009  |           0.0876 |            -0.0059 |          0.0323 |       0.7602 |      0.1784 |
| 2026 (seen before; not a holdout) |         1 | Chronos-feature LSTM                 |  3.7269 |  2.5888 |   0.4434 |        -0.001  |           0.026  |            -0.0094 |          0.0032 |       0.7602 |      0.4775 |
| 2026 (seen before; not a holdout) |         1 | Chronos-feature LSTM, Chronos masked |  3.7267 |  2.5745 |   0.5339 |        -0.0009 |           0.0194 |            -0.0291 |          0.0069 |     nan      |    nan      |
| 2026 (seen before; not a holdout) |         1 | Random walk                          |  3.725  |  2.5774 | nan      |         0      |           0      |           nan      |        nan      |     nan      |    nan      |
| 2026 (seen before; not a holdout) |         1 | Chronos-2                            |  3.8442 |  2.6346 |   0.5023 |        -0.065  |           0.2333 |            -0.0223 |        nan      |       0.7783 |    nan      |
| test 2022-2025                    |         5 | Original LSTM                        |  4.2783 |  3.0721 |   0.5256 |         0.0113 |           0.0738 |             0.0939 |          0.0245 |     nan      |    nan      |
| test 2022-2025                    |         5 | Gated residual LSTM                  |  4.4085 |  3.096  |   0.5376 |        -0.0498 |           0.1633 |             0.0104 |          0.0215 |       0.7984 |      0.116  |
| test 2022-2025                    |         5 | Chronos-feature LSTM                 |  4.3211 |  3.1066 |   0.4905 |        -0.0086 |           0.0553 |            -0.0249 |          0.0131 |       0.8074 |      0.4882 |
| test 2022-2025                    |         5 | Chronos-feature LSTM, Chronos masked |  4.3379 |  3.1097 |   0.5155 |        -0.0164 |           0.0522 |            -0.018  |          0.0107 |     nan      |    nan      |
| test 2022-2025                    |         5 | Random walk                          |  4.3026 |  3.0924 | nan      |         0      |           0      |           nan      |        nan      |     nan      |    nan      |
| test 2022-2025                    |         5 | Chronos-2                            |  4.4271 |  3.0908 |   0.5597 |        -0.0587 |           0.2019 |             0.0347 |        nan      |       0.7693 |    nan      |
| 2026 (seen before; not a holdout) |         5 | Original LSTM                        |  7.6467 |  5.9218 |   0.4661 |        -0.0125 |           0.1805 |             0.0306 |          0.1141 |     nan      |    nan      |
| 2026 (seen before; not a holdout) |         5 | Gated residual LSTM                  |  7.7228 |  5.9421 |   0.5385 |        -0.0328 |           0.1418 |            -0.0843 |          0.0781 |       0.6923 |      0.0957 |
| 2026 (seen before; not a holdout) |         5 | Chronos-feature LSTM                 |  7.6476 |  5.9226 |   0.5249 |        -0.0128 |           0.0732 |            -0.0682 |          0.1182 |       0.733  |      0.4763 |
| 2026 (seen before; not a holdout) |         5 | Chronos-feature LSTM, Chronos masked |  7.6642 |  5.9523 |   0.552  |        -0.0171 |           0.0553 |            -0.1134 |          0.1108 |     nan      |    nan      |
| 2026 (seen before; not a holdout) |         5 | Random walk                          |  7.5993 |  5.8611 | nan      |         0      |           0      |           nan      |        nan      |     nan      |    nan      |
| 2026 (seen before; not a holdout) |         5 | Chronos-2                            |  7.7377 |  5.9189 |   0.5339 |        -0.0368 |           0.1703 |            -0.0553 |        nan      |       0.733  |    nan      |
| test 2022-2025                    |        20 | Original LSTM                        |  7.2777 |  5.5434 |   0.4945 |        -0.0762 |           0.2138 |            -0.0863 |          0.2914 |     nan      |    nan      |
| test 2022-2025                    |        20 | Gated residual LSTM                  |  7.5748 |  5.758  |   0.4945 |        -0.1659 |           0.2466 |            -0.1096 |          0.5033 |       0.7743 |      0.2396 |
| test 2022-2025                    |        20 | Chronos-feature LSTM                 |  7.5584 |  5.753  |   0.5005 |        -0.1608 |           0.307  |            -0.0985 |          0.251  |       0.7513 |      0.4408 |
| test 2022-2025                    |        20 | Chronos-feature LSTM, Chronos masked |  7.7315 |  5.8807 |   0.4875 |        -0.2146 |           0.2928 |            -0.1236 |          0.3284 |     nan      |    nan      |
| test 2022-2025                    |        20 | Random walk                          |  7.0153 |  5.3914 | nan      |         0      |           0      |           nan      |        nan      |     nan      |    nan      |
| test 2022-2025                    |        20 | Chronos-2                            |  7.2848 |  5.5564 |   0.4865 |        -0.0783 |           0.1871 |             0.0201 |        nan      |       0.8034 |    nan      |
| 2026 (seen before; not a holdout) |        20 | Original LSTM                        | 14.5339 | 11.1546 |   0.3258 |        -0.0364 |           0.0782 |            -0.2627 |          0.4585 |     nan      |    nan      |
| 2026 (seen before; not a holdout) |        20 | Gated residual LSTM                  | 15.0457 | 11.5393 |   0.4344 |        -0.1107 |           0.1572 |            -0.1593 |          0.8684 |       0.638  |      0.2285 |
| 2026 (seen before; not a holdout) |        20 | Chronos-feature LSTM                 | 14.1621 | 10.9646 |   0.5385 |         0.0159 |           0.1183 |             0.1437 |          0.6938 |       0.6471 |      0.4635 |
| 2026 (seen before; not a holdout) |        20 | Chronos-feature LSTM, Chronos masked | 14.1917 | 10.8137 |   0.5294 |         0.0118 |           0.1234 |             0.0102 |          0.9525 |     nan      |    nan      |
| 2026 (seen before; not a holdout) |        20 | Random walk                          | 14.2763 | 10.8795 | nan      |         0      |           0      |           nan      |        nan      |     nan      |    nan      |
| 2026 (seen before; not a holdout) |        20 | Chronos-2                            | 14.6437 | 11.242  |   0.4706 |        -0.0521 |           0.1637 |             0.0742 |        nan      |       0.6923 |    nan      |

| period                            |   horizon | model                | vs                                   |      DM |   p_value |   p_holm_primary |       CW |   CW_p_value |
|:----------------------------------|----------:|:---------------------|:-------------------------------------|--------:|----------:|-----------------:|---------:|-------------:|
| test 2022-2025                    |         1 | Chronos-feature LSTM | Original LSTM                        |  0.7788 |    0.4363 |           0.4363 | nan      |     nan      |
| test 2022-2025                    |         1 | Chronos-feature LSTM | Chronos-2                            | -2.2542 |    0.0244 |         nan      | nan      |     nan      |
| test 2022-2025                    |         1 | Chronos-feature LSTM | Gated residual LSTM                  | -1.7798 |    0.0754 |         nan      | nan      |     nan      |
| test 2022-2025                    |         1 | Chronos-feature LSTM | Chronos-feature LSTM, Chronos masked | -0.5286 |    0.5972 |         nan      | nan      |     nan      |
| test 2022-2025                    |         1 | Chronos-feature LSTM | Random walk                          |  0.0267 |    0.9787 |         nan      |   0.2598 |       0.3975 |
| 2026 (seen before; not a holdout) |         1 | Chronos-feature LSTM | Original LSTM                        |  0.205  |    0.8378 |           1      | nan      |     nan      |
| 2026 (seen before; not a holdout) |         1 | Chronos-feature LSTM | Chronos-2                            | -1.0512 |    0.2943 |         nan      | nan      |     nan      |
| 2026 (seen before; not a holdout) |         1 | Chronos-feature LSTM | Gated residual LSTM                  | -0.3699 |    0.7118 |         nan      | nan      |     nan      |
| 2026 (seen before; not a holdout) |         1 | Chronos-feature LSTM | Chronos-feature LSTM, Chronos masked |  0.0164 |    0.9869 |         nan      | nan      |     nan      |
| 2026 (seen before; not a holdout) |         1 | Chronos-feature LSTM | Random walk                          |  0.2325 |    0.8164 |         nan      |  -0.0708 |       0.5282 |
| test 2022-2025                    |         5 | Chronos-feature LSTM | Original LSTM                        |  1.7731 |    0.0765 |           0.2296 | nan      |     nan      |
| test 2022-2025                    |         5 | Chronos-feature LSTM | Chronos-2                            | -0.8817 |    0.3781 |         nan      | nan      |     nan      |
| test 2022-2025                    |         5 | Chronos-feature LSTM | Gated residual LSTM                  | -0.8447 |    0.3985 |         nan      | nan      |     nan      |
| test 2022-2025                    |         5 | Chronos-feature LSTM | Chronos-feature LSTM, Chronos masked | -0.8149 |    0.4153 |         nan      | nan      |     nan      |
| test 2022-2025                    |         5 | Chronos-feature LSTM | Random walk                          |  1.1983 |    0.2311 |         nan      |  -0.7756 |       0.781  |
| 2026 (seen before; not a holdout) |         5 | Chronos-feature LSTM | Original LSTM                        |  0.0054 |    0.9957 |           1      | nan      |     nan      |
| 2026 (seen before; not a holdout) |         5 | Chronos-feature LSTM | Chronos-2                            | -0.3962 |    0.6923 |         nan      | nan      |     nan      |
| 2026 (seen before; not a holdout) |         5 | Chronos-feature LSTM | Gated residual LSTM                  | -0.4388 |    0.6612 |         nan      | nan      |     nan      |
| 2026 (seen before; not a holdout) |         5 | Chronos-feature LSTM | Chronos-feature LSTM, Chronos masked | -0.305  |    0.7607 |         nan      | nan      |     nan      |
| 2026 (seen before; not a holdout) |         5 | Chronos-feature LSTM | Random walk                          |  0.4651 |    0.6423 |         nan      |  -0.086  |       0.5343 |
| test 2022-2025                    |        20 | Chronos-feature LSTM | Original LSTM                        |  1.416  |    0.1571 |           0.3142 | nan      |     nan      |
| test 2022-2025                    |        20 | Chronos-feature LSTM | Chronos-2                            |  1.0908 |    0.2756 |         nan      | nan      |     nan      |
| test 2022-2025                    |        20 | Chronos-feature LSTM | Gated residual LSTM                  | -0.0941 |    0.925  |         nan      | nan      |     nan      |
| test 2022-2025                    |        20 | Chronos-feature LSTM | Chronos-feature LSTM, Chronos masked | -1.2233 |    0.2215 |         nan      | nan      |     nan      |
| test 2022-2025                    |        20 | Chronos-feature LSTM | Random walk                          |  2.1352 |    0.033  |         nan      |  -1.0859 |       0.8612 |
| 2026 (seen before; not a holdout) |        20 | Chronos-feature LSTM | Original LSTM                        | -1.5875 |    0.1138 |           0.3415 | nan      |     nan      |
| 2026 (seen before; not a holdout) |        20 | Chronos-feature LSTM | Chronos-2                            | -0.6082 |    0.5437 |         nan      | nan      |     nan      |
| 2026 (seen before; not a holdout) |        20 | Chronos-feature LSTM | Gated residual LSTM                  | -1.1742 |    0.2416 |         nan      | nan      |     nan      |
| 2026 (seen before; not a holdout) |        20 | Chronos-feature LSTM | Chronos-feature LSTM, Chronos masked | -0.0995 |    0.9209 |         nan      | nan      |     nan      |
| 2026 (seen before; not a holdout) |        20 | Chronos-feature LSTM | Random walk                          | -0.4026 |    0.6876 |         nan      |   0.8061 |       0.2101 |

| period                            |   horizon | model                |   hit_rate |      PT |   PT_p_value |   n_nonoverlap |
|:----------------------------------|----------:|:---------------------|-----------:|--------:|-------------:|---------------:|
| test 2022-2025                    |         1 | Chronos-feature LSTM |     0.4935 | -0.5618 |       0.7129 |            997 |
| test 2022-2025                    |         1 | Original LSTM        |     0.5426 |  2.6071 |       0.0046 |            997 |
| test 2022-2025                    |         1 | Chronos-2            |     0.5236 |  1.5369 |       0.0622 |            997 |
| 2026 (seen before; not a holdout) |         1 | Chronos-feature LSTM |     0.4434 | -1.2417 |       0.8928 |            221 |
| 2026 (seen before; not a holdout) |         1 | Original LSTM        |     0.4751 | -0.0644 |       0.5257 |            221 |
| 2026 (seen before; not a holdout) |         1 | Chronos-2            |     0.5023 | -0.016  |       0.5064 |            221 |
| test 2022-2025                    |         5 | Chronos-feature LSTM |     0.505  |  0.0896 |       0.4643 |            200 |
| test 2022-2025                    |         5 | Original LSTM        |     0.56   |  1.7587 |       0.0393 |            200 |
| test 2022-2025                    |         5 | Chronos-2            |     0.605  |  2.9644 |       0.0015 |            200 |
| 2026 (seen before; not a holdout) |         5 | Chronos-feature LSTM |     0.5333 |  0.1778 |       0.4294 |             45 |
| 2026 (seen before; not a holdout) |         5 | Original LSTM        |     0.4222 | -0.968  |       0.8335 |             45 |
| 2026 (seen before; not a holdout) |         5 | Chronos-2            |     0.5556 |  0.9259 |       0.1773 |             45 |
| test 2022-2025                    |        20 | Chronos-feature LSTM |     0.44   | -0.6082 |       0.7285 |             50 |
| test 2022-2025                    |        20 | Original LSTM        |     0.46   | -0.8064 |       0.79   |             50 |
| test 2022-2025                    |        20 | Chronos-2            |     0.44   | -0.473  |       0.6819 |             50 |
| 2026 (seen before; not a holdout) |        20 | Chronos-feature LSTM |     0.5833 |  0      |       0.5    |             12 |
| 2026 (seen before; not a holdout) |        20 | Original LSTM        |     0.3333 | -1.2792 |       0.8996 |             12 |
| 2026 (seen before; not a holdout) |        20 | Chronos-2            |     0.25   | -1.4771 |       0.9302 |             12 |

|   year |   corr_old_vs_pit |   mean_abs_diff_log |   share_days_differ |   mean_abs_revision_log |   share_revised |
|-------:|------------------:|--------------------:|--------------------:|------------------------:|----------------:|
|   2022 |            0.3311 |              0.4106 |              0.9761 |                  0.1066 |          0.9789 |
|   2023 |            0.3317 |              0.3896 |              1      |                  0.1147 |          1      |
|   2024 |            0.3024 |              0.3098 |              1      |                  0.1239 |          0.9973 |
|   2025 |            0.2869 |              0.3612 |              1      |                  0.1422 |          0.9945 |
|   2026 |            0.4955 |              0.3374 |              1      |                  0.1259 |          0.9511 |
