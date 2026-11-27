| period         |   horizon | model               |    RMSE |     MAE |   DirAcc |   R2_oos_vs_RW |   pred_std_ratio |   corr_pred_actual |   seed_RMSE_std |   coverage80 |   gate_mean |
|:---------------|----------:|:--------------------|--------:|--------:|---------:|---------------:|-----------------:|-------------------:|----------------:|-------------:|------------:|
| test 2022-2025 |         1 | Original LSTM       |  1.9642 |  1.4061 |   0.5406 |         0.0049 |           0.0561 |             0.056  |          0.0056 |     nan      |    nan      |
| test 2022-2025 |         1 | Gated residual LSTM |  2.0285 |  1.4372 |   0.5035 |        -0.0614 |           0.189  |            -0.0372 |          0.0175 |       0.8144 |      0.2191 |
| test 2022-2025 |         1 | Random walk         |  1.969  |  1.4096 | nan      |         0      |           0      |           nan      |        nan      |     nan      |    nan      |
| test 2022-2025 |         1 | Chronos-2           |  2.0807 |  1.4622 |   0.5226 |        -0.1168 |           0.2981 |            -0.0119 |        nan      |       0.7914 |    nan      |
| holdout 2026   |         1 | Original LSTM       |  3.7239 |  2.5946 |   0.4751 |         0.0006 |           0.0489 |             0.0249 |          0.0065 |     nan      |    nan      |
| holdout 2026   |         1 | Gated residual LSTM |  3.7416 |  2.5713 |   0.5339 |        -0.009  |           0.0876 |            -0.0059 |          0.0323 |       0.7602 |      0.1784 |
| holdout 2026   |         1 | Random walk         |  3.725  |  2.5774 | nan      |         0      |           0      |           nan      |        nan      |     nan      |    nan      |
| holdout 2026   |         1 | Chronos-2           |  3.8442 |  2.6346 |   0.5023 |        -0.065  |           0.2333 |            -0.0223 |        nan      |       0.7783 |    nan      |
| test 2022-2025 |         5 | Original LSTM       |  4.2783 |  3.0721 |   0.5256 |         0.0113 |           0.0738 |             0.0939 |          0.0245 |     nan      |    nan      |
| test 2022-2025 |         5 | Gated residual LSTM |  4.4085 |  3.096  |   0.5376 |        -0.0498 |           0.1633 |             0.0104 |          0.0215 |       0.7984 |      0.116  |
| test 2022-2025 |         5 | Random walk         |  4.3026 |  3.0924 | nan      |         0      |           0      |           nan      |        nan      |     nan      |    nan      |
| test 2022-2025 |         5 | Chronos-2           |  4.4271 |  3.0908 |   0.5597 |        -0.0587 |           0.2019 |             0.0347 |        nan      |       0.7693 |    nan      |
| holdout 2026   |         5 | Original LSTM       |  7.6467 |  5.9218 |   0.4661 |        -0.0125 |           0.1805 |             0.0306 |          0.1141 |     nan      |    nan      |
| holdout 2026   |         5 | Gated residual LSTM |  7.7228 |  5.9421 |   0.5385 |        -0.0328 |           0.1418 |            -0.0843 |          0.0781 |       0.6923 |      0.0957 |
| holdout 2026   |         5 | Random walk         |  7.5993 |  5.8611 | nan      |         0      |           0      |           nan      |        nan      |     nan      |    nan      |
| holdout 2026   |         5 | Chronos-2           |  7.7377 |  5.9189 |   0.5339 |        -0.0368 |           0.1703 |            -0.0553 |        nan      |       0.733  |    nan      |
| test 2022-2025 |        20 | Original LSTM       |  7.2777 |  5.5434 |   0.4945 |        -0.0762 |           0.2138 |            -0.0863 |          0.2914 |     nan      |    nan      |
| test 2022-2025 |        20 | Gated residual LSTM |  7.5748 |  5.758  |   0.4945 |        -0.1659 |           0.2466 |            -0.1096 |          0.5033 |       0.7743 |      0.2396 |
| test 2022-2025 |        20 | Random walk         |  7.0153 |  5.3914 | nan      |         0      |           0      |           nan      |        nan      |     nan      |    nan      |
| test 2022-2025 |        20 | Chronos-2           |  7.2848 |  5.5564 |   0.4865 |        -0.0783 |           0.1871 |             0.0201 |        nan      |       0.8034 |    nan      |
| holdout 2026   |        20 | Original LSTM       | 14.5339 | 11.1546 |   0.3258 |        -0.0364 |           0.0782 |            -0.2627 |          0.4585 |     nan      |    nan      |
| holdout 2026   |        20 | Gated residual LSTM | 15.0457 | 11.5393 |   0.4344 |        -0.1107 |           0.1572 |            -0.1593 |          0.8684 |       0.638  |      0.2285 |
| holdout 2026   |        20 | Random walk         | 14.2763 | 10.8795 | nan      |         0      |           0      |           nan      |        nan      |     nan      |    nan      |
| holdout 2026   |        20 | Chronos-2           | 14.6437 | 11.242  |   0.4706 |        -0.0521 |           0.1637 |             0.0742 |        nan      |       0.6923 |    nan      |

| period         |   horizon | model               | vs            |      DM |   p_value |   p_holm_primary |       CW |   CW_p_value |
|:---------------|----------:|:--------------------|:--------------|--------:|----------:|-----------------:|---------:|-------------:|
| test 2022-2025 |         1 | Gated residual LSTM | Original LSTM |  2.1269 |    0.0337 |           0.101  | nan      |     nan      |
| test 2022-2025 |         1 | Gated residual LSTM | Chronos-2     | -2.8487 |    0.0045 |         nan      | nan      |     nan      |
| test 2022-2025 |         1 | Gated residual LSTM | Random walk   |  1.9684 |    0.0493 |         nan      |  -0.6185 |       0.7319 |
| test 2022-2025 |         1 | Original LSTM       | Random walk   | -1.47   |    0.1419 |         nan      |   2.1109 |       0.0174 |
| test 2022-2025 |         1 | Chronos-2           | Random walk   |  2.413  |    0.016  |         nan      |  -0.2814 |       0.6108 |
| holdout 2026   |         1 | Gated residual LSTM | Original LSTM |  0.3912 |    0.6961 |           1      | nan      |     nan      |
| holdout 2026   |         1 | Gated residual LSTM | Chronos-2     | -1.2266 |    0.2213 |         nan      | nan      |     nan      |
| holdout 2026   |         1 | Gated residual LSTM | Random walk   |  0.4385 |    0.6614 |         nan      |  -0.0634 |       0.5253 |
| holdout 2026   |         1 | Original LSTM       | Random walk   | -0.0529 |    0.9578 |         nan      |   0.3572 |       0.3605 |
| holdout 2026   |         1 | Chronos-2           | Random walk   |  1.1027 |    0.2714 |         nan      |  -0.1838 |       0.5729 |
| test 2022-2025 |         5 | Gated residual LSTM | Original LSTM |  1.2924 |    0.1965 |           0.3931 | nan      |     nan      |
| test 2022-2025 |         5 | Gated residual LSTM | Chronos-2     | -0.9718 |    0.3314 |         nan      | nan      |     nan      |
| test 2022-2025 |         5 | Gated residual LSTM | Random walk   |  1.041  |    0.2981 |         nan      |  -0.3871 |       0.6507 |
| test 2022-2025 |         5 | Original LSTM       | Random walk   | -1.317  |    0.1881 |         nan      |   1.8463 |       0.0324 |
| test 2022-2025 |         5 | Chronos-2           | Random walk   |  1.0511 |    0.2935 |         nan      |  -0.1803 |       0.5715 |
| holdout 2026   |         5 | Gated residual LSTM | Original LSTM |  0.6366 |    0.525  |           1      | nan      |     nan      |
| holdout 2026   |         5 | Gated residual LSTM | Chronos-2     | -0.2057 |    0.8372 |         nan      | nan      |     nan      |
| holdout 2026   |         5 | Gated residual LSTM | Random walk   |  0.8261 |    0.4096 |         nan      |  -0.4031 |       0.6566 |
| holdout 2026   |         5 | Original LSTM       | Random walk   |  0.2947 |    0.7685 |         nan      |   0.5873 |       0.2785 |
| holdout 2026   |         5 | Chronos-2           | Random walk   |  0.7578 |    0.4494 |         nan      |  -0.1842 |       0.5731 |
| test 2022-2025 |        20 | Gated residual LSTM | Original LSTM |  1.1698 |    0.2424 |           0.3931 | nan      |     nan      |
| test 2022-2025 |        20 | Gated residual LSTM | Chronos-2     |  1.4762 |    0.1402 |         nan      | nan      |     nan      |
| test 2022-2025 |        20 | Gated residual LSTM | Random walk   |  1.7896 |    0.0738 |         nan      |  -1.0946 |       0.8632 |
| test 2022-2025 |        20 | Original LSTM       | Random walk   |  1.3791 |    0.1682 |         nan      |  -0.4745 |       0.6824 |
| test 2022-2025 |        20 | Chronos-2           | Random walk   |  1.0834 |    0.2789 |         nan      |  -0.4946 |       0.6895 |
| holdout 2026   |        20 | Gated residual LSTM | Original LSTM |  0.8174 |    0.4146 |           1      | nan      |     nan      |
| holdout 2026   |        20 | Gated residual LSTM | Chronos-2     |  1.6211 |    0.1064 |         nan      | nan      |     nan      |
| holdout 2026   |        20 | Gated residual LSTM | Random walk   |  1.3109 |    0.1913 |         nan      |  -1.2905 |       0.9016 |
| holdout 2026   |        20 | Original LSTM       | Random walk   |  1.4293 |    0.1543 |         nan      |  -1.4011 |       0.9194 |
| holdout 2026   |        20 | Chronos-2           | Random walk   |  0.5809 |    0.5619 |         nan      |  -0.2175 |       0.5861 |

| period         |   horizon | model               |   hit_rate |      PT |   PT_p_value |   n_nonoverlap |
|:---------------|----------:|:--------------------|-----------:|--------:|-------------:|---------------:|
| test 2022-2025 |         1 | Gated residual LSTM |     0.5045 |  0.3215 |       0.3739 |            997 |
| test 2022-2025 |         1 | Original LSTM       |     0.5426 |  2.6071 |       0.0046 |            997 |
| test 2022-2025 |         1 | Chronos-2           |     0.5236 |  1.5369 |       0.0622 |            997 |
| holdout 2026   |         1 | Gated residual LSTM |     0.5339 |  0.7771 |       0.2185 |            221 |
| holdout 2026   |         1 | Original LSTM       |     0.4751 | -0.0644 |       0.5257 |            221 |
| holdout 2026   |         1 | Chronos-2           |     0.5023 | -0.016  |       0.5064 |            221 |
| test 2022-2025 |         5 | Gated residual LSTM |     0.56   |  1.6817 |       0.0463 |            200 |
| test 2022-2025 |         5 | Original LSTM       |     0.56   |  1.7587 |       0.0393 |            200 |
| test 2022-2025 |         5 | Chronos-2           |     0.605  |  2.9644 |       0.0015 |            200 |
| holdout 2026   |         5 | Gated residual LSTM |     0.5333 |  0.4054 |       0.3426 |             45 |
| holdout 2026   |         5 | Original LSTM       |     0.4222 | -0.968  |       0.8335 |             45 |
| holdout 2026   |         5 | Chronos-2           |     0.5556 |  0.9259 |       0.1773 |             45 |
| test 2022-2025 |        20 | Gated residual LSTM |     0.48   | -0.0119 |       0.5048 |             50 |
| test 2022-2025 |        20 | Original LSTM       |     0.46   | -0.8064 |       0.79   |             50 |
| test 2022-2025 |        20 | Chronos-2           |     0.44   | -0.473  |       0.6819 |             50 |
| holdout 2026   |        20 | Gated residual LSTM |     0.4167 | -0.4324 |       0.6673 |             12 |
| holdout 2026   |        20 | Original LSTM       |     0.3333 | -1.2792 |       0.8996 |             12 |
| holdout 2026   |        20 | Chronos-2           |     0.25   | -1.4771 |       0.9302 |             12 |
