Validation period 2019-2021, errors in USD/bbl, mean and std over seeds.

| sample                       | sample_start   |   n_rows | inputs     |   n_inputs | model   |   horizon |   val_RMSE_mean |   val_RMSE_std |   val_MAE_mean |   val_MAE_std |   val_DirAcc_mean |   val_DirAcc_std |
|:-----------------------------|:---------------|---------:|:-----------|-----------:|:--------|----------:|----------------:|---------------:|---------------:|--------------:|------------------:|-----------------:|
| sample2006_with_usd_broad    | 2006-01-02     |     5217 | WTI only   |          1 | LSTM    |         1 |          1.5934 |         0.0092 |         1.0047 |        0.0071 |            0.4779 |           0.0148 |
| sample2006_with_usd_broad    | 2006-01-02     |     5217 | All series |         13 | LSTM    |         1 |          1.6167 |         0.0088 |         1.012  |        0.0033 |            0.5243 |           0.0034 |
| sample2003_without_usd_broad | 2003-01-02     |     5998 | WTI only   |          1 | LSTM    |         1 |          1.5946 |         0.0066 |         1.0019 |        0.0035 |            0.5072 |           0.0066 |
| sample2003_without_usd_broad | 2003-01-02     |     5998 | All series |         12 | LSTM    |         1 |          1.6249 |         0.0109 |         1.0231 |        0.0022 |            0.5208 |           0.0107 |