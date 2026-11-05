"""oilfc: shared code for the crude-oil forecasting project.

Modules
-------
utils    : config loading, seeding, device selection, paths
data     : loading, transforms, chronological splits, scaling, windowing
models   : LSTM / GRU forecasters (PyTorch)
train    : training loop with early stopping, prediction helper
metrics  : RMSE, MAE, MAPE, Diebold-Mariano test
"""
