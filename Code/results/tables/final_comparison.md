|   horizon | model       | features      |   RMSE |   MAE |   MAPE |   DirAcc |
|----------:|:------------|:--------------|-------:|------:|-------:|---------:|
|         1 | Random walk | price_only    |  1.925 | 1.347 |  1.665 |    0.046 |
|         1 | ARIMA       | price_only    |  2.121 | 1.512 |  1.87  |    0.5   |
|         1 | LSTM        | price_only    |  1.925 | 1.35  |  1.669 |    0.478 |
|         1 | LSTM        | all           |  1.977 | 1.389 |  1.723 |    0.484 |
|         1 | LSTM        | shap_selected |  1.931 | 1.349 |  1.667 |    0.523 |
|         1 | GRU         | price_only    |  1.92  | 1.348 |  1.668 |    0.482 |
|         1 | GRU         | all           |  1.981 | 1.4   |  1.736 |    0.489 |
|         1 | GRU         | shap_selected |  1.932 | 1.354 |  1.674 |    0.506 |
|         5 | Random walk | price_only    |  4.217 | 3.01  |  3.74  |    0.001 |
|         5 | ARIMA       | price_only    |  4.174 | 3.016 |  3.748 |    0.529 |
|         5 | LSTM        | price_only    |  4.203 | 2.996 |  3.719 |    0.516 |
|         5 | LSTM        | all           |  4.291 | 3.078 |  3.839 |    0.489 |
|         5 | LSTM        | shap_selected |  4.29  | 3.047 |  3.794 |    0.504 |
|         5 | GRU         | price_only    |  4.19  | 3.004 |  3.723 |    0.491 |
|         5 | GRU         | all           |  4.346 | 3.13  |  3.905 |    0.509 |
|         5 | GRU         | shap_selected |  4.246 | 3.003 |  3.739 |    0.538 |
|        20 | Random walk | price_only    |  6.801 | 5.192 |  6.384 |    0.001 |
|        20 | ARIMA       | price_only    |  6.802 | 5.255 |  6.456 |    0.497 |
|        20 | LSTM        | price_only    |  6.844 | 5.271 |  6.528 |    0.513 |
|        20 | LSTM        | all           |  7.748 | 6.084 |  7.628 |    0.435 |
|        20 | LSTM        | shap_selected |  8.06  | 6.284 |  7.916 |    0.547 |
|        20 | GRU         | price_only    |  6.775 | 5.218 |  6.442 |    0.521 |
|        20 | GRU         | all           |  9.124 | 7.253 |  9.154 |    0.477 |
|        20 | GRU         | shap_selected |  7.699 | 5.943 |  7.475 |    0.488 |