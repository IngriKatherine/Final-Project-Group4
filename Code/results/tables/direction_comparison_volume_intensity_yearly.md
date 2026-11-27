|   horizon | subset              | model                                 |    n |   Acc |    F1 |   share_up_pred |   PT_stat |    PT_p |     AUC |
|----------:|:--------------------|:--------------------------------------|-----:|------:|------:|----------------:|----------:|--------:|--------:|
|         1 | all                 | Image CNN / volume_intensity (yearly) |  995 | 0.464 | 0.463 |           0.539 |    -2.336 |   0.99  |   0.465 |
|         1 | |ret|>5%            | Image CNN / volume_intensity (yearly) |   41 | 0.439 | 0.404 |           0.366 |    -1.247 |   0.894 |   0.415 |
|         1 | russia_ukraine_2022 | Image CNN / volume_intensity (yearly) |   62 | 0.371 | 0.367 |           0.532 |    -2.112 |   0.983 |   0.369 |
|         1 | all                 | Random walk (= always up)             |  995 | 0.516 | 0.34  |           1     |   nan     | nan     | nan     |
|         1 | |ret|>5%            | Random walk (= always up)             |   41 | 0.39  | 0.281 |           1     |   nan     | nan     | nan     |
|         1 | russia_ukraine_2022 | Random walk (= always up)             |   62 | 0.548 | 0.354 |           1     |   nan     | nan     | nan     |
|         1 | all                 | ARIMA                                 |  995 | 0.508 | 0.508 |           0.485 |     0.505 |   0.307 | nan     |
|         1 | |ret|>5%            | ARIMA                                 |   41 | 0.585 | 0.581 |           0.512 |     1.17  |   0.121 | nan     |
|         1 | russia_ukraine_2022 | ARIMA                                 |   62 | 0.581 | 0.58  |           0.484 |     1.312 |   0.095 | nan     |
|         1 | all                 | ARIMAX                                |  995 | 0.521 | 0.521 |           0.492 |     1.316 |   0.094 | nan     |
|         1 | |ret|>5%            | ARIMAX                                |   41 | 0.512 | 0.51  |           0.537 |     0.27  |   0.394 | nan     |
|         1 | russia_ukraine_2022 | ARIMAX                                |   62 | 0.403 | 0.39  |           0.597 |    -1.726 |   0.958 | nan     |
|         1 | all                 | LSTM / price_only                     |  995 | 0.502 | 0.493 |           0.357 |     0.393 |   0.347 | nan     |
|         1 | |ret|>5%            | LSTM / price_only                     |   41 | 0.683 | 0.653 |           0.317 |     2.039 |   0.021 | nan     |
|         1 | russia_ukraine_2022 | LSTM / price_only                     |   62 | 0.565 | 0.559 |           0.339 |     1.35  |   0.088 | nan     |
|         1 | all                 | LSTM / all                            |  995 | 0.508 | 0.508 |           0.477 |     0.521 |   0.301 | nan     |
|         1 | |ret|>5%            | LSTM / all                            |   41 | 0.512 | 0.51  |           0.537 |     0.27  |   0.394 | nan     |
|         1 | russia_ukraine_2022 | LSTM / all                            |   62 | 0.516 | 0.514 |           0.516 |     0.233 |   0.408 | nan     |
|         1 | all                 | LSTM / shap_selected                  |  995 | 0.548 | 0.533 |           0.662 |     2.85  |   0.002 | nan     |
|         1 | |ret|>5%            | LSTM / shap_selected                  |   41 | 0.439 | 0.434 |           0.707 |    -0.226 |   0.589 | nan     |
|         1 | russia_ukraine_2022 | LSTM / shap_selected                  |   62 | 0.548 | 0.544 |           0.548 |     0.7   |   0.242 | nan     |
|         1 | all                 | GRU / price_only                      |  995 | 0.506 | 0.505 |           0.449 |     0.451 |   0.326 | nan     |
|         1 | |ret|>5%            | GRU / price_only                      |   41 | 0.585 | 0.576 |           0.463 |     1.031 |   0.151 | nan     |
|         1 | russia_ukraine_2022 | GRU / price_only                      |   62 | 0.613 | 0.612 |           0.484 |     1.827 |   0.034 | nan     |
|         1 | all                 | GRU / all                             |  995 | 0.513 | 0.512 |           0.507 |     0.781 |   0.218 | nan     |
|         1 | |ret|>5%            | GRU / all                             |   41 | 0.415 | 0.414 |           0.634 |    -0.771 |   0.78  | nan     |
|         1 | russia_ukraine_2022 | GRU / all                             |   62 | 0.468 | 0.468 |           0.468 |    -0.466 |   0.679 | nan     |
|         1 | all                 | GRU / shap_selected                   |  995 | 0.531 | 0.531 |           0.478 |     1.98  |   0.024 | nan     |
|         1 | |ret|>5%            | GRU / shap_selected                   |   41 | 0.488 | 0.483 |           0.512 |    -0.127 |   0.55  | nan     |
|         1 | russia_ukraine_2022 | GRU / shap_selected                   |   62 | 0.5   | 0.5   |           0.435 |     0.1   |   0.46  | nan     |
|         5 | all                 | Image CNN / volume_intensity (yearly) | 1042 | 0.447 | 0.442 |           0.399 |    -3.366 |   1     |   0.44  |
|         5 | |ret|>5%            | Image CNN / volume_intensity (yearly) |  270 | 0.422 | 0.398 |           0.319 |    -2.988 |   0.999 |   0.4   |
|         5 | russia_ukraine_2022 | Image CNN / volume_intensity (yearly) |   64 | 0.422 | 0.352 |           0.141 |    -1.294 |   0.902 |   0.353 |
|         5 | all                 | Random walk (= always up)             | 1042 | 0.509 | 0.337 |           1     |   nan     | nan     | nan     |
|         5 | |ret|>5%            | Random walk (= always up)             |  270 | 0.481 | 0.325 |           1     |   nan     | nan     | nan     |
|         5 | russia_ukraine_2022 | Random walk (= always up)             |   64 | 0.531 | 0.347 |           1     |   nan     | nan     | nan     |
|         5 | all                 | ARIMA                                 | 1042 | 0.496 | 0.496 |           0.481 |    -0.227 |   0.59  | nan     |
|         5 | |ret|>5%            | ARIMA                                 |  270 | 0.552 | 0.552 |           0.5   |     1.708 |   0.044 | nan     |
|         5 | russia_ukraine_2022 | ARIMA                                 |   64 | 0.609 | 0.609 |           0.484 |     1.784 |   0.037 | nan     |
|         5 | all                 | ARIMAX                                | 1042 | 0.523 | 0.523 |           0.496 |     1.492 |   0.068 | nan     |
|         5 | |ret|>5%            | ARIMAX                                |  270 | 0.493 | 0.492 |           0.485 |    -0.262 |   0.603 | nan     |
|         5 | russia_ukraine_2022 | ARIMAX                                |   64 | 0.5   | 0.492 |           0.594 |    -0.096 |   0.538 | nan     |
|         5 | all                 | LSTM / price_only                     | 1042 | 0.516 | 0.408 |           0.063 |     3.164 |   0.001 | nan     |
|         5 | |ret|>5%            | LSTM / price_only                     |  270 | 0.585 | 0.483 |           0.074 |     4.366 |   0     | nan     |
|         5 | russia_ukraine_2022 | LSTM / price_only                     |   64 | 0.484 | 0.351 |           0.016 |     0.954 |   0.17  | nan     |
|         5 | all                 | LSTM / all                            | 1042 | 0.489 | 0.469 |           0.689 |    -0.965 |   0.833 | nan     |
|         5 | |ret|>5%            | LSTM / all                            |  270 | 0.485 | 0.457 |           0.744 |    -0.218 |   0.586 | nan     |
|         5 | russia_ukraine_2022 | LSTM / all                            |   64 | 0.562 | 0.504 |           0.812 |     0.889 |   0.187 | nan     |
|         5 | all                 | LSTM / shap_selected                  | 1042 | 0.505 | 0.496 |           0.624 |     0.177 |   0.43  | nan     |
|         5 | |ret|>5%            | LSTM / shap_selected                  |  270 | 0.444 | 0.437 |           0.637 |    -1.729 |   0.958 | nan     |
|         5 | russia_ukraine_2022 | LSTM / shap_selected                  |   64 | 0.453 | 0.436 |           0.641 |    -0.937 |   0.826 | nan     |
|         5 | all                 | GRU / price_only                      | 1042 | 0.491 | 0.329 |           0     |   nan     | nan     | nan     |
|         5 | |ret|>5%            | GRU / price_only                      |  270 | 0.519 | 0.341 |           0     |   nan     | nan     | nan     |
|         5 | russia_ukraine_2022 | GRU / price_only                      |   64 | 0.469 | 0.319 |           0     |   nan     | nan     | nan     |
|         5 | all                 | GRU / all                             | 1042 | 0.51  | 0.506 |           0.573 |     0.544 |   0.293 | nan     |
|         5 | |ret|>5%            | GRU / all                             |  270 | 0.522 | 0.514 |           0.648 |     0.956 |   0.17  | nan     |
|         5 | russia_ukraine_2022 | GRU / all                             |   64 | 0.578 | 0.56  |           0.672 |     1.159 |   0.123 | nan     |
|         5 | all                 | GRU / shap_selected                   | 1042 | 0.538 | 0.532 |           0.606 |     2.417 |   0.008 | nan     |
|         5 | |ret|>5%            | GRU / shap_selected                   |  270 | 0.537 | 0.534 |           0.596 |     1.363 |   0.086 | nan     |
|         5 | russia_ukraine_2022 | GRU / shap_selected                   |   64 | 0.656 | 0.653 |           0.562 |     2.481 |   0.007 | nan     |
|        20 | all                 | Image CNN / volume_intensity (yearly) | 1042 | 0.53  | 0.524 |           0.437 |     1.533 |   0.063 |   0.524 |
|        20 | |ret|>5%            | Image CNN / volume_intensity (yearly) |  553 | 0.537 | 0.535 |           0.483 |     1.683 |   0.046 |   0.537 |
|        20 | russia_ukraine_2022 | Image CNN / volume_intensity (yearly) |   64 | 0.578 | 0.527 |           0.625 |     0.498 |   0.309 |   0.488 |
|        20 | all                 | Random walk (= always up)             | 1042 | 0.45  | 0.31  |           1     |   nan     | nan     | nan     |
|        20 | |ret|>5%            | Random walk (= always up)             |  553 | 0.458 | 0.314 |           1     |   nan     | nan     | nan     |
|        20 | russia_ukraine_2022 | Random walk (= always up)             |   64 | 0.703 | 0.413 |           1     |   nan     | nan     | nan     |
|        20 | all                 | ARIMA                                 | 1042 | 0.51  | 0.507 |           0.48  |     0.493 |   0.311 | nan     |
|        20 | |ret|>5%            | ARIMA                                 |  553 | 0.521 | 0.519 |           0.485 |     0.921 |   0.178 | nan     |
|        20 | russia_ukraine_2022 | ARIMA                                 |   64 | 0.531 | 0.514 |           0.484 |     0.664 |   0.253 | nan     |
|        20 | all                 | ARIMAX                                | 1042 | 0.477 | 0.476 |           0.499 |    -1.501 |   0.933 | nan     |
|        20 | |ret|>5%            | ARIMAX                                |  553 | 0.494 | 0.493 |           0.505 |    -0.281 |   0.611 | nan     |
|        20 | russia_ukraine_2022 | ARIMAX                                |   64 | 0.484 | 0.435 |           0.594 |    -0.965 |   0.833 | nan     |
|        20 | all                 | LSTM / price_only                     | 1042 | 0.513 | 0.461 |           0.862 |     4.66  |   0     | nan     |
|        20 | |ret|>5%            | LSTM / price_only                     |  553 | 0.539 | 0.476 |           0.89  |     5.429 |   0     | nan     |
|        20 | russia_ukraine_2022 | LSTM / price_only                     |   64 | 0.703 | 0.413 |           1     |   nan     | nan     | nan     |
|        20 | all                 | LSTM / all                            | 1042 | 0.436 | 0.421 |           0.709 |    -3.103 |   0.999 | nan     |
|        20 | |ret|>5%            | LSTM / all                            |  553 | 0.407 | 0.386 |           0.725 |    -3.915 |   1     | nan     |
|        20 | russia_ukraine_2022 | LSTM / all                            |   64 | 0.688 | 0.407 |           0.984 |    -0.66  |   0.745 | nan     |
|        20 | all                 | LSTM / shap_selected                  | 1042 | 0.548 | 0.546 |           0.489 |     3.047 |   0.001 | nan     |
|        20 | |ret|>5%            | LSTM / shap_selected                  |  553 | 0.557 | 0.557 |           0.521 |     2.777 |   0.003 | nan     |
|        20 | russia_ukraine_2022 | LSTM / shap_selected                  |   64 | 0.609 | 0.379 |           0.906 |    -1.685 |   0.954 | nan     |
|        20 | all                 | GRU / price_only                      | 1042 | 0.521 | 0.506 |           0.727 |     3.194 |   0.001 | nan     |
|        20 | |ret|>5%            | GRU / price_only                      |  553 | 0.546 | 0.527 |           0.741 |     3.595 |   0     | nan     |
|        20 | russia_ukraine_2022 | GRU / price_only                      |   64 | 0.672 | 0.443 |           0.938 |    -0.214 |   0.585 | nan     |
|        20 | all                 | GRU / all                             | 1042 | 0.477 | 0.477 |           0.551 |    -1.172 |   0.879 | nan     |
|        20 | |ret|>5%            | GRU / all                             |  553 | 0.461 | 0.461 |           0.559 |    -1.612 |   0.947 | nan     |
|        20 | russia_ukraine_2022 | GRU / all                             |   64 | 0.75  | 0.561 |           0.953 |     2.752 |   0.003 | nan     |
|        20 | all                 | GRU / shap_selected                   | 1042 | 0.488 | 0.483 |           0.656 |     0.28  |   0.39  | nan     |
|        20 | |ret|>5%            | GRU / shap_selected                   |  553 | 0.477 | 0.46  |           0.72  |    -0.207 |   0.582 | nan     |
|        20 | russia_ukraine_2022 | GRU / shap_selected                   |   64 | 0.703 | 0.413 |           1     |   nan     | nan     | nan     |