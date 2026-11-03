# LSTMOilPriceForecasting
Forecast crude oil prices (WTI/Brent) with LSTM and GRU networks, and use SHAP to understand and select the input features.

- Data: oil prices plus candidate drivers (dollar index, gold, S&P 500, interest rates) from FRED and Yahoo Finance, all free

Reference papers:
1. The Interpretability of LSTM Models for Predicting Oil Company Stocks: Impact of Correlated Features - https://arxiv.org/html/2201.00350v5
2. An explainable deep learning framework for marine gas oil price forecasting: integrating LSTM with SHAP-based feature importance - https://link.springer.com/article/10.1186/s41072-026-00237-3

The two papers point in different directions. Firouzjaee & Khalilian found that adding correlated assets (crude oil, gold, the US dollar) to an LSTM mostly did not improve its predictions of oil company stocks. A 2026 paper in the Journal of Shipping and Trade instead used SHAP-based feature selection with an LSTM to forecast marine gas oil prices 8 weeks ahead, and it beat ARIMA, ARIMAX and standard LSTM setups. Our project would test whether SHAP-guided feature selection helps where simply adding more features doesn't.
