"""
Everything a presenter might want to edit lives here: names, the repo link,
slide titles, headlines and speaker notes. The slide files only hold layout
and charts, so you can rewrite the story without touching any code.
"""

PROJECT_TITLE = "Forecasting crude oil with LSTM, GRU and SHAP feature selection"
COURSE = "Deep Learning and Fundamentals of AI, final project"
GROUP = "Group 4"

# TODO: replace with the real names before presenting.
TEAM = ["[Name 1]", "[Name 2]", "[Name 3]"]

# Who owns which part (shown on the title slide and next to each slide's notes).
ROLES = {
    "data": "[Name 1]",      # data pipeline and baselines
    "model": "[Name 2]",     # LSTM / GRU modeling
    "shap": "[Name 3]",      # SHAP analysis
}

# TODO: paste the GitHub URL once the repository is public.
REPO_URL = "https://github.com/<owner>/Final-Project-Group4"

RESEARCH_QUESTION = (
    "Does SHAP-guided feature selection improve recurrent-network forecasts of "
    "crude oil prices in cases where simply adding more features does not?"
)

# -----------------------------------------------------------------------------
# The deck. Order here is the order of the presentation.
#   key      : file name in slides/ (without .py)
#   nav      : short label in the sidebar
#   section  : group in the sidebar and in the progress rail
#   headline : the claim the slide makes (big title on the slide)
#   lede     : one sentence under the headline
#   speaker  : key of ROLES, or "all"
#   minutes  : time budget, used by the presenter timer
#   notes    : speaker notes (shown when "Speaker notes" is on in the sidebar)
# -----------------------------------------------------------------------------
SLIDES = [
    dict(key="01_title", nav="Title", icon=":material/oil_barrel:", section="Story",
         headline=PROJECT_TITLE,
         lede="", speaker="all", minutes=0.3,
         notes="Introduce the group and the one question we answer. Point at the WTI line: "
               "20 years of a price that refuses to be predicted."),
    dict(key="02_motivation", nav="Why oil prices?", icon=":material/show_chart:", section="Story",
         headline="Oil prices behave almost like a random walk",
         lede="Daily returns carry almost no memory, so 'tomorrow equals today' is the forecast to beat.",
         speaker="data", minutes=0.75,
         notes="Use the range buttons to show 2008, 2014-16, April 2020 and 2022. Then the "
               "autocorrelation bars: every lag sits inside the noise band. That is why the random "
               "walk is a strong benchmark and why beating it is a real result."),
    dict(key="03_question", nav="Two papers, one question", icon=":material/balance:", section="Story",
         headline="Two papers disagree about adding more inputs",
         lede="One found extra drivers useless, the other found SHAP-selected drivers decisive. We test both on crude oil.",
         speaker="all", minutes=1.0,
         notes="Paper 1: correlated assets did not help an LSTM forecast oil-company stocks. Paper 2: "
               "SHAP selection helped an LSTM forecast marine gas oil. Our three input sets put both "
               "ideas in one experiment. Read the three hypotheses."),
    dict(key="04_data", nav="Data", icon=":material/database:", section="Data",
         headline="Thirteen public series, twenty years, one chronological split",
         lede="FRED and Yahoo Finance, aligned to business days from 2006 to 2025.",
         speaker="data", minutes=1.0,
         notes="Sources table first. Then Explore: switch to 'Indexed to 100' to compare series on "
               "one axis. Correlations tab: daily returns of most drivers are only weakly related to WTI."),
    dict(key="05_preprocessing", nav="Preprocessing", icon=":material/tune:", section="Data",
         headline="Every window only sees the past",
         lede="Returns instead of levels, a clipped negative price, and splits assigned by target date.",
         speaker="data", minutes=0.75,
         notes="Pipeline diagram, then the April 2020 zoom: WTI settled below zero and we floor it at "
               "$1 so log returns stay defined. Finish with the window demo: move the origin and show "
               "which 30 days the network reads and which day it predicts."),
    dict(key="06_baselines", nav="Baselines", icon=":material/flag:", section="Methods",
         headline="ARIMA cannot reliably beat 'no change' either",
         lede="The random walk and a rolling ARIMA set the bar on the same test origins the networks use.",
         speaker="data", minutes=0.75,
         notes="RMSE grows with the horizon for both. ARIMA is a hair better at 5 days and worse at 1 "
               "day. The random walk predicts 'no change', so directional accuracy does not apply to it."),
    dict(key="07_networks", nav="LSTM and GRU", icon=":material/hub:", section="Methods",
         headline="Gated recurrent networks decide what to remember",
         lede="A stacked LSTM or GRU reads a 30-day window and predicts one h-step log return.",
         speaker="model", minutes=1.25,
         notes="Equations for both cells, then our architecture. Use the parameter calculator to show "
               "that the default LSTM with all 13 inputs has 53,569 weights. The cell playground shows "
               "the forget gate at work."),
    dict(key="08_training", nav="Training", icon=":material/model_training:", section="Methods",
         headline="Validation loss bottoms out within a few epochs",
         lede="Adam, early stopping and three seeds per configuration, trained on the AWS GPU.",
         speaker="model", minutes=1.0,
         notes="Training settings, then learning curves. Point out the best epoch marker: the network "
               "finds what little signal there is quickly and then starts to overfit, so early "
               "stopping does the heavy lifting."),
    dict(key="09_shap", nav="SHAP", icon=":material/insights:", section="Methods",
         headline="SHAP ranks the drivers the network actually uses",
         lede="Expected-gradient SHAP on validation windows, averaged over lags, samples and seeds.",
         speaker="shap", minutes=2.25,
         notes="One line on Shapley values. Switch horizons: Brent and the energy ETF lead at 1 day, "
               "the S&P 500 rises at 20 days. The lag heatmap shows the most recent days matter most. "
               "Stability tab: how much the seeds agree."),
    dict(key="10_results", nav="Main results", icon=":material/leaderboard:", section="Results",
         headline="Main result: price only vs. all features vs. SHAP-selected",
         lede="Seed-averaged test forecasts for 2022 to 2025, compared on prices in USD per barrel.",
         speaker="all", minutes=2.25,
         notes="Skill vs. random walk is the headline number: below zero means the network loses to "
               "'no change'. Then the significance tab: Diebold-Mariano tests against the baselines "
               "and between feature sets."),
    dict(key="11_ablations", nav="Ablations", icon=":material/science:", section="Results",
         headline="How much do depth, width, lookback and dropout matter?",
         lede="One setting changes at a time; settings are judged on validation error only.",
         speaker="model", minutes=1.0,
         notes="Pick each parameter in turn. Validation RMSE picks the setting; test RMSE is shown "
               "for information only."),
    dict(key="12_forecasts", nav="Forecast explorer", icon=":material/timeline:", section="Results",
         headline="Where the forecasts go wrong",
         lede="Actual vs. predicted prices over the test period, with rolling error to show when models struggle.",
         speaker="all", minutes=0.75,
         notes="Zoom into the 2022 spike with the period buttons. The rolling RMSE chart shows every "
               "model failing at the same time: the errors come from shocks no input anticipated."),
    dict(key="13_lab", nav="Model lab", icon=":material/experiment:", section="Results",
         headline="Ask the trained network a what-if question",
         lede="Load a saved checkpoint, pick a day, shock one input, and watch the forecast move.",
         speaker="shap", minutes=0.75,
         notes="Live inference with the saved LSTM. Show the input heatmap (what the network sees), "
               "then the sensitivity chart: a 2-sigma move in each input over the last 5 days. It should "
               "roughly agree with the SHAP ranking."),
    dict(key="14_conclusions", nav="Conclusions", icon=":material/task_alt:", section="Wrap-up",
         headline="What we can say, and what we cannot",
         lede="Our answer to the research question, the limits of one test period, and what comes next.",
         speaker="all", minutes=1.0,
         notes="Read the verdict on each hypothesis, two limitations, two next steps, then open for questions."),
]

SECTIONS = ["Story", "Data", "Methods", "Results", "Wrap-up"]

# -----------------------------------------------------------------------------
# Papers and references
# -----------------------------------------------------------------------------
PAPERS = [
    dict(
        short="Firouzjaee & Khalilian (2022)",
        title="The interpretability of LSTM models for predicting oil company stocks: impact of correlated features",
        venue="arXiv:2201.00350",
        url="https://arxiv.org/html/2201.00350v5",
        target="Oil-company stock prices",
        method="LSTM, with crude oil, gold and the US dollar added as inputs",
        finding="The extra correlated inputs mostly did not improve the forecasts.",
        lesson="More features is not the same as more information.",
    ),
    dict(
        short="Okasha & Zaher (2026)",
        title="An explainable deep learning framework for marine gas oil price forecasting: "
              "integrating LSTM with SHAP-based feature importance",
        venue="Journal of Shipping and Trade, 11, 15",
        url="https://doi.org/10.1186/s41072-026-00237-3",
        target="Marine gas oil price, 8 weeks ahead",
        method="LSTM on the drivers ranked highest by SHAP",
        finding="Beat ARIMA, ARIMAX and an LSTM without feature selection.",
        lesson="Choosing the right few features can matter more than adding all of them.",
    ),
]

HYPOTHESES = [
    dict(key="H1", text="An LSTM or GRU that sees only the oil price does not beat the random walk at a 1-day horizon."),
    dict(key="H2", text="Adding every candidate driver does not reliably improve on the price-only network (as in paper 1)."),
    dict(key="H3", text="SHAP-selected drivers improve on both the price-only and the all-features network (as in paper 2)."),
]

REFERENCES = [
    "Firouzjaee, J. T., & Khalilian, P. (2022). The interpretability of LSTM models for predicting oil company stocks: "
    "impact of correlated features. arXiv:2201.00350.",
    "Okasha, S. H., & Zaher, H. (2026). An explainable deep learning framework for marine gas oil price forecasting: "
    "integrating LSTM with SHAP-based feature importance. Journal of Shipping and Trade, 11, 15.",
    "Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. Neural Computation, 9(8).",
    "Cho, K., et al. (2014). Learning phrase representations using RNN encoder-decoder for statistical machine translation. EMNLP.",
    "Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. NeurIPS.",
    "Diebold, F. X., & Mariano, R. S. (1995). Comparing predictive accuracy. Journal of Business & Economic Statistics, 13(3).",
    "Data: Federal Reserve Bank of St. Louis (FRED); Yahoo Finance via yfinance.",
    "App structure follows the course's Streamlit examples (Data-Visualization/streamlit, A. Jafari).",
]

LIMITATIONS = [
    "One commodity and one test period (2022 to 2025), which opens with an unusual war-driven price spike.",
    "Daily data: paper 2 worked with weekly prices 8 weeks ahead, where slower drivers have more room to matter.",
    "SHAP on correlated inputs (two dollar indices, two yields, WTI and Brent) splits credit between near-duplicates.",
    "Models are trained once; a rolling re-training scheme would adapt to new regimes.",
]

FUTURE_WORK = [
    "Weekly frequency and longer horizons, to match paper 2's setting.",
    "Attention or Transformer encoders, and a multi-horizon output head.",
    "Rolling-origin re-training and SHAP over time: does the dollar matter more in 2022 than in 2019?",
    "News and sentiment features, and grouped SHAP for correlated inputs.",
]
