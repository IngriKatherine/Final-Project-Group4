"""
Everything a presenter might want to edit lives here: names, the repo link,
slide titles, headlines and speaker notes. The slide files only hold layout
and charts, so you can rewrite the story without touching any code.
"""

PROJECT_TITLE = "Forecasting crude oil prices, from price history to ships seen from space"
COURSE = "Deep Learning and Fundamentals of AI, final project"
GROUP = "Group 4"

TEAM = [
    "Ingri Katherine Quevedo Rocha",
    "Aditi Shukla",
    "Aidan Carslisle",
    "Constantine Zhang",
]

# Who presents which part (shown next to each slide's speaker notes).
# TODO: put the right name on each part.
ROLES = {
    "stats": "[ARIMA / ARIMAX owner]",          # random walk, ARIMA, ARIMAX
    "rnn": "[LSTM / GRU owner]",                # recurrent networks
    "image": "[image CNN owner]",               # price-chart images (Ren et al., 2024)
    "satellite": "Ingri Katherine Quevedo Rocha",   # Strait of Hormuz satellite CNN and 2026 forecasts
}

# TODO: paste the GitHub URL once the repository is public.
REPO_URL = "https://github.com/<owner>/Final-Project-Group4"

GOAL = (
    "Forecast crude oil prices, and test whether information beyond the price history (market drivers, "
    "price-chart images, and oil supply measured from satellite images of the Strait of Hormuz) "
    "makes the forecasts better, including through the 2026 Hormuz shock."
)

# The four model families, in the order the talk introduces them.
#   files: glob patterns under Code/ that show the family has results (status chip on slide 3)
FAMILIES = [
    dict(key="stats", name="Statistical baselines", models="Random walk, ARIMA, ARIMAX",
         input="Price history (ARIMAX adds market drivers)",
         output="Price h days ahead", color="#5F6B74",
         files=["results/predictions/baseline_arima_h*.csv", "results/predictions/baseline_arimax_h*.csv"]),
    dict(key="rnn", name="Recurrent networks", models="LSTM, GRU",
         input="30-day windows of daily returns (price only, or with market drivers)",
         output="Log return h days ahead", color="#2a78d6",
         files=["results/predictions/lstm_*_h*_s*.csv", "results/predictions/gru_*_h*_s*.csv"]),
    dict(key="image", name="Image CNN", models="CNN on price-chart images",
         input="The recent price path drawn as a picture",
         output="Direction of the next move (up / down)", color="#eda100",
         files=["results/predictions/imgcnn_*_h*_s*.csv"]),
    dict(key="satellite", name="Satellite CNN", models="Faster R-CNN ship detector + LSTM/GRU",
         input="Sentinel-2 images of the Strait of Hormuz; ship counts calibrated with IMF PortWatch",
         output="Oil-supply series, then the 2026 price forecast", color="#1baf7a",
         files=["satellite/data/series/supply_by_scene.csv", "results/predictions_2026/*.csv"]),
]

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
         notes="Introduce the four of us and the goal: forecast crude oil prices, and find out which kinds of "
               "information help. Point at the WTI line: twenty years of a price that refuses to be predicted."),
    dict(key="02_motivation", nav="Why oil prices?", icon=":material/show_chart:", section="Story",
         headline="Oil prices behave almost like a random walk",
         lede="Daily returns carry almost no memory, so 'tomorrow equals today' is the forecast to beat.",
         speaker="stats", minutes=0.75,
         notes="Use the range buttons for 2008, 2014-16, April 2020 and 2022. Then the autocorrelation bars: "
               "every lag sits inside the noise band. That is why the random walk is a strong benchmark and "
               "why beating it is a real result."),
    dict(key="03_approach", nav="Our approach", icon=":material/route:", section="Story",
         headline="Four ways to forecast one price",
         lede="Statistical models, recurrent networks, a CNN that reads price charts, and a CNN that counts "
              "tankers in the Strait of Hormuz.",
         speaker="all", minutes=1.0,
         notes="One sentence per family, left to right: what it reads and what it predicts. Then the four "
               "hypotheses; the verdicts fill in from the results files."),
    dict(key="04_data", nav="Data", icon=":material/database:", section="Data",
         headline="Thirteen market series, twenty years, one chronological split",
         lede="FRED and Yahoo Finance, aligned to business days; the satellite data come later in the talk.",
         speaker="stats", minutes=0.75,
         notes="Sources table first. Then Explore: switch to 'Indexed to 100' to compare series on one axis. "
               "Correlations tab: daily returns of most drivers are only weakly related to WTI, which is why "
               "ARIMAX and the multi-input networks have a hard job."),
    dict(key="05_preprocessing", nav="Preprocessing", icon=":material/tune:", section="Data",
         headline="Every window only sees the past",
         lede="Returns instead of levels, a repaired negative price, and splits assigned by target date.",
         speaker="stats", minutes=0.6,
         notes="Pipeline diagram, then the April 2020 zoom: WTI settled below zero for one day. Finish with "
               "the window demo: move the origin and show which 30 days the network reads and which day it predicts."),
    dict(key="06_baselines", nav="ARIMA and ARIMAX", icon=":material/flag:", section="Models",
         headline="ARIMA and ARIMAX: can statistics beat 'no change'?",
         lede="The random walk, a rolling ARIMA, and ARIMAX with market drivers, on the same test days as every network.",
         speaker="stats", minutes=1.0,
         notes="Equations first. ARIMAX adds the market drivers as regressors. RMSE grows with the horizon for "
               "all of them; the question is whether any statistical model gets under the random walk."),
    dict(key="07_networks", nav="LSTM and GRU", icon=":material/hub:", section="Models",
         headline="Gated recurrent networks decide what to remember",
         lede="A stacked LSTM or GRU reads a 30-day window and predicts one h-step log return.",
         speaker="rnn", minutes=1.0,
         notes="Equations for both cells, then our architecture. The parameter calculator shows how quickly "
               "weights grow with inputs. The cell playground shows the forget gate at work."),
    dict(key="08_training", nav="Training", icon=":material/model_training:", section="Models",
         headline="Validation loss bottoms out within a few epochs",
         lede="Adam, early stopping and three seeds per configuration, trained on the AWS GPU.",
         speaker="rnn", minutes=0.75,
         notes="Training settings, then learning curves. The best-epoch marker shows the network finds what "
               "little signal there is quickly and then starts to overfit, so early stopping does the heavy lifting."),
    dict(key="09_images", nav="Price charts as images", icon=":material/image:", section="Models",
         headline="Reading price charts like a trader: a CNN on images",
         lede="Following Ren et al. (2024), each window of prices becomes a small chart image and a CNN "
              "predicts whether the price goes up or down.",
         speaker="image", minutes=1.25,
         notes="Show the image builder: the same 20 days as a line and as a picture. The CNN never sees "
               "numbers, only pixels. Then the examples of up and down charts, the architecture, and the "
               "directional accuracy against a coin flip and against the LSTM/GRU."),
    dict(key="10_results", nav="Main results", icon=":material/leaderboard:", section="Results",
         headline="Which model forecasts best, 2022 to 2025?",
         lede="Seed-averaged test forecasts, compared on prices in USD per barrel against the random walk.",
         speaker="rnn", minutes=1.5,
         notes="Skill vs. random walk is the headline number: below zero means the model loses to 'no change'. "
               "Then the significance tab: Diebold-Mariano tests against the baselines and between input sets."),
    dict(key="11_ablations", nav="Ablations", icon=":material/science:", section="Results",
         headline="How much do depth, width, lookback and dropout matter?",
         lede="One setting changes at a time; settings are judged on validation error only.",
         speaker="rnn", minutes=0.6,
         notes="Pick each setting in turn. Validation RMSE picks the setting; test RMSE is shown for information only."),
    dict(key="12_forecasts", nav="Forecast explorer", icon=":material/timeline:", section="Results",
         headline="Where the forecasts go wrong",
         lede="Actual vs. predicted prices over the test period, with rolling error to show when models struggle.",
         speaker="all", minutes=0.6,
         notes="Zoom into the 2022 spike. The rolling RMSE chart shows every model failing at the same time: the "
               "errors come from shocks that no price-based input anticipated. That motivates the satellite data."),
    dict(key="13_satellite", nav="Ships from space", icon=":material/satellite_alt:", section="2026 and Hormuz",
         headline="Counting tankers in the Strait of Hormuz from space",
         lede="A CNN finds ships in Sentinel-2 images; IMF PortWatch transit data provide the labels that turn "
              "ship counts into an oil-supply series.",
         speaker="satellite", minutes=1.75,
         notes="Why Hormuz: about a fifth of the world's oil, effectively closed after 28 February 2026, and many "
               "tankers crossed with their AIS transponders off. Pipeline left to right. Show one image with the "
               "detections, the detector's accuracy, then the supply series against PortWatch."),
    dict(key="14_forecast2026", nav="2026 forecast", icon=":material/query_stats:", section="2026 and Hormuz",
         headline="Forecasting 2026 with oil supply measured from space",
         lede="The networks read the price history plus the satellite supply series; 2026 is held out as the test year.",
         speaker="satellite", minutes=1.5,
         notes="First the chart: price and satellite supply in 2026. Then the table: price only vs. price + AIS "
               "supply vs. price + satellite supply, all on the same 2026 days. If the satellite series beats "
               "AIS, the images carried information that ship tracking did not."),
    dict(key="15_conclusions", nav="Conclusions", icon=":material/task_alt:", section="Wrap-up",
         headline="What we can say, and what we cannot",
         lede="The verdict on each hypothesis, the limits of our test periods, and what comes next.",
         speaker="all", minutes=0.9,
         notes="Read the verdict on each hypothesis, two limitations, two next steps, then open for questions."),
    dict(key="16_lab", nav="Model lab (backup)", icon=":material/experiment:", section="Appendix",
         headline="Ask the trained network a what-if question",
         lede="Load a saved checkpoint, pick a day, shock one input, and watch the forecast move.",
         speaker="rnn", minutes=0,
         notes="Backup for questions: live inference with a saved LSTM or GRU checkpoint."),
]

SECTIONS = ["Story", "Data", "Models", "Results", "2026 and Hormuz", "Wrap-up", "Appendix"]

# -----------------------------------------------------------------------------
# Hypotheses (verdicts are computed in loaders.verdicts from the results files)
# -----------------------------------------------------------------------------
HYPOTHESES = [
    dict(key="H1", text="No model beats the random walk on price error 1 day ahead (2022 to 2025)."),
    dict(key="H2", text="ARIMAX, which adds market drivers, is no more accurate than ARIMA."),
    dict(key="H3", text="A CNN reading price-chart images calls the direction of the move better than a coin flip."),
    dict(key="H4", text="In 2026, adding the satellite oil-supply series lowers forecast error compared with price only."),
]

# The image-CNN design shown on slide 9. Edit to match the implementation.
IMAGE_CNN = dict(
    window=20,                  # days drawn in one image
    size=(64, 64),              # image height x width in pixels
    layers=[
        "Conv 5x3, 64 filters, batch norm, LeakyReLU, max-pool 2x1",
        "Conv 5x3, 128 filters, batch norm, LeakyReLU, max-pool 2x1",
        "Conv 5x3, 256 filters, batch norm, LeakyReLU, max-pool 2x1",
        "Dropout 0.5, fully connected, softmax over {down, up}",
    ],
    loss="Cross-entropy on the sign of the h-day return",
)

REFERENCES = [
    "Ren, X., Jiang, W., Ji, Q., & Zhai, P. (2024). Seeing is believing: Forecasting crude oil price trend from the "
    "perspective of images. Journal of Forecasting, 43(7), 2809-2821. https://doi.org/10.1002/for.3149",
    "Box, G. E. P., Jenkins, G. M., Reinsel, G. C., & Ljung, G. M. (2015). Time Series Analysis: Forecasting and Control (5th ed.). Wiley.",
    "Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. Neural Computation, 9(8).",
    "Cho, K., et al. (2014). Learning phrase representations using RNN encoder-decoder for statistical machine translation. EMNLP.",
    "Ren, S., He, K., Girshick, R., & Sun, J. (2015). Faster R-CNN: Towards real-time object detection with region proposal networks. NeurIPS.",
    "Beukema, P., et al. (2023). Satellite imagery and AI: A new era in ocean conservation, from research to deployment "
    "and impact (Ai2 Sentinel-2 vessel dataset). arXiv:2312.03207.",
    "IMF PortWatch (2026). Daily chokepoint transit calls and trade volume estimates. https://portwatch.imf.org",
    "Diebold, F. X., & Mariano, R. S. (1995). Comparing predictive accuracy. Journal of Business & Economic Statistics, 13(3).",
    "Data: FRED (Federal Reserve Bank of St. Louis); Yahoo Finance via yfinance; Copernicus Sentinel-2 L2A via the AWS "
    "Registry of Open Data (Earth Search).",
    "App structure follows the course's Streamlit examples (Data-Visualization/streamlit, A. Jafari).",
]

LIMITATIONS = [
    "Two test periods (2022 to 2025, and 2026) that both open with war-driven price spikes.",
    "Sentinel-2 is optical and daytime only: cloudy and dusty dates are dropped, so the supply series has gaps.",
    "At 10 m a tanker cannot be told from a container ship by shape; ship length is the proxy for oil capacity.",
    "The satellite series starts in 2023, so the 2026 models train on about three years of weekly data.",
]

FUTURE_WORK = [
    "Sentinel-1 radar (through NASA's ASF archive) to see through cloud and at night.",
    "More chokepoints and export terminals (Bab el-Mandeb, Ras Tanura, Kharg Island) as extra supply series.",
    "Attention or Transformer encoders that read chart images and supply series together.",
    "Rolling re-training through 2026, so the models adapt as the strait reopens.",
]
