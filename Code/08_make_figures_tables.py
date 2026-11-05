"""
08_make_figures_tables.py  -  Final comparison tables and figures for the report (shared)

  * Combines baselines (03) and RNN runs (06) into one results table per horizon
  * Diebold-Mariano tests: each RNN configuration vs. the random walk and vs. ARIMA
    (seed-averaged forecasts, same test origins)
  * Figures: test-period forecasts vs. actual, RMSE by feature set and horizon,
    learning curves

Outputs
  results/tables/final_comparison.csv   and  final_comparison.md (paste into the report)
  results/tables/dm_tests.csv
  results/figures/forecast_vs_actual_h{h}.png
  results/figures/rmse_by_featureset.png
  results/figures/learning_curves_h{h}.png

Usage
  python 08_make_figures_tables.py
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from oilfc.metrics import all_metrics, diebold_mariano
from oilfc.utils import load_config, load_json, path


def seed_averaged(cfg, model, features, h, seeds) -> pd.DataFrame | None:
    frames = []
    for s in seeds:
        f = path(cfg, "results_dir", "predictions", f"{model}_{features}_h{h}_s{s}.csv")
        if f.exists():
            frames.append(pd.read_csv(f, parse_dates=["origin_date", "target_date"]))
    if not frames:
        return None
    out = frames[0].copy()
    out["pred_price"] = sum(fr["pred_price"] for fr in frames) / len(frames)
    return out


def main():
    cfg = load_config()
    e = cfg["experiments"]
    rows, dm_rows = [], []

    for h in cfg["framing"]["horizons"]:
        base = {}
        for name in ("rw", "arima"):
            f = path(cfg, "results_dir", "predictions", f"baseline_{name}_h{h}.csv")
            if f.exists():
                base[name] = pd.read_csv(f, parse_dates=["origin_date", "target_date"])
                b = base[name]
                rows.append({"horizon": h, "model": {"rw": "Random walk", "arima": "ARIMA"}[name],
                             "features": "price_only", **all_metrics(b.true_price, b.pred_price, b.base_price)})

        fig, ax = plt.subplots(figsize=(11, 4))
        plotted_actual = False
        for m in e["models"]:
            for fs in e["feature_sets"]:
                p = seed_averaged(cfg, m, fs, h, e["seeds"])
                if p is None:
                    continue
                rows.append({"horizon": h, "model": m.upper(), "features": fs,
                             **all_metrics(p.true_price, p.pred_price, p.base_price)})
                for bname, b in base.items():
                    j = p.merge(b, on="origin_date", suffixes=("", "_b"))
                    dm_rows.append({"horizon": h, "model": m.upper(), "features": fs, "vs": bname,
                                    **diebold_mariano(j.true_price, j.pred_price, j.pred_price_b, h=h)})
                if not plotted_actual:
                    ax.plot(p.target_date, p.true_price, color="black", lw=1, label="Actual")
                    plotted_actual = True
                ax.plot(p.target_date, p.pred_price, lw=0.8, label=f"{m.upper()} / {fs}")
        if plotted_actual:
            ax.set_title(f"Test period: {h}-step-ahead forecasts ({cfg['data']['target']})")
            ax.set_ylabel("USD / bbl"); ax.legend(fontsize=7, ncol=3); fig.tight_layout()
            fig.savefig(path(cfg, "results_dir", "figures", f"forecast_vs_actual_h{h}.png"), dpi=150)
        plt.close(fig)

        # learning curves for seed 0
        fig, axes = plt.subplots(1, len(e["feature_sets"]), figsize=(4 * len(e["feature_sets"]), 3), sharey=True)
        for ax, fs in zip(axes, e["feature_sets"]):
            for m in e["models"]:
                f = path(cfg, "results_dir", "tables", "runs", f"{m}_{fs}_h{h}_s{e['seeds'][0]}.json")
                if f.exists():
                    hist = load_json(f)["history"]
                    ax.plot(hist["train_loss"], label=f"{m} train")
                    ax.plot(hist["val_loss"], ls="--", label=f"{m} val")
            ax.set_title(fs); ax.set_xlabel("epoch")
        axes[0].set_ylabel("loss (scaled)"); axes[0].legend(fontsize=7); fig.tight_layout()
        fig.savefig(path(cfg, "results_dir", "figures", f"learning_curves_h{h}.png"), dpi=150)
        plt.close(fig)

    table = pd.DataFrame(rows)
    table.to_csv(path(cfg, "results_dir", "tables", "final_comparison.csv"), index=False)
    with open(path(cfg, "results_dir", "tables", "final_comparison.md"), "w") as f:
        f.write(table.round(3).to_markdown(index=False))
    pd.DataFrame(dm_rows).to_csv(path(cfg, "results_dir", "tables", "dm_tests.csv"), index=False)

    if not table.empty:
        rnn = table[~table.model.isin(["Random walk", "ARIMA"])]
        if not rnn.empty:
            piv = rnn.pivot_table(index=["horizon", "model"], columns="features", values="RMSE")
            ax = piv.plot.bar(figsize=(8, 4), rot=0)
            rw = table[table.model == "Random walk"].set_index("horizon")["RMSE"]
            ax.set_ylabel("test RMSE (USD/bbl)")
            ax.set_title("Test RMSE by feature set (random-walk RMSE per horizon: "
                         + ", ".join(f"h{k}={v:.2f}" for k, v in rw.items()) + ")", fontsize=9)
            plt.tight_layout()
            plt.savefig(path(cfg, "results_dir", "figures", "rmse_by_featureset.png"), dpi=150)
        print(table.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
