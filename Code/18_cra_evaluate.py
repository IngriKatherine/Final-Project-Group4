"""
18_cra_evaluate.py  -  Compare the gated residual LSTM, the original LSTM, Chronos-2 and the random walk (issue #10)

Seed-averaged forecasts on identical origins, separately for the test period
(test_years) and the post-release holdout (holdout_years).
  * Accuracy: RMSE, MAE, MAPE, directional accuracy, out-of-sample R^2 vs RW,
    prediction std ratio, seed dispersion, 80% interval coverage, mean gate
  * Primary test: DM-HLN, gated residual LSTM vs original LSTM, Holm-adjusted over horizons
  * Also: DM vs Chronos-2 and vs RW; Clark-West vs RW (nested); Pesaran-Timmermann
    on non-overlapping origins

Outputs
  results/cra/tables/metrics.csv, tests.csv, rmse_by_year.csv, variable_weights.csv, summary.md
  results/cra/figures/rmse_vs_rw.png, forecast_holdout_h{h}.png

Usage
  python 18_cra_evaluate.py
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from oilfc.cra_data import load_prior
from oilfc.cra_metrics import holm, interval_coverage, pesaran_timmermann
from oilfc.metrics import all_metrics, clark_west, diebold_mariano, oos_r2
from oilfc.utils import CODE_DIR, load_config, load_json, path

LABELS = {"rw": "Random walk", "chronos": "Chronos-2", "lstm": "Original LSTM", "cra": "Gated residual LSTM"}


def load_runs(cfg, model, h, years) -> list[pd.DataFrame]:
    """One frame per seed, all years concatenated."""
    out = []
    for s in cfg["evaluation"]["seeds"]:
        fr = [pd.read_csv(path(cfg, "results_dir", "predictions", f"{model}_h{h}_y{y}_s{s}.csv"),
                          parse_dates=["origin_date", "target_date"]) for y in years]
        out.append(pd.concat(fr, ignore_index=True))
    return out


def seed_mean(frames: list[pd.DataFrame]) -> pd.DataFrame:
    out = frames[0].copy()
    for c in ("pred_price", "q10_price", "q90_price", "gate"):
        if c in out:
            out[c] = np.mean([f[c].values for f in frames], axis=0)
    return out


def main():
    cfg = load_config(CODE_DIR / "cra_config.yaml")
    ev = cfg["evaluation"]
    years = ev["test_years"] + ev["holdout_years"]
    periods = {"test " + f"{ev['test_years'][0]}-{ev['test_years'][-1]}": ev["test_years"],
               "holdout " + "-".join(map(str, ev["holdout_years"])): ev["holdout_years"]}
    prior = load_prior(cfg)
    metric_rows, test_rows, year_rows, w_rows = [], [], [], []

    for h in cfg["framing"]["horizons"]:
        seeds = {m: load_runs(cfg, m, h, years) for m in ("lstm", "cra")}
        f = {m: seed_mean(fr) for m, fr in seeds.items()}
        assert f["lstm"].origin_date.equals(f["cra"].origin_date), "models must share origins"
        base = f["cra"][["origin_date", "target_date", "base_price", "true_price"]]
        q = prior.xs(h, level="horizon").reindex(base.origin_date)
        f["rw"] = base.assign(pred_price=base.base_price)
        f["chronos"] = base.assign(pred_price=base.base_price.values * np.exp(q.q50.values),
                                   q10_price=base.base_price.values * np.exp(q.q10.values),
                                   q90_price=base.base_price.values * np.exp(q.q90.values))

        for pname, pyears in periods.items():
            sel = base.target_date.dt.year.isin(pyears).values
            g = {m: d.loc[sel].reset_index(drop=True) for m, d in f.items()}
            y, b = g["rw"].true_price.values, g["rw"].base_price.values
            act = np.log(y / b)
            for m, d in g.items():
                pr = np.log(d.pred_price.values / b)
                row = {"period": pname, "horizon": h, "model": LABELS[m], "n": int(sel.sum()),
                       **all_metrics(y, d.pred_price, b),
                       "R2_oos_vs_RW": oos_r2(y, d.pred_price, b),
                       "pred_std_ratio": pr.std() / act.std(),
                       "corr_pred_actual": np.corrcoef(pr, act)[0, 1] if pr.std() > 0 else np.nan}
                if m in seeds:
                    rm = [np.sqrt(np.mean((fr.loc[sel].true_price - fr.loc[sel].pred_price) ** 2)) for fr in seeds[m]]
                    row["seed_RMSE_std"] = float(np.std(rm))
                if "q10_price" in d:
                    row["coverage80"] = interval_coverage(y, d.q10_price, d.q90_price)
                if "gate" in d:
                    row["gate_mean"] = float(d.gate.mean())
                metric_rows.append(row)

            def add(model, vs, nested):
                r = {"period": pname, "horizon": h, "model": LABELS[model], "vs": LABELS[vs],
                     **diebold_mariano(y, g[model].pred_price, g[vs].pred_price, h=h)}
                if nested:
                    r.update(clark_west(y, g[vs].pred_price, g[model].pred_price, h=h))
                test_rows.append(r)

            add("cra", "lstm", False)
            add("cra", "chronos", False)
            for m in ("cra", "lstm", "chronos"):
                add(m, "rw", True)
            step = np.arange(0, len(y), h)                   # non-overlapping origins
            for m in ("cra", "lstm", "chronos"):
                pt = pesaran_timmermann(act[step], np.log(g[m].pred_price.values / b)[step])
                test_rows.append({"period": pname, "horizon": h, "model": LABELS[m], "vs": "direction (PT)",
                                  "PT": pt["PT"], "PT_p_value": pt["PT_p_value"], "hit_rate": pt["hit_rate"],
                                  "n_nonoverlap": pt["n"]})

        for m, d in f.items():
            for yr, dd in d.groupby(d.target_date.dt.year):
                year_rows.append({"horizon": h, "year": yr, "model": LABELS[m],
                                  "RMSE": np.sqrt(np.mean((dd.true_price - dd.pred_price) ** 2))})
        for s in ev["seeds"]:
            for yr in ev["test_years"]:
                w = load_json(path(cfg, "results_dir", "runs", f"cra_h{h}_y{yr}_s{s}.json"))["vsn_weights"]
                w_rows.append({"horizon": h, **w})

    metrics, tests = pd.DataFrame(metric_rows), pd.DataFrame(test_rows)
    primary = (tests.model == LABELS["cra"]) & (tests.vs == LABELS["lstm"])
    tests["p_holm_primary"] = np.nan
    for pname in periods:
        k = primary & (tests.period == pname)
        tests.loc[k, "p_holm_primary"] = holm(tests.loc[k, "p_value"].values)
    by_year = pd.DataFrame(year_rows).pivot_table(index=["horizon", "year"], columns="model", values="RMSE")
    weights = pd.DataFrame(w_rows).groupby("horizon").mean().T

    tab = path(cfg, "results_dir", "tables")
    tab.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(tab / "metrics.csv", index=False)
    tests.to_csv(tab / "tests.csv", index=False)
    by_year.to_csv(tab / "rmse_by_year.csv")
    weights.to_csv(tab / "variable_weights.csv")
    cols = ["period", "horizon", "model", "RMSE", "MAE", "DirAcc", "R2_oos_vs_RW", "pred_std_ratio",
            "corr_pred_actual", "seed_RMSE_std", "coverage80", "gate_mean"]
    dm_cols = ["period", "horizon", "model", "vs", "DM", "p_value", "p_holm_primary", "CW", "CW_p_value"]
    with open(tab / "summary.md", "w") as fh:
        fh.write(metrics[cols].round(4).to_markdown(index=False) + "\n\n")
        fh.write(tests[tests.DM.notna()][dm_cols].round(4).to_markdown(index=False) + "\n\n")
        fh.write(tests[tests.vs == "direction (PT)"][["period", "horizon", "model", "hit_rate", "PT", "PT_p_value",
                                                       "n_nonoverlap"]].round(4).to_markdown(index=False) + "\n")
    plots(cfg, metrics, f, periods)
    print(metrics[cols].round(4).to_string(index=False))
    print(tests[tests.DM.notna()][dm_cols].round(4).to_string(index=False))


def plots(cfg, metrics, f_last, periods):
    fig_dir = path(cfg, "results_dir", "figures")
    fig_dir.mkdir(parents=True, exist_ok=True)
    m = metrics.copy()
    rw = m[m.model == LABELS["rw"]].set_index(["period", "horizon"]).RMSE
    m["ratio"] = m.RMSE.values / rw.reindex(pd.MultiIndex.from_frame(m[["period", "horizon"]])).values
    m = m[m.model != LABELS["rw"]]
    fig, axes = plt.subplots(1, len(periods), figsize=(5 * len(periods), 3.5), sharey=True)
    for ax, p in zip(np.atleast_1d(axes), periods):
        m[m.period == p].pivot(index="horizon", columns="model", values="ratio").plot.bar(ax=ax, rot=0)
        ax.axhline(1.0, color="black", lw=0.8)
        ax.set_title(p); ax.set_ylabel("RMSE / random-walk RMSE"); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(fig_dir / "rmse_vs_rw.png", dpi=150); plt.close(fig)

    h = cfg["framing"]["horizons"][-1]
    d = f_last["cra"]
    hold = d.target_date.dt.year.isin(list(periods.values())[-1])
    fig, ax = plt.subplots(figsize=(10, 3.8))
    dd = d[hold]
    ax.plot(dd.target_date, dd.true_price, color="black", lw=1, label="Actual")
    ax.plot(dd.target_date, f_last["lstm"][hold].pred_price, lw=0.8, label=LABELS["lstm"])
    ax.plot(dd.target_date, f_last["chronos"][hold].pred_price, lw=0.8, label=LABELS["chronos"])
    ax.plot(dd.target_date, dd.pred_price, lw=0.9, label=LABELS["cra"])
    ax.fill_between(dd.target_date, dd.q10_price, dd.q90_price, alpha=0.2, label="80% interval")
    ax.set_title(f"Holdout: {h}-day-ahead WTI forecasts"); ax.set_ylabel("USD / bbl"); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(fig_dir / f"forecast_holdout_h{h}.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    main()
