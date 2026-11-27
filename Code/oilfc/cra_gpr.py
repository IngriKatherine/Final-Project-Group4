"""Point-in-time alignment of the daily GPR index (issue #10 follow-up).

The daily file is updated every Monday (the next business day if that Monday is
a federal holiday) and each update includes data through the release day.
At trading day t the usable value is the last observation of the latest release
made before t:
  * releases from the first archived vintage (2022-02-24) on: the vintage itself,
    so preliminary values and later revisions are as they were seen then;
  * earlier releases: the release schedule applied to the current file.
"""
from __future__ import annotations

import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay

from .cra_data import asof_on_calendar, load_cra_data, make_features
from .utils import path


def release_dates(dates) -> pd.DatetimeIndex:
    """Scheduled release that first contains each day."""
    d = pd.DatetimeIndex(dates)
    mon = d + pd.to_timedelta((-d.weekday) % 7, unit="D")
    hol = USFederalHolidayCalendar().holidays(mon.min(), mon.max() + pd.Timedelta(days=14))
    bday = CustomBusinessDay(holidays=hol)
    return pd.DatetimeIndex([m + bday if m in hol else m for m in mon])


def gpr_releases(final: pd.Series, vintages: pd.DataFrame) -> pd.DataFrame:
    """One row per release: release date, last date covered, latest value."""
    final = final.dropna().sort_index()
    first_v = vintages.vintage.min()
    rel = release_dates(final.index)
    sched = pd.DataFrame({"release": rel, "date": final.index, "GPR": final.values})
    sched = sched[sched.release < first_v].groupby("release").last().reset_index()
    vint = (vintages.sort_values("date").groupby("vintage").last().reset_index()
            .rename(columns={"vintage": "release"}))
    return pd.concat([sched.assign(source="schedule"), vint.assign(source="vintage")], ignore_index=True)


def gpr_point_in_time(final: pd.Series, vintages: pd.DataFrame, cal: pd.DatetimeIndex,
                      usable_after_days: int, staleness_rows: int) -> pd.Series:
    r = gpr_releases(final, vintages)
    s = pd.Series(r.GPR.values, index=r.release + pd.Timedelta(days=usable_after_days)).sort_index()
    return asof_on_calendar(s, cal, staleness_rows)


def load_followup_data(cfg: dict, fu: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Levels and features of cra_config.yaml with GPR replaced by its point-in-time version."""
    lv, _ = load_cra_data(cfg)
    final = pd.read_csv(path(cfg, "raw_dir", "gpr.csv"), index_col=0, parse_dates=True)["GPR"]
    vint = pd.read_csv(path(cfg, "raw_dir", "gpr_vintages.csv"), parse_dates=["vintage", "date"])
    g = fu["gpr"]
    lv = lv.copy()
    lv["GPR"] = gpr_point_in_time(final, vint, lv.index, g["usable_after_days"], g["staleness_rows"])
    if lv["GPR"].isna().any():
        raise ValueError(f"point-in-time GPR missing on {lv.index[lv['GPR'].isna()][:5].tolist()}")
    return lv, make_features(lv)
