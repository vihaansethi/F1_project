"""Per-(race, driver) rolling features computed from past results only."""
from __future__ import annotations

import numpy as np
import pandas as pd


def build_driver_features(results: pd.DataFrame, races: pd.DataFrame) -> pd.DataFrame:
    """Return one row per (raceId, driverId) with driver form features.

    All features look strictly backwards in time — for race ``r``, we only use
    finishes that took place before race ``r``. This keeps the features
    leak-free for downstream models.
    """
    df = (
        results[["raceId", "driverId", "grid", "position", "points", "statusId"]]
        .merge(races[["raceId", "year", "round", "date"]], on="raceId", how="left")
        .sort_values(["driverId", "date", "raceId"])
        .reset_index(drop=True)
    )

    df["_finished"] = df["position"].notna().astype(int)
    df["_win"] = df["position"].eq(1).fillna(False).astype(int)
    df["_podium"] = df["position"].le(3).fillna(False).astype(int)
    df["_points"] = df["points"].fillna(0).astype(float)
    df["_position_for_avg"] = df["position"].fillna(21.0)  # treat DNF as worse than last

    g = df.groupby("driverId", sort=False)

    out = pd.DataFrame({"raceId": df["raceId"].values, "driverId": df["driverId"].values})
    out["driver_career_starts"] = g.cumcount().values
    out["driver_career_wins"] = _shifted_cumsum(g["_win"]).values
    out["driver_career_podiums"] = _shifted_cumsum(g["_podium"]).values
    out["driver_career_points"] = _shifted_cumsum(g["_points"]).values
    out["driver_career_finishes"] = _shifted_cumsum(g["_finished"]).values
    out["driver_last5_avg_position"] = _shifted_rolling_mean(g["_position_for_avg"], 5).values
    out["driver_last5_avg_points"] = _shifted_rolling_mean(g["_points"], 5).values
    out["driver_last5_finish_rate"] = _shifted_rolling_mean(g["_finished"], 5).values
    out["driver_last10_win_rate"] = _shifted_rolling_mean(g["_win"], 10).values
    out["driver_last10_podium_rate"] = _shifted_rolling_mean(g["_podium"], 10).values

    out["driver_career_win_rate"] = _safe_div(out["driver_career_wins"], out["driver_career_starts"])
    out["driver_career_podium_rate"] = _safe_div(out["driver_career_podiums"], out["driver_career_starts"])
    return out


def _shifted_cumsum(grouped_series) -> pd.Series:
    """Cumulative sum within each group, lagged one row so the current race is excluded."""
    return grouped_series.transform(lambda s: s.shift(1).fillna(0).cumsum())


def _shifted_rolling_mean(grouped_series, window: int) -> pd.Series:
    """Rolling mean within each group over the previous ``window`` rows."""
    return grouped_series.transform(
        lambda s: s.shift(1).rolling(window, min_periods=1).mean()
    )


def _safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
    den = den.replace(0, np.nan)
    return num / den
