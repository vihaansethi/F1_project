"""Constructor (team) form features computed from past results only."""
from __future__ import annotations

import numpy as np
import pandas as pd


def build_constructor_features(results: pd.DataFrame, races: pd.DataFrame) -> pd.DataFrame:
    """Return one row per (raceId, constructorId) with team form features.

    Per-race aggregates are computed first (sum of points, best finish across
    both drivers), then lagged rolling stats are computed within each
    constructor so the current race is excluded.
    """
    merged = (
        results[["raceId", "constructorId", "position", "points"]]
        .merge(races[["raceId", "year", "round", "date"]], on="raceId", how="left")
    )

    # One row per (race, constructor)
    agg = (
        merged.groupby(["constructorId", "raceId", "date", "year"], dropna=False)
        .agg(
            team_race_points=("points", "sum"),
            team_best_position=("position", "min"),
        )
        .reset_index()
        .sort_values(["constructorId", "date", "raceId"])
        .reset_index(drop=True)
    )

    agg["_win"] = agg["team_best_position"].eq(1).fillna(False).astype(int)
    agg["_podium"] = agg["team_best_position"].le(3).fillna(False).astype(int)
    agg["_points"] = agg["team_race_points"].fillna(0)

    g = agg.groupby("constructorId", sort=False)

    out = pd.DataFrame({"raceId": agg["raceId"].values, "constructorId": agg["constructorId"].values})
    out["constructor_career_starts"] = g.cumcount().values
    out["constructor_career_wins"] = g["_win"].transform(lambda s: s.shift(1).fillna(0).cumsum()).values
    out["constructor_career_podiums"] = g["_podium"].transform(lambda s: s.shift(1).fillna(0).cumsum()).values
    out["constructor_last5_avg_points"] = (
        g["_points"].transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean()).values
    )
    out["constructor_last5_win_rate"] = (
        g["_win"].transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean()).values
    )
    out["constructor_last10_podium_rate"] = (
        g["_podium"].transform(lambda s: s.shift(1).rolling(10, min_periods=1).mean()).values
    )
    out["constructor_season_points_to_date"] = (
        agg.groupby(["constructorId", "year"], sort=False)["_points"]
        .transform(lambda s: s.shift(1).fillna(0).cumsum())
        .values
    )

    den = out["constructor_career_starts"].replace(0, np.nan)
    out["constructor_career_win_rate"] = out["constructor_career_wins"] / den
    out["constructor_career_podium_rate"] = out["constructor_career_podiums"] / den
    return out
