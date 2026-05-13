"""Top-level feature builder that joins per-(race, driver) features into a model-ready frame."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

from .circuit_features import (
    build_circuit_features,
    build_circuit_pace_baseline,
    build_circuit_static_features,
)
from .constructor_features import build_constructor_features
from .driver_features import build_driver_features


@dataclass
class FeatureFrames:
    """All model targets and features in one frame, plus column groupings."""

    df: pd.DataFrame
    feature_cols: list[str]
    target_cols: dict[str, str]


def build_feature_table(
    tables: Dict[str, pd.DataFrame],
    min_year: Optional[int] = None,
) -> FeatureFrames:
    """Join cleaned tables into a single feature/target frame.

    Parameters
    ----------
    tables:
        Output of :class:`f1ml.data.DataCleaner` — a dict of cleaned tables.
    min_year:
        Optional inclusive lower bound on race year (e.g. 2010 to focus on
        the modern era).
    """
    required = {"results", "races", "drivers", "constructors", "circuits"}
    missing = required - tables.keys()
    if missing:
        raise KeyError(f"missing cleaned tables: {sorted(missing)}")

    results = tables["results"]
    races = tables["races"]
    drivers = tables["drivers"]
    constructors = tables["constructors"]
    circuits = tables["circuits"]
    qualifying = tables.get("qualifying")
    lap_times = tables.get("lap_times")

    driver_form = build_driver_features(results, races)
    constructor_form = build_constructor_features(results, races)
    circuit_form = build_circuit_features(results, races)
    circuit_static = build_circuit_static_features(circuits, races)
    circuit_pace = build_circuit_pace_baseline(races, lap_times=lap_times, results=results)

    base_cols = [
        "raceId", "driverId", "constructorId", "grid", "position", "points",
        "statusId", "milliseconds", "laps",
    ]
    base = results[base_cols].copy()
    base = base.merge(
        races[["raceId", "circuitId", "year", "round", "date", "name"]].rename(columns={"name": "race_name"}),
        on="raceId",
        how="left",
    )

    df = (
        base.merge(driver_form, on=["raceId", "driverId"], how="left")
        .merge(constructor_form, on=["raceId", "constructorId"], how="left")
        .merge(circuit_form, on=["raceId", "driverId"], how="left")
        .merge(circuit_static, on=["raceId", "circuitId"], how="left")
        .merge(circuit_pace, on="raceId", how="left")
    )

    if qualifying is not None and "best_quali_ms" in qualifying.columns:
        df = df.merge(
            qualifying[["raceId", "driverId", "best_quali_ms", "position"]].rename(
                columns={"position": "quali_position"}
            ),
            on=["raceId", "driverId"],
            how="left",
        )

    df = df.merge(
        drivers[["driverId", "driver_name", "nationality", "dob"]].rename(
            columns={"nationality": "driver_nationality"}
        ),
        on="driverId",
        how="left",
    )
    df = df.merge(
        constructors[["constructorId", "name", "nationality"]].rename(
            columns={"name": "constructor_name", "nationality": "constructor_nationality"}
        ),
        on="constructorId",
        how="left",
    )

    df["driver_age_years"] = (
        (df["date"] - df["dob"]).dt.days / 365.25
    ).astype(float)

    # ----- Targets ----------------------------------------------------------
    df["target_win"] = df["position"].eq(1).astype(int)
    df["target_podium"] = df["position"].le(3).fillna(False).astype(int)
    df["target_points_finish"] = df["points"].fillna(0).gt(0).astype(int)
    # Median per-lap pace (ms). Computed from lap_times when available (much
    # cleaner than total race time, which is polluted by safety cars and red
    # flags); falls back to (milliseconds / laps) otherwise.
    if lap_times is not None and not lap_times.empty:
        pace = (
            lap_times.groupby(["raceId", "driverId"])["milliseconds"]
            .median()
            .reset_index()
            .rename(columns={"milliseconds": "target_avg_lap_ms"})
        )
        df = df.merge(pace, on=["raceId", "driverId"], how="left")
    else:
        df["target_avg_lap_ms"] = np.where(
            df["laps"].fillna(0) > 0, df["milliseconds"] / df["laps"].replace(0, np.nan), np.nan
        )

    if min_year is not None:
        df = df[df["year"] >= min_year].copy()

    feature_cols = [
        "grid",
        "driver_age_years",
        "driver_career_starts",
        "driver_career_wins",
        "driver_career_podiums",
        "driver_career_points",
        "driver_career_win_rate",
        "driver_career_podium_rate",
        "driver_last5_avg_position",
        "driver_last5_avg_points",
        "driver_last5_finish_rate",
        "driver_last10_win_rate",
        "driver_last10_podium_rate",
        "constructor_career_starts",
        "constructor_career_wins",
        "constructor_career_podiums",
        "constructor_career_win_rate",
        "constructor_career_podium_rate",
        "constructor_last5_avg_points",
        "constructor_last5_win_rate",
        "constructor_last10_podium_rate",
        "constructor_season_points_to_date",
        "driver_circuit_starts",
        "driver_circuit_wins",
        "driver_circuit_podiums",
        "driver_circuit_avg_position",
        "driver_circuit_avg_points",
        "driver_circuit_win_rate",
        "driver_circuit_podium_rate",
        "circuit_alt",
        "circuit_lat",
        "circuit_lng",
        "circuit_pace_baseline_ms",
    ]
    if "best_quali_ms" in df.columns:
        feature_cols.extend(["best_quali_ms", "quali_position"])

    target_cols = {
        "win": "target_win",
        "podium": "target_podium",
        "points_finish": "target_points_finish",
        "avg_lap_ms": "target_avg_lap_ms",
    }
    return FeatureFrames(df=df, feature_cols=feature_cols, target_cols=target_cols)
