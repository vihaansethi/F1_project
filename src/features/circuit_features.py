"""Circuit-level features: how a driver and team have historically done here."""
from __future__ import annotations

import numpy as np
import pandas as pd


def build_circuit_features(results: pd.DataFrame, races: pd.DataFrame) -> pd.DataFrame:
    """Return one row per (raceId, driverId) with circuit-history features."""
    merged = (
        results[["raceId", "driverId", "constructorId", "grid", "position", "points"]]
        .merge(races[["raceId", "circuitId", "year", "round", "date"]], on="raceId", how="left")
        .sort_values(["driverId", "circuitId", "date", "raceId"])
        .reset_index(drop=True)
    )

    merged["_win"] = merged["position"].eq(1).fillna(False).astype(int)
    merged["_podium"] = merged["position"].le(3).fillna(False).astype(int)
    merged["_points"] = merged["points"].fillna(0)
    merged["_position_for_avg"] = merged["position"].fillna(21.0)

    driver_circuit = merged.groupby(["driverId", "circuitId"], sort=False)
    merged["driver_circuit_starts"] = driver_circuit.cumcount()
    merged["driver_circuit_wins"] = driver_circuit["_win"].transform(
        lambda s: s.shift(1).fillna(0).cumsum()
    )
    merged["driver_circuit_podiums"] = driver_circuit["_podium"].transform(
        lambda s: s.shift(1).fillna(0).cumsum()
    )
    merged["driver_circuit_avg_position"] = driver_circuit["_position_for_avg"].transform(
        lambda s: s.shift(1).expanding(min_periods=1).mean()
    )
    merged["driver_circuit_avg_points"] = driver_circuit["_points"].transform(
        lambda s: s.shift(1).expanding(min_periods=1).mean()
    )

    out = merged[
        [
            "raceId",
            "driverId",
            "driver_circuit_starts",
            "driver_circuit_wins",
            "driver_circuit_podiums",
            "driver_circuit_avg_position",
            "driver_circuit_avg_points",
        ]
    ].copy()

    den = out["driver_circuit_starts"].replace(0, np.nan)
    out["driver_circuit_win_rate"] = out["driver_circuit_wins"] / den
    out["driver_circuit_podium_rate"] = out["driver_circuit_podiums"] / den
    return out


def build_circuit_static_features(circuits: pd.DataFrame, races: pd.DataFrame) -> pd.DataFrame:
    """Per-race static features that depend only on the circuit (lat/lng/altitude)."""
    return (
        races[["raceId", "circuitId"]]
        .merge(
            circuits[["circuitId", "lat", "lng", "alt", "country"]].rename(
                columns={
                    "lat": "circuit_lat",
                    "lng": "circuit_lng",
                    "alt": "circuit_alt",
                    "country": "circuit_country",
                }
            ),
            on="circuitId",
            how="left",
        )
    )


def build_circuit_pace_baseline(
    races: pd.DataFrame,
    lap_times: pd.DataFrame | None = None,
    results: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Median per-lap pace at each circuit, averaged over *prior* races.

    Acts as the circuit's typical pace baseline; without it, a regression model
    on lap times mostly learns "Monaco is slow, Monza is fast" and nothing
    interesting beyond that.

    Prefers ``lap_times`` (cleaner — direct per-lap median). Falls back to
    ``results['milliseconds'] / results['laps']`` if ``lap_times`` is absent.
    """
    if lap_times is not None and not lap_times.empty:
        per_race = (
            lap_times.groupby("raceId")["milliseconds"]
            .median()
            .reset_index(name="avg_lap_ms")
        )
    elif results is not None:
        per_race = results[["raceId", "milliseconds", "laps"]].copy()
        per_race["avg_lap_ms"] = np.where(
            per_race["laps"].fillna(0) > 0,
            per_race["milliseconds"] / per_race["laps"].replace(0, np.nan),
            np.nan,
        )
        per_race = (
            per_race.dropna(subset=["avg_lap_ms"]).groupby("raceId")["avg_lap_ms"].median().reset_index()
        )
    else:
        raise ValueError("Provide either lap_times or results to build the pace baseline")

    per_race = per_race.merge(races[["raceId", "circuitId", "date"]], on="raceId", how="left")
    per_race = per_race.sort_values(["circuitId", "date", "raceId"]).reset_index(drop=True)
    per_race["circuit_pace_baseline_ms"] = (
        per_race.groupby("circuitId")["avg_lap_ms"]
        .transform(lambda s: s.shift(1).expanding(min_periods=1).mean())
    )
    return per_race[["raceId", "circuit_pace_baseline_ms"]]
