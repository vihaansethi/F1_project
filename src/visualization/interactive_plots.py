"""Plotly interactive plots for the F1 dataset."""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def world_map_of_circuits(circuits: pd.DataFrame) -> go.Figure:
    """Geo scatter of every circuit, colored by altitude."""
    df = circuits.copy()
    df["alt_for_size"] = df["alt"].abs().clip(lower=1)
    fig = px.scatter_geo(
        df,
        lat="lat",
        lon="lng",
        hover_name="name",
        hover_data={"location": True, "country": True, "alt": True, "lat": False, "lng": False},
        color="alt",
        size="alt_for_size",
        color_continuous_scale="RdYlBu_r",
        size_max=15,
        title="F1 Circuits Around the World",
        labels={"alt": "Altitude (m)"},
    )
    fig.update_layout(
        height=600,
        geo=dict(showland=True, landcolor="rgb(243, 243, 243)", coastlinecolor="rgb(204, 204, 204)"),
    )
    return fig


def driver_career_timeline(tables: Dict[str, pd.DataFrame], driver_id: int) -> go.Figure:
    """Line chart of a single driver's points by race over their whole career."""
    results = tables["results"]
    races = tables["races"]
    drivers = tables["drivers"]

    driver_row = drivers.loc[drivers["driverId"] == driver_id]
    if driver_row.empty:
        raise ValueError(f"unknown driverId: {driver_id}")
    name = driver_row.iloc[0].get("driver_name") or f"Driver {driver_id}"

    df = (
        results[results["driverId"] == driver_id]
        .merge(races[["raceId", "year", "round", "name", "date"]], on="raceId", how="left")
        .sort_values("date")
    )
    df["cumulative_points"] = df["points"].fillna(0).cumsum()
    df["cumulative_wins"] = df["position"].eq(1).cumsum()

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        subplot_titles=["Cumulative points", "Cumulative wins"],
        vertical_spacing=0.1,
    )
    fig.add_trace(
        go.Scatter(x=df["date"], y=df["cumulative_points"], mode="lines", name="Points", line=dict(color="#1f77b4")),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=df["date"], y=df["cumulative_wins"], mode="lines", name="Wins", line=dict(color="#d62728")),
        row=2, col=1,
    )
    fig.update_layout(height=600, title_text=f"{name} — Career Progression", showlegend=False)
    return fig


def championship_standings_chart(
    tables: Dict[str, pd.DataFrame], year: int, top_n: int = 10
) -> go.Figure:
    """Driver standings evolution across a single season."""
    standings = tables["driver_standings"]
    races = tables["races"]
    drivers = tables["drivers"]

    season_races = races[races["year"] == year][["raceId", "round", "date", "name"]].sort_values("round")
    if season_races.empty:
        raise ValueError(f"no races for year {year}")

    merged = (
        standings.merge(season_races, on="raceId", how="inner")
        .merge(drivers[["driverId", "driver_name"]], on="driverId", how="left")
    )
    final_standings = merged[merged["round"] == merged["round"].max()].nlargest(top_n, "points")
    top_driver_ids = final_standings["driverId"].tolist()
    top_merged = merged[merged["driverId"].isin(top_driver_ids)]

    fig = px.line(
        top_merged.sort_values(["driverId", "round"]),
        x="round",
        y="points",
        color="driver_name",
        markers=True,
        title=f"{year} Drivers' Championship — Top {top_n}",
        labels={"round": "Race round", "points": "Championship points", "driver_name": "Driver"},
    )
    fig.update_layout(height=500)
    return fig


def predicted_winner_panel(predictions: pd.DataFrame, drivers: pd.DataFrame, race_id: int) -> go.Figure:
    """Bar chart of predicted win probabilities for a single race."""
    race_preds = predictions[predictions["raceId"] == race_id].copy()
    if race_preds.empty:
        raise ValueError(f"no predictions for raceId {race_id}")
    race_preds = race_preds.merge(drivers[["driverId", "driver_name"]], on="driverId", how="left")
    race_preds = race_preds.sort_values("win_prob", ascending=True)

    fig = px.bar(
        race_preds,
        x="win_prob",
        y="driver_name",
        orientation="h",
        title=f"Predicted Win Probabilities — Race {race_id}",
        labels={"win_prob": "Win probability", "driver_name": "Driver"},
    )
    fig.update_layout(height=500, xaxis_tickformat=".0%")
    return fig


def feature_importance_plot(model, feature_cols, top_n: int = 20) -> go.Figure:
    """Horizontal bar of the top features for a trained tree model."""
    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        raise ValueError("model has no feature_importances_")
    df = (
        pd.DataFrame({"feature": feature_cols, "importance": importances})
        .sort_values("importance", ascending=False)
        .head(top_n)
        .iloc[::-1]
    )
    fig = px.bar(df, x="importance", y="feature", orientation="h", title="Top features (gain importance)")
    fig.update_layout(height=600)
    return fig
