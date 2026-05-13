"""Matplotlib/Seaborn plots for exploring the F1 dataset.

Every plot takes a dict of cleaned tables (the output of
:class:`f1ml.data.DataCleaner`) and returns a matplotlib Figure so callers
can ``fig.show()``, ``fig.savefig(...)``, or embed it in a report.
"""
from __future__ import annotations

from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 100


def plot_circuits_by_country(tables: Dict[str, pd.DataFrame], top_n: int = 15) -> plt.Figure:
    """Horizontal bar plot of top countries by circuit count."""
    circuits = tables["circuits"]
    counts = (
        circuits.groupby("country")
        .size()
        .reset_index(name="circuit_count")
        .sort_values("circuit_count", ascending=False)
        .head(top_n)
    )
    fig, ax = plt.subplots(figsize=(10, 8))
    palette = sns.color_palette("viridis", n_colors=len(counts))
    sns.barplot(data=counts, x="circuit_count", y="country", palette=palette, ax=ax)
    for i, (_, row) in enumerate(counts.iterrows()):
        ax.text(row["circuit_count"] + 0.1, i, str(row["circuit_count"]), va="center")
    ax.set_title(f"Top {top_n} Countries by Number of F1 Circuits", pad=20)
    ax.set_xlabel("Number of Circuits")
    ax.set_ylabel("")
    fig.tight_layout()
    return fig


def plot_constructor_wins_heatmap(tables: Dict[str, pd.DataFrame], since_year: int = 2010) -> plt.Figure:
    """Heatmap of race wins by constructor and year."""
    results = tables["results"]
    races = tables["races"]
    constructors = tables["constructors"]

    wins = results[results["position"] == 1].merge(
        races[["raceId", "year"]], on="raceId", how="left"
    ).merge(constructors[["constructorId", "name"]], on="constructorId", how="left")
    wins = wins.dropna(subset=["year", "name"])
    wins = wins[wins["year"] >= since_year]

    wins_by_year = wins.groupby(["name", "year"]).size().reset_index(name="wins")
    top = (
        wins_by_year.groupby("name")["wins"].sum().nlargest(8).index.tolist()
    )
    plot_data = wins_by_year[wins_by_year["name"].isin(top)]
    pivot = (
        plot_data.pivot(index="name", columns="year", values="wins")
        .fillna(0)
        .assign(total=lambda d: d.sum(axis=1))
        .sort_values("total", ascending=False)
        .drop(columns="total")
    )

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".0f",
        cmap="RdYlBu_r",
        cbar_kws={"label": "Wins"},
        linewidths=0.5,
        linecolor="gray",
        ax=ax,
    )
    ax.set_title(f"Constructor Race Wins ({since_year}-present)", pad=20, weight="bold")
    ax.set_xlabel("Year")
    ax.set_ylabel("Constructor")
    fig.tight_layout()
    return fig


def plot_driver_podiums(tables: Dict[str, pd.DataFrame], top_n: int = 20) -> plt.Figure:
    """Stacked horizontal bar chart of top drivers by podium finishes."""
    results = tables["results"].copy()
    drivers = tables["drivers"]
    results["position"] = pd.to_numeric(results["position"], errors="coerce")

    podiums = results[results["position"].isin([1, 2, 3])].copy()
    podiums = podiums.merge(
        drivers[["driverId", "forename", "surname"]], on="driverId", how="left"
    )
    podiums["driver"] = (podiums["forename"].fillna("") + " " + podiums["surname"].fillna("")).str.strip()

    stats = (
        podiums.assign(
            wins=lambda d: (d["position"] == 1).astype(int),
            seconds=lambda d: (d["position"] == 2).astype(int),
            thirds=lambda d: (d["position"] == 3).astype(int),
        )
        .groupby("driver")[["wins", "seconds", "thirds"]]
        .sum()
        .assign(total=lambda d: d.sum(axis=1))
        .sort_values("total", ascending=False)
        .head(top_n)
        .drop(columns="total")
    )

    fig, ax = plt.subplots(figsize=(10, 12))
    stats.plot(kind="barh", stacked=True, color=["gold", "silver", "#CD7F32"], ax=ax)
    ax.invert_yaxis()
    ax.set_title(f"Top {top_n} F1 Drivers by Podium Finishes", pad=20)
    ax.set_xlabel("Podiums")
    ax.set_ylabel("")
    ax.legend(["1st", "2nd", "3rd"], loc="lower right")
    fig.tight_layout()
    return fig


def plot_pit_stop_trends(tables: Dict[str, pd.DataFrame]) -> plt.Figure:
    """Two-panel plot: average / median pit stop duration, plus fastest stop per year."""
    pit_stops = tables["pit_stops"]
    races = tables["races"]

    df = pit_stops.merge(races[["raceId", "year"]], on="raceId", how="left")
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df[(df["year"] >= 2010) & (df["year"] <= 2024)].copy()
    df["seconds"] = df["milliseconds"] / 1_000

    yearly = (
        df.groupby("year")["seconds"]
        .agg(
            mean="mean",
            median="median",
            fastest="min",
            count="count",
            q1=lambda s: s.quantile(0.25),
            q3=lambda s: s.quantile(0.75),
        )
        .reset_index()
    )
    yearly = yearly[yearly["count"] >= 30]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    ax1.plot(yearly["year"], yearly["mean"], marker="o", linewidth=3, label="Average", color="#2E86AB")
    ax1.plot(yearly["year"], yearly["median"], marker="s", linewidth=3, label="Median", color="#A23B72")
    ax1.fill_between(yearly["year"], yearly["q1"], yearly["q3"], alpha=0.3, color="gray", label="IQR")
    ax1.set_ylabel("Duration (s)")
    ax1.set_title("F1 Pit Stop Duration Trends (2010–2024)", pad=20, weight="bold")
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3, linestyle="--")
    ax1.set_ylim(10, 30)

    ax2.plot(yearly["year"], yearly["fastest"], marker="^", linewidth=3, color="#F18F01", label="Fastest")
    for _, row in yearly.iterrows():
        ax2.annotate(
            f"{row['fastest']:.2f}s",
            (row["year"], row["fastest"]),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=9,
            color="#F18F01",
            weight="bold",
        )
    ax2.set_xlabel("Year")
    ax2.set_ylabel("Duration (s)")
    ax2.set_title("Fastest Pit Stop Each Year", weight="bold")
    ax2.grid(True, alpha=0.3, linestyle="--")
    ax2.set_ylim(yearly["fastest"].min() - 0.5, yearly["fastest"].max() + 1)
    ax2.set_xticks(range(int(yearly["year"].min()), int(yearly["year"].max()) + 1))
    ax2.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    return fig


def plot_constructor_dominance(tables: Dict[str, pd.DataFrame], since_year: int = 2014) -> plt.Figure:
    """Heatmap of constructor points by year (hybrid era by default)."""
    results = tables["results"].merge(tables["races"][["raceId", "year"]], on="raceId").merge(
        tables["constructors"][["constructorId", "name"]], on="constructorId"
    )
    results = results[results["year"] >= since_year]

    points = results.groupby(["name", "year"])["points"].sum().reset_index()
    top = points.groupby("name")["points"].sum().nlargest(8).index
    pivot = (
        points[points["name"].isin(top)]
        .pivot(index="name", columns="year", values="points")
        .fillna(0)
    )
    pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=False).index]

    fig, ax = plt.subplots(figsize=(14, 8))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".0f",
        cmap="YlOrRd",
        cbar_kws={"label": "Points"},
        linewidths=0.5,
        linecolor="gray",
        ax=ax,
    )
    ax.set_title(f"Constructor Points by Year (since {since_year})", pad=20)
    ax.set_xlabel("Year")
    ax.set_ylabel("Constructor")
    fig.tight_layout()
    return fig


def plot_model_metrics(metrics_by_model: Dict[str, Dict[str, float]]) -> plt.Figure:
    """Bar plot of headline metrics for a set of trained models."""
    rows = []
    for name, metrics in metrics_by_model.items():
        for metric, value in metrics.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                rows.append({"model": name, "metric": metric, "value": value})
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=df, x="metric", y="value", hue="model", ax=ax)
    ax.set_title("Model Evaluation Metrics", pad=20, weight="bold")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    return fig
