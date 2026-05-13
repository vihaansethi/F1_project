"""Load and clean the raw Kaggle F1 dataset."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


CSV_TABLES: tuple[str, ...] = (
    "circuits",
    "constructors",
    "constructor_results",
    "constructor_standings",
    "drivers",
    "driver_standings",
    "lap_times",
    "pit_stops",
    "qualifying",
    "races",
    "results",
    "seasons",
    "sprint_results",
    "status",
)


class DataCleaner:
    """Loads the raw F1 CSV files and applies cleaning steps.

    Use ``load_all_data`` then ``clean_all_data`` (or just ``run``) to get a
    dict of cleaned DataFrames keyed by table name.
    """

    def __init__(self, data_path: str | Path, tables: Optional[Iterable[str]] = None):
        self.data_path = Path(data_path)
        self.tables = tuple(tables) if tables is not None else CSV_TABLES
        self.data: Dict[str, pd.DataFrame] = {}

    # ------------------------------------------------------------------ load
    def load_all_data(self) -> Dict[str, pd.DataFrame]:
        """Load every requested CSV; missing files are warned, not fatal."""
        if not self.data_path.exists():
            raise FileNotFoundError(f"Raw data path does not exist: {self.data_path}")

        for table in self.tables:
            file_path = self.data_path / f"{table}.csv"
            if not file_path.exists():
                logger.warning("missing CSV: %s", file_path.name)
                continue
            # The Kaggle export uses '\N' as the null sentinel.
            df = pd.read_csv(file_path, na_values=["\\N", "\\\\N", ""], low_memory=False)
            self.data[table] = df
            logger.info("loaded %s (%s rows)", table, f"{len(df):,}")
        return self.data

    # ------------------------------------------------------------ summaries
    def get_summary(self) -> pd.DataFrame:
        """Return a per-table row/column/null summary."""
        rows = []
        for name, df in self.data.items():
            rows.append(
                {
                    "table": name,
                    "rows": len(df),
                    "columns": df.shape[1],
                    "null_cells": int(df.isna().sum().sum()),
                    "memory_mb": round(df.memory_usage(deep=True).sum() / 1024**2, 2),
                }
            )
        return pd.DataFrame(rows).sort_values("rows", ascending=False).reset_index(drop=True)

    # -------------------------------------------------------------- cleaners
    def clean_circuits_data(self) -> None:
        if "circuits" not in self.data:
            return
        df = self.data["circuits"].copy()

        city_to_country = {
            "Catalunya": "Spain", "Valencia": "Spain", "Barcelona": "Spain",
            "Indianapolis": "USA", "Phoenix": "USA", "Detroit": "USA",
            "Las Vegas": "USA", "Miami": "USA", "Austin": "USA",
            "Silverstone": "UK", "Brands Hatch": "UK",
            "Monza": "Italy", "Imola": "Italy",
            "Monte-Carlo": "Monaco", "Spa": "Belgium",
            "Suzuka": "Japan", "Shanghai": "China",
            "Marina Bay": "Singapore", "Yas Marina": "UAE",
            "Sakhir": "Bahrain", "Istanbul": "Turkey",
            "Budapest": "Hungary", "Interlagos": "Brazil",
            "Montreal": "Canada", "Melbourne": "Australia",
            "Baku": "Azerbaijan", "Sochi": "Russia",
        }
        for city, country in city_to_country.items():
            df.loc[df["location"].str.contains(city, na=False), "country"] = country
            df.loc[df["country"] == city, "country"] = country
        df["country"] = df["country"].replace(
            {
                "UK": "United Kingdom",
                "USA": "United States",
                "UAE": "United Arab Emirates",
                "Korea": "South Korea",
            }
        )
        self.data["circuits"] = df
        logger.info("circuits cleaned: %s unique countries", df["country"].nunique())

    def clean_lap_times_data(self) -> None:
        if "lap_times" not in self.data:
            return
        df = self.data["lap_times"]
        original = len(df)
        # Lap times between 60s and 180s — drops formation laps, red flag stops, parse errors.
        mask = (df["milliseconds"] >= 60_000) & (df["milliseconds"] <= 180_000)
        self.data["lap_times"] = df[mask].copy()
        logger.info("lap_times cleaned: removed %s outliers", f"{original - mask.sum():,}")

    def clean_pit_stops_data(self) -> None:
        if "pit_stops" not in self.data:
            return
        df = self.data["pit_stops"]
        original = len(df)
        # Pit stop durations from 1.5s (modern record territory) up to 60s.
        mask = (df["milliseconds"] >= 1_500) & (df["milliseconds"] <= 60_000)
        self.data["pit_stops"] = df[mask].copy()
        logger.info("pit_stops cleaned: removed %s outliers", f"{original - mask.sum():,}")

    def clean_results_data(self) -> None:
        if "results" not in self.data:
            return
        df = self.data["results"].copy()
        numeric_cols = [
            "position", "milliseconds", "points", "grid", "laps", "fastestLap", "rank",
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        self.data["results"] = df
        logger.info("results cleaned")

    def clean_qualifying_data(self) -> None:
        if "qualifying" not in self.data:
            return
        df = self.data["qualifying"].copy()
        for col in ("q1", "q2", "q3"):
            if col in df.columns:
                df[f"{col}_ms"] = df[col].apply(_qualifying_time_to_ms)
        df["best_quali_ms"] = df[[c for c in ("q1_ms", "q2_ms", "q3_ms") if c in df.columns]].min(axis=1)
        self.data["qualifying"] = df
        logger.info("qualifying cleaned: parsed q1/q2/q3 to milliseconds")

    def clean_races_data(self) -> None:
        if "races" not in self.data:
            return
        df = self.data["races"].copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
        df["round"] = pd.to_numeric(df["round"], errors="coerce").astype("Int64")
        self.data["races"] = df
        logger.info("races cleaned: parsed dates")

    def clean_drivers_data(self) -> None:
        if "drivers" not in self.data:
            return
        df = self.data["drivers"].copy()
        df["dob"] = pd.to_datetime(df["dob"], errors="coerce")
        df["driver_name"] = df["forename"].fillna("") + " " + df["surname"].fillna("")
        df["driver_name"] = df["driver_name"].str.strip()
        self.data["drivers"] = df
        logger.info("drivers cleaned: parsed dob, added driver_name")

    # ------------------------------------------------------------------ run
    def clean_all_data(self) -> Dict[str, pd.DataFrame]:
        logger.info("cleaning all data...")
        self.clean_races_data()
        self.clean_drivers_data()
        self.clean_circuits_data()
        self.clean_lap_times_data()
        self.clean_pit_stops_data()
        self.clean_results_data()
        self.clean_qualifying_data()
        logger.info("cleaning complete")
        return self.data

    def run(self) -> Dict[str, pd.DataFrame]:
        """Convenience: load and clean in one call."""
        self.load_all_data()
        return self.clean_all_data()


def _qualifying_time_to_ms(value) -> float:
    """Parse a qualifying time like ``"1:25.123"`` into milliseconds.

    Returns ``np.nan`` for missing or unparsable values.
    """
    if pd.isna(value):
        return np.nan
    text = str(value).strip()
    if not text or text in {"\\N", "0"}:
        return np.nan
    try:
        if ":" in text:
            minutes, seconds = text.split(":", 1)
            return int(minutes) * 60_000 + float(seconds) * 1_000
        return float(text) * 1_000
    except ValueError:
        return np.nan
