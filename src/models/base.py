"""Shared training and evaluation helpers for F1 prediction models."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd


@dataclass
class TrainedModel:
    """Bundle of artifacts produced by a training run."""

    model: Any
    feature_cols: List[str]
    target_col: str
    metrics: Dict[str, float]
    model_kind: str
    extra: Dict[str, Any] = field(default_factory=dict)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "TrainedModel":
        return joblib.load(path)


def time_based_split(
    df: pd.DataFrame,
    year_col: str = "year",
    test_years: int = 2,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Hold out the most recent ``test_years`` seasons as the test set."""
    if year_col not in df.columns:
        raise KeyError(f"{year_col} not in dataframe")
    years = sorted(df[year_col].dropna().unique())
    if len(years) <= test_years:
        raise ValueError("not enough seasons in data to split")
    cutoff = years[-test_years]
    train = df[df[year_col] < cutoff].copy()
    test = df[df[year_col] >= cutoff].copy()
    return train, test


def prepare_xy(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    drop_target_na: bool = True,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Materialize X/y for a model, optionally dropping rows missing the target."""
    available = [c for c in feature_cols if c in df.columns]
    work = df[available + [target_col]].copy()
    if drop_target_na:
        work = work.dropna(subset=[target_col])
    return work[available], work[target_col]
