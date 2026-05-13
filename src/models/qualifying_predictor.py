"""Regression model for race pace (average lap time in milliseconds).

Acts as a stand-in for a qualifying / lap-time predictor: we predict the
driver's average race lap time and rank drivers within a race by their
predicted pace.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

from .base import TrainedModel, prepare_xy, time_based_split


def train_lap_pace_model(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = "target_avg_lap_ms",
    test_years: int = 2,
    params: Optional[Dict] = None,
) -> TrainedModel:
    """Train an XGBoost regressor on average lap time in milliseconds."""
    train_df, test_df = time_based_split(df, test_years=test_years)
    X_train, y_train = prepare_xy(train_df, feature_cols, target_col)
    X_test, y_test = prepare_xy(test_df, feature_cols, target_col)

    model = XGBRegressor(**_default_params(params))
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae_ms = float(mean_absolute_error(y_test, preds))
    rmse_ms = float(np.sqrt(mean_squared_error(y_test, preds)))
    metrics = {
        "test_mae_ms": mae_ms,
        "test_rmse_ms": rmse_ms,
        "test_mae_seconds": mae_ms / 1_000,
        "test_r2": float(r2_score(y_test, preds)),
        "test_rank_correlation": float(_race_rank_correlation(test_df.loc[X_test.index], preds)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }

    return TrainedModel(
        model=model,
        feature_cols=list(X_train.columns),
        target_col=target_col,
        metrics=metrics,
        model_kind="lap_pace",
    )


def predict_lap_pace(trained: TrainedModel, race_df: pd.DataFrame) -> pd.DataFrame:
    """Return predicted average lap time (ms) and within-race pace rank."""
    X = race_df[trained.feature_cols].copy()
    preds = trained.model.predict(X)
    out = race_df[["raceId", "driverId"]].copy()
    out["predicted_avg_lap_ms"] = preds
    out["predicted_avg_lap_seconds"] = preds / 1_000
    out["pace_rank"] = out.groupby("raceId")["predicted_avg_lap_ms"].rank(method="min")
    return out.sort_values(["raceId", "pace_rank"]).reset_index(drop=True)


def _default_params(overrides: Optional[Dict]) -> Dict:
    params = {
        "n_estimators": 600,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "reg_lambda": 1.0,
        "objective": "reg:squarederror",
        "tree_method": "hist",
        "n_jobs": -1,
        "random_state": 42,
    }
    if overrides:
        params.update(overrides)
    return params


def _race_rank_correlation(test_df: pd.DataFrame, preds: np.ndarray) -> float:
    """Average Spearman rank correlation between predicted and actual pace."""
    work = test_df.copy()
    work["_p"] = preds
    correlations = []
    for _, race_rows in work.groupby("raceId"):
        rr = race_rows.dropna(subset=["target_avg_lap_ms"])
        if len(rr) < 5:
            continue
        actual_rank = rr["target_avg_lap_ms"].rank()
        pred_rank = rr["_p"].rank()
        correlations.append(actual_rank.corr(pred_rank, method="spearman"))
    return float(np.nanmean(correlations)) if correlations else float("nan")
