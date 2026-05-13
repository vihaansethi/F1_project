"""Predict whether a driver finishes on the podium (top-3) of a given race."""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from xgboost import XGBClassifier

from .base import TrainedModel, prepare_xy, time_based_split


def train_podium_model(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = "target_podium",
    test_years: int = 2,
    params: Optional[Dict] = None,
) -> TrainedModel:
    train_df, test_df = time_based_split(df, test_years=test_years)
    X_train, y_train = prepare_xy(train_df, feature_cols, target_col)
    X_test, y_test = prepare_xy(test_df, feature_cols, target_col)

    model = XGBClassifier(**_default_params(params))
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]
    preds = (proba >= 0.5).astype(int)
    metrics = {
        "test_logloss": float(log_loss(y_test, proba, labels=[0, 1])),
        "test_auc": float(roc_auc_score(y_test, proba)) if y_test.nunique() > 1 else float("nan"),
        "test_accuracy": float(accuracy_score(y_test, preds)),
        "test_top3_hit_rate": float(_podium_top3_hit_rate(test_df.loc[X_test.index], proba)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }

    return TrainedModel(
        model=model,
        feature_cols=list(X_train.columns),
        target_col=target_col,
        metrics=metrics,
        model_kind="podium",
    )


def predict_podium_probs(trained: TrainedModel, race_df: pd.DataFrame) -> pd.DataFrame:
    """Return per-driver podium probabilities, sorted high → low within race."""
    if "raceId" not in race_df.columns or "driverId" not in race_df.columns:
        raise ValueError("race_df must contain raceId and driverId")
    X = race_df[trained.feature_cols].copy()
    proba = trained.model.predict_proba(X)[:, 1]
    out = race_df[["raceId", "driverId"]].copy()
    out["podium_prob"] = proba
    return out.sort_values(["raceId", "podium_prob"], ascending=[True, False]).reset_index(drop=True)


def _default_params(overrides: Optional[Dict]) -> Dict:
    params = {
        "n_estimators": 400,
        "max_depth": 5,
        "learning_rate": 0.05,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "reg_lambda": 1.0,
        "objective": "binary:logistic",
        "tree_method": "hist",
        "eval_metric": "logloss",
        "n_jobs": -1,
        "random_state": 42,
    }
    if overrides:
        params.update(overrides)
    return params


def _podium_top3_hit_rate(test_df: pd.DataFrame, proba: np.ndarray) -> float:
    """Average overlap between predicted top-3 and actual top-3 (out of 3)."""
    work = test_df.copy()
    work["_p"] = proba
    overlaps = []
    for _, race_rows in work.groupby("raceId"):
        actual = set(race_rows.loc[race_rows["target_podium"] == 1, "driverId"])
        if len(actual) < 3:
            continue
        predicted = set(race_rows.nlargest(3, "_p")["driverId"])
        overlaps.append(len(actual & predicted) / 3)
    return float(np.mean(overlaps)) if overlaps else float("nan")
