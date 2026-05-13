"""Predict who wins a given race.

The model is a binary classifier on (race, driver) rows. At prediction time
we softmax-normalize win probabilities within each race so they sum to 1.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from xgboost import XGBClassifier

from .base import TrainedModel, prepare_xy, time_based_split


def train_race_winner_model(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = "target_win",
    test_years: int = 2,
    params: Optional[Dict] = None,
) -> TrainedModel:
    """Train a race-winner binary classifier with a time-based holdout."""
    train_df, test_df = time_based_split(df, test_years=test_years)
    X_train, y_train = prepare_xy(train_df, feature_cols, target_col)
    X_test, y_test = prepare_xy(test_df, feature_cols, target_col)

    model = XGBClassifier(**_default_params(params))
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]
    metrics = {
        "test_logloss": float(log_loss(y_test, proba, labels=[0, 1])),
        "test_auc": float(roc_auc_score(y_test, proba)) if y_test.nunique() > 1 else float("nan"),
        "test_top1_accuracy": float(_race_top1_accuracy(test_df.loc[X_test.index], proba)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }

    return TrainedModel(
        model=model,
        feature_cols=list(X_train.columns),
        target_col=target_col,
        metrics=metrics,
        model_kind="race_winner",
    )


def predict_race_winner_probs(
    trained: TrainedModel,
    race_df: pd.DataFrame,
) -> pd.DataFrame:
    """Return a per-driver win probability frame for the given race rows.

    Probabilities are normalized within the race so they sum to 1.0.
    """
    if "raceId" not in race_df.columns or "driverId" not in race_df.columns:
        raise ValueError("race_df must contain raceId and driverId")
    X = race_df[trained.feature_cols].copy()
    raw = trained.model.predict_proba(X)[:, 1]
    out = race_df[["raceId", "driverId"]].copy()
    out["win_prob_raw"] = raw
    out["win_prob"] = out.groupby("raceId")["win_prob_raw"].transform(
        lambda s: s / s.sum() if s.sum() > 0 else s
    )
    return out.sort_values(["raceId", "win_prob"], ascending=[True, False]).reset_index(drop=True)


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


def _race_top1_accuracy(test_df: pd.DataFrame, proba: np.ndarray) -> float:
    """Per-race top-1 hit rate: did the highest-probability driver actually win?"""
    work = test_df.copy()
    work["_p"] = proba
    hits = 0
    races = 0
    for _, race_rows in work.groupby("raceId"):
        if not race_rows["target_win"].any():
            continue
        races += 1
        pred_winner = race_rows.loc[race_rows["_p"].idxmax(), "driverId"]
        actual_winner = race_rows.loc[race_rows["target_win"].idxmax(), "driverId"]
        if pred_winner == actual_winner:
            hits += 1
    return hits / races if races else float("nan")
