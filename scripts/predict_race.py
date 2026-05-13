"""Generate predictions for a held-out race using trained models."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models import (  # noqa: E402
    TrainedModel,
    predict_lap_pace,
    predict_podium_probs,
    predict_race_winner_probs,
)
from src.utils import load_table, setup_logger  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict outcomes for a single race.")
    parser.add_argument("--features", default="data/processed/features.parquet")
    parser.add_argument("--models", default="data/models", help="Directory with the .joblib bundles.")
    parser.add_argument(
        "--race-id", type=int, help="raceId to predict. If omitted, the most recent race is used.",
    )
    parser.add_argument("--top", type=int, default=10, help="How many drivers to show per model.")
    args = parser.parse_args()

    logger = setup_logger()

    feature_path = Path(args.features)
    if not feature_path.exists():
        raise SystemExit("feature table missing — run build_features.py first")

    df = load_table(feature_path)
    race_id = args.race_id if args.race_id is not None else int(df.sort_values("date").iloc[-1]["raceId"])
    race_df = df[df["raceId"] == race_id].copy()
    if race_df.empty:
        raise SystemExit(f"no rows for raceId {race_id}")

    race_name = race_df.iloc[0].get("race_name", "<unknown>")
    year = race_df.iloc[0].get("year", "")
    logger.info("predicting raceId=%s (%s %s)", race_id, year, race_name)

    models_dir = Path(args.models)
    winner = TrainedModel.load(models_dir / "race_winner.joblib")
    podium = TrainedModel.load(models_dir / "podium.joblib")
    pace = TrainedModel.load(models_dir / "lap_pace.joblib")

    winner_preds = predict_race_winner_probs(winner, race_df)
    podium_preds = predict_podium_probs(podium, race_df)
    pace_preds = predict_lap_pace(pace, race_df)

    names = race_df[["driverId", "driver_name", "constructor_name", "grid"]].drop_duplicates("driverId")

    print(f"\n=== Race {race_id} ({year} {race_name}) ===")
    print("\n-- Win probabilities --")
    print(_pretty(winner_preds.merge(names, on="driverId"), "win_prob", args.top))

    print("\n-- Podium probabilities --")
    print(_pretty(podium_preds.merge(names, on="driverId"), "podium_prob", args.top))

    print("\n-- Predicted pace ranking --")
    print(_pretty_pace(pace_preds.merge(names, on="driverId"), args.top))


def _pretty(df: pd.DataFrame, prob_col: str, top: int) -> str:
    out = df[["driver_name", "constructor_name", "grid", prob_col]].head(top).copy()
    out[prob_col] = out[prob_col].map(lambda v: f"{v:.2%}")
    return out.to_string(index=False)


def _pretty_pace(df: pd.DataFrame, top: int) -> str:
    out = df.sort_values("pace_rank").head(top)
    out = out[["driver_name", "constructor_name", "grid", "predicted_avg_lap_seconds", "pace_rank"]].copy()
    out["predicted_avg_lap_seconds"] = out["predicted_avg_lap_seconds"].map(lambda v: f"{v:.3f}s")
    out["pace_rank"] = out["pace_rank"].astype(int)
    return out.to_string(index=False)


if __name__ == "__main__":
    main()
