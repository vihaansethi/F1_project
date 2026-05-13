"""Train the three F1 prediction models and save them to disk."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models import train_lap_pace_model, train_podium_model, train_race_winner_model  # noqa: E402
from src.utils import ensure_dir, load_table, setup_logger  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Train all F1 prediction models.")
    parser.add_argument("--features", default="data/processed/features.parquet")
    parser.add_argument("--metadata", default="data/processed/feature_metadata.json")
    parser.add_argument("--output", default="data/models", help="Directory to save .joblib bundles.")
    parser.add_argument("--test-years", type=int, default=2, help="Number of recent seasons held out.")
    args = parser.parse_args()

    logger = setup_logger()
    feature_path = Path(args.features)
    metadata_path = Path(args.metadata)
    if not feature_path.exists() or not metadata_path.exists():
        raise SystemExit("feature table or metadata missing — run build_features.py first")

    df = load_table(feature_path)
    metadata = json.loads(metadata_path.read_text())
    feature_cols = metadata["feature_cols"]
    logger.info("loaded features: %s rows, %s feature cols", f"{len(df):,}", len(feature_cols))

    out_dir = ensure_dir(args.output)
    all_metrics = {}

    logger.info("training race winner model...")
    winner = train_race_winner_model(df, feature_cols, test_years=args.test_years)
    winner.save(out_dir / "race_winner.joblib")
    all_metrics["race_winner"] = winner.metrics
    _log_metrics(logger, "race_winner", winner.metrics)

    logger.info("training podium model...")
    podium = train_podium_model(df, feature_cols, test_years=args.test_years)
    podium.save(out_dir / "podium.joblib")
    all_metrics["podium"] = podium.metrics
    _log_metrics(logger, "podium", podium.metrics)

    logger.info("training lap-pace regressor...")
    pace = train_lap_pace_model(df, feature_cols, test_years=args.test_years)
    pace.save(out_dir / "lap_pace.joblib")
    all_metrics["lap_pace"] = pace.metrics
    _log_metrics(logger, "lap_pace", pace.metrics)

    (out_dir / "metrics.json").write_text(json.dumps(all_metrics, indent=2))
    logger.info("wrote metrics.json")


def _log_metrics(logger, name, metrics):
    parts = ", ".join(
        f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in metrics.items()
    )
    logger.info("  %s: %s", name, parts)


if __name__ == "__main__":
    main()
