"""Build the (race, driver) feature/target frame and write it to disk."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.features.feature_builder import build_feature_table  # noqa: E402
from src.utils import ensure_dir, load_interim_tables, save_table, setup_logger  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the modeling feature table.")
    parser.add_argument("--interim", default="data/interim", help="Where cleaned tables live.")
    parser.add_argument("--output", default="data/processed", help="Output directory.")
    parser.add_argument(
        "--format", choices=["parquet", "pickle", "csv"], default="parquet",
        help="Both the interim format to read and the processed format to write.",
    )
    parser.add_argument("--min-year", type=int, default=2010, help="Drop seasons earlier than this.")
    args = parser.parse_args()

    logger = setup_logger()
    logger.info("loading interim tables from %s (%s)", args.interim, args.format)
    tables = load_interim_tables(args.interim, fmt=args.format)
    if not tables:
        raise SystemExit("no interim tables found — run clean_data.py first")

    logger.info("building feature table (min_year=%s)", args.min_year)
    frames = build_feature_table(tables, min_year=args.min_year)

    out_dir = ensure_dir(args.output)
    feature_path = save_table(frames.df, out_dir / "features", fmt=args.format)
    logger.info("wrote %s (%s rows, %s cols)", feature_path.name, f"{len(frames.df):,}", frames.df.shape[1])

    import json
    metadata = {
        "feature_cols": frames.feature_cols,
        "target_cols": frames.target_cols,
        "rows": len(frames.df),
        "min_year": args.min_year,
    }
    (out_dir / "feature_metadata.json").write_text(json.dumps(metadata, indent=2))
    logger.info("wrote feature_metadata.json")


if __name__ == "__main__":
    main()
