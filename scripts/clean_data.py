"""Clean the raw F1 CSVs and write interim tables to disk.

Usage:
    python scripts/clean_data.py --input data/raw --output data/interim --format parquet
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.data_cleaner import DataCleaner  # noqa: E402
from src.utils import ensure_dir, save_table, setup_logger  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean raw F1 CSVs into interim tables.")
    parser.add_argument("--input", default="data/raw", help="Directory holding the raw Kaggle CSVs.")
    parser.add_argument("--output", default="data/interim", help="Where to write the cleaned tables.")
    parser.add_argument(
        "--format", choices=["parquet", "pickle", "csv"], default="parquet",
        help="Output file format.",
    )
    args = parser.parse_args()

    logger = setup_logger()
    logger.info("cleaning data from %s -> %s (%s)", args.input, args.output, args.format)

    cleaner = DataCleaner(args.input)
    tables = cleaner.run()

    out_dir = ensure_dir(args.output)
    for name, df in tables.items():
        path = save_table(df, out_dir / name, fmt=args.format)
        logger.info("wrote %s (%s rows)", path.name, f"{len(df):,}")

    summary_path = out_dir / "_summary.csv"
    cleaner.get_summary().to_csv(summary_path, index=False)
    logger.info("wrote summary: %s", summary_path)


if __name__ == "__main__":
    main()
