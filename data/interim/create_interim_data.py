"""
Script to create cleaned data from the raw csv files

Usage:
    python scripts/create_interim_data.py --input data/raw --output data/interim
"""

import os
import sys
import argparse
import logging
from pathlib import Path
import pandas as pd
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.data_cleaner import F1DataCleaner
from src.utils.logger import setup_logger

logger = setup_logger('create_interim_data')

def create_interim_data(raw_data_path: str, interim_data_path: str, format: str = 'pickle'):
    """
    Load raw data, clean it and save as interim data
    
    Args:
        raw_data_path: path to raw csv files
        interim_data_path: path to save cleaned data
        format: output format ('pickle' or 'csv')
    """
    logger.info("="*60)
    logger.info("Creating interim data for F1 analysis")
    logger.info("="*60)

    interim_path = Path(interim_data_path)
    interim_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading data from: {raw_data_path}")
    cleaner = F1DataCleaner(raw_data_path)

    raw_data = cleaner.load_all_data()

    logger.into("\nRaw Data Summary:")
    summary = cleaner.get_summary()

    print(summary.to_string())

    cleaned_data = cleaner.clean_all_data()

    logger.info(f"\nSaving interim data to: {interim_data_path}")

    for table_name, df in cleaned_data.items():
        if format == 'pickle':
            file_path = interim_path / f"{table_name}.pkl"
            df.to_pickle(file_path)
            logger.info(f" Saved {table_name}.pkl ({len(df):,} rows)")
        else:
            file_path = interim_path / f"{table_name}_cleaned.csv"
            df.to_csv(file_path, index=False)
            logger.info(f" Saved {table_name}_cleaned.csv ({len(df):,} rows)")
    metadata = {
        'created_at': datetime.now().isoformat(),
        'raw_data_path': str(raw_data_path),
        'tables': list(cleaned_data_keys()),
        'format': format,
        'row_counts': {name: len(df) for name, df in cleaned_data.items()}
    }

    metadata_df = pd.DataFrame([metadata])
    metadata_path = interim_path / 'metadata.csv'
    metadata_df.to_csv(metadata_path, index=False)
    logger.info(f"\n Created metadata file: {metadata_path}")

    logger.info("\nData Cleaning Summary:")
    logger.info(f" - Total tables processed: {len(cleaned_data)}")
    logger.info(f"  - Total rows: {sum(len(df) for df in cleaned_data.values()):,}")
    logger.info(f" - Output format: {format}")

    logger.info("\n Interim data creation complete")
    return cleaned_data
def main():
    """main function with argument parsing"""
    parser = argparse.ArgumentParser(
        description='Create interim (cleaned_ F1 data from raw CSV files'
    )
    parser.add_argument(
        '--input',
        type=str,
        default='data/raw',
        help='Path to raw data directory (default: data/raw)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/interim',
        help='Path to save interim data (default: data/interim)'
    )

    parser.add_argument(
        '--format',
        type=str,
        choices=['pickle', 'csv'],
        default='pickle',
        help='Output format (default: pickle)'
    )

    args = parser.parse_args()
    create_interim_data(args.input, args.output, args.format)

if __name__ == "__main__":
    main()
    



