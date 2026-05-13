"""Shared helpers: config loading, paths, dataset I/O."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"


def load_config(path: Optional[str | Path] = None) -> Dict[str, Any]:
    """Load YAML config; returns ``{}`` if the file is missing or empty."""
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not config_path.exists():
        return {}
    with config_path.open() as fh:
        loaded = yaml.safe_load(fh)
    return loaded or {}


def project_path(*parts: str) -> Path:
    """Build an absolute path relative to the project root."""
    return PROJECT_ROOT.joinpath(*parts)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_table(df: pd.DataFrame, path: str | Path, fmt: str = "parquet") -> Path:
    """Save a DataFrame as parquet, pickle or csv depending on ``fmt``."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "parquet":
        out = out.with_suffix(".parquet")
        df.to_parquet(out, index=False)
    elif fmt == "pickle":
        out = out.with_suffix(".pkl")
        df.to_pickle(out)
    elif fmt == "csv":
        out = out.with_suffix(".csv")
        df.to_csv(out, index=False)
    else:
        raise ValueError(f"unsupported format: {fmt!r}")
    return out


def load_table(path: str | Path) -> pd.DataFrame:
    """Load a DataFrame by file suffix (.parquet/.pkl/.csv)."""
    p = Path(path)
    if p.suffix == ".parquet":
        return pd.read_parquet(p)
    if p.suffix == ".pkl":
        return pd.read_pickle(p)
    if p.suffix == ".csv":
        return pd.read_csv(p)
    raise ValueError(f"cannot infer reader for {p.suffix}")


def load_interim_tables(interim_dir: str | Path, fmt: str = "parquet") -> Dict[str, pd.DataFrame]:
    """Read every cleaned table from ``interim_dir``."""
    suffix = {"parquet": ".parquet", "pickle": ".pkl", "csv": ".csv"}[fmt]
    interim = Path(interim_dir)
    out: Dict[str, pd.DataFrame] = {}
    for path in sorted(interim.glob(f"*{suffix}")):
        out[path.stem] = load_table(path)
    return out
