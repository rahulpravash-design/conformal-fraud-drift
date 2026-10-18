"""Loading BAF variants. Raw files are never modified."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROW_ID = "row_id"


def find_variant_file(data_dir: str | Path, variant: str, file_format: str = "auto") -> Path:
    data_dir = Path(data_dir)
    exts = ["parquet", "csv"] if file_format == "auto" else [file_format]
    for ext in exts:
        p = data_dir / f"{variant}.{ext}"
        if p.exists():
            return p
    raise FileNotFoundError(
        f"No {variant}.{'/'.join(exts)} in {data_dir}. See data/README.md for acquisition instructions."
    )


def read_variant(path: str | Path) -> pd.DataFrame:
    """Read a BAF file as the official README does (csv has an unnamed index column).

    A `row_id` column holding the 0-based position in the file is added so that
    partition disjointness can be verified. File order carries no temporal meaning
    within a month.
    """
    path = Path(path)
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    elif path.suffix == ".csv":
        df = pd.read_csv(path, index_col=0)
    else:
        raise ValueError(f"Unsupported extension: {path.suffix}")
    df = df.reset_index(drop=True)
    df.insert(0, ROW_ID, range(len(df)))
    return df


def validate_schema(df: pd.DataFrame, label: str, time_column: str, categorical: list[str], group_columns: list[str]) -> None:
    missing = [c for c in [label, time_column, *group_columns] if c not in df.columns]
    if missing:
        raise ValueError(f"Required columns missing from dataset: {missing}")
    if not set(pd.unique(df[label])) <= {0, 1}:
        raise ValueError(f"Label column {label!r} must be binary 0/1; found {sorted(pd.unique(df[label]))[:10]}")
    if df[label].isna().any() or df[time_column].isna().any():
        raise ValueError("NaN in label or time column")
    t = df[time_column]
    if not (t == t.round()).all():
        raise ValueError(f"Time column {time_column!r} has non-integer values")
    absent_cat = [c for c in categorical if c not in df.columns]
    if absent_cat:
        raise ValueError(f"Configured categorical features not in data: {absent_cat}")


def feature_columns(df: pd.DataFrame, label: str, time_column: str, exclude: list[str]) -> list[str]:
    drop = {label, time_column, ROW_ID, *exclude}
    return [c for c in df.columns if c not in drop]
