from __future__ import annotations

from pathlib import Path

import pandas as pd


def ensure_parent(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def read_table(path: str | Path, **kwargs) -> pd.DataFrame:
    path = Path(path)
    suffixes = "".join(path.suffixes).lower()
    if suffixes.endswith(".parquet"):
        return pd.read_parquet(path, **kwargs)
    if suffixes.endswith(".tsv") or suffixes.endswith(".txt"):
        return pd.read_csv(path, sep="\t", **kwargs)
    if suffixes.endswith(".csv"):
        return pd.read_csv(path, **kwargs)
    return pd.read_csv(path, sep=None, engine="python", **kwargs)


def write_table(df: pd.DataFrame, path: str | Path, index: bool = False) -> None:
    path = Path(path)
    ensure_parent(path)
    suffixes = "".join(path.suffixes).lower()
    if suffixes.endswith(".parquet"):
        df.to_parquet(path, index=index)
    elif suffixes.endswith(".tsv") or suffixes.endswith(".txt"):
        df.to_csv(path, sep="\t", index=index)
    else:
        df.to_csv(path, index=index)

