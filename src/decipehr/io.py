from __future__ import annotations

from pathlib import Path

import pandas as pd

COMPRESSION_SUFFIXES = {".gz", ".bz2", ".xz", ".zip", ".zst"}
TABULAR_SUFFIXES = {".tsv", ".txt", ".psam"}


def ensure_parent(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def _data_suffix(path: Path) -> str:
    suffixes = [suffix.lower() for suffix in path.suffixes]
    while suffixes and suffixes[-1] in COMPRESSION_SUFFIXES:
        suffixes.pop()
    return suffixes[-1] if suffixes else ""


def read_table(path: str | Path, **kwargs) -> pd.DataFrame:
    path = Path(path)
    suffix = _data_suffix(path)
    if suffix == ".parquet":
        return pd.read_parquet(path, **kwargs)
    if suffix in TABULAR_SUFFIXES:
        kwargs.setdefault("sep", "\t")
        return pd.read_csv(path, **kwargs)
    if suffix == ".csv":
        return pd.read_csv(path, **kwargs)
    kwargs.setdefault("sep", None)
    kwargs.setdefault("engine", "python")
    return pd.read_csv(path, **kwargs)


def write_table(df: pd.DataFrame, path: str | Path, index: bool = False, **kwargs) -> None:
    path = Path(path)
    ensure_parent(path)
    suffix = _data_suffix(path)
    if suffix == ".parquet":
        df.to_parquet(path, index=index, **kwargs)
    elif suffix in TABULAR_SUFFIXES:
        df.to_csv(path, sep="\t", index=index, **kwargs)
    else:
        df.to_csv(path, index=index, **kwargs)
