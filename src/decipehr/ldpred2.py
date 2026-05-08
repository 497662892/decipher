from __future__ import annotations

from itertools import product

import pandas as pd

DEFAULT_H2_GRID = [0.05, 0.1, 0.2, 0.4, 0.8, 1.0]
DEFAULT_P_GRID = [1e-4, 1e-3, 1e-2, 0.1, 1.0]
DEFAULT_SPARSE_GRID = [False, True]


def float_tag(value: float) -> str:
    text = f"{value:g}"
    return text.replace("-", "m").replace(".", "p").replace("+", "")


def sparse_tag(value: bool) -> str:
    return "sparse1" if value else "sparse0"


def ldpred2_parameter_grid(
    h2_grid: list[float] | None = None,
    p_grid: list[float] | None = None,
    sparse_grid: list[bool] | None = None,
) -> pd.DataFrame:
    h2_grid = h2_grid or DEFAULT_H2_GRID
    p_grid = p_grid or DEFAULT_P_GRID
    sparse_grid = DEFAULT_SPARSE_GRID if sparse_grid is None else sparse_grid
    return pd.DataFrame(
        [
            {"h2": float(h2), "p": float(p), "sparse": bool(sparse)}
            for h2, p, sparse in product(h2_grid, p_grid, sparse_grid)
        ]
    )


def ldpred2_model_tag(h2: float, p: float, sparse: bool) -> str:
    return f"h2{float_tag(h2)}_p{float_tag(p)}_{sparse_tag(bool(sparse))}"
