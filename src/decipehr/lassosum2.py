from __future__ import annotations

from itertools import product

import pandas as pd

DEFAULT_DELTA_GRID = [0.001, 0.01, 0.1, 1.0]
DEFAULT_NLAMBDA = 30
DEFAULT_LAMBDA_MIN_RATIO = 0.01
DEFAULT_DFMAX = 200_000
DEFAULT_MAXITER = 1000
DEFAULT_TOL = 1e-5


def float_tag(value: float) -> str:
    text = f"{value:g}"
    return text.replace("-", "m").replace(".", "p").replace("+", "")


def lassosum2_parameter_grid(
    delta_grid: list[float] | None = None,
    nlambda: int = DEFAULT_NLAMBDA,
) -> pd.DataFrame:
    delta_grid = delta_grid or DEFAULT_DELTA_GRID
    return pd.DataFrame(
        [
            {"delta": float(delta), "lambda_index": int(lambda_index)}
            for delta, lambda_index in product(delta_grid, range(1, nlambda + 1))
        ]
    )


def lassosum2_model_tag(delta: float, lambda_value: float) -> str:
    return f"delta{float_tag(delta)}_lambda{float_tag(lambda_value)}"
