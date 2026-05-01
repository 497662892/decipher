from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


@dataclass(frozen=True)
class MetricResult:
    auc: float
    auprc: float
    brier: float
    prevalence: float
    n: int


def _safe_metric(fn, y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return math.nan
    try:
        return float(fn(y_true, y_score))
    except ValueError:
        return math.nan


def binary_metrics(y_true: pd.Series | np.ndarray, y_prob: pd.Series | np.ndarray) -> MetricResult:
    y = np.asarray(y_true).astype(int)
    p = np.asarray(y_prob).astype(float)
    return MetricResult(
        auc=_safe_metric(roc_auc_score, y, p),
        auprc=_safe_metric(average_precision_score, y, p),
        brier=float(brier_score_loss(y, p)),
        prevalence=float(y.mean()) if len(y) else math.nan,
        n=int(len(y)),
    )


def odds_ratio_per_sd(
    y_true: pd.Series | np.ndarray,
    score: pd.Series | np.ndarray,
    covariates: pd.DataFrame | None = None,
) -> float:
    y = pd.Series(y_true).astype(float).reset_index(drop=True)
    x = pd.Series(score).astype(float).reset_index(drop=True)
    sd = x.std(ddof=0)
    if not np.isfinite(sd) or sd == 0 or y.nunique() < 2:
        return math.nan
    design = pd.DataFrame({"score": (x - x.mean()) / sd})
    if covariates is not None and not covariates.empty:
        design = pd.concat([design, covariates.reset_index(drop=True)], axis=1)
        design = pd.get_dummies(design, drop_first=True)
    design = sm.add_constant(design, has_constant="add")
    try:
        result = sm.Logit(y, design.astype(float)).fit(disp=False)
    except Exception:
        return math.nan
    return float(np.exp(result.params["score"]))


def _group_or(cases_a: int, total_a: int, cases_b: int, total_b: int) -> float:
    non_cases_a = total_a - cases_a
    non_cases_b = total_b - cases_b
    return float(((cases_a + 0.5) * (non_cases_b + 0.5)) / ((non_cases_a + 0.5) * (cases_b + 0.5)))


def risk_stratification_or(y_true: pd.Series | np.ndarray, score: pd.Series | np.ndarray) -> dict[str, float]:
    frame = pd.DataFrame({"y": np.asarray(y_true).astype(int), "score": np.asarray(score).astype(float)})
    frame = frame.dropna()
    if frame.empty or frame["score"].nunique() < 3:
        return {"top_vs_bottom": math.nan, "top_vs_middle": math.nan, "bottom_vs_middle": math.nan}
    q10 = frame["score"].quantile(0.1)
    q40 = frame["score"].quantile(0.4)
    q60 = frame["score"].quantile(0.6)
    q90 = frame["score"].quantile(0.9)
    bottom = frame[frame["score"] <= q10]
    middle = frame[(frame["score"] >= q40) & (frame["score"] <= q60)]
    top = frame[frame["score"] >= q90]
    if min(len(bottom), len(middle), len(top)) == 0:
        return {"top_vs_bottom": math.nan, "top_vs_middle": math.nan, "bottom_vs_middle": math.nan}
    return {
        "top_vs_bottom": _group_or(int(top["y"].sum()), len(top), int(bottom["y"].sum()), len(bottom)),
        "top_vs_middle": _group_or(int(top["y"].sum()), len(top), int(middle["y"].sum()), len(middle)),
        "bottom_vs_middle": _group_or(int(bottom["y"].sum()), len(bottom), int(middle["y"].sum()), len(middle)),
    }

