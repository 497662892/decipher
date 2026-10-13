"""Post hoc diagnostics for frozen binary predictions; no predictive model fitting."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score
from statsmodels.stats.contingency_tables import Table2x2
from statsmodels.stats.proportion import proportion_confint

from decipehr.metrics import binary_metrics, nagelkerke_r2, top_decile_odds_ratio


def validate_predictions(frame: pd.DataFrame) -> None:
    required = ["sampleid", "disease", "label", "score", "direct_probability"]
    if set(required) - set(frame):
        raise ValueError("Missing required prediction columns")
    if frame[required].isna().any().any():
        raise ValueError("Missing prediction values")
    if frame.duplicated(["sampleid", "disease"]).any():
        raise ValueError("Duplicate patient-disease predictions")
    if "fold" in frame and frame.groupby("sampleid").fold.nunique().gt(1).any():
        raise ValueError("A patient has different folds across diseases")
    if not frame.label.isin([0, 1]).all():
        raise ValueError("Non-binary labels")
    if not np.isfinite(frame[["score", "direct_probability"]].to_numpy()).all():
        raise ValueError("Non-finite predictions")
    if not frame.direct_probability.between(0, 1, inclusive="neither").all():
        raise ValueError("Probabilities must be strictly between zero and one")


def log_likelihood_terms(y: np.ndarray, p: np.ndarray | float) -> np.ndarray:
    p = np.clip(p, 1e-15, 1 - 1e-15)
    return y * np.log(p) + (1 - y) * np.log1p(-p)


def metric_row(y: np.ndarray, p: np.ndarray, score: np.ndarray | None = None) -> dict:
    base = binary_metrics(y, p)
    top = top_decile_odds_ratio(y, p if score is None else score)
    return {
        **vars(base), **top, "cases": int(y.sum()),
        "nagelkerke_r2": nagelkerke_r2(y, p), "log_loss": log_loss(y, p),
        "top10_case_rate": top["top10_cases"] / top["top10_n"],
        "lower90_case_rate": top["lower90_cases"] / top["lower90_n"],
        "top10_sensitivity": top["top10_cases"] / y.sum(),
    }


def risk_deciles(score: np.ndarray) -> np.ndarray:
    """Decile 10 contains exactly ceil(n/10) patients, matching repository OR90."""
    n = len(score)
    order = np.argsort(-np.asarray(score), kind="stable")
    boundaries = np.ceil(np.arange(1, 11) * n / 10).astype(int)
    result = np.empty(n, dtype=int)
    result[order] = 10 - np.searchsorted(boundaries, np.arange(n), side="right")
    return result


def decile_table(y: np.ndarray, p: np.ndarray, score: np.ndarray,
                 reference_p: np.ndarray) -> pd.DataFrame:
    ll = log_likelihood_terms(y, p)
    frame = pd.DataFrame({
        "decile": risk_deciles(score), "y": y, "p": p, "score": score,
        "ll_gain_null": ll - log_likelihood_terms(y, y.mean()),
        "ll_gain_covariates": ll - log_likelihood_terms(y, reference_p),
    })
    out = frame.groupby("decile").agg(
        n=("y", "size"), cases=("y", "sum"), observed=("y", "mean"),
        predicted=("p", "mean"), score_min=("score", "min"), score_max=("score", "max"),
        ll_gain_null=("ll_gain_null", "sum"),
        ll_gain_covariates=("ll_gain_covariates", "sum"),
    ).reset_index()
    out["observed_low"], out["observed_high"] = proportion_confint(
        out.cases, out.n, method="wilson"
    )
    return out


def threshold_table(y: np.ndarray, score: np.ndarray, fractions: list[float]) -> pd.DataFrame:
    rows = []
    for fraction in fractions:
        top = top_decile_odds_ratio(y, score, top_fraction=fraction)
        a, b = top["top10_cases"], top["top10_n"] - top["top10_cases"]
        c, d = top["lower90_cases"], top["lower90_n"] - top["lower90_cases"]
        cells = np.asarray([[a, b], [c, d]], dtype=float)
        # Match the repository's all-cell Haldane correction for zero-cell tables.
        table = Table2x2(cells + (0.5 if (cells == 0).any() else 0), shift_zeros=False)
        low, high = table.oddsratio_confint()
        rows.append({
            "top_fraction": fraction, "n_top": a + b, "cases_top": a,
            "n_rest": c + d, "cases_rest": c, "or": top["top10_vs_lower90_or"],
            "or_low": low, "or_high": high, "case_rate": a / (a + b),
            "sensitivity": a / y.sum(), "enrichment": a / (a + b) / y.mean(),
        })
    return pd.DataFrame(rows)


def paired_bootstrap(y: np.ndarray, probabilities: dict[str, np.ndarray],
                     n_bootstrap: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Patient resampling conditional on frozen scores; same draws for both models."""
    rng = np.random.default_rng(seed)
    rows = []
    for repeat in range(n_bootstrap):
        idx = rng.integers(0, len(y), len(y))
        yi = y[idx]
        if len(np.unique(yi)) != 2:
            continue
        for model, p in probabilities.items():
            pi = p[idx]
            rows.append({
                "repeat": repeat, "model": model,
                "auc": roc_auc_score(yi, pi),
                "nagelkerke_r2": nagelkerke_r2(yi, pi),
                "top10_vs_lower90_or": top_decile_odds_ratio(yi, pi)["top10_vs_lower90_or"],
                "brier": np.mean((yi - pi) ** 2),
                "log_loss": -log_likelihood_terms(yi, pi).mean(),
            })
    draws = pd.DataFrame(rows)
    intervals = []
    for model, group in draws.groupby("model"):
        for metric in ["auc", "nagelkerke_r2", "top10_vs_lower90_or", "brier", "log_loss"]:
            low, high = group[metric].quantile([0.025, 0.975])
            intervals.append({"model": model, "metric": metric, "low": low, "high": high,
                              "n_valid": len(group)})
    return draws, pd.DataFrame(intervals)
