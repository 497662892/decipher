"""Label-free agreement diagnostics for two aligned probability vectors."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import logit


def paired_probabilities(frame: pd.DataFrame, student: str, teacher: str = "TabPFN_v3"):
    selected = frame.loc[frame.arm.isin([teacher, student]), ["sampleid", "arm", "probability"]]
    if selected.sampleid.isna().any() or selected.duplicated(["sampleid", "arm"]).any():
        raise ValueError("Prediction pairs require unique, nonmissing patient/model keys")
    wide = selected.pivot(index="sampleid", columns="arm", values="probability").sort_index()
    if set(wide.columns) != {teacher, student} or wide.isna().any().any():
        raise ValueError("Teacher and student must cover the same complete patient set")
    return wide.index.to_numpy(str), wide[teacher].to_numpy(float), wide[student].to_numpy(float)


def agreement(teacher, student, sample_ids, *, replicates=1000, seed=20261105):
    x, y, ids = np.asarray(teacher, float), np.asarray(student, float), np.asarray(sample_ids, str)
    if x.ndim != 1 or x.shape != y.shape or x.shape != ids.shape or len(x) < 3:
        raise ValueError("Agreement requires at least three aligned probability pairs")
    if len(set(ids)) != len(ids) or not np.isfinite([x, y]).all() or ((x < 0) | (x > 1) | (y < 0) | (y > 1)).any():
        raise ValueError("Invalid patient identifiers or probabilities")
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        raise ValueError("Correlation is undefined for constant predictions")

    def statistic(a, b):
        return np.array([stats.pearsonr(a, b).statistic, stats.spearmanr(a, b).statistic,
                         np.mean(np.abs(b - a)), np.mean(b - a)])

    names = ["pearson_probability", "spearman", "mae_probability", "mean_student_minus_teacher"]
    result = dict(zip(names, map(float, statistic(x, y)), strict=True))
    if replicates:
        interval = stats.bootstrap((x, y), statistic, paired=True, vectorized=False,
            n_resamples=replicates, batch=64, method="percentile", rng=np.random.default_rng(seed)).confidence_interval
        if not np.isfinite([interval.low, interval.high]).all():
            raise ValueError("Undefined bootstrap interval")
        for name, lo, hi in zip(names, interval.low, interval.high, strict=True):
            result[f"{name}_ci_low"], result[f"{name}_ci_high"] = float(lo), float(hi)
    k = math.ceil(.1 * len(x))
    a, b = np.lexsort((ids, -x))[:k], np.lexsort((ids, -y))[:k]
    overlap = len(set(a) & set(b))
    result.update(n=len(x), bootstrap_replicates=replicates,
        pearson_logit=float(stats.pearsonr(logit(np.clip(x, 1e-6, 1-1e-6)),
                                         logit(np.clip(y, 1e-6, 1-1e-6))).statistic),
        logit_clip_epsilon=1e-6,
        teacher_logit_clipped=int(((x < 1e-6) | (x > 1-1e-6)).sum()),
        student_logit_clipped=int(((y < 1e-6) | (y > 1-1e-6)).sum()),
        rmse_probability=float(np.sqrt(np.mean((y-x)**2))),
        absolute_difference_p95=float(np.quantile(np.abs(y-x), .95)),
        mean_teacher=float(x.mean()), mean_student=float(y.mean()),
        top_decile_n=k, top_decile_intersection=overlap, top_decile_overlap=overlap/k,
        top_decile_jaccard=overlap/(2*k-overlap),
        teacher_cutoff_ties=int((x == x[a[-1]]).sum()),
        student_cutoff_ties=int((y == y[b[-1]]).sum()))
    return result
