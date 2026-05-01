from __future__ import annotations

from itertools import product

import numpy as np
import pandas as pd

from decipehr.schema import require_columns


def pt_parameter_grid(
    pvalue_thresholds: list[float] | None = None,
    r2_thresholds: list[float] | None = None,
) -> pd.DataFrame:
    pvalue_thresholds = pvalue_thresholds or [1.0, 0.5, 0.05, 5e-4, 5e-6, 5e-8]
    r2_thresholds = r2_thresholds or [0.2, 0.4, 0.6, 0.8]
    return pd.DataFrame(
        [{"p_threshold": p, "clump_r2": r2} for p, r2 in product(pvalue_thresholds, r2_thresholds)]
    )


def compute_prs_from_dosage(
    genotype: pd.DataFrame,
    weights: pd.DataFrame,
    patient_col: str = "patient_id",
    snp_col: str = "SNP",
    beta_col: str = "BETA",
) -> pd.DataFrame:
    require_columns(genotype, [patient_col], "genotype")
    require_columns(weights, [snp_col, beta_col], "weights")
    available = [snp for snp in weights[snp_col].astype(str).tolist() if snp in genotype.columns]
    if not available:
        raise ValueError("No overlapping SNPs between genotype matrix and weights")
    beta = weights.set_index(snp_col).loc[available, beta_col].astype(float)
    dosage = genotype[available].astype(float)
    score = dosage.to_numpy() @ beta.to_numpy()
    return pd.DataFrame({patient_col: genotype[patient_col].to_numpy(), "prs": score})


def standardize_train_val(train_scores: pd.Series, val_scores: pd.Series) -> tuple[pd.Series, pd.Series]:
    mean = float(train_scores.mean())
    sd = float(train_scores.std(ddof=0))
    if not np.isfinite(sd) or sd == 0:
        sd = 1.0
    return (train_scores - mean) / sd, (val_scores - mean) / sd

