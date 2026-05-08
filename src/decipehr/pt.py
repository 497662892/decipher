from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from decipehr.io import read_table, write_table
from decipehr.schema import require_columns


def float_tag(value: float) -> str:
    return f"{value:g}".replace(".", "p").replace("-", "m")


def p_threshold_column(p_threshold: float) -> str:
    return f"BETA_p{float_tag(p_threshold)}"


def r2_tag(r2_threshold: float) -> str:
    return f"r2{float_tag(r2_threshold)}"


def fold_r2_tag(fold: int, r2_threshold: float) -> str:
    return f"fold{fold}_{r2_tag(r2_threshold)}"


def _find_first(columns: list[str], candidates: list[str], table_name: str) -> str:
    normalized = {col.upper().lstrip("#"): col for col in columns}
    for candidate in candidates:
        if candidate.upper().lstrip("#") in normalized:
            return normalized[candidate.upper().lstrip("#")]
    raise ValueError(f"{table_name} is missing one of these columns: {candidates}")


def read_plink_clumps(path: str | Path) -> pd.DataFrame:
    """Read PLINK2 .clumps output and normalize lead variant IDs and p-values."""
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame({"SNP": pd.Series(dtype="string"), "P": pd.Series(dtype="float64")})
    clumps = pd.read_csv(path, sep=r"\s+", engine="python")
    if clumps.empty:
        return pd.DataFrame({"SNP": pd.Series(dtype="string"), "P": pd.Series(dtype="float64")})
    id_col = _find_first(list(clumps.columns), ["ID", "SNP", "SNP_ID", "RSID"], str(path))
    p_col = None
    try:
        p_col = _find_first(list(clumps.columns), ["P", "P_VALUE", "PVALUE", "PVAL"], str(path))
    except ValueError:
        pass
    out = pd.DataFrame({"SNP": clumps[id_col].astype("string").str.strip()})
    if p_col is None:
        out["P"] = np.nan
    else:
        out["P"] = pd.to_numeric(clumps[p_col], errors="coerce")
    out = out.dropna(subset=["SNP"])
    out = out.loc[~out["SNP"].isin(["", "nan", "<NA>"])]
    return out.drop_duplicates(subset=["SNP"], keep="first").reset_index(drop=True)


def build_multithreshold_score_table(
    harmonized: pd.DataFrame,
    clumps: pd.DataFrame,
    p_thresholds: list[float],
) -> tuple[pd.DataFrame, dict[str, int | float | list[float] | dict[str, int]]]:
    """Build one PLINK2 score file with one beta column per p-value threshold."""
    require_columns(harmonized, ["SNP", "A1", "BETA", "P"], "harmonized summary statistics")
    require_columns(clumps, ["SNP"], "PLINK clumps")
    if not p_thresholds:
        raise ValueError("At least one p-value threshold is required")

    work = harmonized.copy()
    work["SNP"] = work["SNP"].astype("string").str.strip()
    work["A1"] = work["A1"].astype("string").str.upper()
    work["BETA"] = pd.to_numeric(work["BETA"], errors="coerce")
    work["P"] = pd.to_numeric(work["P"], errors="coerce")
    work = work.dropna(subset=["SNP", "A1", "BETA", "P"])
    work = work.drop_duplicates(subset=["SNP"], keep="first")
    n_after_required_columns = int(len(work))

    clumped_ids = set(clumps["SNP"].astype("string").dropna().str.strip())
    work = work.loc[work["SNP"].isin(clumped_ids)].copy()
    if {"CHR", "BP"}.issubset(work.columns):
        work = work.sort_values(["CHR", "BP", "SNP"], kind="mergesort")
    else:
        work = work.sort_values(["SNP"], kind="mergesort")

    score = work[["SNP", "A1"]].reset_index(drop=True)
    counts_by_threshold: dict[str, int] = {}
    for threshold in p_thresholds:
        col = p_threshold_column(threshold)
        mask = work["P"].le(float(threshold)).to_numpy()
        score[col] = np.where(mask, work["BETA"].to_numpy(), 0.0)
        counts_by_threshold[col] = int(mask.sum())

    beta_cols = [p_threshold_column(threshold) for threshold in p_thresholds]
    score = score.loc[score[beta_cols].ne(0.0).any(axis=1)].reset_index(drop=True)
    summary: dict[str, int | float | list[float] | dict[str, int]] = {
        "n_harmonized_input": int(len(harmonized)),
        "n_harmonized_after_required_columns": n_after_required_columns,
        "n_clumped_leads": int(len(clumped_ids)),
        "n_harmonized_clumped_overlap": int(len(work)),
        "n_score_variants": int(len(score)),
        "p_thresholds": [float(threshold) for threshold in p_thresholds],
        "n_nonzero_by_threshold": counts_by_threshold,
    }
    return score, summary


def build_multithreshold_score_file(
    harmonized_path: str | Path,
    clumps_path: str | Path,
    out_path: str | Path,
    p_thresholds: list[float],
) -> dict[str, int | float | list[float] | dict[str, int]]:
    harmonized = read_table(harmonized_path)
    clumps = read_plink_clumps(clumps_path)
    score, summary = build_multithreshold_score_table(harmonized, clumps, p_thresholds)
    write_table(score, out_path)
    return summary


def plink_score_columns(n_score_columns: int) -> list[int]:
    if n_score_columns < 1:
        raise ValueError("n_score_columns must be at least 1")
    return list(range(3, 3 + n_score_columns))


_SSCORE_COL_RE = re.compile(r"^SCORE(\d+)_(SUM|AVG)$")


def _plink_score_sum_or_avg(columns: list[str], score_name: str) -> str | None:
    candidates = {col.upper(): col for col in columns}
    return candidates.get(f"{score_name.upper()}_SUM") or candidates.get(
        f"{score_name.upper()}_AVG"
    )


def plink_sscore_columns(
    columns: list[str],
    n_scores: int,
    score_names: list[str] | None = None,
) -> list[str]:
    if score_names is not None and len(score_names) != n_scores:
        raise ValueError("score_names length must match n_scores")

    by_index: dict[int, dict[str, str]] = {}
    for col in columns:
        match = _SSCORE_COL_RE.match(col)
        if match:
            index = int(match.group(1))
            kind = match.group(2)
            by_index.setdefault(index, {})[kind] = col
    selected = []
    for index in range(1, n_scores + 1):
        if score_names is not None:
            named_col = _plink_score_sum_or_avg(columns, score_names[index - 1])
            if named_col is not None:
                selected.append(named_col)
                continue
        options = by_index.get(index, {})
        col = options.get("SUM") or options.get("AVG")
        if col is None:
            raise ValueError(f"PLINK .sscore output is missing SCORE{index}_SUM/AVG")
        selected.append(col)
    return selected


def read_plink_sscore(path: str | Path, patient_col: str = "sampleid") -> pd.DataFrame:
    frame = pd.read_csv(path, sep=r"\s+", engine="python")
    if "#IID" in frame.columns:
        frame = frame.rename(columns={"#IID": patient_col})
    elif "IID" in frame.columns:
        frame = frame.rename(columns={"IID": patient_col})
    require_columns(frame, [patient_col], str(path))
    return frame
