from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from decipehr.io import read_table, write_table
from decipehr.schema import require_columns, validate_unique

DEFAULT_COMBINED_SCORES = (
    "outputs/final_tabpfn3_comparison_rerun_inputs/all8/{disease}.train_long.csv"
)
DEFAULT_TRAIN_PHENOTYPE = "data/processed/phenotype.csv"
DEFAULT_VALIDATION_PHENOTYPE = "data/processed/official_validation_phenotype.csv"
DEFAULT_OUT_DIR = (
    "outputs/ancestry_stratified_best_single_vs_tabpfn3/"
    "tabpfn_best_single_bridge_inputs"
)


@dataclass(frozen=True)
class BridgeInputSpec:
    disease: str
    family: str
    model: str
    source_model: str
    setting: str
    expected_parameters: dict[str, str | float]


SPECS = {
    "cad": BridgeInputSpec(
        disease="cad",
        family="prscs",
        model="prscs",
        source_model="prscs",
        setting="phi=auto",
        expected_parameters={"phi": "auto"},
    ),
    "bc": BridgeInputSpec(
        disease="bc",
        family="lassosum2",
        model="lassosum2",
        source_model="delta1_lambda0p01206377",
        setting="delta=1, lambda=0.0120637653810499",
        expected_parameters={"delta": 1.0, "lambda": 0.0120637653810499},
    ),
    "T2D": BridgeInputSpec(
        disease="T2D",
        family="megaprs_annotated",
        model="megaprs_annotated",
        source_model="megaprs_bayesr",
        setting="model=megaprs_bayesr",
        expected_parameters={"source_model": "megaprs_bayesr"},
    ),
}


def _matches(series: pd.Series, expected: str | float) -> pd.Series:
    if isinstance(expected, float):
        values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
        return pd.Series(
            np.isclose(values, expected, rtol=1e-10, atol=1e-12),
            index=series.index,
        )
    return series.astype("string").str.strip().eq(str(expected))


def select_canonical_scores(
    scores: pd.DataFrame,
    spec: BridgeInputSpec,
    *,
    patient_col: str = "sampleid",
) -> pd.DataFrame:
    require_columns(
        scores,
        [patient_col, "disease", "prs", "model", *spec.expected_parameters],
        f"{spec.disease} combined final-trainval PRS scores",
    )
    selected = scores.loc[
        scores["disease"].astype(str).eq(spec.disease)
        & scores["model"].astype(str).eq(spec.model)
    ].copy()
    for column, expected in spec.expected_parameters.items():
        selected = selected.loc[_matches(selected[column], expected)].copy()
    if selected.empty:
        raise ValueError(
            f"No canonical {spec.disease} {spec.family} rows match {spec.setting}"
        )
    validate_unique(
        selected,
        patient_col,
        f"canonical {spec.disease} {spec.family} combined scores",
    )
    numeric_scores = pd.to_numeric(selected["prs"], errors="coerce")
    if not np.isfinite(numeric_scores).all():
        raise ValueError(f"Canonical {spec.disease} scores contain non-finite values")
    selected["prs"] = numeric_scores.astype(float)
    return selected


def split_bridge_inputs(
    selected_scores: pd.DataFrame,
    train_ids: pd.DataFrame,
    validation_ids: pd.DataFrame,
    spec: BridgeInputSpec,
    *,
    patient_col: str = "sampleid",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    require_columns(train_ids, [patient_col], "official training phenotype")
    require_columns(validation_ids, [patient_col], "official validation phenotype")
    validate_unique(train_ids, patient_col, "official training phenotype")
    validate_unique(validation_ids, patient_col, "official validation phenotype")

    train_set = set(train_ids[patient_col].astype(str))
    validation_set = set(validation_ids[patient_col].astype(str))
    if overlap := train_set.intersection(validation_set):
        examples = sorted(overlap)[:5]
        raise ValueError(f"Training and validation patient sets overlap: {examples}")

    observed = set(selected_scores[patient_col].astype(str))
    expected = train_set.union(validation_set)
    if observed != expected:
        missing = sorted(expected.difference(observed))[:5]
        extra = sorted(observed.difference(expected))[:5]
        raise ValueError(
            "Canonical score patients do not equal the official train + validation "
            f"patients; missing={missing}, extra={extra}"
        )

    base = selected_scores[[patient_col, "prs"]].copy()
    base[patient_col] = base[patient_col].astype(str)
    base.insert(1, "disease", spec.disease)
    base.insert(3, "model", "best_single_prs")
    base["family"] = spec.family
    base["source_model"] = spec.source_model
    base["source_setting"] = spec.setting

    train = train_ids[[patient_col]].assign(
        **{patient_col: train_ids[patient_col].astype(str)}
    ).merge(base, on=patient_col, how="left", validate="one_to_one")
    validation = validation_ids[[patient_col]].assign(
        **{patient_col: validation_ids[patient_col].astype(str)}
    ).merge(base, on=patient_col, how="left", validate="one_to_one")
    if train["prs"].isna().any() or validation["prs"].isna().any():
        raise ValueError("Bridge split unexpectedly produced missing PRS values")
    return train, validation


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build exact single-best-PRS TabPFN bridge inputs by splitting the "
            "canonical final-trainval scores back into official train and validation IDs."
        )
    )
    parser.add_argument("--disease", choices=list(SPECS), required=True)
    parser.add_argument("--combined-scores", default=DEFAULT_COMBINED_SCORES)
    parser.add_argument("--train-phenotype", default=DEFAULT_TRAIN_PHENOTYPE)
    parser.add_argument(
        "--validation-phenotype", default=DEFAULT_VALIDATION_PHENOTYPE
    )
    parser.add_argument("--patient-col", default="sampleid")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    spec = SPECS[args.disease]
    source_path = Path(args.combined_scores.format(disease=args.disease))
    selected = select_canonical_scores(
        read_table(source_path), spec, patient_col=args.patient_col
    )
    train, validation = split_bridge_inputs(
        selected,
        read_table(args.train_phenotype),
        read_table(args.validation_phenotype),
        spec,
        patient_col=args.patient_col,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / f"{args.disease}.train_long.csv"
    validation_path = out_dir / f"{args.disease}.validation_long.csv"
    train_wide_path = out_dir / f"{args.disease}.train_wide.csv"
    validation_wide_path = out_dir / f"{args.disease}.validation_wide.csv"
    write_table(train, train_path)
    write_table(validation, validation_path)
    score_column = f"prs_{args.disease}"
    write_table(
        train[[args.patient_col, "prs"]].rename(columns={"prs": score_column}),
        train_wide_path,
    )
    write_table(
        validation[[args.patient_col, "prs"]].rename(columns={"prs": score_column}),
        validation_wide_path,
    )
    manifest = {
        **asdict(spec),
        "source_score_file": str(source_path),
        "train_phenotype": args.train_phenotype,
        "validation_phenotype": args.validation_phenotype,
        "train_output": str(train_path),
        "validation_output": str(validation_path),
        "train_wide_output": str(train_wide_path),
        "validation_wide_output": str(validation_wide_path),
        "n_combined": int(len(selected)),
        "n_train": int(len(train)),
        "n_validation": int(len(validation)),
    }
    write_table(pd.DataFrame([manifest]), out_dir / f"{args.disease}.manifest.csv")
    print(
        f"Wrote {args.disease} bridge inputs: train={len(train):,}, "
        f"validation={len(validation):,}"
    )


if __name__ == "__main__":
    main()
