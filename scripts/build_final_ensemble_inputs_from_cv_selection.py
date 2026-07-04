from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from decipehr.io import read_table, write_table
from decipehr.schema import parse_csv_list, require_columns


@dataclass(frozen=True)
class FinalFamilySpec:
    name: str
    score_root: str
    train_scores_template: str
    test_scores_template: str
    param_cols: tuple[str, ...]


DEFAULT_FAMILIES: dict[str, FinalFamilySpec] = {
    "pt": FinalFamilySpec(
        name="pt",
        score_root="outputs/pt_final_trainval_test",
        train_scores_template="{disease}/scores/{disease}.pt_full_train_trainval_scores_long.csv",
        test_scores_template="{disease}/scores/{disease}.pt_full_train_holdout_test_scores_long.csv",
        param_cols=("clump_r2", "p_threshold"),
    ),
    "ldpred2": FinalFamilySpec(
        name="ldpred2",
        score_root="outputs/ldpred2_final_trainval_test",
        train_scores_template="{disease}/scores/{disease}.ldpred2_full_train_trainval_scores_long.csv",
        test_scores_template="{disease}/scores/{disease}.ldpred2_full_train_holdout_test_scores_long.csv",
        param_cols=("h2", "p", "sparse"),
    ),
    "lassosum2": FinalFamilySpec(
        name="lassosum2",
        score_root="outputs/lassosum2_ukbb_final_trainval_test",
        train_scores_template="{disease}/scores/{disease}.lassosum2_full_train_trainval_scores_long.csv",
        test_scores_template="{disease}/scores/{disease}.lassosum2_full_train_holdout_test_scores_long.csv",
        param_cols=("delta", "lambda"),
    ),
    "prscs": FinalFamilySpec(
        name="prscs",
        score_root="outputs/prscs_final_trainval_test_ukbb_eur",
        train_scores_template="{disease}/scores/{disease}.prscs_full_train_trainval_scores_long.csv",
        test_scores_template="{disease}/scores/{disease}.prscs_full_train_holdout_test_scores_long.csv",
        param_cols=("phi",),
    ),
    "sdpr": FinalFamilySpec(
        name="sdpr",
        score_root="outputs/sdpr_final_trainval_test",
        train_scores_template="{disease}/scores/{disease}.sdpr_full_train_trainval_scores_long.csv",
        test_scores_template="{disease}/scores/{disease}.sdpr_full_train_holdout_test_scores_long.csv",
        param_cols=("model",),
    ),
    "megaprs_noannot": FinalFamilySpec(
        name="megaprs_noannot",
        score_root="outputs/megaprs_final_trainval_noannot",
        train_scores_template="{disease}/scores/{disease}.megaprs_full_train_trainval_scores_long.csv",
        test_scores_template="{disease}/scores/{disease}.megaprs_full_train_holdout_test_scores_long.csv",
        param_cols=(),
    ),
    "megaprs_annotated": FinalFamilySpec(
        name="megaprs_annotated",
        score_root="outputs/megaprs_final_trainval_annotated",
        train_scores_template="{disease}/scores/{disease}.megaprs_full_train_trainval_scores_long.csv",
        test_scores_template="{disease}/scores/{disease}.megaprs_full_train_holdout_test_scores_long.csv",
        param_cols=(),
    ),
    "sbayesrc": FinalFamilySpec(
        name="sbayesrc",
        score_root="outputs/sbayesrc_7m_final_trainval_eig0p95",
        train_scores_template="{disease}/scores/{disease}.sbayesrc_full_train_trainval_scores_long.csv",
        test_scores_template="{disease}/scores/{disease}.sbayesrc_full_train_holdout_test_scores_long.csv",
        param_cols=(),
    ),
}


def _path(root: str | Path, template: str, disease: str) -> Path:
    return Path(root) / template.format(disease=disease)


def _numeric_scalar(value: Any) -> float | None:
    if pd.isna(value):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(numeric):
        return None
    return numeric


def _normal_string(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().lower()


def _match_param(series: pd.Series, selected_value: Any) -> pd.Series:
    selected_numeric = _numeric_scalar(selected_value)
    if selected_numeric is not None:
        numeric_series = pd.to_numeric(series, errors="coerce")
        numeric_mask = np.isclose(
            numeric_series.to_numpy(dtype=float),
            selected_numeric,
            rtol=1e-8,
            atol=1e-12,
            equal_nan=False,
        )
        if bool(np.any(numeric_mask)):
            return pd.Series(numeric_mask, index=series.index)
    return series.map(_normal_string) == _normal_string(selected_value)


def _selection_row(selection: pd.DataFrame, *, disease: str, family: str) -> pd.Series:
    require_columns(selection, ["disease", "family"], "selection file")
    rows = selection.loc[
        (selection["disease"].astype(str) == disease) & (selection["family"].astype(str) == family)
    ]
    if rows.empty:
        raise ValueError(f"Selection file has no row for disease={disease!r}, family={family!r}")
    if len(rows) > 1:
        raise ValueError(f"Selection file has multiple rows for disease={disease!r}, family={family!r}")
    return rows.iloc[0]


def _select_score_rows(
    scores: pd.DataFrame,
    *,
    selected: pd.Series,
    spec: FinalFamilySpec,
    disease: str,
    patient_col: str,
    disease_col: str,
    score_col: str,
    score_path: Path,
    context: str,
) -> pd.DataFrame:
    require_columns(scores, [patient_col, disease_col, score_col, *spec.param_cols], context)
    mask = scores[disease_col].astype(str) == disease
    for column in spec.param_cols:
        if column not in selected.index or pd.isna(selected[column]):
            raise ValueError(f"Selection row for {spec.name} has no value for parameter {column!r}")
        mask &= _match_param(scores[column], selected[column])
    matched = scores.loc[mask].copy()
    if matched.empty:
        selected_params = {col: selected[col] for col in spec.param_cols}
        raise ValueError(f"No {context} rows matched {spec.name} selected parameters: {selected_params}")
    if matched.duplicated([patient_col, disease_col]).any():
        examples = matched.loc[matched.duplicated([patient_col, disease_col]), [patient_col, disease_col]].head(5)
        raise ValueError(f"Duplicate selected {context} rows: {examples.to_dict('records')}")

    matched[score_col] = pd.to_numeric(matched[score_col], errors="coerce")
    if matched[score_col].isna().any():
        examples = matched.loc[matched[score_col].isna(), [patient_col, score_col]].head(5)
        raise ValueError(f"Selected {context} contains nonnumeric PRS values: {examples.to_dict('records')}")

    output = matched[[patient_col, disease_col, score_col]].copy()
    output = output.rename(columns={disease_col: "disease", score_col: "prs"})
    output["model"] = spec.name
    output["family"] = spec.name
    output["source_model"] = matched["model"].astype(str).to_numpy() if "model" in matched.columns else spec.name
    for column in spec.param_cols:
        if column == "model":
            output["selected_model"] = matched[column].to_numpy()
        else:
            output[column] = matched[column].to_numpy()
    output["source_score_file"] = str(score_path)
    if "selection_metric" in selected.index:
        output["selection_metric"] = selected["selection_metric"]
    if "selection_metric_value" in selected.index:
        output["selection_metric_value"] = selected["selection_metric_value"]
    return output


def _process_family(
    *,
    spec: FinalFamilySpec,
    selected: pd.Series,
    disease: str,
    patient_col: str,
    disease_col: str,
    score_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    train_path = _path(spec.score_root, spec.train_scores_template, disease)
    test_path = _path(spec.score_root, spec.test_scores_template, disease)
    missing = [path for path in [train_path, test_path] if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing final score files for {spec.name} {disease}: {missing}")

    train_scores = read_table(train_path)
    test_scores = read_table(test_path)
    train_selected = _select_score_rows(
        train_scores,
        selected=selected,
        spec=spec,
        disease=disease,
        patient_col=patient_col,
        disease_col=disease_col,
        score_col=score_col,
        score_path=train_path,
        context=f"{spec.name} final train-val scores",
    )
    test_selected = _select_score_rows(
        test_scores,
        selected=selected,
        spec=spec,
        disease=disease,
        patient_col=patient_col,
        disease_col=disease_col,
        score_col=score_col,
        score_path=test_path,
        context=f"{spec.name} final holdout test scores",
    )
    record: dict[str, Any] = {
        "disease": disease,
        "family": spec.name,
        "train_score_file": str(train_path),
        "test_score_file": str(test_path),
    }
    for column in ["selection_metric", "selection_direction", "selection_metric_value", *spec.param_cols]:
        if column in selected.index:
            record[column] = selected[column]
    return train_selected, test_selected, record


def _apply_overrides(args: argparse.Namespace, specs: dict[str, FinalFamilySpec]) -> dict[str, FinalFamilySpec]:
    overrides = {
        "pt": args.pt_score_root,
        "ldpred2": args.ldpred2_score_root,
        "lassosum2": args.lassosum2_score_root,
        "prscs": args.prscs_score_root,
        "sdpr": args.sdpr_score_root,
        "megaprs_noannot": args.megaprs_noannot_score_root,
        "megaprs_annotated": args.megaprs_annotated_score_root,
        "sbayesrc": args.sbayesrc_score_root,
    }
    updated = {}
    for name, spec in specs.items():
        updated[name] = FinalFamilySpec(
            name=spec.name,
            score_root=overrides[name] or spec.score_root,
            train_scores_template=spec.train_scores_template,
            test_scores_template=spec.test_scores_template,
            param_cols=spec.param_cols,
        )
    return updated


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Use CV-selected PRS family parameters to collect full train-val and holdout "
            "test long inputs for final TabPFN ensemble prediction."
        )
    )
    parser.add_argument("--selection-file", default="outputs/tabpfn_cv_ensemble_inputs/selected_single_models.csv")
    parser.add_argument("--disease", default=None)
    parser.add_argument("--labels", default=None)
    parser.add_argument("--families", default="pt,ldpred2,lassosum2,prscs,sdpr")
    parser.add_argument("--patient-col", default="sampleid")
    parser.add_argument("--disease-col", default="disease")
    parser.add_argument("--score-col", default="prs")
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--pt-score-root", default=None)
    parser.add_argument("--ldpred2-score-root", default=None)
    parser.add_argument("--lassosum2-score-root", default=None)
    parser.add_argument("--prscs-score-root", default=None)
    parser.add_argument("--sdpr-score-root", default=None)
    parser.add_argument("--megaprs-noannot-score-root", default=None)
    parser.add_argument("--megaprs-annotated-score-root", default=None)
    parser.add_argument("--sbayesrc-score-root", default=None)
    args = parser.parse_args()

    if bool(args.disease) == bool(args.labels):
        raise ValueError("Use exactly one of --disease or --labels")
    diseases = [args.disease] if args.disease else parse_csv_list(args.labels)
    families = parse_csv_list(args.families)
    unknown = [family for family in families if family not in DEFAULT_FAMILIES]
    if unknown:
        raise ValueError(f"Unknown PRS families: {unknown}; expected one of {sorted(DEFAULT_FAMILIES)}")

    selection = read_table(args.selection_file)
    specs = _apply_overrides(args, DEFAULT_FAMILIES)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_records = []
    for disease in diseases:
        train_parts = []
        test_parts = []
        records = []
        missing_messages = []
        for family in families:
            try:
                selected = _selection_row(selection, disease=disease, family=family)
                train_selected, test_selected, record = _process_family(
                    spec=specs[family],
                    selected=selected,
                    disease=disease,
                    patient_col=args.patient_col,
                    disease_col=args.disease_col,
                    score_col=args.score_col,
                )
            except (FileNotFoundError, ValueError) as exc:
                if args.allow_missing:
                    missing_messages.append(f"{family}: {exc}")
                    continue
                raise
            train_parts.append(train_selected)
            test_parts.append(test_selected)
            records.append(record)
        if not train_parts or not test_parts:
            raise ValueError(f"No final TabPFN ensemble inputs were built for {disease}; missing={missing_messages}")
        train_long = pd.concat(train_parts, ignore_index=True)
        test_long = pd.concat(test_parts, ignore_index=True)
        for label, frame in [("train", train_long), ("test", test_long)]:
            if frame.duplicated([args.patient_col, "disease", "model"]).any():
                examples = frame.loc[
                    frame.duplicated([args.patient_col, "disease", "model"]),
                    [args.patient_col, "disease", "model"],
                ].head(5)
                raise ValueError(f"Duplicate final {label} ensemble rows: {examples.to_dict('records')}")
        write_table(train_long, out_dir / f"{disease}.train_long.csv")
        write_table(test_long, out_dir / f"{disease}.test_long.csv")
        write_table(pd.DataFrame(records), out_dir / f"{disease}.selected_final_models.csv")
        all_records.extend(records)
        print(
            f"Wrote {len(records)} selected final PRS families for {disease} to "
            f"{out_dir / f'{disease}.train_long.csv'} and {out_dir / f'{disease}.test_long.csv'}"
        )
        for message in missing_messages:
            print(f"Skipped missing family for {disease}: {message}")

    if all_records:
        write_table(pd.DataFrame(all_records), out_dir / "selected_final_models.csv")


if __name__ == "__main__":
    main()
