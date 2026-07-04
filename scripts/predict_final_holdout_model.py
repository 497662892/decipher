from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from decipehr.config import load_config
from decipehr.eligibility import eligible_counts, filter_eligible, load_eligibility_rules
from decipehr.io import read_table, write_table
from decipehr.models import fit_logistic_predict_proba
from decipehr.schema import parse_csv_list, require_columns, validate_unique
from decipehr.submission import validate_probability_columns, wide_submission

DEFAULT_CONFIG = "configs/default.yaml"
DEFAULT_ELIGIBILITY_CONFIG = "configs/disease_eligibility.yaml"


def _columns_config(config: dict[str, Any]) -> dict[str, Any]:
    columns = config.get("columns", {})
    if not isinstance(columns, dict):
        raise ValueError("Config key 'columns' must be a mapping")
    return columns


def _csv_or_config(value: str | None, config_value: object, context: str) -> list[str]:
    if value:
        return parse_csv_list(value)
    if isinstance(config_value, str):
        return parse_csv_list(config_value)
    if isinstance(config_value, list):
        out = [str(item) for item in config_value]
        if out:
            return out
    raise ValueError(f"{context} must be supplied or present in the config")


def _patient_col(value: str | None, columns: dict[str, Any]) -> str:
    if value:
        return value
    patient_col = columns.get("patient_id")
    if patient_col in (None, ""):
        raise ValueError("patient column must be supplied or present as columns.patient_id")
    return str(patient_col)


def _class_weight(value: str) -> str | None:
    if value == "balanced":
        return "balanced"
    if value == "none":
        return None
    raise ValueError(f"Unsupported class weight: {value}")


def _parse_selected_prs_params(
    value: str | None,
    param_cols: list[str],
) -> list[dict[str, str]]:
    if not value:
        return []
    if not param_cols:
        raise ValueError("--selected-prs-params requires --prs-param-cols")
    selected: list[dict[str, str]] = []
    for raw_spec in value.split(";"):
        raw_spec = raw_spec.strip()
        if not raw_spec:
            continue
        spec: dict[str, str] = {}
        for raw_part in raw_spec.replace("|", ",").split(","):
            if "=" not in raw_part:
                raise ValueError(
                    "Selected PRS parameter specs must use key=value pairs, "
                    f"got {raw_part!r}"
                )
            key, raw_val = raw_part.split("=", 1)
            key = key.strip()
            raw_val = raw_val.strip()
            if key not in param_cols:
                raise ValueError(
                    f"Selected PRS parameter {key!r} is not in --prs-param-cols {param_cols}"
                )
            spec[key] = raw_val
        missing = [col for col in param_cols if col not in spec]
        if missing:
            raise ValueError(f"Selected PRS parameter spec is missing: {missing}")
        selected.append(spec)
    if not selected:
        raise ValueError("--selected-prs-params did not contain any parameter specs")
    return selected


def _param_match_mask(series: pd.Series, raw_value: str) -> pd.Series:
    numeric_series = pd.to_numeric(series, errors="coerce")
    numeric_value = pd.to_numeric(pd.Series([raw_value]), errors="coerce").iloc[0]
    if pd.notna(numeric_value) and numeric_series.notna().any():
        tolerance = max(1e-12, abs(float(numeric_value)) * 1e-9)
        return (numeric_series - float(numeric_value)).abs() <= tolerance
    return series.astype(str) == raw_value


def _filter_long_prs_params(
    prs_long: pd.DataFrame,
    selected: list[dict[str, str]],
    param_cols: list[str],
    context: str,
) -> pd.DataFrame:
    if not selected:
        return prs_long
    require_columns(prs_long, param_cols, context)
    keep_mask = pd.Series(False, index=prs_long.index)
    for spec in selected:
        spec_mask = pd.Series(True, index=prs_long.index)
        for column in param_cols:
            spec_mask &= _param_match_mask(prs_long[column], spec[column])
        keep_mask |= spec_mask
    filtered = prs_long.loc[keep_mask].copy()
    if filtered.empty:
        raise ValueError(f"No {context} rows matched selected PRS params: {selected}")
    return filtered


def _long_prs_group_map(
    prs_long: pd.DataFrame | None,
    disease: str,
    patient_col: str,
    param_cols: list[str],
    context: str,
) -> dict[tuple[object, ...], tuple[dict[str, object], pd.DataFrame]]:
    if prs_long is None:
        return {}
    disease_scores = prs_long.loc[prs_long["disease"] == disease].copy()
    if disease_scores.empty:
        return {}
    require_columns(disease_scores, [patient_col, "prs", *param_cols], context)
    if not param_cols:
        return {(): ({}, disease_scores[[patient_col, "prs"]].copy())}

    groups: dict[tuple[object, ...], tuple[dict[str, object], pd.DataFrame]] = {}
    for values, frame in disease_scores.groupby(param_cols, dropna=False, sort=True):
        if not isinstance(values, tuple):
            values = (values,)
        params = dict(zip(param_cols, values, strict=True))
        groups[values] = (params, frame[[patient_col, "prs"]].copy())
    return groups


def _merge_scores(
    frame: pd.DataFrame,
    scores: pd.DataFrame,
    patient_col: str,
    context: str,
) -> pd.DataFrame:
    merged = frame.merge(scores, on=patient_col, how="left", validate="one_to_one")
    missing = int(merged["prs"].isna().sum())
    if missing:
        raise ValueError(f"{context} is missing PRS scores for {missing} eligible patients")
    return merged


def _wide_prs_score_frame(
    scores: pd.DataFrame | None,
    *,
    disease: str,
    patient_col: str,
    prs_prefix: str,
    context: str,
) -> pd.DataFrame | None:
    if scores is None:
        return None
    score_col = f"{prs_prefix}{disease}"
    require_columns(scores, [patient_col, score_col], context)
    validate_unique(scores, patient_col, context)
    out = scores[[patient_col, score_col]].rename(columns={score_col: "prs"}).copy()
    out["prs"] = pd.to_numeric(out["prs"], errors="coerce")
    return out


def _predict(
    *,
    train: pd.DataFrame,
    test: pd.DataFrame,
    disease: str,
    method: str,
    feature_cols: list[str],
    patient_col: str,
    class_weight: str | None,
    model_name: str,
    params: dict[str, object] | None = None,
) -> pd.DataFrame:
    params = params or {}
    prob = fit_logistic_predict_proba(
        train=train,
        val=test,
        label_col=disease,
        feature_cols=feature_cols,
        patient_col=patient_col,
        class_weight=class_weight,
    )
    pred = pd.DataFrame(
        {
            patient_col: test[patient_col].to_numpy(),
            "model_name": model_name,
            "disease": disease,
            "method": method,
            "y_prob": prob,
        }
    )
    for column, value in params.items():
        pred[column] = value
    return pred


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Fit final train-val logistic PRS models and predict the unlabelled holdout test set."
        )
    )
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--config-override", default=None)
    parser.add_argument("--train-phenotype", default="data/processed/final/train_val_phenotype.csv")
    parser.add_argument("--test-covariates", default="data/processed/final/holdout_test_covariates.csv")
    parser.add_argument("--patient-col", default=None)
    parser.add_argument("--labels", default=None)
    parser.add_argument("--covariates", default=None)
    parser.add_argument("--train-prs-long-scores", default=None)
    parser.add_argument("--test-prs-long-scores", default=None)
    parser.add_argument("--prs-param-cols", default="")
    parser.add_argument(
        "--selected-prs-params",
        default=None,
        help=(
            "Optional semicolon-separated key=value specs, e.g. "
            "'delta=1.0,lambda=0.003' or 'delta=1.0|lambda=0.003'. "
            "Only matching long-score parameter sets are used."
        ),
    )
    parser.add_argument("--train-prs-scores", default=None)
    parser.add_argument("--test-prs-scores", default=None)
    parser.add_argument("--prs-prefix", default="prs_")
    parser.add_argument("--eligibility-config", default=DEFAULT_ELIGIBILITY_CONFIG)
    parser.add_argument("--class-weight", choices=["none", "balanced"], default="none")
    parser.add_argument("--include-covariate-only", action="store_true")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--predictions-out", default=None)
    parser.add_argument("--submission-out", default=None)
    args = parser.parse_args()

    config = load_config(args.config, args.config_override)
    columns = _columns_config(config)
    patient_col = _patient_col(args.patient_col, columns)
    labels = _csv_or_config(args.labels, columns.get("labels"), "labels")
    covariates = _csv_or_config(args.covariates, columns.get("covariates"), "covariates")
    param_cols = parse_csv_list(args.prs_param_cols) if args.prs_param_cols else []
    selected_prs_params = _parse_selected_prs_params(args.selected_prs_params, param_cols)
    class_weight = _class_weight(args.class_weight)

    has_train_prs_long = bool(args.train_prs_long_scores)
    has_test_prs_long = bool(args.test_prs_long_scores)
    has_train_prs_wide = bool(args.train_prs_scores)
    has_test_prs_wide = bool(args.test_prs_scores)
    if has_train_prs_long != has_test_prs_long:
        raise ValueError("--train-prs-long-scores and --test-prs-long-scores must be supplied together")
    if has_train_prs_wide != has_test_prs_wide:
        raise ValueError("--train-prs-scores and --test-prs-scores must be supplied together")
    if has_train_prs_long and has_train_prs_wide:
        raise ValueError("Use either long PRS scores or wide PRS scores, not both")

    train_phenotype = read_table(args.train_phenotype)
    test_covariates = read_table(args.test_covariates)
    require_columns(train_phenotype, [patient_col, *labels, *covariates], "train phenotype")
    require_columns(test_covariates, [patient_col, *covariates], "test covariates")

    train_prs_long = read_table(args.train_prs_long_scores) if args.train_prs_long_scores else None
    test_prs_long = read_table(args.test_prs_long_scores) if args.test_prs_long_scores else None
    if train_prs_long is not None:
        require_columns(train_prs_long, [patient_col, "disease", "prs", *param_cols], "training PRS")
        require_columns(test_prs_long, [patient_col, "disease", "prs", *param_cols], "test PRS")
        train_prs_long = _filter_long_prs_params(
            train_prs_long,
            selected_prs_params,
            param_cols,
            "training PRS",
        )
        test_prs_long = _filter_long_prs_params(
            test_prs_long,
            selected_prs_params,
            param_cols,
            "test PRS",
        )
    train_prs_wide = read_table(args.train_prs_scores) if args.train_prs_scores else None
    test_prs_wide = read_table(args.test_prs_scores) if args.test_prs_scores else None
    if train_prs_wide is not None:
        require_columns(
            train_prs_wide,
            [patient_col, *[f"{args.prs_prefix}{disease}" for disease in labels]],
            "training wide PRS",
        )
        require_columns(
            test_prs_wide,
            [patient_col, *[f"{args.prs_prefix}{disease}" for disease in labels]],
            "test wide PRS",
        )
        validate_unique(train_prs_wide, patient_col, "training wide PRS")
        validate_unique(test_prs_wide, patient_col, "test wide PRS")

    eligibility_rules = load_eligibility_rules(args.eligibility_config)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if selected_prs_params and train_prs_long is not None and test_prs_long is not None:
        write_table(train_prs_long, out_dir / "selected_train_prs_long_scores.csv")
        write_table(test_prs_long, out_dir / "selected_test_prs_long_scores.csv")

    prediction_frames: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []
    for disease in labels:
        train_disease = filter_eligible(train_phenotype, disease, eligibility_rules).reset_index(drop=True)
        test_disease = filter_eligible(test_covariates, disease, eligibility_rules).reset_index(drop=True)
        if train_disease.empty:
            raise ValueError(f"No eligible train-val patients remain for disease {disease}")
        if test_disease.empty:
            raise ValueError(f"No eligible holdout test patients remain for disease {disease}")

        if args.include_covariate_only or (train_prs_long is None and train_prs_wide is None):
            prediction_frames.append(
                _predict(
                    train=train_disease,
                    test=test_disease,
                    disease=disease,
                    method="covariate_only",
                    feature_cols=covariates,
                    patient_col=patient_col,
                    class_weight=class_weight,
                    model_name=args.model_name,
                )
            )
            summary_rows.append(
                {
                    "model_name": args.model_name,
                    "disease": disease,
                    "method": "covariate_only",
                    "n_fit": int(len(train_disease)),
                    "n_prediction": int(len(test_disease)),
                    "fit_cases": int(train_disease[disease].astype(int).sum()),
                }
            )

        train_wide_scores = _wide_prs_score_frame(
            train_prs_wide,
            disease=disease,
            patient_col=patient_col,
            prs_prefix=args.prs_prefix,
            context="training wide PRS scores",
        )
        test_wide_scores = _wide_prs_score_frame(
            test_prs_wide,
            disease=disease,
            patient_col=patient_col,
            prs_prefix=args.prs_prefix,
            context="test wide PRS scores",
        )
        if train_wide_scores is not None and test_wide_scores is not None:
            fit_data = _merge_scores(
                train_disease,
                train_wide_scores,
                patient_col,
                f"training wide PRS disease={disease}",
            )
            pred_data = _merge_scores(
                test_disease,
                test_wide_scores,
                patient_col,
                f"test wide PRS disease={disease}",
            )
            prediction_frames.append(
                _predict(
                    train=fit_data,
                    test=pred_data,
                    disease=disease,
                    method=f"prs_covariate_{disease}",
                    feature_cols=[*covariates, "prs"],
                    patient_col=patient_col,
                    class_weight=class_weight,
                    model_name=args.model_name,
                )
            )
            summary_rows.append(
                {
                    "model_name": args.model_name,
                    "disease": disease,
                    "method": f"prs_covariate_{disease}",
                    "n_fit": int(len(fit_data)),
                    "n_prediction": int(len(pred_data)),
                    "fit_cases": int(fit_data[disease].astype(int).sum()),
                }
            )
            continue

        train_groups = _long_prs_group_map(
            train_prs_long,
            disease,
            patient_col,
            param_cols,
            "training PRS scores",
        )
        test_groups = _long_prs_group_map(
            test_prs_long,
            disease,
            patient_col,
            param_cols,
            "test PRS scores",
        )
        for key, (params, test_score_frame) in test_groups.items():
            if key not in train_groups:
                raise ValueError(f"Missing training PRS scores for disease={disease}, params={params}")
            _, train_score_frame = train_groups[key]
            fit_data = _merge_scores(
                train_disease,
                train_score_frame,
                patient_col,
                f"training disease={disease} params={params}",
            )
            pred_data = _merge_scores(
                test_disease,
                test_score_frame,
                patient_col,
                f"test disease={disease} params={params}",
            )
            prediction_frames.append(
                _predict(
                    train=fit_data,
                    test=pred_data,
                    disease=disease,
                    method=f"prs_covariate_{disease}",
                    feature_cols=[*covariates, "prs"],
                    patient_col=patient_col,
                    class_weight=class_weight,
                    model_name=args.model_name,
                    params=params,
                )
            )
            summary_rows.append(
                {
                    "model_name": args.model_name,
                    "disease": disease,
                    "method": f"prs_covariate_{disease}",
                    **params,
                    "n_fit": int(len(fit_data)),
                    "n_prediction": int(len(pred_data)),
                    "fit_cases": int(fit_data[disease].astype(int).sum()),
                }
            )

    predictions = pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()
    predictions_out = Path(args.predictions_out) if args.predictions_out else out_dir / "predictions.csv"
    write_table(predictions, predictions_out)
    write_table(pd.DataFrame(summary_rows), out_dir / "fit_prediction_summary.csv")
    write_table(eligible_counts(train_phenotype, labels, eligibility_rules), out_dir / "train_eligibility_counts.csv")
    write_table(eligible_counts(test_covariates, labels, eligibility_rules), out_dir / "test_eligibility_counts.csv")
    (out_dir / "summary.json").write_text(
        json.dumps(summary_rows, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    if args.submission_out:
        duplicate_keys = predictions.duplicated([patient_col, "disease"], keep=False)
        if duplicate_keys.any():
            raise ValueError(
                "Submission output requires exactly one prediction per patient and disease; "
                "rerun with one selected parameter set and without --include-covariate-only."
            )
        submission = wide_submission(predictions, patient_col, disease_col="disease", probability_col="y_prob")
        validate_probability_columns(submission, patient_col)
        write_table(submission, args.submission_out)

    print(f"Wrote final holdout predictions to {predictions_out}")


if __name__ == "__main__":
    main()
