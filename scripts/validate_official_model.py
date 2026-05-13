from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

from decipehr.config import load_config
from decipehr.schema import parse_csv_list

DEFAULT_CONFIG = "configs/default.yaml"
DEFAULT_ELIGIBILITY_CONFIG = "configs/disease_eligibility.yaml"
DEFAULT_TRAIN_PHENOTYPE = (
    "/gpfs/data/deciphEHRlab/deciphEHRdream/training_phase/phenotype_files/"
    "Training_Cov_file.txt"
)


def _safe_name(value: str) -> str:
    safe = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value.strip())
    if not safe:
        raise ValueError("model name must contain at least one alphanumeric, '-' or '_' character")
    return safe


def _default_out_dir(model_name: str) -> str:
    safe = _safe_name(model_name)
    if safe in {"baseline", "covariate_only"}:
        return "outputs/official_validation_baseline"
    return f"outputs/official_validation_{safe}"


def _require_mapping(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key, {})
    if not isinstance(value, dict):
        raise ValueError(f"Config key {key!r} must be a mapping")
    return value


def _required_value(mapping: dict[str, Any], key: str, context: str) -> str:
    value = mapping.get(key)
    if value in (None, ""):
        raise ValueError(f"Missing required config value {context}.{key}")
    return str(value)


def _csv_from_config(value: object, context: str) -> str:
    if isinstance(value, str):
        parsed = parse_csv_list(value)
    elif isinstance(value, list):
        parsed = [str(item) for item in value]
    else:
        raise ValueError(f"Config value {context} must be a string or list")
    if not parsed:
        raise ValueError(f"Config value {context} must not be empty")
    return ",".join(parsed)


def _labels(args: argparse.Namespace, columns: dict[str, Any]) -> str:
    if args.labels and args.disease:
        raise ValueError("Use either --labels or --disease, not both")
    if args.disease:
        return args.disease
    if args.labels:
        return ",".join(parse_csv_list(args.labels))
    return _csv_from_config(columns.get("labels"), "columns.labels")


def _prs_param_cols(model_name: str, value: str | None) -> str:
    if value:
        return ",".join(parse_csv_list(value))
    if model_name.lower() in {"ldpred2_auto", "ldpred2-auto", "ldpred2auto"}:
        return "model,h2,p,sparse"
    if model_name.lower() in {"ldpred2", "ldpred2_prs"}:
        return "h2,p,sparse"
    return "clump_r2,p_threshold"


def _add_optional_pair(
    cmd: list[str],
    train_flag: str,
    train_path: str | None,
    validation_flag: str,
    validation_path: str | None,
    description: str,
) -> None:
    if bool(train_path) != bool(validation_path):
        raise ValueError(f"{description} requires both training and validation score files")
    if train_path and validation_path:
        cmd.extend([train_flag, train_path, validation_flag, validation_path])


def _build_command(args: argparse.Namespace) -> list[str]:
    has_wide_prs = bool(args.train_prs_scores or args.validation_prs_scores)
    has_long_prs = bool(args.train_prs_long_scores or args.validation_prs_long_scores)
    if has_wide_prs and has_long_prs:
        raise ValueError("Use either wide PRS scores or long PRS scores, not both")

    config = load_config(args.config, args.config_override)
    paths = _require_mapping(config, "paths")
    columns = _require_mapping(config, "columns")
    evaluation = _require_mapping(config, "evaluation")

    labels = _labels(args, columns)
    covariates = args.covariates or _csv_from_config(columns.get("covariates"), "columns.covariates")
    patient_col = args.patient_col or _required_value(columns, "patient_id", "columns")
    n_bootstraps = args.n_bootstraps or int(evaluation.get("n_bootstraps", 1000))
    out_dir = args.out_dir or _default_out_dir(args.model_name)
    validation_phenotype_out = args.validation_phenotype_out or str(
        paths.get("official_validation_phenotype", "data/processed/official_validation_phenotype.csv")
    )
    train_phenotype = args.train_phenotype or str(
        paths.get("official_train_phenotype", DEFAULT_TRAIN_PHENOTYPE)
    )

    script = Path(__file__).resolve().with_name("run_official_validation.py")
    cmd = [
        sys.executable,
        str(script),
        "--model-name",
        args.model_name,
        "--train-phenotype",
        train_phenotype,
        "--labelled-phenotype",
        args.labelled_phenotype or _required_value(paths, "official_labelled_phenotype", "paths"),
        "--validation-covariates",
        args.validation_covariates
        or _required_value(paths, "official_validation_covariates", "paths"),
        "--patient-col",
        patient_col,
        "--labels",
        labels,
        "--covariates",
        covariates,
        "--eligibility-config",
        args.eligibility_config,
        "--n-bootstraps",
        str(n_bootstraps),
        "--seed",
        str(args.seed if args.seed is not None else config.get("project", {}).get("seed", 42)),
        "--class-weight",
        args.class_weight,
        "--validation-phenotype-out",
        validation_phenotype_out,
        "--out-dir",
        out_dir,
    ]
    if args.bootstrap_size is not None:
        cmd.extend(["--bootstrap-size", str(args.bootstrap_size)])
    if args.write_predictions:
        cmd.append("--write-predictions")

    _add_optional_pair(
        cmd,
        "--train-prs-scores",
        args.train_prs_scores,
        "--prs-scores",
        args.validation_prs_scores,
        "wide PRS validation",
    )
    _add_optional_pair(
        cmd,
        "--train-prs-long-scores",
        args.train_prs_long_scores,
        "--prs-long-scores",
        args.validation_prs_long_scores,
        "long PRS validation",
    )
    if args.validation_prs_long_scores:
        cmd.extend(["--prs-param-cols", _prs_param_cols(args.model_name, args.prs_param_cols)])
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the fixed official validation workflow for one model. Official "
            "phenotype paths, covariates, labels, eligibility, and bootstrap settings "
            "come from the project config; model-specific PRS score files are the only "
            "required model inputs."
        )
    )
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--config-override", default=None)
    parser.add_argument("--disease", default=None, help="Validate a single disease.")
    parser.add_argument("--labels", default=None, help="Comma-separated label override.")
    parser.add_argument("--covariates", default=None, help="Comma-separated covariate override.")
    parser.add_argument("--patient-col", default=None)
    parser.add_argument("--train-phenotype", default=None)
    parser.add_argument("--labelled-phenotype", default=None)
    parser.add_argument("--validation-covariates", default=None)
    parser.add_argument("--eligibility-config", default=DEFAULT_ELIGIBILITY_CONFIG)
    parser.add_argument("--n-bootstraps", type=int, default=None)
    parser.add_argument("--bootstrap-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--class-weight", choices=["none", "balanced"], default="none")
    parser.add_argument("--train-prs-scores", default=None)
    parser.add_argument("--validation-prs-scores", "--prs-scores", dest="validation_prs_scores", default=None)
    parser.add_argument("--train-prs-long-scores", default=None)
    parser.add_argument(
        "--validation-prs-long-scores",
        "--prs-long-scores",
        dest="validation_prs_long_scores",
        default=None,
    )
    parser.add_argument("--prs-param-cols", default=None)
    parser.add_argument("--validation-phenotype-out", default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--write-predictions", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Print the resolved command without running it.")
    args = parser.parse_args()

    cmd = _build_command(args)
    print("Resolved official validation command:")
    print(shlex.join(cmd))
    if args.dry_run:
        return
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
