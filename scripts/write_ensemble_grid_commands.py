from __future__ import annotations

import argparse
import shlex
from pathlib import Path

import pandas as pd

from decipehr.io import write_table
from decipehr.schema import parse_csv_list


def _safe_token(value: object) -> str:
    text = str(value).strip().replace(".", "p").replace("-", "m")
    return "".join(char if char.isalnum() or char in {"_", "p", "m"} else "_" for char in text)


def _c_values(value: str) -> list[float]:
    values = [float(item) for item in parse_csv_list(value)]
    if not values:
        raise ValueError("--c-grid must contain at least one value")
    if any(item <= 0 for item in values):
        raise ValueError("All --c-grid values must be positive")
    return values


def _penalties(value: str) -> list[str]:
    penalties = parse_csv_list(value)
    invalid = [item for item in penalties if item not in {"l1", "l2", "elasticnet", "none"}]
    if invalid:
        raise ValueError(f"Unsupported penalties: {invalid}")
    if not penalties:
        raise ValueError("--penalties must contain at least one value")
    return penalties


def _l1_ratios(value: str) -> list[float]:
    values = [float(item) for item in parse_csv_list(value)]
    if not values:
        raise ValueError("--l1-ratio-grid must contain at least one value")
    if any(item < 0 or item > 1 for item in values):
        raise ValueError("All --l1-ratio-grid values must be between 0 and 1")
    return values


def _setting_id(
    *,
    penalty: str,
    c_value: float,
    l1_ratio: float | None,
    class_weight: str,
    strategies: str,
) -> str:
    strategy_token = _safe_token(strategies.replace(",", "_"))
    if penalty == "none":
        return f"penalty_none_class_{class_weight}_{strategy_token}"
    ratio_token = f"_l1r{_safe_token(l1_ratio)}" if penalty == "elasticnet" else ""
    return f"penalty_{penalty}_C{_safe_token(c_value)}{ratio_token}_class_{class_weight}_{strategy_token}"


def _build_command(
    *,
    args: argparse.Namespace,
    disease: str,
    setting_id: str,
    penalty: str,
    c_value: float,
    l1_ratio: float | None,
    class_weight: str,
) -> list[str]:
    setting_dir = Path(args.out_root) / disease / setting_id
    command = [
        "uv",
        "run",
        "python",
        "scripts/build_ensemble_prs.py",
        "--config",
        args.config,
        "--disease",
        disease,
        "--model-name",
        f"ensemble_{setting_id}",
        "--train-prs-long-scores",
        str(Path(args.ensemble_input_dir) / f"{disease}.train_long.csv"),
        "--validation-prs-long-scores",
        str(Path(args.ensemble_input_dir) / f"{disease}.validation_long.csv"),
        "--strategies",
        args.strategies,
        "--penalty",
        penalty,
        "--class-weight",
        class_weight,
        "--eligibility-config",
        args.eligibility_config,
        "--out-dir",
        str(setting_dir),
    ]
    if penalty != "none":
        command.extend(["--C", str(c_value)])
    if penalty == "elasticnet":
        if l1_ratio is None:
            raise ValueError("elasticnet command generation requires l1_ratio")
        command.extend(["--l1-ratio", str(l1_ratio)])
    if args.config_override:
        command.extend(["--config-override", args.config_override])
    if args.covariates:
        command.extend(["--covariates", args.covariates])
    if args.patient_col:
        command.extend(["--patient-col", args.patient_col])
    return command


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Write build-and-validate commands for an ensemble hyperparameter grid. "
            "Inputs are the selected long PRS files produced by build_ensemble_inputs.py."
        )
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--config-override", default=None)
    parser.add_argument("--eligibility-config", default="configs/disease_eligibility.yaml")
    parser.add_argument("--disease", default=None)
    parser.add_argument("--labels", default=None)
    parser.add_argument("--ensemble-input-dir", default="outputs/ensemble_inputs")
    parser.add_argument("--out-root", default="outputs/ensemble_grid")
    parser.add_argument("--strategies", default="prs_only,covariate_adjusted")
    parser.add_argument("--penalties", default="l2")
    parser.add_argument("--c-grid", default="0.01,0.03,0.1,0.3,1,3,10")
    parser.add_argument("--l1-ratio-grid", default="0.25,0.5,0.75")
    parser.add_argument("--class-weights", default="none")
    parser.add_argument("--covariates", default=None)
    parser.add_argument("--patient-col", default=None)
    parser.add_argument("--out", default=None, help="Shell script path.")
    args = parser.parse_args()

    if bool(args.disease) == bool(args.labels):
        raise ValueError("Use exactly one of --disease or --labels")
    diseases = [args.disease] if args.disease else parse_csv_list(args.labels)
    penalties = _penalties(args.penalties)
    c_grid = _c_values(args.c_grid)
    l1_ratio_grid = _l1_ratios(args.l1_ratio_grid)
    class_weights = parse_csv_list(args.class_weights)
    if not class_weights:
        raise ValueError("--class-weights must contain at least one value")
    invalid_class_weights = [item for item in class_weights if item not in {"none", "balanced"}]
    if invalid_class_weights:
        raise ValueError(f"Unsupported class weights: {invalid_class_weights}")

    out_root = Path(args.out_root)
    script_path = Path(args.out) if args.out else out_root / "ensemble_grid.commands.sh"
    task_path = script_path.with_suffix(".array_tasks.txt")
    manifest_path = script_path.with_suffix(".tasks.csv")
    script_path.parent.mkdir(parents=True, exist_ok=True)

    script_lines = ["#!/usr/bin/env bash", "set -euo pipefail", ""]
    task_lines = []
    manifest_rows = []
    for disease in diseases:
        for penalty in penalties:
            c_values = c_grid if penalty != "none" else [1.0]
            l1_ratios = l1_ratio_grid if penalty == "elasticnet" else [None]
            for c_value in c_values:
                for l1_ratio in l1_ratios:
                    for class_weight in class_weights:
                        setting_id = _setting_id(
                            penalty=penalty,
                            c_value=c_value,
                            l1_ratio=l1_ratio,
                            class_weight=class_weight,
                            strategies=args.strategies,
                        )
                        setting_dir = out_root / disease / setting_id
                        build_command = _build_command(
                            args=args,
                            disease=disease,
                            setting_id=setting_id,
                            penalty=penalty,
                            c_value=c_value,
                            l1_ratio=l1_ratio,
                            class_weight=class_weight,
                        )
                        validate_script = setting_dir / "validate_ensemble_commands.sh"
                        command_text = f"{shlex.join(build_command)} && bash {shlex.quote(str(validate_script))}"
                        script_lines.append(command_text)
                        task_lines.append(command_text)
                        manifest_rows.append(
                            {
                                "disease": disease,
                                "setting_id": setting_id,
                                "penalty": penalty,
                                "C": c_value if penalty != "none" else "",
                                "l1_ratio": l1_ratio if penalty == "elasticnet" else "",
                                "class_weight": class_weight,
                                "strategies": args.strategies,
                                "out_dir": str(setting_dir),
                                "train_prs_long_scores": str(
                                    Path(args.ensemble_input_dir) / f"{disease}.train_long.csv"
                                ),
                                "validation_prs_long_scores": str(
                                    Path(args.ensemble_input_dir) / f"{disease}.validation_long.csv"
                                ),
                                "command": command_text,
                            }
                        )
    script_path.write_text("\n\n".join(script_lines) + "\n", encoding="utf-8")
    task_path.write_text("\n".join(task_lines) + "\n", encoding="utf-8")
    write_table(pd.DataFrame(manifest_rows), manifest_path)
    print(f"Wrote ensemble grid commands to {script_path}")
    print(f"Wrote array tasks to {task_path}")
    print(f"Wrote task manifest to {manifest_path}")


if __name__ == "__main__":
    main()
