#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from decipehr.config import load_yaml
from decipehr.metadata_score_ablations import (
    MetadataScoreAblation,
    build_metadata_score_ablations,
    metadata_ablation_task,
)


def _mapping(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Config key {key!r} must be a mapping")
    return value


def _list(mapping: dict[str, Any], key: str) -> list[str]:
    value = mapping.get(key)
    if not isinstance(value, list) or not value:
        raise ValueError(f"Config value {key!r} must be a non-empty list")
    return [str(item) for item in value]


def _resolve_task(
    *,
    task_id: int | None,
    requested_disease: str | None,
    requested_ablation: str | None,
    diseases: list[str],
    ablations: tuple[MetadataScoreAblation, ...],
) -> tuple[str, MetadataScoreAblation, int]:
    if task_id is not None:
        if requested_disease is not None or requested_ablation is not None:
            raise ValueError("Use --task-id or --disease/--ablation, not both")
        disease, ablation = metadata_ablation_task(task_id, diseases, ablations)
        return disease, ablation, int(task_id)
    if requested_disease is None or requested_ablation is None:
        raise ValueError("Provide --task-id or both --disease and --ablation")
    if requested_disease not in diseases:
        raise ValueError(f"Unknown disease: {requested_disease}")
    by_name = {ablation.name: ablation for ablation in ablations}
    if requested_ablation not in by_name:
        raise ValueError(
            f"Unknown ablation: {requested_ablation}; expected one of {list(by_name)}"
        )
    resolved_task_id = diseases.index(requested_disease) * len(ablations) + list(
        by_name
    ).index(requested_ablation)
    return requested_disease, by_name[requested_ablation], resolved_task_id


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run one leakage-safe TabPFN metadata-score ablation. TabPFN score "
            "covariates vary by arm; the final logistic covariates stay fixed."
        )
    )
    parser.add_argument(
        "--config", default="configs/tabpfn_metadata_score_ablations.yaml"
    )
    parser.add_argument("--task-id", type=int, default=None)
    parser.add_argument("--disease", default=None)
    parser.add_argument("--ablation", default=None)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    config = load_yaml(args.config)
    project = _mapping(config, "project")
    paths = _mapping(config, "paths")
    metadata = _mapping(config, "metadata")
    tabpfn = _mapping(config, "tabpfn")
    diseases = _list(project, "diseases")
    final_covariates = _list(metadata, "final")
    ablations = build_metadata_score_ablations(
        sex=str(metadata["sex"]),
        age=str(metadata["age"]),
        pcs=_list(metadata, "pcs"),
        full_covariates=final_covariates,
    )
    disease, ablation, task_id = _resolve_task(
        task_id=args.task_id,
        requested_disease=args.disease,
        requested_ablation=args.ablation,
        diseases=diseases,
        ablations=ablations,
    )

    model_path = Path(str(tabpfn["model_path"]))
    if not model_path.is_file():
        raise FileNotFoundError(f"TabPFN checkpoint does not exist: {model_path}")
    output_dir = Path(str(paths["output_root"])) / ablation.name / disease
    output_dir.mkdir(parents=True, exist_ok=True)
    classifier_kwargs = {
        "n_estimators": int(tabpfn["n_estimators"]),
        "model_path": str(model_path),
        "random_state": int(project["seed"]),
    }
    evaluator = Path(__file__).resolve().with_name("evaluate_tabpfn_ensemble_cv.py")
    command = [
        sys.executable,
        str(evaluator),
        "--config",
        str(paths["base_config"]),
        "--model-name",
        f"tabpfn3_selected_score_{ablation.name}",
        "--disease",
        disease,
        "--score-covariates",
        ",".join(ablation.score_covariates),
        "--final-covariates",
        ",".join(final_covariates),
        "--patient-col",
        "sampleid",
        "--phenotype",
        str(paths["phenotype"]),
        "--splits",
        str(paths["splits"]),
        "--eligibility-config",
        str(paths["eligibility_config"]),
        "--train-prs-long-scores",
        str(paths["prs_long_template"]),
        "--strategies",
        "covariate_adjusted",
        "--inner-folds",
        str(int(project["n_inner_folds"])),
        "--train-prediction-mode",
        "oof",
        "--calibration-methods",
        "prs_covariate",
        "--class-weight",
        str(tabpfn["class_weight"]),
        "--clip-eps",
        str(float(tabpfn["clip_eps"])),
        "--score-scale",
        str(tabpfn["score_scale"]),
        "--device",
        args.device,
        "--tabpfn-model-version",
        str(tabpfn["model_version"]),
        "--tabpfn-kwargs",
        json.dumps(classifier_kwargs, separators=(",", ":")),
        "--tabpfn-model-cache-dir",
        str(paths["model_cache"]),
        "--seed",
        str(int(project["seed"])),
        "--out-dir",
        str(output_dir),
    ]
    print(
        f"Running task {task_id}: disease={disease}, ablation={ablation.name}, "
        f"score_covariates={list(ablation.score_covariates)}, "
        f"final_covariates={final_covariates}",
        flush=True,
    )
    subprocess.run(command, check=True)
    run_record = {
        "config": args.config,
        "task_id": task_id,
        "disease": disease,
        "ablation": ablation.name,
        "description": ablation.description,
        "omitted_from_score": list(ablation.omitted_from_score),
        "score_covariates": list(ablation.score_covariates),
        "final_covariates": final_covariates,
        "model_definition": str(project["model_definition"]),
        "prs_long_scores": str(paths["prs_long_template"]),
        "prs_selection_manifest": str(paths["prs_selection_manifest"]),
        "outer_folds": int(project["n_outer_folds"]),
        "inner_folds": int(project["n_inner_folds"]),
        "tabpfn": classifier_kwargs,
        "command": command,
    }
    (output_dir / "ablation_run_record.json").write_text(
        json.dumps(run_record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
