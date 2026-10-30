"""Prepare or run one matched train+development outer fold on frozen PRS scores."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path

import pandas as pd
from evaluate_external_pgs_cv import _tabpfn_factory

from decipehr.eligibility import load_eligibility_rules
from decipehr.external_pgs import sha256_file, write_json
from decipehr.external_pgs_benchmark import prepare_disease_data
from decipehr.frozen_prs_comparison import ARMS, OUTCOMES, evaluate_frozen_fold

DEFAULT_RUN = "outputs/external_pgs/frozen_internal_external_trainval_20261013"
INTERNAL = Path("outputs/tabpfn3_final_trainval_selected_drop3_drop2_drop3_inputs")
HISTORICAL = Path("submissions/final_phase/Go Gene!_audited_pipeline_with_trainval_5cv")
SELECTION = Path("outputs/external_pgs/holdout_wrapper_20261012/SELECTION.json")
CHECKPOINT = Path("outputs/tabpfn-model-cache/tabpfn-v3-classifier-v3_default.ckpt")


def read_csv(path):
    return pd.read_csv(path, sep="\t" if Path(path).suffix == ".tsv" else ",",
                       dtype={"sampleid": str, "IID": str}, float_precision="round_trip",
                       low_memory=False)


def verify_inputs(protocol: dict) -> None:
    for path, digest in protocol["input_sha256"].items():
        if sha256_file(path) != digest:
            raise ValueError(f"Frozen comparison input changed: {path}")


def prepare(root: Path, eligibility_config: str) -> None:
    if root.exists():
        raise FileExistsError("Use a new run directory; preparation never overwrites a run")
    paths = {
        "phenotype": Path("data/processed/final/train_val_phenotype.csv"),
        "original_train": Path("data/processed/phenotype.csv"),
        "original_development": Path("data/processed/official_validation_phenotype.csv"),
        "splits": Path("data/processed/final/splits/train_val_5fold_1000g_hard_diagnostic_seed42.disease_long.csv"),
        "ancestry": Path("outputs/final_ancestry_1000g/combined/target_1000G_hard_labels_pc10_margin0.5.train_val.tsv"),
        "eligibility": Path(eligibility_config), "checkpoint": CHECKPOINT,
        "external_selection": SELECTION, "internal_manifest": INTERNAL / "final_selection_manifest.csv",
        "historical_predictions": HISTORICAL / "trainval_5cv_scores_long.csv",
        "historical_run": HISTORICAL / "run_record.json",
        "official_sumstats_manifest": Path("configs/official_sumstats.yaml"),
    }
    phenotype, splits, ancestry = [read_csv(paths[key]) for key in ["phenotype", "splits", "ancestry"]]
    original = pd.concat([read_csv(paths[key]) for key in ["original_train", "original_development"]])
    if original.sampleid.duplicated().any():
        raise ValueError("Train and development overlap")
    pd.testing.assert_frame_equal(
        original[phenotype.columns].sort_values("sampleid").reset_index(drop=True),
        phenotype.sort_values("sampleid").reset_index(drop=True), check_dtype=False,
    )
    selection = json.loads(SELECTION.read_text())
    rules = load_eligibility_rules(eligibility_config)
    prepared, models, audits, missing = {}, {}, [], []
    for disease in OUTCOMES:
        selected = selection["diseases"][disease]
        bank = Path(selected["bank"])
        for name, key in [("scores.long.csv", "scores_sha256"),
                          ("weights.scored.lock.json", "weight_lock_sha256")]:
            scoring = json.loads((bank / "scoring_run.json").read_text())
            if sha256_file(bank / name) != scoring[key]:
                raise ValueError(f"Changed audited external bank: {disease}")
        paths[f"{disease}_external_scores"] = bank / "scores.long.csv"
        paths[f"{disease}_external_weights"] = bank / "weights.scored.lock.json"
        paths[f"{disease}_external_scoring"] = bank / "scoring_run.json"
        paths[f"{disease}_internal_scores"] = INTERNAL / f"{disease}.train_long.csv"
        paths[f"{disease}_internal_selection"] = INTERNAL / f"{disease}.selected_final_models.csv"
        old_selection = read_csv(paths[f"{disease}_internal_selection"])
        missing.extend({"disease": disease, "family": row.family, "path": row.train_score_file}
                       for row in old_selection.itertuples() if not Path(row.train_score_file).exists())
        models[disease] = {
            "internal": sorted(old_selection.family.tolist()), "external": selected["model_ids"],
        }
        for arm in ARMS:
            data = prepare_disease_data(
                phenotype=phenotype, splits=splits, scores=read_csv(paths[f"{disease}_{arm}_scores"]),
                soft_weights=None, hard_labels=ancestry, disease=disease,
                model_ids=models[disease][arm], eligibility_rules=rules,
            )
            prepared[disease, arm] = data
            audits.extend({"disease": disease, "arm": arm, "fold": int(fold),
                           "n": len(part), "cases": int(part[disease].sum()),
                           "n_scores": len(models[disease][arm])}
                          for fold, part in data.groupby("fold"))
        common = ["sampleid", *OUTCOMES, "fold", "hard_label", "age_sd", "sex",
                  *[f"PC{i}_sd" for i in range(1, 11)]]
        pd.testing.assert_frame_equal(prepared[disease, "internal"][common],
                                      prepared[disease, "external"][common])
    # Hash the implementation as well as the data before any model fitting.
    code_paths = [Path(__file__), Path("scripts/evaluate_external_pgs_cv.py"),
                  Path("scripts/report_frozen_prs_comparison.py"),
                  Path("slurm/frozen_prs_comparison.sbatch")]
    code_paths += [Path("src/decipehr") / f"{name}.py" for name in [
        "frozen_prs_comparison", "external_pgs_holdout", "external_pgs_benchmark",
        "tabpfn_ensemble", "models", "eligibility", "ancestry_performance",
    ]]
    paths.update({str(path): path for path in code_paths})
    hashes = {str(path): sha256_file(path) for path in paths.values()}
    root.mkdir(parents=True, exist_ok=False)
    for (disease, arm), data in prepared.items():
        target = root / "inputs" / disease / f"{arm}.parquet"
        target.parent.mkdir(parents=True, exist_ok=True)
        data.to_parquet(target, index=False)
        hashes[str(target)] = sha256_file(target)
    pd.DataFrame(audits).to_csv(root / "cohort_counts.csv", index=False)
    write_json(root / "PROTOCOL.json", {
        "status": "prepared", "cohort": "original train plus original development",
        "original_union_verified": True, "outer_folds": 5, "shared_patient_folds": True,
        "seed": 42, "wrapper_inner_folds": 3, "wrapper_clip_eps": 0.001,
        "model": "TabPFN v3 default, 16 estimators; PRS + age + sex + continuous PC1-PC10",
        "checkpoint": str(CHECKPOINT), "models": models, "input_sha256": hashes,
        "paths": {key: str(value) for key, value in paths.items()},
        "primary_endpoint": "pooled OOF wrapper AUC; raw AUC is supplementary",
        "bootstrap": {"resamples": 2000, "seed": 42, "stratified_by_outcome": True,
                      "paired": True, "family_size": 6, "contrasts": "three diseases x two output scales"},
        "user_scope": "Use existing frozen PRS scores; do not pursue missing historical candidate files",
        "end_to_end_nested_prs_cv": False, "holdout_used": False,
        "missing_historical_candidate_files": missing,
        "limitations": [
            "Both score panels were previously selected using development evidence; selection is not nested anew.",
            "Historical PRS construction, genotype imputation and candidate selection are not replayed.",
            "Only downstream preprocessing, TabPFN context and wrapper are fitted within each outer training fold.",
            "This is retrospective development evidence conditional on frozen panels, not an independent test.",
            "Patient bootstrap conditions on saved predictions; it omits refitting, selection and family dependence.",
        ],
    })
    print(pd.DataFrame(audits).groupby(["disease", "arm"])[["n", "cases"]].sum().to_string(), flush=True)


def main() -> None:
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default=DEFAULT_RUN)
    parser.add_argument("--eligibility-config", default="configs/disease_eligibility.yaml")
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--disease", choices=OUTCOMES)
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--fold", type=int, choices=range(5))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    root = Path(args.run_dir)
    if args.prepare:
        prepare(root, args.eligibility_config)
        return
    if args.disease is None or args.arm is None or args.fold is None:
        parser.error("Running requires --disease, --arm and --fold")
    protocol = json.loads((root / "PROTOCOL.json").read_text())
    verify_inputs(protocol)
    if str(Path(args.eligibility_config)) != protocol["paths"]["eligibility"]:
        raise ValueError("Eligibility configuration differs from preparation")
    data = pd.read_parquet(root / "inputs" / args.disease / f"{args.arm}.parquet")
    out = root / "folds" / args.disease / args.arm / str(args.fold)
    out.mkdir(parents=True, exist_ok=False)
    print(f"Starting {args.disease}/{args.arm}/fold{args.fold}: {len(data)} eligible patients", flush=True)
    frames = evaluate_frozen_fold(
        data, disease=args.disease, arm=args.arm, fold=args.fold,
        model_ids=protocol["models"][args.disease][args.arm], seed=protocol["seed"],
        tabpfn_factory=_tabpfn_factory(args.device, protocol["checkpoint"]),
    )
    files = ["predictions.csv", "wrapper_training_oof.csv", "fit_audit.csv", "preprocessing.csv"]
    for name, frame in zip(files, frames, strict=True):
        frame.to_csv(out / name, index=False)
    verify_inputs(protocol)
    write_json(out / "COMPLETE.json", {
        "status": "complete", "arguments": vars(args), "protocol_sha256": sha256_file(root / "PROTOCOL.json"),
        "output_sha256": {name: sha256_file(out / name) for name in files},
        "versions": {name: importlib.metadata.version(name) for name in ["numpy", "pandas", "scikit-learn", "torch", "tabpfn"]},
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "hostname": os.uname().nodename,
    })
    print(f"Completed {args.disease}/{args.arm}/fold{args.fold}", flush=True)


if __name__ == "__main__":
    main()
