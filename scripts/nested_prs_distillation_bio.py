"""Strict train-only outer-fold sensitivity with fold-local teacher soft targets."""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from run_prs_distillation_bio import run, verify

from decipehr.external_pgs import sha256_file, write_json
from decipehr.prs_distillation.config import load_config
from decipehr.prs_distillation.data import make_split_manifest, verify_files
from decipehr.prs_distillation.experiment import load_inputs
from decipehr.prs_distillation.preprocessing import assert_boundary, id_hash
from decipehr.prs_distillation.teacher import make_teacher, teacher_oof, teacher_predict


def read(path):
    return json.loads(Path(path).read_text())


def prepare(parent, fold):
    protocol, cfg = verify(parent)
    source = Path(cfg["source_run"])
    root = parent / "nested" / f"outer{fold}"
    if (root / "PREPARED.json").exists():
        verify_files(root, read(root / "PREPARED.json")["output_sha256"])
        return root
    root.mkdir(parents=True, exist_ok=True)
    local_source = root / "source"
    local_source.mkdir(exist_ok=True)
    audit = read(source / "audit_report.json")
    phenotype_path = Path(audit["paths"]["training"])
    assert sha256_file(phenotype_path) == audit["source_sha256"][str(phenotype_path)]
    phenotypes = pd.read_csv(phenotype_path, dtype={"sampleid": str}, usecols=["sampleid", "T2D", "cad", "bc"])
    d0, _, _ = load_inputs(source, "T2D")
    table = d0[["sampleid", "kinship_group", "hard_label"]].merge(phenotypes, on="sampleid", validate="one_to_one")
    outer = pd.read_parquet(source / "split_manifest.parquet").set_index("sampleid").loc[table.sampleid]
    held = outer.final_teacher_fold.to_numpy() == fold
    assert_boundary(table.loc[~held], table.loc[held])
    manifest, split_receipt = make_split_manifest(table.loc[~held], table.loc[held], 20261101+fold)
    manifest.to_parquet(local_source / "split_manifest.parquet", index=False)
    write_json(local_source / "split_receipt.json", {**split_receipt, "outer_fold": fold,
        "sha256": sha256_file(local_source / "split_manifest.parquet")})
    write_json(local_source / "audit_report.json", {"cohort_policy": "nested_train_only_group_disjoint",
        "paths": {"eligibility": protocol["eligibility"]}, "outer_fold": fold,
        "original_official_validation_accessed": False})
    assignments = manifest.set_index("sampleid")
    teacher_cfg = load_config(source / "resolved_config.yaml")
    inputs, scopes = {}, []
    for trait in cfg["traits"]:
        d, _, features = load_inputs(source, trait)
        roles = assignments.loc[d.sampleid]
        train = d.loc[roles.split.eq("official_train").to_numpy()]
        evaluation = d.loc[roles.split.eq("official_validation").to_numpy()].drop(columns="y")
        assert_boundary(train, evaluation)
        destination = local_source / trait
        destination.mkdir(exist_ok=True)
        train.to_parquet(destination / "training.parquet", index=False)
        evaluation.to_parquet(destination / "evaluation_features.parquet", index=False)
        (destination / "feature_manifest.tsv").write_bytes((source / trait / "feature_manifest.tsv").read_bytes())
        write_json(destination / "input_receipt.json", {p.name: sha256_file(p) for p in destination.iterdir()
                                                       if p.suffix in [".parquet", ".tsv"]})
        subset = assignments.loc[train.sampleid]
        u = train.loc[subset.internal_role.eq("U").to_numpy()]
        h = train.loc[subset.internal_role.eq("H").to_numpy()]
        contract = read(source / "core" / trait / "teacher_contract.json")
        verify_files(Path.cwd(), contract["implementation_sha256"])
        assert sha256_file(contract["checkpoint"]) == contract["checkpoint_sha256"]
        common = dict(contract=contract, trait=trait, manifest_hash=sha256_file(destination / "feature_manifest.tsv"),
                      factory=lambda: make_teacher(teacher_cfg))
        cache = root / "teacher_cache" / trait
        q_u = teacher_oof(u, subset.loc[u.sampleid, "u_teacher_fold"].to_numpy(), features.column.tolist(),
                          directory=cache / "U_oof", **common)
        q_d = teacher_oof(train, subset.final_teacher_fold.to_numpy(), features.column.tolist(),
                          directory=cache / "R_oof", **common)
        q_h = teacher_predict(u, h.drop(columns="y"), features.column.tolist(), directory=cache / "U_to_H",
                              fold_id=f"outer{fold}_U_to_H", **common)
        # The original D-OOF fold has exactly this R -> held-out-fold scope.
        def cache_only():
            raise RuntimeError("Original outer-fold teacher cache mismatch")
        q_e = teacher_predict(train, evaluation, features.column.tolist(),
            directory=source / "core" / trait / "teacher_cache/D_oof" / str(fold),
            fold_id=fold, **{**common, "factory": cache_only})
        target_dir = root / "inputs" / trait
        target_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(target_dir / "targets.npz", q_U=q_u, q_D=q_d, q_H=q_h, q_E=q_e)
        inputs[str((target_dir / "targets.npz").relative_to(root))] = sha256_file(target_dir / "targets.npz")
        scopes.append({"trait": trait, "outer_fold": fold, "R_n": len(train), "U_n": len(u), "H_n": len(h),
                       "outer_query_n": len(evaluation), "R_ids_hash": id_hash(train), "query_ids_hash": id_hash(evaluation),
                       "teacher_query_label_exclusion": True, "group_disjoint": True})
        print(f"Prepared outer {fold}/{trait}: independent U/R teacher targets", flush=True)
    nested = copy.deepcopy(protocol)
    nested["identity"]["config"]["source_run"] = str(local_source)
    nested["identity"]["code_sha256"][str(Path(__file__).resolve().relative_to(Path.cwd()))] = sha256_file(__file__)
    nested["input_sha256"] = inputs
    nested["cohort_policy"] = "nested_train_only_group_disjoint"
    nested["known_cross_groups_retained"] = 0
    nested["scope"] = "All fits, preprocessing, teacher contexts and H selection exclude the outer fold"
    nested["new_teacher_fits"] = True
    nested["outer_fold"] = fold
    nested["gate_policy"] = "Predeclared common rho=0.001; avoids inheriting the old full-train H-tuned gate ridge"
    for trait in cfg["traits"]:
        nested["priors"][trait]["common_gate_ridge"] = .001
    write_json(root / "PROTOCOL.json", nested)
    pd.DataFrame(scopes).to_csv(root / "scope_audit.csv", index=False)
    write_json(root / "PREPARED.json", {"outer_fold": fold, "status": "fold_local_teacher_targets_complete",
        "output_sha256": {str(p.relative_to(root)): sha256_file(p) for p in root.rglob("*") if p.is_file()}})
    return root


def main():
    os.umask(0o077)
    torch.set_num_threads(1)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["prepare", "train"])
    p.add_argument("--parent-run", type=Path, required=True)
    p.add_argument("--fold", type=int, choices=range(5), required=True)
    p.add_argument("--trait", choices=["T2D", "cad", "bc"])
    a = p.parse_args()
    if a.command == "prepare":
        print(prepare(a.parent_run, a.fold))
    else:
        if a.trait is None:
            p.error("Trait required for CPU student training")
        root = a.parent_run / "nested" / f"outer{a.fold}"
        verify_files(root, read(root / "PREPARED.json")["output_sha256"])
        smoke = a.parent_run / "smoke" / a.trait
        verify_files(smoke, read(smoke / "PREDICTIONS_COMPLETE.json")["output_sha256"])
        run(root, a.trait)


if __name__ == "__main__":
    main()
