"""Resolve existing frozen sources; never load evaluation outcomes during preparation."""

from __future__ import annotations

import importlib.metadata
import json
import platform
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import StratifiedGroupKFold

from decipehr.eligibility import filter_eligible, load_eligibility_rules
from decipehr.external_pgs import load_weight_lock, sha256_file, write_json
from decipehr.prs_distillation.config import Config, digest
from decipehr.prs_distillation.preprocessing import AGE, PCS, assert_boundary, frame_hash, id_hash

TRAITS = ["T2D", "cad", "bc"]


def read_csv(path, **kwargs):
    return pd.read_csv(path, sep="\t" if Path(path).suffix == ".tsv" else ",",
                       dtype={"sampleid": str, "IID": str, "kinship_group": str},
                       float_precision="round_trip", **kwargs)


def verify_files(root, hashes):
    for name, expected in hashes.items():
        if sha256_file(Path(root) / name) != expected:
            raise ValueError(f"Frozen input/artifact changed: {name}")


def code_hashes():
    files = [*Path("src/decipehr/prs_distillation").glob("*.py"),
             Path("scripts/run_prs_distillation.py"), Path("configs/disease_eligibility.yaml"),
             Path("src/decipehr/tabpfn_ensemble.py")]
    return {str(p): sha256_file(p) for p in sorted(files) if p.exists()}


def environment():
    git = "/gpfs/share/apps/git/2.49.0/bin/git"
    commit = subprocess.check_output([git, "rev-parse", "HEAD"], text=True).strip()
    status = subprocess.check_output([git, "status", "--short"], text=True).strip()
    return {"git_commit": commit, "git_status": status, "hostname": platform.node(),
            "python": platform.python_version(), "code_sha256": code_hashes(),
            "packages": {p: importlib.metadata.version(p) for p in
                         ["numpy", "pandas", "scipy", "scikit-learn", "torch", "tabpfn"]}}


def grouped_folds(frame, seed, labels):
    """One deterministic assignment, with an outcome-only fallback for rare joint strata."""
    for column in ["sampleid", "kinship_group", *labels]:
        if frame[column].isna().any():
            raise ValueError(f"Missing split field: {column}")
    if not frame.sampleid.is_unique or frame.kinship_group.nunique() < 5:
        raise ValueError("Five grouped folds require unique patients and at least five groups")
    strata = frame[labels].astype(str).agg("|".join, axis=1)
    joint = strata + "|" + frame.hard_label.astype(str)
    group_counts = pd.DataFrame({"s": joint, "g": frame.kinship_group}).groupby("s").g.nunique()
    use_ancestry = bool(group_counts.min() >= 5)
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    result = np.full(len(frame), -1, dtype=int)
    for fold, (fit, query) in enumerate(splitter.split(frame, joint if use_ancestry else strata,
                                                      frame.kinship_group)):
        assert_boundary(frame.iloc[fit], frame.iloc[query])
        for label in labels:
            if frame.iloc[fit][label].nunique() != 2 or frame.iloc[query][label].nunique() != 2:
                raise ValueError(f"Insufficient groups/classes in fold {fold} for {label}")
        result[query] = fold
    return result, "outcome_and_ancestry" if use_ancestry else "outcome_only_rare_joint_strata"


def make_split_manifest(training, evaluation, seed, *, allow_evaluation_kinship_overlap=False):
    labels = [c for c in TRAITS if c in training] or ["y"]
    final_fold, final_rule = grouped_folds(training, seed, labels)
    internal_role = np.where(final_fold == 0, "H", "U")
    u = training.loc[internal_role == "U"].copy()
    inner_fold, inner_rule = grouped_folds(u, seed + 1, labels)
    manifest = training[["sampleid", "kinship_group"]].copy()
    manifest["split"] = "official_train"
    manifest["internal_role"] = internal_role
    manifest["final_teacher_fold"] = final_fold
    manifest["u_teacher_fold"] = -1
    manifest.loc[internal_role == "U", "u_teacher_fold"] = inner_fold
    query = evaluation[["sampleid", "kinship_group"]].copy().assign(
        split="official_validation", internal_role="E", final_teacher_fold=-1, u_teacher_fold=-1)
    assert_boundary(training, evaluation, require_group_disjoint=not allow_evaluation_kinship_overlap)
    return pd.concat([manifest, query], ignore_index=True), {
        "final_stratification": final_rule, "u_stratification": inner_rule,
        "seed": seed, "teacher_oof_folds": 5, "tuning_fold": 0,
        "all_traits_share_patient_assignments": True,
        "evaluation_kinship_overlap_allowed": allow_evaluation_kinship_overlap,
    }


def attach_full_sensitivity_groups(frame, groups):
    """Keep every patient; missing QC-valid kinship is explicitly unresolved, not unrelated."""
    merged = frame.merge(groups, on="sampleid", how="left", validate="one_to_one")
    missing = merged.kinship_group.isna()
    merged.loc[missing, "kinship_group"] = (
        "UNRESOLVED_QC::" + merged.loc[missing, "sampleid"].astype(str)
    )
    return merged


def resolve_inputs(cfg: Config, eligibility_config):
    """Audit known repository sources; missing provenance is a blocker, never an inference."""
    env = environment()
    issues = []
    if env["packages"]["tabpfn"] != cfg.teacher_package:
        issues.append("blocked_teacher_package_version")
    if not cfg.allow_standardized_age:
        issues.append("blocked_age_years_mapping_unavailable: only age_sd is supplied")
    paths = {"training": cfg.train_path, "evaluation": cfg.evaluation_path,
             "panels": cfg.panel_selection, "ancestry": cfg.ancestry_path,
             "eligibility": eligibility_config, "checkpoint": cfg.checkpoint,
             "qc_manifest": str(Path(cfg.qc_prepared) / "manifest.json")}
    # Selection contains prior development evidence and fixed panels, no blind patient labels.
    selection = json.loads(Path(cfg.panel_selection).read_text())
    training = read_csv(cfg.train_path, usecols=["sampleid", *TRAITS, "sex", AGE, *PCS])
    evaluation = read_csv(cfg.evaluation_path, usecols=["sampleid", "sex", AGE, *PCS])
    for frame in (training, evaluation):
        if frame.sampleid.isna().any() or not frame.sampleid.is_unique:
            raise ValueError("Missing or duplicate official patient ID")
    if set(training.sampleid) & set(evaluation.sampleid):
        raise ValueError("Official patient sets overlap")
    if not training[TRAITS].isin([0, 1]).all().all():
        raise ValueError("Training labels are not complete binary outcomes")
    ancestry = read_csv(cfg.ancestry_path)
    if "sampleid" not in ancestry:
        ancestry = ancestry.rename(columns={"IID": "sampleid"})
    qc_root = Path(cfg.qc_prepared)
    qc_manifest = json.loads((qc_root / "manifest.json").read_text())
    for name in ("sample_qc.csv", "kinship_groups.csv"):
        verify_files(qc_root, {name: qc_manifest["output_sha256"][name]})
        paths[name] = str(qc_root / name)
    # Verify the linkage back to the same official patients and frozen kinship result.
    for key in ("train", "development", "kinship"):
        if sha256_file(qc_manifest["source_paths"][key]) != qc_manifest["source_sha256"][key]:
            raise ValueError(f"Existing QC input receipt changed: {key}")
    if sha256_file(cfg.train_path) != qc_manifest["source_sha256"]["train"] or \
            sha256_file(cfg.evaluation_path) != qc_manifest["source_sha256"]["development"]:
        raise ValueError("QC receipt does not describe the current official cohort")
    qc = read_csv(qc_root / "sample_qc.csv")
    groups = read_csv(qc_root / "kinship_groups.csv")
    original_counts = {"training": len(training), "evaluation": len(evaluation)}
    retained = set(qc.loc[~qc.exclude_high_heterozygosity, "sampleid"])
    all_ids = set(training.sampleid) | set(evaluation.sampleid)
    if set(qc.sampleid) != all_ids or set(groups.sampleid) != retained:
        raise ValueError("QC/group coverage mismatch")
    train_groups = set(groups.loc[groups.sampleid.isin(training.sampleid), "kinship_group"])
    eval_groups = set(groups.loc[groups.sampleid.isin(evaluation.sampleid), "kinship_group"])
    cross_groups = train_groups & eval_groups
    if cfg.cohort_policy == "audited_qc":
        training = training.loc[training.sampleid.isin(retained)].merge(groups, validate="one_to_one")
        evaluation = evaluation.loc[evaluation.sampleid.isin(retained)].merge(groups, validate="one_to_one")
        training = training.loc[~training.kinship_group.isin(eval_groups)].copy()
        assert_boundary(training, evaluation)
    elif cfg.cohort_policy == "original_full_sensitivity":
        training = attach_full_sensitivity_groups(training, groups)
        evaluation = attach_full_sensitivity_groups(evaluation, groups)
        assert_boundary(training, evaluation, require_group_disjoint=False)
    else:
        issues.extend([
            f"blocked_incomplete_validated_kinship: {len(all_ids - retained)} QC-excluded patients",
            f"blocked_official_group_overlap: {len(cross_groups)} validated groups cross splits",
        ])
    rules = load_eligibility_rules(eligibility_config)
    if "bc" not in rules:
        raise ValueError("BC requires --eligibility-config containing its female eligibility rule")
    prepared, manifests, counts = {}, {}, []
    for trait in cfg.traits:
        item = selection["diseases"][trait]
        bank, columns = Path(item["bank"]), item["model_ids"]
        receipt = json.loads((bank / "scoring_run.json").read_text())
        for filename, key in [("scores.long.csv", "scores_sha256"),
                              ("weights.scored.lock.json", "weight_lock_sha256")]:
            if sha256_file(bank / filename) != receipt[key]:
                raise ValueError(f"Score provenance changed: {trait}/{filename}")
        for name in ["scores.long.csv", "weights.scored.lock.json", "scoring_run.json"]:
            paths[f"{trait}/{name}"] = str(bank / name)
        lock = load_weight_lock(bank / "weights.scored.lock.json", verify_files=False)
        resources = {r["id"]: r for r in lock["resources"]}
        if len(set(columns)) != len(columns) or any(c not in resources for c in columns):
            raise ValueError("Panel selection differs from audited weight manifest")
        manifest = []
        for name in columns:
            r = resources[name]
            meta, header = r.get("metadata", {}), r["header"]
            # PGS000296 reports its weight type as NR. Its published positive disease-risk
            # direction is explicit (Wang et al., PMID 32762905); do not relabel NR as beta.
            known_nr = name == "PGS000296" and header["weight_type"] == "NR"
            if r["disease"] != trait or (header["weight_type"] != "beta" and not known_nr):
                raise ValueError("Unsupported trait/weight direction provenance")
            manifest.append({
                "trait": trait, "column": name, "display_name": header["pgs_name"],
                "source_gwas": json.dumps(meta.get("samples_variants", r.get("publication_url"))),
                "source_ancestry": json.dumps(meta.get("ancestry_distribution", {}).get("gwas",
                                                                                         r.get("component"))),
                "construction_method": meta.get("method_name", r["family"]),
                "risk_direction": ("published positive disease-risk direction; no target-label flipping"),
                "published_weight_type": header["weight_type"],
                "direction_reference": ("https://pubmed.ncbi.nlm.nih.gov/32762905/" if known_nr
                                        else r.get("catalog_url", r.get("publication_url", ""))),
                "correction_status": "context-standardized raw published score; no PC residualization",
                "correction_fit_scope": "each allowed context separately, mean/SD ddof=0",
                "correction_artifact": "saved preprocessing.joblib and scope metadata",
                "scale_description": "existing TabPFNFeatureTransformer numeric scaling",
                "weight_sha256": r["sha256"], "scoring_receipt": str(bank / "scoring_run.json"),
                "genome_build": lock["build"],
            })
        manifests[trait] = pd.DataFrame(manifest)
        scores = read_csv(bank / "scores.long.csv")
        scores = scores.loc[(scores.disease == trait) & scores.model.isin(columns)]
        if scores.duplicated(["sampleid", "model"]).any():
            raise ValueError("Duplicate patient-score rows")
        wide = scores.pivot(index="sampleid", columns="model", values="prs").reset_index()
        pair = []
        for role, source in [("D", training), ("E", evaluation)]:
            frame = filter_eligible(source, trait, rules)
            frame = frame.merge(wide[["sampleid", *columns]], how="left", validate="one_to_one")
            frame = frame.merge(ancestry[["sampleid", "hard_label"]], how="left", validate="one_to_one")
            if frame[[*columns, AGE, *PCS, "sex", "hard_label"]].isna().any().any():
                raise ValueError("Missing score/covariate/group; no model-specific row deletion allowed")
            if not np.isfinite(frame[[*columns, AGE, *PCS]].to_numpy(float)).all():
                raise ValueError("Nonfinite source feature")
            if role == "D":
                frame["y"] = frame[trait].astype(int)
            order = ["sampleid"] + (["kinship_group"] if "kinship_group" in frame else [])
            order += ["hard_label", *columns, AGE, *PCS, "sex"] + (["y"] if role == "D" else [])
            frame = frame[order].sort_values("sampleid").reset_index(drop=True)
            pair.append(frame)
            counts.append({"trait": trait, "scope": role, "n": len(frame), "n_prs": len(columns),
                           "events": int(frame.y.sum()) if role == "D" else None})
        prepared[trait] = pair
    if not issues:
        master = training.merge(ancestry[["sampleid", "hard_label"]], validate="one_to_one")
        splits, split_info = make_split_manifest(
            master, evaluation, cfg.split_seed,
            allow_evaluation_kinship_overlap=cfg.cohort_policy == "original_full_sensitivity",
        )
    else:
        splits, split_info = None, None
    report = {
        "status": "blocked" if issues else "prepared", "blockers": issues,
        "original_counts": original_counts, "cohort_policy": cfg.cohort_policy,
        "qc_excluded": 0 if cfg.cohort_policy == "original_full_sensitivity" else len(all_ids - retained),
        "source_qc_flagged": len(all_ids - retained), "original_cross_groups": len(cross_groups),
        "known_cross_groups_retained": len(cross_groups)
        if cfg.cohort_policy == "original_full_sensitivity" else 0,
        "full_kinship_isolation": cfg.cohort_policy == "audited_qc",
        "unresolved_kinship_retained": {
            role: int((~frame.sampleid.isin(retained)).sum())
            for role, frame in [("training", training), ("evaluation", evaluation)]
        },
        "kinship_policy": (
            "Full original patients retained. Known QC-valid families stay together in training "
            "folds; QC-excluded patients have UNRESOLVED_QC patient placeholders, not evidence "
            "of unrelatedness. Train/validation family overlap is retained. Group-bootstrap "
            "intervals cannot account for unresolved relatedness."
            if cfg.cohort_policy == "original_full_sensitivity" else
            "QC-valid kinship groups isolated across all context/query scopes."
        ),
        "counts": counts, "split_info": split_info, "paths": paths,
        "source_sha256": {p: sha256_file(p) for p in paths.values()},
        "feature_hashes": {t: [frame_hash(x) for x in pair] for t, pair in prepared.items()},
        "age_units": "age_sd; unknown original mean/SD; not years",
        "binary_outcome": "existing competition disease indicator; not a survival endpoint",
        "index_time": "not available in processed competition phenotype; no new onset definition",
        "prs_policy": "existing raw-score mean/SD scaling; no added ancestry residualization",
        "genotype_scope": "frozen train+development variant/QC construction; not fully inductive",
        "pc_scope": "provided clinical PC1_sd-PC10_sd; original fit population undocumented",
        "evaluation_status": "previously reused development cohort; post-hoc direct diagnostics",
        "phase_b": "not implemented or executed", "blind_data_opened": False,
    }
    return env, report, manifests, prepared, splits


def prepare_run(cfg, eligibility_config):
    env, report, manifests, prepared, splits = resolve_inputs(cfg, eligibility_config)
    signature = digest({"config": cfg.to_dict(), "code": env["code_sha256"],
                        "features": report["feature_hashes"]})[:12]
    root = Path(cfg.output_root) / f"{cfg.run_name}_{env['git_commit'][:8]}_{signature}"
    if root.exists():
        saved = json.loads((root / "audit_report.json").read_text())
        if saved != report:
            raise ValueError("Existing run audit differs; use a new run_name")
        return root, report
    root.mkdir(parents=True, exist_ok=False)
    write_json(root / "environment.json", env)
    write_json(root / "audit_report.json", report)
    (root / "resolved_config.yaml").write_text(yaml.safe_dump(cfg.to_dict(), sort_keys=False))
    text = ["# Fixed PRS distillation discovery", "", f"Status: {report['status']}",
            f"Cohort policy: {cfg.cohort_policy}; age: {report['age_units']}.",
            "", "The current frozen external panels are inherited from the existing selection receipt.",
            "Selection previously used development outcomes; this is post-hoc evaluation.",
            "Raw published scores use existing context-only standard scaling, with no PC residualization.",
            "PRS provenance includes GRCh38 allele-matching and frozen weights in the scoring receipts.",
            "Clinical PC fitting population and original age conversion are undocumented.",
            "No evaluation labels have been read; blind data are excluded from this pipeline.",
            "Phase B / historical Stage-2 compatibility is not implemented.", "",
            "Kinship groups and outcome-blind heterozygosity QC are inherited from the audited NYU run.",
            "The audited_qc option removes QC failures and training relatives of development patients.",
            "It retains the existing female BC eligibility rule and does not move anyone across splits.",
            "", "Counts (evaluation event counts deliberately unavailable before reporting):", ""]
    if cfg.cohort_policy == "original_full_sensitivity":
        text = [line for line in text if line !=
                "The audited_qc option removes QC failures and training relatives of development patients."]
        text += ["FULL ORIGINAL COHORT SENSITIVITY: the previous QC exclusions are not applied.",
                 report["kinship_policy"], ""]
    text += [json.dumps(row) for row in report["counts"]]
    text += ["", "Blockers:", *report["blockers"], "", "Verified source paths:",
             *[f"- {key}: `{value}`" for key, value in report["paths"].items()]]
    (root / "discovery_report.md").write_text("\n".join(text) + "\n")
    for trait, manifest in manifests.items():
        out = root / trait
        out.mkdir()
        manifest.to_csv(out / "feature_manifest.tsv", sep="\t", index=False)
        if report["status"] == "prepared":
            d, e = prepared[trait]
            d.to_parquet(out / "training.parquet", index=False)
            e.to_parquet(out / "evaluation_features.parquet", index=False)
            write_json(out / "input_receipt.json", {
                "training.parquet": sha256_file(out / "training.parquet"),
                "evaluation_features.parquet": sha256_file(out / "evaluation_features.parquet"),
                "feature_manifest.tsv": sha256_file(out / "feature_manifest.tsv"),
                "training_ids_hash": id_hash(d), "evaluation_ids_hash": id_hash(e),
            })
    if splits is not None:
        splits.to_parquet(root / "split_manifest.parquet", index=False)
        write_json(root / "split_receipt.json", {"sha256": sha256_file(root / "split_manifest.parquet")})
    return root, report
