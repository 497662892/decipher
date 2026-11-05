"""Export the verified reference-PC models and their exact genetic input resources."""

from __future__ import annotations

import csv
import importlib.metadata
import json
import os
import shutil
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml

from decipehr.external_pgs import sha256_file, write_json
from decipehr.prs_distillation.data import verify_files
from decipehr.prs_distillation.preprocessing import frame_hash

TRAITS = ("T2D", "cad", "bc")
SCOPES = ("train_only", "trainval_refit")
AGE_CONTRACT = "decipehr_provided_age_sd_v1"
AUTOSOMES = {str(i) for i in range(1, 23)}


def read_json(path):
    return json.loads(Path(path).read_text())


def contained(root, name):
    """Bundle paths are relative and may not escape the portable directory."""
    root, name = Path(root).resolve(), Path(name)
    path = (root / name).resolve()
    if name.is_absolute() or not path.is_relative_to(root):
        raise ValueError(f"Nonportable bundle path: {name}")
    return path


def load_bundle(path, *, verify=True):
    root = Path(path).resolve()
    manifest = read_json(root / "BUNDLE.json")
    if manifest.get("schema_version") != 1 or manifest.get("status") != "complete":
        raise ValueError("Incomplete or unsupported external model bundle")
    if manifest["genome_build"] != "GRCh38" or manifest["pc_mode"] != "reference_pc" or manifest["oadp"]:
        raise ValueError("Expected the frozen GRCh38 standard reference-PC models")
    required = [manifest[name] for name in ["reference_frequencies", "required_variants", "variant_map"]]
    required += [manifest["pc"][name] for name in ["weights", "frequencies", "variants"]]
    required += [part[name] for part in manifest["score_parts"] for name in ["weights", "variants"]]
    if "traits" in manifest:
        required += [manifest["eligibility"], manifest["teacher"]["checkpoint"]]
        for scopes in manifest["traits"].values():
            for packet in scopes.values():
                required += [packet[name] for name in ["context", "teacher_transformer", "teacher_preprocessing", "feature_reference"]]
                for directory in packet["students"].values():
                    files = [str(p.relative_to(root)) for p in contained(root, directory).rglob("*") if p.is_file()]
                    if not files:
                        raise ValueError("Student bundle has no saved files")
                    required += files
    if set(required) - manifest["file_sha256"].keys():
        raise ValueError("Bundle resources lack recorded checksums")
    for name, digest in manifest["file_sha256"].items():
        p = contained(root, name)
        if verify and sha256_file(p) != digest:
            raise ValueError(f"Bundle file changed: {name}")
    return manifest


def export_bundle(sources, out_dir, *, scopes=SCOPES):
    """Copy private assets locally; no downloads, retraining, or external writes."""
    os.umask(0o077)
    source_file = Path(sources).resolve()
    config = yaml.safe_load(source_file.read_text())
    if config.get("schema_version") != 1 or not scopes or set(scopes) - set(SCOPES):
        raise ValueError("Invalid source schema or model scope")
    repo = (source_file.parent / config.get("repository_root", "..")).resolve()
    paths = {key: (source_file.parent / config[key]).resolve() for key in
             ["agreement_run", "train_baseline", "scoring_receipt", "pc_selection", "eligibility"]}
    run, baseline = paths["agreement_run"], paths["train_baseline"]
    prepared = read_json(run / "PREPARED.json")
    verify_files(repo, prepared["input_sha256"])
    if read_json(run / "FINAL_REVIEW.json")["status"] != "completed_and_verified":
        raise ValueError("Export requires the reviewed, completed two-scope experiment")
    selection = read_json(paths["pc_selection"])
    if selection["selected_mode"] != "reference_pc" or selection["oadp_enabled"]:
        raise ValueError("The selected PC method changed")
    out = Path(out_dir).resolve()
    if out.exists():
        raise FileExistsError("Export to a new bundle directory; existing bundles are immutable")
    out.mkdir(parents=True)
    origins = {}

    def copy(source, name, expected=None):
        source = Path(source)
        digest = sha256_file(source)
        if expected is not None and digest != expected:
            raise ValueError(f"Source checksum mismatch: {source}")
        target = contained(out, name)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        if sha256_file(target) != digest:
            raise ValueError(f"Copy verification failed: {name}")
        origins[name] = {"source": str(source), "sha256": digest}
        return name

    def original(name):
        return (repo / name).resolve()

    available = pd.read_csv(run / "availability.csv").set_index("trait")
    models, expected_prs, shared_teacher = {}, {}, None
    for trait in TRAITS:
        info = read_json(run / "private" / trait / "INPUTS.json")
        expected_prs[trait] = info["prs"]
        if len(info["prs"]) != int(available.loc[trait, "n_prs"]):
            raise ValueError("PRS feature count changed")
        models[trait] = {}
        for scope in scopes:
            source = run / "private" / trait / scope
            receipt = read_json(source / "COMPLETE.json")
            verify_files(source, receipt["output_sha256"])
            context = pd.read_parquet(run / "private" / trait /
                                      ("train.parquet" if scope == "train_only" else "trainval.parquet"))
            if frame_hash(context) != receipt["context_hash"]:
                raise ValueError("Teacher context changed")
            teacher_path = source / "teacher_holdout"
            contract = read_json(teacher_path / "cache.json")["fingerprint"]["teacher"]
            verify_files(repo, contract["implementation_sha256"])
            if contract["feature_pipeline"] != "historical_external" or contract["model_version"] != "V3":
                raise ValueError("Unexpected teacher feature pipeline or model")
            if shared_teacher is None:
                shared_teacher = contract
                copy(contract["checkpoint"], "teacher/checkpoint.ckpt", contract["checkpoint_sha256"])
            elif contract != shared_teacher:
                raise ValueError("Teacher contracts differ across traits/scopes")
            dest = f"models/{scope}/{trait}"
            prep_name = copy(teacher_path / "preprocessing.joblib", f"{dest}/teacher_preprocessing.joblib")
            transformer_name = copy(teacher_path / "historical_teacher_preprocessing.joblib",
                                    f"{dest}/teacher_transformer.joblib")
            transformer = joblib.load(out / transformer_name)
            values = transformer.transform(context).to_numpy()
            if not np.isfinite(values).all():
                raise ValueError("Nonfinite frozen teacher context")
            np.savez_compressed(out / dest / "teacher_context.npz", x=values,
                                y=context.y.to_numpy(), sampleid=context.sampleid.to_numpy(dtype=str))
            numeric = context[[*info["prs"], "age_sd", *[f"PC{i}_sd" for i in range(1, 11)]]]
            reference = pd.DataFrame({"feature": numeric.columns, "mean": numeric.mean().to_numpy(),
                "sd": numeric.std(ddof=0).to_numpy(), "min": numeric.min().to_numpy(),
                "max": numeric.max().to_numpy(), "q01": numeric.quantile(.01).to_numpy(),
                "q99": numeric.quantile(.99).to_numpy()})
            reference.to_csv(out / dest / "feature_reference.csv", index=False)
            source_models = (baseline / "private" / trait / "reference_pc/final/models"
                             if scope == "train_only" else source / "models")
            students = {}
            for arm in dict.fromkeys(["M0_KD", info["selected_student"]]):
                for path in sorted((source_models / arm).iterdir()):
                    if path.is_file():
                        copy(path, f"{dest}/students/{arm}/{path.name}")
                students[arm] = f"{dest}/students/{arm}"
            models[trait][scope] = {
                "n_context": len(context), "context_source_hash": receipt["context_hash"],
                "context": f"{dest}/teacher_context.npz", "teacher_transformer": transformer_name,
                "feature_reference": f"{dest}/feature_reference.csv",
                "teacher_preprocessing": prep_name, "teacher_columns": list(transformer.output_columns),
                "feature_columns": [*info["prs"], "age_sd", *[f"PC{i}_sd" for i in range(1, 11)], "sex"],
                "students": students, "selected_student": info["selected_student"],
            }
    assert shared_teacher is not None
    copy(paths["eligibility"], "eligibility.yaml")
    projection = Path(selection["reference_projection_directory"])
    pc_source = read_json(projection / "projection.json")
    verify_files(projection, pc_source["output_sha256"])
    pc = {name: copy(projection / f"target.{suffix}", f"genetics/pc.{suffix}")
          for name, suffix in [("weights", "loadings.tsv"), ("frequencies", "afreq"), ("variants", "snplist")]}
    pc["n_variants"] = pc_source["n_scored_variants"]
    required = set((out / pc["variants"]).read_text().splitlines())
    score_source = read_json(paths["scoring_receipt"])
    parts, resources, source_pvar, aliases, frequencies = [], [], None, {}, None
    for source_index, record in enumerate(score_source["sources"]):
        source = original(record["directory"])
        lock = read_json(source / "weights.scored.lock.json")
        qc = pd.read_csv(source / "scoring_qc.csv").set_index("model")
        scoring = read_json(source / "scoring_run.json")
        for name, key in [("weights.scored.lock.json", "weight_lock_sha256"),
                          ("scores.long.csv", "scores_sha256"), ("scoring_qc.csv", "qc_sha256")]:
            if sha256_file(source / name) != scoring[key]:
                raise ValueError(f"Frozen score source changed: {source / name}")
        current_pvar = Path(scoring["pfile"] + ".pvar")
        if source_pvar is not None and current_pvar != source_pvar:
            raise ValueError("Frozen score matrices must share the audited variant axis")
        source_pvar = current_pvar
        for item in lock["resources"]:
            metadata = item.get("metadata", {})
            resources.append({"id": item["id"], "disease": item["disease"],
                "original_weights_sha256": item["sha256"], "catalog_url": item.get("catalog_url"),
                "metadata": metadata, "source_type": item.get("source_type", "PGS Catalog"),
                "source_url": item.get("source_url"), "source_doi": item.get("source_doi"),
                "review_for_new_external_cohort": "not_established; prior decipEHR overlap review does not transfer"})
            aliases[str(qc.loc[item["id"], "matcher_accession"])] = item["id"]
        for part_index, matrix in enumerate(record["score_matrices"]):
            command = matrix["command"]
            extract = original(command[command.index("--extract")+1])
            ids = extract.read_text().splitlines()
            used_path = original(command[command.index("--out")+1] + ".sscore.vars")
            if len(ids) != len(set(ids)) or set(ids) != set(used_path.read_text().splitlines()) or len(ids) != matrix["n_scored_variants"]:
                raise ValueError("Frozen extraction list differs from the original scored markers")
            required.update(ids)
            name = f"genetics/score_{source_index}_{part_index}"
            parts.append({"weights": copy(original(matrix["path"]), name + ".tsv.gz", matrix["sha256"]),
                "variants": copy(extract, name + ".snplist"), "n_variants": len(ids),
                "score_columns": [int(x) for x in command[command.index("--score-col-nums")+1].split(",")]})
            if frequencies is None:
                frequencies = copy(original(command[command.index("--read-freq")+1]),
                                   "genetics/prs.afreq", score_source["reference_frequency_sha256"])
            elif sha256_file(original(command[command.index("--read-freq")+1])) != score_source["reference_frequency_sha256"]:
                raise ValueError("PRS reference-frequency definitions differ")
    if {r["id"] for r in resources} != {p for names in expected_prs.values() for p in names}:
        raise ValueError("Exported score resources differ from the model feature panels")
    write_json(out / "resources.json", resources)
    assert source_pvar is not None
    missing, count = required.copy(), 0
    with source_pvar.open() as stream, (out / "genetics/variants.bim").open("w") as bim, \
            (out / "genetics/required.snplist").open("w") as snps:
        first = next(line for line in stream if not line.startswith("##"))
        if first.split()[:5] != ["#CHROM", "POS", "ID", "REF", "ALT"]:
            raise ValueError("Unexpected frozen PVAR schema")
        writer = csv.writer(bim, delimiter="\t", lineterminator="\n")
        for line in stream:
            chrom, pos, key, ref, alt = line.rstrip("\r\n").split("\t", 5)[:5]
            if key not in required:
                continue
            if key not in missing:
                raise ValueError("Duplicate required variant in the original genotype axis")
            if chrom.removeprefix("chr") not in AUTOSOMES or len(ref) != 1 or len(alt) != 1:
                raise ValueError("Frozen scoring bundle must contain autosomal biallelic SNPs")
            writer.writerow([chrom.removeprefix("chr"), key, 0, pos, alt, ref])
            snps.write(key + "\n")
            missing.remove(key)
            count += 1
    if missing:
        raise ValueError(f"{len(missing)} frozen markers lack source coordinates")
    print(f"Exported {len(resources)} raw PRS definitions and {count:,} required variants", flush=True)
    code = {str(p.relative_to(repo / "src/decipehr")): sha256_file(p)
            for p in [repo / name for name in shared_teacher["implementation_sha256"] if name.startswith("src/decipehr/")]}
    for name in ["prs_distillation_bio.py", "prs_distillation_v3.py", "prs_monotone_boosting.py"]:
        code[name] = sha256_file(repo / "src/decipehr" / name)
    runtime = out / "runtime"
    shutil.copytree(repo / "src/decipehr", runtime / "src/decipehr", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for name in ["pyproject.toml", "uv.lock", "README.md", "scripts/external_inference.py", "slurm/external_inference.sbatch"]:
        copy(repo / name, f"runtime/{name}")
    for folder, pattern in [("docs", "EXTERNAL*.md"), ("configs", "external_cohort*.yaml"),
                            ("scripts", "export_aou*.py")]:
        for path in sorted((repo / folder).glob(pattern)):
            copy(path, f"runtime/{folder}/{path.name}")
    write_json(out / "EXPORT_PROVENANCE.json", {"source_config_sha256": sha256_file(source_file),
        "agreement_review_sha256": sha256_file(run / "FINAL_REVIEW.json"), "source_files": origins,
        "age_mapping_deferred": True, "outcomes_used_for_new_fits": False, "new_model_fits": 0})
    manifest = {"schema_version": 1, "status": "complete", "genome_build": "GRCh38",
        "pc_mode": "reference_pc", "oadp": False, "scopes": list(scopes), "default_scope": "trainval_refit" if "trainval_refit" in scopes else scopes[0],
        "age_contract": {"id": AGE_CONTRACT, "units": "provided age_sd; not years",
                         "years_conversion_available": False, "target_cohort_standardization_allowed": False},
        "traits": models, "prs": expected_prs, "score_parts": parts, "score_aliases": aliases,
        "reference_frequencies": frequencies, "pc": pc, "variant_map": "genetics/variants.bim",
        "required_variants": "genetics/required.snplist", "n_required_variants": count,
        "eligibility": "eligibility.yaml", "teacher": {"checkpoint": "teacher/checkpoint.ckpt",
            "package": shared_teacher["package"], "n_estimators": 16, "seed": 42, "output": "direct_positive_class_probability"},
        "environment": {p: importlib.metadata.version(p) for p in ["numpy", "pandas", "scikit-learn", "scipy", "torch", "tabpfn"]},
        "implementation_sha256": code, "contains_private_training_context": True,
        "external_cohort_independence": "not established by this export",
        "file_sha256": {str(p.relative_to(out)): sha256_file(p) for p in sorted(out.rglob("*")) if p.is_file()},
    }
    write_json(out / "BUNDLE.json", manifest)
    load_bundle(out)
    return manifest
