"""Prediction-only adapters for portable reference-PC teacher/student bundles."""

from __future__ import annotations

import gc
import importlib.metadata
import os
import time
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
from scipy.special import logit

import decipehr
from decipehr.eligibility import filter_eligible, load_eligibility_rules
from decipehr.external_bundle import AGE_CONTRACT, contained, load_bundle, read_json
from decipehr.external_cohort import digest, prepare_cohort, read_covariates
from decipehr.external_pgs import sha256_file, write_json
from decipehr.prs_distillation.teacher import make_teacher, positive_probability
from decipehr.prs_distillation_bio import load as load_student
from decipehr.prs_distillation_bio import predict as predict_student
from decipehr.prs_monotone_boosting import predict as predict_hgb
from decipehr.prs_monotone_boosting import raw_predict


def inference_options(config, bundle):
    options = {"scope": bundle["default_scope"], "models": ["tabpfn", "student"], "student": "M0_KD",
               "traits": ["T2D", "cad", "bc"], "device": "cuda", "batch_size": 1024, "name": "primary"}
    if set(config["inference"]) - set(options):
        raise ValueError("Unknown inference option")
    options.update(config["inference"])
    if options["scope"] not in bundle["scopes"] or options["student"] not in {"M0_KD", "selected", "both"}:
        raise ValueError("Scope or student selection is unavailable")
    for key, allowed in [("models", {"tabpfn", "student"}), ("traits", {"T2D", "cad", "bc"})]:
        if not isinstance(options[key], list) or not options[key] or len(set(options[key])) != len(options[key]) or set(options[key]) - allowed:
            raise ValueError(f"Invalid or repeated inference {key}")
    if options["device"] not in {"cuda", "cpu"} or not isinstance(options["batch_size"], int) or options["batch_size"] < 1:
        raise ValueError("Inference requires cpu/cuda and a positive integer batch_size")
    if not options["name"] or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in options["name"]):
        raise ValueError("Inference name must use letters, digits, underscore, or hyphen")
    return options


def check_runtime(bundle, options):
    packages = ["numpy", "pandas", "scikit-learn", "scipy", "torch"]
    if "tabpfn" in options["models"]:
        packages.append("tabpfn")
    actual = {name: importlib.metadata.version(name) for name in packages}
    for name, value in actual.items():
        expected = bundle["environment"][name]
        if (value.split("+")[0] if name == "torch" else value) != (expected.split("+")[0] if name == "torch" else expected):
            raise ValueError(f"Runtime {name}={value}; frozen bundle requires {expected}. Use the bundled uv.lock")
    package_root = Path(decipehr.__file__).parent
    for name, expected in bundle["implementation_sha256"].items():
        if sha256_file(package_root / name) != expected:
            raise ValueError(f"Frozen model implementation differs: {name}")
    return actual


def compatible_age(config, bundle):
    age = config["age"]
    if age["mode"] != "training_scale":
        raise ValueError("Genetic preparation is ready; inference awaits compatible age_sd. Age conversion is deferred. Do not use raw years or target-cohort z-scores")
    if age.get("contract_id") != AGE_CONTRACT or age.get("contract_id") != bundle["age_contract"]["id"] or not str(age.get("provenance", "")).strip():
        raise ValueError("training_scale age needs the bundle contract_id and documented provenance")


def _cached_part(path, fingerprint, ids):
    receipt = Path(str(path) + ".json")
    if receipt.exists():
        saved = read_json(receipt)
        if saved["fingerprint"] != fingerprint or not path.exists() or sha256_file(path) != saved["sha256"]:
            raise ValueError(f"Prediction cache changed: {path}")
        frame = pd.read_parquet(path)
        if frame.sampleid.tolist() != ids:
            raise ValueError("Prediction cache patient order changed")
        return frame
    if path.exists():
        path.rename(path.with_name(path.name + f".incomplete-{time.time_ns()}"))
    return None


def _write_part(path, frame, fingerprint):
    if not np.isfinite(frame[["probability", "logit"]].to_numpy()).all() or not frame.probability.between(0, 1).all():
        raise ValueError("Invalid model predictions")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)
    write_json(str(path) + ".json", {"fingerprint": fingerprint, "sha256": sha256_file(path), "n": len(frame)})


def _distribution(query, reference, trait, scope):
    rows = []
    for item in reference.to_dict("records"):
        values = query[item["feature"]].to_numpy(float)
        scale = item["sd"] or 1
        rows.append({"trait": trait, "scope": scope, **{f"context_{k}": v for k, v in item.items() if k != "feature"},
            "feature": item["feature"], "n_query": len(values), "query_mean": values.mean(), "query_sd": values.std(),
            "mean_shift_in_context_sd": (values.mean()-item["mean"])/scale,
            "query_q01": np.quantile(values, .01), "query_q99": np.quantile(values, .99),
            "outside_context_range_fraction": np.mean((values < item["min"]) | (values > item["max"]))})
    return rows


def predict_cohort(config, bundle=None):
    import torch

    torch.set_num_threads(config["runtime"]["threads"])
    root, bundle_root = Path(config["run_dir"]), Path(config["bundle"])
    bundle = bundle or load_bundle(bundle_root)
    options = inference_options(config, bundle)
    preparation = prepare_cohort(config, bundle)
    compatible_age(config, bundle)
    environment = check_runtime(bundle, options)
    if "tabpfn" in options["models"] and options["device"] == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("TabPFN cuda inference requires an allocated GPU; genetic preparation remains reusable")
    covariates = read_covariates(config)
    genetics = pd.read_parquet(root / "preparation/genetic_features.parquet")
    if set(genetics.sampleid) != set(covariates.sampleid):
        raise ValueError("Covariate and prepared genetic patient sets differ")
    features = covariates.merge(genetics, on="sampleid", validate="one_to_one", how="left")
    rules = load_eligibility_rules(bundle_root / bundle["eligibility"])
    if "bc" not in rules:
        raise ValueError("Female BC eligibility is required")
    contract = {"bundle_sha256": sha256_file(bundle_root / "BUNDLE.json"), "preparation": preparation["output_sha256"],
                "covariates_sha256": sha256_file(config["covariates"]), "age": config["age"], "columns": config["columns"],
                "sex_mapping": config["sex_mapping"], "options": options, "environment": environment,
                "prediction_code_sha256": sha256_file(__file__)}
    fingerprint = digest(contract)
    output = root / "inference" / options["name"]
    output.mkdir(parents=True, exist_ok=True)
    if (output / "CONTRACT.json").exists() and read_json(output / "CONTRACT.json") != contract:
        raise ValueError("Inference inputs/options changed; choose a new inference.name to retain prior outputs")
    write_json(output / "CONTRACT.json", contract)
    if (output / "COMPLETE.json").exists():
        saved = read_json(output / "COMPLETE.json")
        for name, expected in saved["output_sha256"].items():
            if sha256_file(output / name) != expected:
                raise ValueError(f"Completed prediction file changed: {name}")
        return saved
    frames, distributions, counts, timings = [], [], [], []
    started = time.monotonic()
    for trait in options["traits"]:
        packet = bundle["traits"][trait][options["scope"]]
        eligible = filter_eligible(features, trait, rules)
        query = eligible[["sampleid", *packet["feature_columns"]]].reset_index(drop=True)
        counts.append({"trait": trait, "n_input": len(features), "n_eligible": len(query), "n_ineligible": len(features)-len(query)})
        if query.empty:
            continue
        distributions += _distribution(query, pd.read_csv(bundle_root / packet["feature_reference"]), trait, options["scope"])
        arms = []
        if "tabpfn" in options["models"]:
            arms.append("TabPFN_v3")
        if "student" in options["models"]:
            arms.extend({"M0_KD": ["M0_KD"], "selected": [packet["selected_student"]],
                         "both": list(dict.fromkeys(["M0_KD", packet["selected_student"]]))}[options["student"]])
        for arm in arms:
            model_started = time.monotonic()
            pending, existing = [], {}
            for start in range(0, len(query), options["batch_size"]):
                batch = query.iloc[start:start+options["batch_size"]]
                path = output / trait / arm / f"part{start:09d}.parquet"
                cached = _cached_part(path, fingerprint, batch.sampleid.tolist())
                if cached is None:
                    pending.append((start, path))
                else:
                    existing[start] = cached
            classifier = None
            if pending:
                if arm == "TabPFN_v3":
                    context = np.load(contained(bundle_root, packet["context"]), allow_pickle=False)
                    if set(context["sampleid"]) & set(query.sampleid):
                        raise ValueError("Query and teacher-context patient identifiers overlap")
                    teacher_prep = joblib.load(bundle_root / packet["teacher_preprocessing"])
                    transformer = joblib.load(bundle_root / packet["teacher_transformer"])
                    if list(transformer.output_columns) != packet["teacher_columns"]:
                        raise ValueError("Saved teacher feature order changed")
                    teacher = bundle["teacher"]
                    classifier = make_teacher(SimpleNamespace(checkpoint=str(bundle_root / teacher["checkpoint"]),
                        teacher_package=teacher["package"], teacher_estimators=teacher["n_estimators"],
                        teacher_device=options["device"], teacher_seed=teacher["seed"]))
                    classifier.fit(context["x"], context["y"])
                elif arm.startswith("HGB"):
                    saved = joblib.load(bundle_root / packet["students"][arm] / "model.joblib")
                    prep = saved["features"].prep
                else:
                    student, prep, _ = load_student(bundle_root / packet["students"][arm])
                for start, path in pending:
                    batch = query.iloc[start:start+options["batch_size"]]
                    if arm == "TabPFN_v3":
                        transformed = teacher_prep.transform(batch)
                        probability, _ = positive_probability(classifier, transformer.transform(batch).to_numpy())
                        eta = logit(np.clip(probability, 1e-6, 1-1e-6))
                    elif arm.startswith("HGB"):
                        transformed = prep.transform(batch)
                        values = saved["features"].transform(batch)
                        probability, eta = predict_hgb(saved, values), raw_predict(saved, values)
                    else:
                        transformed = prep.transform(batch)
                        prediction = predict_student(student, transformed)
                        probability, eta = prediction["probability"], prediction["logit"]
                    result = pd.DataFrame({"sampleid": batch.sampleid.to_numpy(), "trait": trait,
                        "model": arm, "training_scope": options["scope"], "probability": probability, "logit": eta,
                        "age_out_of_support": transformed["age_out_of_support"], "sex_out_of_support": transformed["sex_out_of_support"]})
                    _write_part(path, result, fingerprint)
                    existing[start] = result
                if classifier is not None:
                    del classifier
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            frames.extend(existing[start] for start in sorted(existing))
            timings.append({"trait": trait, "model": arm, "seconds": time.monotonic()-model_started,
                            "new_prediction_batches": len(pending), "reused_prediction_batches": len(existing)-len(pending)})
            print(f"Predicted {trait}/{arm}: {len(query)} eligible patients", flush=True)
    columns = ["sampleid", "trait", "model", "training_scope", "probability", "logit", "age_out_of_support", "sex_out_of_support"]
    predictions = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=columns)
    if predictions.duplicated(["sampleid", "trait", "model"]).any():
        raise ValueError("Duplicate final predictions")
    predictions.to_csv(output / "predictions.csv", index=False)
    pd.DataFrame(counts).to_csv(output / "cohort_counts.csv", index=False)
    pd.DataFrame(distributions).to_csv(output / "feature_distribution.csv", index=False)
    write_json(output / "timing.json", {"total_seconds": time.monotonic()-started, "models": timings})
    receipt = {"status": "prediction_complete", "n_patients": len(features), "n_predictions": len(predictions),
        "counts": counts, "scope": options["scope"], "models": options["models"], "student": options["student"],
        "outcomes_used": False, "student_fits": 0, "context": "frozen source-training features and labels only",
        "age_contract": config["age"], "query_scalers_fitted": 0, "pc_mode": "reference_pc", "oadp": False,
        "teacher_logit_clip_epsilon": 1e-6, "student_logit": "native model logit",
        "environment": environment, "device": options["device"],
        "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "fingerprint": fingerprint,
        "external_predictive_performance_established": False,
        "output_sha256": {str(p.relative_to(output)): sha256_file(p) for p in sorted(output.rglob("*")) if p.is_file()}}
    write_json(output / "COMPLETE.json", receipt)
    return receipt
