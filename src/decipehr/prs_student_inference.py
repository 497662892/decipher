"""Prediction-only access to the saved biological-prior student experiments."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from decipehr.eligibility import filter_eligible, load_eligibility_rules
from decipehr.external_pgs import sha256_file
from decipehr.prs_distillation_bio import load as load_bio
from decipehr.prs_distillation_bio import predict as predict_bio
from decipehr.prs_monotone_boosting import predict as predict_hgb
from decipehr.prs_monotone_boosting import raw_predict
from decipehr.schema import require_columns

ARMS = ("M0_KD", "M0_SUP", "G_A6_KD", "HGB_PAIR_KD", "BIO_POLICY")


def predict_student(
    run_dir: str | Path,
    frame: pd.DataFrame,
    *,
    trait: str,
    arm: str = "M0_KD",
    seed: int = 17,
    eligibility_config: str | Path = "configs/disease_eligibility.yaml",
) -> pd.DataFrame:
    """Use trusted local bundles, frozen transforms and no labels or fitting.

    ``run_dir`` is the biological-prior experiment root. BIO_POLICY uses its
    locked original-validation policy; it never selects using query outcomes.
    Rows ineligible for the requested trait are omitted, preserving input order.
    """
    if trait not in {"T2D", "cad", "bc"} or arm not in ARMS:
        raise ValueError("Unsupported trait or student arm")
    require_columns(frame, ["sampleid"], "student query")
    if frame.columns.duplicated().any() or frame.sampleid.isna().any() or not frame.sampleid.is_unique:
        raise ValueError("Query needs unique columns and nonmissing, unique sample IDs")
    rules = load_eligibility_rules(eligibility_config)
    if trait == "bc" and "bc" not in rules:
        raise ValueError("Breast cancer requires an explicit eligibility rule")
    query = filter_eligible(frame, trait, rules).copy()
    if query.empty:
        raise ValueError("No eligible query patients")
    root = Path(run_dir)
    if arm == "BIO_POLICY":
        policy = json.loads((root / "report/val/POLICY_LOCK.json").read_text())
        if policy["assessment_labels_used_for_selection"] is not False:
            raise ValueError("Student policy must be selected within training only")
        arm = policy["choices"][trait]["-1"]
        if arm not in ARMS or arm == "BIO_POLICY":
            raise ValueError("Unsupported arm in the frozen policy")
    if arm.startswith("HGB_"):
        if seed != 17:
            raise ValueError("The saved HGB models use seed 17 only")
        directory = root / "low_order" / trait / "models" / arm
        fit = json.loads((directory / "FIT.json").read_text())
        if sha256_file(directory / "model.joblib") != fit["model_sha256"]:
            raise ValueError("HGB model checksum mismatch")
        bundle = joblib.load(directory / "model.joblib")
        features = bundle["features"]
        prep = features.prep
    else:
        directory = root / "core" / trait / "models" / arm / f"seed{seed}"
        model, prep, metadata = load_bio(directory)
        if metadata["arm"] != arm or metadata["seed"] != seed:
            raise ValueError("Student arm or seed differs from saved metadata")
    columns = prep.age.fixed.source_columns
    require_columns(query, columns, "student features")
    # Explicitly restore the saved order and exclude every extra column/label.
    inputs = query[["sampleid", *columns]]
    transformed = prep.transform(inputs)
    if arm.startswith("HGB_"):
        x = features.transform(inputs)
        probability, logit = predict_hgb(bundle, x), raw_predict(bundle, x)
    else:
        result = predict_bio(model, transformed)
        probability, logit = result["probability"], result["logit"]
    if (not np.isfinite(logit).all() or not np.isfinite(probability).all()
            or ((probability < 0) | (probability > 1)).any()):
        raise ValueError("Student produced invalid predictions")
    result = pd.DataFrame({
        "sampleid": query.sampleid.to_numpy(), "trait": trait, "arm": arm, "seed": seed,
        "probability": probability, "logit": logit,
        "age_out_of_support": transformed["age_out_of_support"],
        "sex_out_of_support": transformed["sex_out_of_support"],
    })
    manifest = directory / ("FIT.json" if arm.startswith("HGB_") else "HASHES.json")
    result.attrs["model_directory"] = str(directory)
    result.attrs["model_manifest_sha256"] = sha256_file(manifest)
    return result
