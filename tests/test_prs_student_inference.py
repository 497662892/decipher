"""Exercise saved bundles, outcome isolation and eligibility at the inference boundary."""

import json
import subprocess
import sys

import joblib
import numpy as np
import pandas as pd
import pytest
import torch

from decipehr.external_pgs import sha256_file
from decipehr.prs_distillation_bio import BioPreprocessor, BioStudent, predict, save
from decipehr.prs_monotone_boosting import Features, bundle_model, fit_booster
from decipehr.prs_monotone_boosting import predict as hgb_predict
from decipehr.prs_student_inference import predict_student


@pytest.fixture
def saved(tmp_path):
    rng = np.random.default_rng(42)
    n = 100
    frame = pd.DataFrame({
        "sampleid": [f"synthetic-inference-{i}" for i in range(n)],
        "PRS_A": rng.normal(size=n), "PRS_B": rng.normal(size=n),
        "age_sd": np.linspace(-2, 2, n),
        **{f"PC{i}_sd": rng.normal(size=n) for i in range(1, 11)},
        "sex": ["Female", "Male"]*(n//2), "y": rng.integers(0, 2, n),
    })
    prep = BioPreprocessor.fit(frame, ["PRS_A", "PRS_B"])
    x = prep.transform(frame)
    model = BioStudent(2, 10, x["nonage"].shape[1], prep.age.n_segments)
    with torch.no_grad():
        model.age_slopes.fill_(.3)
    for trait in ["T2D", "bc"]:
        save(tmp_path / "core" / trait / "models/M0_KD/seed17", model, prep,
             {"arm": "M0_KD", "seed": 17})
    return tmp_path, frame, predict(model, x)["probability"]


def test_bio_prediction_replays_bundle_without_labels_or_context(saved):
    root, frame, expected = saved
    result = predict_student(root, frame.drop(columns="y"), trait="T2D")
    np.testing.assert_array_equal(result.probability, expected)
    # Arbitrary input order and changed outcome columns cannot change predictions.
    altered = frame.assign(y=99, T2D=-10, cad=1, bc=0)
    changed = predict_student(root, altered[altered.columns[::-1]], trait="T2D")
    pd.testing.assert_frame_equal(result, changed)
    assert result.sampleid.tolist() == frame.sampleid.tolist()


def test_breast_cancer_filters_only_query_and_requires_rule(saved, tmp_path):
    root, frame, expected = saved
    result = predict_student(root, frame, trait="bc")
    mask = frame.sex.eq("Female")
    np.testing.assert_array_equal(result.probability, expected[mask])
    assert result.sampleid.tolist() == frame.loc[mask, "sampleid"].tolist()
    empty_rules = tmp_path / "empty.yaml"
    empty_rules.write_text("{}")
    with pytest.raises(ValueError, match="eligibility rule"):
        predict_student(root, frame, trait="bc", eligibility_config=empty_rules)


@pytest.mark.parametrize("error", ["duplicate", "missing_id", "missing_prs", "nonfinite"])
def test_invalid_query_fails(saved, error):
    root, frame, _ = saved
    frame = frame.copy()
    if error == "duplicate":
        frame.loc[1, "sampleid"] = frame.loc[0, "sampleid"]
    elif error == "missing_id":
        frame.loc[0, "sampleid"] = None
    elif error == "missing_prs":
        frame = frame.drop(columns="PRS_A")
    else:
        frame.loc[0, "PRS_A"] = np.inf
    with pytest.raises(ValueError):
        predict_student(root, frame, trait="T2D")


def test_policy_hgb_replay_and_checksum(saved):
    root, frame, _ = saved
    features = Features.fit(frame, ["PRS_A", "PRS_B"])
    x = features.transform(frame)
    model = fit_booster(x, .2+.6*frame.y, features, family="PAIR", leaves=4, l2=1, iterations=2)
    bundle = bundle_model(model, features, "PAIR", 2, .5, {})
    directory = root / "low_order/cad/models/HGB_PAIR_KD"
    directory.mkdir(parents=True)
    path = directory / "model.joblib"
    joblib.dump(bundle, path)
    (directory / "FIT.json").write_text(json.dumps({"model_sha256": sha256_file(path)}))
    policy = root / "report/val/POLICY_LOCK.json"
    policy.parent.mkdir(parents=True)
    policy.write_text(json.dumps({"assessment_labels_used_for_selection": False,
                                 "choices": {"cad": {"-1": "HGB_PAIR_KD"}}}))
    result = predict_student(root, frame.drop(columns="y"), trait="cad", arm="BIO_POLICY")
    np.testing.assert_array_equal(result.probability, hgb_predict(bundle, x))
    assert result.arm.eq("HGB_PAIR_KD").all()
    with pytest.raises(ValueError, match="seed 17"):
        predict_student(root, frame, trait="cad", arm="BIO_POLICY", seed=29)
    with path.open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        predict_student(root, frame, trait="cad", arm="BIO_POLICY")


def test_cli_receipt_and_existing_output_protection(saved):
    root, frame, expected = saved
    query, output = root / "query.csv", root / "predictions.csv"
    frame = frame.assign(sampleid=[f"{i:05d}" for i in range(len(frame))])
    frame.drop(columns="y").to_csv(query, index=False)
    command = [sys.executable, "scripts/predict_prs_student.py", "--run-dir", str(root),
               "--trait", "T2D", "--input", str(query), "--output", str(output),
               "--eligibility-config", "configs/disease_eligibility.yaml"]
    subprocess.run(command, check=True, capture_output=True, text=True)
    result = pd.read_csv(output, dtype={"sampleid": str})
    np.testing.assert_allclose(result.probability, expected, atol=1e-15)
    assert result.sampleid.tolist() == frame.sampleid.tolist()
    receipt = json.loads((root / "predictions.csv.json").read_text())
    assert receipt["n_predicted"] == len(frame) and receipt["new_fits"] == 0
    assert receipt["labels_used"] is False
    assert receipt["output_sha256"] == sha256_file(output)
    assert receipt["model_manifest_sha256"] == sha256_file(root / "core/T2D/models/M0_KD/seed17/HASHES.json")
    before = output.read_bytes()
    retry = subprocess.run(command, capture_output=True, text=True)
    assert retry.returncode != 0 and output.read_bytes() == before
