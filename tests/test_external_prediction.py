"""Portable saved-student inference, cohort boundaries, and resumable outputs."""

import importlib.metadata
import runpy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from test_external_cohort import make_case
from test_external_cohort import plink as plink_fixture

from decipehr.external_bundle import AGE_CONTRACT, load_bundle
from decipehr.external_cohort import prepare_cohort
from decipehr.external_pgs import sha256_file, write_json
from decipehr.external_prediction import predict_cohort
from decipehr.prs_distillation_bio import BioPreprocessor, BioStudent, predict, save

plink = plink_fixture


def prediction_case(tmp_path, plink):
    _, config, _, _ = make_case(tmp_path, plink)
    root = Path(config["bundle"])
    manifest = load_bundle(root)
    manifest.update(default_scope="trainval_refit", scopes=["trainval_refit"], traits={},
                    eligibility="eligibility.yaml", teacher={"checkpoint": "checkpoint"}, implementation_sha256={},
                    environment={p: importlib.metadata.version(p) for p in ["numpy", "pandas", "scikit-learn", "scipy", "torch"]})
    (root / "checkpoint").write_text("unused for student inference")
    (root / "context").write_text("unused for student inference")
    (root / "transformer").write_text("unused for student inference")
    # Use the actual project eligibility schema.
    (root / "eligibility.yaml").write_bytes(Path("configs/disease_eligibility.yaml").read_bytes())
    rng = np.random.default_rng(143)
    truth = {}
    for trait, prs in manifest["prs"].items():
        context = pd.DataFrame({"sampleid": [f"donor{i}" for i in range(24)], prs[0]: rng.normal(size=24),
                                "age_sd": np.linspace(-2, 2, 24), "sex": ["Female", "Male"] * 12,
                                **{f"PC{i}_sd": rng.normal(size=24) for i in range(1, 11)}})
        context = context[["sampleid", *prs, "age_sd", *[f"PC{i}_sd" for i in range(1, 11)], "sex"]]
        prep = BioPreprocessor.fit(context, prs, 3)
        model = BioStudent(1, 10, prep.transform(context)["nonage"].shape[1], prep.age.n_segments, variant="M0")
        dest = root / trait
        save(dest / "M0_KD", model, prep, {"synthetic": True})
        numeric = context.drop(columns=["sampleid", "sex"])
        pd.DataFrame({"feature": numeric.columns, "mean": numeric.mean(), "sd": numeric.std(ddof=0),
                      "min": numeric.min(), "max": numeric.max(), "q01": numeric.quantile(.01),
                      "q99": numeric.quantile(.99)}).to_csv(dest / "reference.csv", index=False)
        manifest["traits"][trait] = {"trainval_refit": {
            "feature_columns": [*prs, "age_sd", *[f"PC{i}_sd" for i in range(1, 11)], "sex"],
            "feature_reference": f"{trait}/reference.csv", "students": {"M0_KD": f"{trait}/M0_KD"},
            "selected_student": "M0_KD", "context": "context", "teacher_transformer": "transformer",
            "teacher_preprocessing": f"{trait}/M0_KD/preprocessing.joblib"}}
        truth[trait] = model, prep
    manifest["file_sha256"] = {str(p.relative_to(root)): sha256_file(p) for p in root.rglob("*") if p.is_file() and p.name != "BUNDLE.json"}
    write_json(root / "BUNDLE.json", manifest)
    covariates = pd.read_csv(config["covariates"], dtype={"sampleid": str}).assign(age_sd=[0, .2, .3, -.3])
    covariates.to_csv(config["covariates"], index=False)
    config["age"] = {"mode": "training_scale", "contract_id": AGE_CONTRACT, "provenance": "synthetic values only"}
    config["inference"] = {"models": ["student"], "device": "cpu", "batch_size": 1}
    prepare_cohort(config)
    return config, truth


def test_saved_student_inference_resume_eligibility_and_evaluation(tmp_path, plink, monkeypatch):
    config, truth = prediction_case(tmp_path, plink)
    def forbidden(*args, **kwargs):
        raise AssertionError("Prediction must not fit any query preprocessor")
    monkeypatch.setattr(BioPreprocessor, "fit", forbidden)
    receipt = predict_cohort(config)
    assert receipt["n_predictions"] == 10  # Four T2D, four CAD, two female BC.
    assert receipt["student_fits"] == receipt["query_scalers_fitted"] == 0
    output = Path(config["run_dir"]) / "inference/primary"
    frame = pd.read_csv(output / "predictions.csv", dtype={"sampleid": str})
    genetics = pd.read_parquet(Path(config["run_dir"]) / "preparation/genetic_features.parquet")
    cov = pd.read_csv(config["covariates"], dtype={"sampleid": str})
    for trait, (model, prep) in truth.items():
        query = cov.merge(genetics, on="sampleid")
        if trait == "bc":
            query = query.loc[query.sex.eq("Female")]
        columns = load_bundle(config["bundle"])["traits"][trait]["trainval_refit"]["feature_columns"]
        query = query[["sampleid", *columns]]
        actual = frame.loc[frame.trait.eq(trait)]
        assert actual.sampleid.tolist() == query.sampleid.tolist()
        np.testing.assert_allclose(actual.probability, predict(model, prep.transform(query))["probability"], atol=1e-15)
    before = (output / "predictions.csv").stat().st_mtime_ns
    assert predict_cohort(config) == receipt
    assert (output / "predictions.csv").stat().st_mtime_ns == before
    # Reassemble a partially completed run from verified batches.
    (output / "COMPLETE.json").unlink()
    resumed = predict_cohort(config)
    assert resumed["n_predictions"] == receipt["n_predictions"]
    evaluate = runpy.run_path("scripts/external_inference.py")["evaluate"]
    labels = tmp_path / "outcomes.csv"
    cov[["sampleid"]].assign(T2D=[0, 1, 0, 1], cad=[1, 0, 1, 0], bc=[1, None, None, 0]).to_csv(labels, index=False)
    result = evaluate(SimpleNamespace(prediction_dir=output, outcomes=labels, phenotype_note="synthetic", out_dir=tmp_path / "metrics"))
    assert result["model_or_calibration_fits"] == 0
    metrics = pd.read_csv(tmp_path / "metrics/metrics.csv").set_index("trait")
    assert metrics.loc["bc", "n_labeled"] == 2
    with (output / "predictions.csv").open("a") as stream:
        stream.write("changed\n")
    with pytest.raises(ValueError, match="Completed prediction file changed"):
        predict_cohort(config)


def test_empty_female_cohort_and_age_gate(tmp_path, plink):
    config, _ = prediction_case(tmp_path, plink)
    config["age"] = {"mode": "deferred"}
    with pytest.raises(ValueError, match="awaits compatible age"):
        predict_cohort(config)
    config["age"] = {"mode": "training_scale", "contract_id": AGE_CONTRACT, "provenance": "synthetic"}
    cov = pd.read_csv(config["covariates"], dtype={"sampleid": str}).assign(sex="Male")
    cov.to_csv(config["covariates"], index=False)
    config["inference"]["traits"] = ["bc"]
    result = predict_cohort(config)
    assert result["n_predictions"] == 0
    assert result["counts"] == [{"trait": "bc", "n_input": 4, "n_eligible": 0, "n_ineligible": 4}]

