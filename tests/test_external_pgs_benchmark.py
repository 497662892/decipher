import gzip
import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

import decipehr.external_pgs_benchmark as benchmark
from decipehr.external_pgs import read_weight_header, sha256_file, write_json
from decipehr.external_pgs_benchmark import (
    DEMOGRAPHICS,
    HARD_ANCESTRY_METHOD,
    METHODS,
    PC_ONLY_LR_METHOD,
    PCS,
    SOFT,
    evaluate_disease,
    evaluate_official_validation,
    feature_matrices,
    predict_method,
    prepare_disease_data,
    prepare_official_validation_data,
    summarize_benchmark,
    validate_shared_folds,
)


@pytest.fixture
def inputs():
    rng = np.random.default_rng(37)
    n = 90
    ids = [f"{i:03d}" for i in range(n)]
    phenotype = pd.DataFrame({
        "sampleid": ids, "age_sd": rng.normal(size=n),
        "sex": np.where(np.arange(n) % 3 == 0, "Male", "Female"),
        "T2D": np.arange(n) % 2, "bc": np.arange(n) % 2, "cad": np.arange(n) % 2,
    })
    for pc in PCS:
        phenotype[pc] = rng.normal(size=n)
    splits = pd.concat([
        pd.DataFrame({"sampleid": ids, "disease": disease, "fold": np.arange(n) % 5})
        for disease in ("T2D", "bc", "cad")
    ], ignore_index=True)
    scores = pd.concat([
        pd.DataFrame({
            "sampleid": ids, "disease": disease, "model": accession,
            "prs": phenotype[disease] * effect + rng.normal(size=n),
        })
        for disease in ("T2D", "bc", "cad")
        for accession, effect in (("PGS000001", 0.8), ("PGS000002", 0.3))
    ], ignore_index=True)
    weights = pd.DataFrame(rng.dirichlet(np.ones(5), n), columns=SOFT).assign(sampleid=ids)
    hard = pd.DataFrame({
        "sampleid": ids,
        "hard_label": np.where(np.arange(n) % 3 == 0, "AMBIGUOUS", "EUR-like"),
    })
    return dict(phenotype=phenotype, splits=splits, scores=scores, soft_weights=weights, hard_labels=hard)


def data_for(inputs, disease="T2D"):
    return prepare_disease_data(
        **inputs, disease=disease, model_ids=["PGS000001", "PGS000002"],
        eligibility_rules={"bc": {"column": "sex", "allowed_values": ["Female"]}},
    )


def test_shared_folds_reject_cross_trait_leakage(inputs):
    validate_shared_folds(inputs["splits"])
    bad = inputs["splits"].copy()
    bad.loc[0, "fold"] = 1
    with pytest.raises(ValueError, match="different folds across diseases"):
        validate_shared_folds(bad)
    bad = pd.concat([inputs["splits"], inputs["splits"].iloc[[0]]])
    with pytest.raises(ValueError, match="Duplicate patient-disease"):
        validate_shared_folds(bad)


def test_pc_only_linear_ignores_soft_features_and_query_labels(inputs):
    data = data_for(inputs)
    context, query = data.iloc[:60].copy(), data.iloc[60:].copy()
    ids = ["PGS000001", "PGS000002"]
    assert benchmark.method_features(PC_ONLY_LR_METHOD, ids) == benchmark.method_features("tabpfn_prs_pc", ids)
    options = dict(disease="T2D", method=PC_ONLY_LR_METHOD, model_ids=ids, seed=42)
    original, selection, _ = predict_method(context, query, **options)
    context = context.drop(columns=SOFT)
    query = query.drop(columns=SOFT).assign(T2D=1 - query.T2D)
    repeated, repeated_selection, _ = predict_method(context, query, **options)
    np.testing.assert_array_equal(original, repeated)
    assert selection == repeated_selection
    evaluated = evaluate_official_validation(
        context, query, disease="T2D", model_ids=ids, methods=(PC_ONLY_LR_METHOD,),
    )
    assert len(evaluated.predictions) == len(query)
    assert set(evaluated.fits.method) == {PC_ONLY_LR_METHOD}


def test_bc_eligibility_and_complete_matched_cohort(inputs):
    data = data_for(inputs, "bc")
    assert len(data) == 60
    assert set(data.sex) == {"Female"}
    with pytest.raises(ValueError, match="eligibility-config"):
        prepare_disease_data(
            **inputs, disease="bc", model_ids=["PGS000001"], eligibility_rules={},
        )
    inputs["scores"] = inputs["scores"].iloc[1:]
    with pytest.raises(ValueError, match="finite external scores"):
        data_for(inputs)


def test_missing_folds_and_invalid_ancestry_weights_fail(inputs):
    original = inputs["splits"]
    inputs["splits"] = original.iloc[1:]
    with pytest.raises(ValueError, match="Every eligible patient"):
        data_for(inputs)
    inputs["splits"] = original
    inputs["soft_weights"].loc[0, SOFT[0]] = -0.1
    with pytest.raises(ValueError, match="nonnegative and sum"):
        data_for(inputs)


@pytest.mark.parametrize("normalization", ["none", "soft_z"])
def test_preprocessing_fits_context_only_and_rejects_patient_overlap(inputs, normalization):
    data = data_for(inputs)
    context, query = data.iloc[:60], data.iloc[60:].copy()
    options = dict(prs=["PGS000001"], covariates=[*DEMOGRAPHICS, *PCS, *SOFT],
                   interactions=True, normalization=normalization)
    x_context, _, parameters = feature_matrices(context, query, **options)
    query["PGS000001"] = 1e6
    query["age_sd"] = -1e6
    x_again, _, parameters_again = feature_matrices(context, query, **options)
    np.testing.assert_allclose(x_context, x_again)
    pd.testing.assert_frame_equal(parameters, parameters_again)
    with pytest.raises(ValueError, match="patients overlap"):
        feature_matrices(context, context.iloc[:2], **options)


def test_query_labels_do_not_affect_score_selection_or_predictions(inputs):
    data = data_for(inputs)
    train, query = data.iloc[:60], data.iloc[60:].copy()
    args = dict(disease="T2D", method="best_single_lr", model_ids=["PGS000001", "PGS000002"],
                seed=42, c_grid=(0.1, 1.0))
    predicted, selection, _ = predict_method(train, query, **args)
    query["T2D"] = 1 - query["T2D"]
    repeated, repeated_selection, _ = predict_method(train, query, **args)
    np.testing.assert_allclose(predicted, repeated)
    assert selection == repeated_selection


def test_end_to_end_oof_matrix_calibration_and_paired_report(inputs):
    data = data_for(inputs)
    training_sizes = []

    def fake_tabpfn(seed):
        class QueryOnlyClassifier(LogisticRegression):
            def fit(self, x, y):
                training_sizes.append(len(y))
                self.context_rows = {tuple(row) for row in x}
                return super().fit(x, y)

            def predict_proba(self, x):
                assert not self.context_rows.intersection(tuple(row) for row in x)
                return super().predict_proba(x)

        return QueryOnlyClassifier(C=1, max_iter=2000, random_state=seed)

    result = evaluate_disease(
        data, disease="T2D", model_ids=["PGS000001", "PGS000002"],
        tabpfn_factory=fake_tabpfn, c_grid=(1.0,), methods=METHODS,
    )
    predictions = result.predictions
    assert len(predictions) == len(data) * len(METHODS)
    assert not predictions.duplicated(["sampleid", "disease", "method"]).any()
    assert predictions.groupby("sampleid")["fold"].nunique().eq(1).all()
    assert np.isfinite(predictions[["y_prob", "wrapper_prob"]]).all().all()
    assert max(training_sizes) == 72
    assert set(result.fits.fit_scope) == {
        "outer_context", "wrapper_inner_0", "wrapper_inner_1", "wrapper_inner_2",
    }
    assert (result.fits.context_ids_sha256 != result.fits.query_ids_sha256).all()
    assert "AMBIGUOUS" in set(predictions.ancestry_group)
    metrics, contrasts = summarize_benchmark(predictions, n_bootstraps=30)
    assert set(metrics.prediction_scale) == {"raw", "official_wrapper"}
    assert set(contrasts.comparison) == {"full_vs_pc", "full_vs_linear_interactions"}
    assert contrasts.family_size.eq(6).all()
    assert (contrasts.familywise_ci_low <= contrasts.ci_low).all()
    assert (contrasts.familywise_ci_high >= contrasts.ci_high).all()
    incomplete = predictions.iloc[1:]
    with pytest.raises(ValueError, match="different OOF patient sets"):
        summarize_benchmark(incomplete, n_bootstraps=2)



def official_data_for(inputs, disease="T2D"):
    return prepare_official_validation_data(
        train_phenotype=inputs["phenotype"].iloc[:60],
        validation_phenotype=inputs["phenotype"].iloc[60:],
        scores=inputs["scores"], soft_weights=inputs["soft_weights"], hard_labels=inputs["hard_labels"],
        disease=disease, model_ids=["PGS000001", "PGS000002"],
        eligibility_rules={"bc": {"column": "sex", "allowed_values": ["Female"]}},
    )


def test_official_split_preserves_membership_eligibility_and_complete_score_coverage(inputs):
    train, val = official_data_for(inputs)
    assert len(train) == 60 and len(val) == 30
    assert set(train.sampleid) == set(inputs["phenotype"].iloc[:60].sampleid)
    assert set(val.sampleid) == set(inputs["phenotype"].iloc[60:].sampleid)
    assert "fold" not in train and "fold" not in val
    bc_train, bc_val = official_data_for(inputs, "bc")
    assert len(bc_train) == 40 and len(bc_val) == 20
    assert set(bc_train.sex) == set(bc_val.sex) == {"Female"}
    # Even overlap in a male who is ineligible for BC must fail before filtering.
    inputs["phenotype"] = pd.concat([inputs["phenotype"], inputs["phenotype"].iloc[[0]]])
    with pytest.raises(ValueError, match="patients overlap"):
        official_data_for(inputs, "bc")
    inputs["phenotype"] = inputs["phenotype"].iloc[:-1]
    inputs["scores"] = inputs["scores"].drop(index=60)
    with pytest.raises(ValueError, match="finite external scores"):
        official_data_for(inputs)


def test_official_validation_labels_never_affect_any_fit_or_prediction(inputs, monkeypatch):
    train, val = official_data_for(inputs)
    train_ids, val_ids = set(train.sampleid), set(val.sampleid)
    original_predict = benchmark._predict
    original_wrapper = benchmark.fit_logistic_predict_proba
    final_context_sizes = []

    def audited_predict(context, query, **kwargs):
        assert set(context.sampleid).issubset(train_ids)
        assert not set(context.sampleid).intersection(query.sampleid)
        if set(query.sampleid).intersection(val_ids):
            assert set(context.sampleid) == train_ids
            assert set(query.sampleid) == val_ids
            assert "T2D" not in query
            final_context_sizes.append(len(context))
        return original_predict(context, query, **kwargs)

    def audited_wrapper(*, train, val, **kwargs):
        assert set(train.sampleid) == train_ids
        assert set(val.sampleid) == val_ids
        assert "T2D" not in val
        return original_wrapper(train=train, val=val, **kwargs)

    monkeypatch.setattr(benchmark, "_predict", audited_predict)
    monkeypatch.setattr(benchmark, "fit_logistic_predict_proba", audited_wrapper)
    options = dict(
        disease="T2D", model_ids=["PGS000001", "PGS000002"], c_grid=(1.0,),
        tabpfn_factory=lambda seed: LogisticRegression(max_iter=2000, random_state=seed),
        normalization="soft_z",
        methods=METHODS,
    )
    result = evaluate_official_validation(train, val, **options)
    changed = evaluate_official_validation(train, val.assign(T2D=1 - val.T2D), **options)
    pd.testing.assert_frame_equal(result.fits, changed.fits)
    pd.testing.assert_frame_equal(result.preprocessing, changed.preprocessing)
    pd.testing.assert_frame_equal(
        result.predictions.drop(columns="y_true"), changed.predictions.drop(columns="y_true"),
    )
    assert final_context_sizes == [60] * (2 * len(METHODS))
    assert len(result.predictions) == len(val) * len(METHODS)
    assert set(result.predictions.sampleid) == val_ids
    assert result.predictions.evaluation_protocol.eq("official_train_val").all()
    final_fits = result.fits.loc[result.fits.fit_scope == "official_train_context"]
    assert len(final_fits) == len(METHODS) and final_fits.n_context.eq(len(train)).all()
    metrics, contrasts = summarize_benchmark(result.predictions, n_bootstraps=10)
    assert metrics.evaluation_protocol.eq("official_train_val").all()
    assert contrasts.inference_scope.eq("conditional_on_frozen_validation_predictions").all()
    mixed = result.predictions.copy()
    mixed.loc[0, "evaluation_protocol"] = "shared_5fold_cv"
    with pytest.raises(ValueError, match="separately"):
        summarize_benchmark(mixed, n_bootstraps=2)


def test_pc_default_predictions_are_identical_without_soft_weights(inputs):
    train_with_soft, val_with_soft = official_data_for(inputs)
    inputs["soft_weights"] = None
    train, val = official_data_for(inputs)
    assert not set(SOFT).intersection(train.columns)
    options = dict(
        disease="T2D", model_ids=["PGS000001", "PGS000002"],
        tabpfn_factory=lambda seed: LogisticRegression(max_iter=2000, random_state=seed),
    )
    result = evaluate_official_validation(train, val, **options)
    reference = evaluate_official_validation(train_with_soft, val_with_soft, **options)
    pd.testing.assert_frame_equal(result.predictions, reference.predictions)
    pd.testing.assert_frame_equal(result.preprocessing, reference.preprocessing)
    assert set(result.predictions.method) == {"tabpfn_prs_pc"}
    assert len(result.fits) == 4
    metrics, comparisons = summarize_benchmark(result.predictions, n_bootstraps=2)
    assert not metrics.empty and comparisons.empty


def test_hard_ancestry_uses_train_categories_and_ignores_pcs_soft_and_query_labels(inputs):
    inputs["soft_weights"] = None
    train, val = official_data_for(inputs)
    options = dict(
        disease="T2D", method=HARD_ANCESTRY_METHOD,
        model_ids=["PGS000001", "PGS000002"], seed=42,
        tabpfn_factory=lambda seed: LogisticRegression(max_iter=2000, random_state=seed),
    )
    predicted, selection, parameters = predict_method(train, val, **options)
    assert not set(parameters.source).intersection([*PCS, *SOFT])
    ancestry = parameters.loc[parameters.source.eq("hard_label")]
    assert ancestry.encoding.eq("one_hot").all()
    assert set(ancestry.category) == {"AMBIGUOUS", "EUR-like"}
    changed_train, changed_val = train.copy(), val.copy()
    changed_train[PCS] = 1e6
    changed_val[PCS] = -1e6
    changed_val["T2D"] = 1 - changed_val.T2D
    repeated, repeated_selection, repeated_parameters = predict_method(
        changed_train, changed_val, **options,
    )
    np.testing.assert_array_equal(predicted, repeated)
    assert selection == repeated_selection
    pd.testing.assert_frame_equal(parameters, repeated_parameters)
    val["hard_label"] = "QUERY_ONLY_CATEGORY"
    _, _, parameters_unseen = predict_method(train, val, **options)
    pd.testing.assert_frame_equal(parameters, parameters_unseen)


def test_official_validation_cli_writes_validation_outputs_and_rejects_overlap(inputs, tmp_path):
    paths = {name: tmp_path / (name + ".csv") for name in ("train", "validation", "scores", "soft", "hard")}
    for name, frame in (
        ("train", inputs["phenotype"].iloc[:60]), ("validation", inputs["phenotype"].iloc[60:]),
        ("scores", inputs["scores"]), ("soft", inputs["soft_weights"]), ("hard", inputs["hard_labels"]),
    ):
        frame.to_csv(paths[name], index=False)
    weight = tmp_path / "synthetic.txt.gz"
    with gzip.open(weight, "wt") as stream:
        stream.write("#pgs_id=PGS000001\n#HmPOS_build=GRCh38\n#variants_number=1\nSNP\tBETA\nrs1\t1\n")
    lock_path, scoring_run = tmp_path / "weights.lock.json", tmp_path / "scoring_run.json"
    write_json(lock_path, {
        "schema_version": 1, "build": "GRCh38", "target_cohort": "synthetic", "use_context": "research",
        "resources": [{
            "id": "PGS000001", "name": "synthetic", "disease": "T2D",
            "path": str(weight), "sha256": sha256_file(weight), "weight_url": "https://example.org/synthetic",
            "header": read_weight_header(weight, accession="PGS000001", build="GRCh38"),
            "review": {"include": True, "trait_match": "confirmed", "allowed_use": "allowed",
                       "sample_overlap": "excluded", "evidence": ["synthetic only"]},
        }],
    })
    write_json(scoring_run, {"weight_lock_sha256": sha256_file(lock_path), "scores_sha256": sha256_file(paths["scores"])})
    out = tmp_path / "result"
    command = [
        sys.executable, "scripts/evaluate_external_pgs_validation.py",
        "--weight-lock", str(lock_path), "--scores", str(paths["scores"]),
        "--scoring-run", str(scoring_run), "--train-phenotype", str(paths["train"]),
        "--validation-phenotype", str(paths["validation"]),
        "--ancestry-weights", str(paths["soft"]), "--ancestry-labels", str(paths["hard"]),
        "--diseases", "T2D", "--methods", "multi_prs_lr", "--n-bootstraps", "2", "--out-dir", str(out),
    ]
    subprocess.run(command, check=True, capture_output=True, text=True)
    predictions = pd.read_csv(out / "validation_predictions.csv", dtype={"sampleid": str})
    assert set(predictions.sampleid) == set(inputs["phenotype"].iloc[60:].sampleid)
    assert not (out / "oof_predictions.csv").exists()
    run = json.loads((out / "run.json").read_text())
    assert run["status"] == "complete" and run["evaluation_protocol"] == "official_train_val"
    assert "validation_phenotype" in run["input_sha256"] and "splits" not in run["input_sha256"]
    assert pd.read_csv(out / "cohort_counts.csv")["n"].tolist() == [60, 30]
    pc_command = command.copy()
    for flag in ("--ancestry-weights", "--methods"):
        index = pc_command.index(flag)
        del pc_command[index:index + 2]
    pc_out = tmp_path / "pc_preflight"
    pc_command[-1] = str(pc_out)
    subprocess.run([*pc_command, "--preflight-only"], check=True, capture_output=True, text=True)
    preflight = json.loads((pc_out / "preflight.json").read_text())
    assert preflight["primary_method"] == "tabpfn_prs_pc"
    assert preflight["arguments"]["methods"] == "tabpfn_prs_pc"
    assert "ancestry_weights" not in preflight["input_sha256"]
    pc_lr_command = pc_command.copy()
    pc_lr_command[-1] = str(tmp_path / "pc_lr_preflight")
    subprocess.run([*pc_lr_command, "--methods", PC_ONLY_LR_METHOD, "--preflight-only"],
                   check=True, capture_output=True, text=True)
    pc_lr_preflight = json.loads((tmp_path / "pc_lr_preflight/preflight.json").read_text())
    assert "ancestry_weights" not in pc_lr_preflight["input_sha256"]
    for extra in (("--methods", "all"), ("--normalization", "soft_z")):
        missing = subprocess.run([*pc_command, *extra, "--preflight-only"], capture_output=True, text=True)
        assert missing.returncode != 0 and "--ancestry-weights is required" in missing.stderr
    command[command.index("--validation-phenotype") + 1] = str(paths["train"])
    command[-1] = str(tmp_path / "overlap")
    failed = subprocess.run(command, capture_output=True, text=True)
    assert failed.returncode != 0 and "patients overlap" in failed.stderr
    assert not (tmp_path / "overlap").exists()
