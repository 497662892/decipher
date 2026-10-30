"""Synthetic-only algebra, leakage, cache, selection and deployment checks."""

from __future__ import annotations

import copy
import json

import numpy as np
import pandas as pd
import pytest
import torch
from scipy.special import expit
from torch.nn import functional as F

from decipehr.prs_distillation.config import Config, Optimization, load_config
from decipehr.prs_distillation.data import attach_full_sensitivity_groups, make_split_manifest
from decipehr.prs_distillation.experiment import select_candidates
from decipehr.prs_distillation.preprocessing import (
    AGE,
    PCS,
    Preprocessor,
    assert_boundary,
    id_hash,
)
from decipehr.prs_distillation.reporting import (
    basic_metrics,
    calibration,
    fidelity,
    paired_bootstrap,
)
from decipehr.prs_distillation.student import (
    Student,
    StudentBundle,
    fit_student,
    load_student_bundle,
    predict,
    soft_targets,
    tensors,
)
from decipehr.prs_distillation.teacher import positive_probability, teacher_oof, teacher_predict

torch.set_num_threads(1)


def synthetic(n=120, offset=0):
    rng = np.random.default_rng(812 + offset)
    frame = pd.DataFrame({"sampleid": [f"synthetic-{i + offset}" for i in range(n)],
                          "kinship_group": [f"family-{(i + offset) // 2}" for i in range(n)],
                          "hard_label": ["groupA" if i % 3 else "groupB" for i in range(n)],
                          "PRS_A": rng.normal(size=n), "PRS_B": rng.normal(size=n),
                          AGE: rng.normal(size=n)})
    for col in PCS:
        frame[col] = rng.normal(size=n)
    frame["sex"] = ["Female" if i % 2 else "Male" for i in range(n)]
    frame["y"] = np.tile([0, 1, 1, 0], n // 4)
    return frame


@pytest.fixture
def data():
    frame = synthetic()
    prep = Preprocessor.fit(frame, ["PRS_A", "PRS_B"])
    return frame, prep, prep.transform(frame)


def model_for(data, **kwargs):
    return Student(data["z"].shape[1], data["a"].shape[1], data["phi"].shape[1],
                   data["baseline"].shape[1], **kwargs)


@pytest.mark.parametrize("variant", ["full", "static_gate", "constant_scale"])
def test_exact_algebra_and_deletion_ablations(data, variant):
    _, _, x = data
    model = model_for(x, variant=variant)
    with torch.no_grad():
        model.baseline[1:] = torch.linspace(-0.2, 0.3, len(model.baseline) - 1)
        model.u0.fill_(np.log(0.7))
    p = predict(model, x)
    assert (p["weights"] >= 0).all()
    np.testing.assert_allclose(p["weights"].sum(axis=1), 1)
    np.testing.assert_allclose(p["effective_coefficients"].sum(axis=1), p["age_scale"])
    np.testing.assert_allclose(p["logit"], p["baseline_logit"] + p["contributions"].sum(axis=1))
    if variant == "static_gate":
        np.testing.assert_allclose(p["weights"], np.tile(p["weights"][0], (len(p["weights"]), 1)))
        assert not hasattr(model, "w") and np.std(p["baseline_logit"]) > 0
    if variant == "constant_scale":
        np.testing.assert_allclose(p["age_scale"], 0.7)
        assert not hasattr(model, "theta_s") and model.u0.requires_grad


def test_gate_only_depends_on_ancestry_and_nonnegative_effect(data):
    frame, prep, x = data
    model = model_for(x)
    changed = frame.copy()
    changed[AGE] += 0.6
    changed["PRS_A"] += 5
    a, b = predict(model, x), predict(model, prep.transform(changed))
    np.testing.assert_allclose(a["weights"], b["weights"])
    changed = copy.deepcopy(x)
    changed["z"][:, 0] += 1
    assert (predict(model, changed)["probability"] >= a["probability"]).all()
    changed["z"][:] = 0
    np.testing.assert_allclose(predict(model, changed)["logit"], a["baseline_logit"])


def test_single_prototype_and_parameter_budget(data):
    _, _, x = data
    x = copy.deepcopy(x)
    x["z"] = x["z"][:, :1]
    model = model_for(x)
    assert set(dict(model.named_parameters())) == {"baseline", "u0", "theta_s"}
    np.testing.assert_allclose(predict(model, x)["weights"], 1)


def test_prs_permutation_softmax_gauge_and_export(data):
    _, _, x = data
    model = model_for(x)
    expected = predict(model, x)
    with torch.no_grad():
        model.w.add_(3)
        model.c.add_(4)
    np.testing.assert_allclose(predict(model, x)["logit"], expected["logit"], atol=1e-12)
    assert abs(model.canonical_state()["w"].sum().item()) < 1e-12
    with torch.no_grad():
        model.w.copy_(model.w.flip(0))
        model.c.copy_(model.c.flip(0))
    changed = copy.deepcopy(x)
    changed["z"] = changed["z"][:, ::-1].copy()
    np.testing.assert_allclose(predict(model, changed)["logit"], expected["logit"], atol=1e-12)


def test_soft_bce_identity_extremes_and_detach():
    y = torch.tensor([0, 1, 0, 1], dtype=torch.float64)
    q = torch.tensor([0, 1, 0.2, 0.7], dtype=torch.float64, requires_grad=True)
    logits = torch.tensor([-1000, 1000, -2, 3], dtype=torch.float64, requires_grad=True)
    target = soft_targets(y, q, 0.25)
    a = F.binary_cross_entropy_with_logits(logits, target)
    b = 0.75 * F.binary_cross_entropy_with_logits(logits, y) + \
        0.25 * F.binary_cross_entropy_with_logits(logits, q.detach())
    torch.testing.assert_close(a, b)
    a.backward()
    assert q.grad is None and logits.grad is not None
    torch.testing.assert_close(soft_targets(y, None, 0), y)
    torch.testing.assert_close(soft_targets(None, q, 1), q.detach())
    with pytest.raises(ValueError, match="Targets"):
        soft_targets([1.2], None, 0)


def test_gradients_reach_every_effect(data):
    frame, _, x = data
    model = model_for(x)
    loss = F.binary_cross_entropy_with_logits(model(tensors(x))["logit"],
                                               soft_targets(frame.y, None, 0))
    loss.backward()
    for parameter in model.parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0


@pytest.mark.parametrize("lam", [0, 1])
def test_endpoint_training_independence(data, lam):
    frame, _, x = data
    options = Optimization(max_epochs=20, evaluate_every=5)
    q = expit(frame.PRS_A.to_numpy())
    first, _ = fit_student(x, frame.y.to_numpy(), q, lam=lam, gate_ridge=0.001,
                           seed=17, options=options, epochs=20)
    second, _ = fit_student(x, 1 - frame.y.to_numpy() if lam == 1 else frame.y.to_numpy(),
                            1 - q if lam == 0 else q, lam=lam, gate_ridge=0.001,
                            seed=17, options=options, epochs=20)
    np.testing.assert_array_equal(predict(first, x)["probability"], predict(second, x)["probability"])


def test_fit_scope_scaling_fallback_and_input_contract(data):
    frame, prep, _ = data
    assert prep.fit_n == len(frame) and prep.fit_ids_hash == id_hash(frame)
    assert prep.age_basis(frame[AGE]).shape == (len(frame), 3)
    assert np.max(np.abs(prep.age_basis(frame[AGE]).mean(axis=0))) < 1e-14
    extreme = frame.copy()
    extreme[AGE] = 1e6
    before = prep.metadata()
    assert prep.transform(extreme)["age_out_of_support"].all()
    assert prep.metadata() == before
    np.testing.assert_allclose(prep.age_basis([1e6]), prep.age_basis([prep.age_max]))
    with pytest.raises(ValueError, match="order"):
        prep.transform(frame[list(reversed(frame.columns))])
    with pytest.raises(ValueError, match="Missing"):
        prep.transform(frame.drop(columns="PRS_A"))
    bad = frame.copy()
    bad.loc[0, "sex"] = "unseen"
    with pytest.raises(ValueError, match="Unsupported"):
        prep.transform(bad)
    two_ages = frame.copy()
    two_ages[AGE] = np.arange(len(frame)) % 2
    fallback = Preprocessor.fit(two_ages, ["PRS_A", "PRS_B"])
    assert fallback.spline is None and fallback.metadata()["basis"] == "logged_linear_fallback"


def test_duplicate_and_constant_prototypes_are_outcome_blind(data):
    frame, _, _ = data
    frame = frame.copy()
    frame["PRS_DUP"] = frame.PRS_A
    frame["PRS_CONST"] = 4
    cols = ["PRS_A", "PRS_B", "PRS_DUP", "PRS_CONST"]
    frame = frame[["sampleid", "kinship_group", "hard_label", *cols, AGE, *PCS, "sex", "y"]]
    prep = Preprocessor.fit(frame, cols)
    assert prep.kept_prs == ["PRS_A", "PRS_B"]
    assert prep.dropped_prs == {"PRS_DUP": "exact_duplicate_of:PRS_A", "PRS_CONST": "constant_in_fit_scope"}
    frame.y = 1 - frame.y
    assert Preprocessor.fit(frame, cols).metadata() == prep.metadata()


def test_declared_unknown_sex_matches_existing_encoder_without_relabeling(data):
    from decipehr.tabpfn_ensemble import TabPFNFeatureTransformer

    frame, prep, _ = data
    query = frame.iloc[:2].copy()
    query.loc[query.index[0], "sex"] = "Unknown"
    transformed = prep.transform(query)
    existing = TabPFNFeatureTransformer.fit(frame, prs_cols=["PRS_A", "PRS_B"],
                                           covariates=[AGE, *PCS, "sex"], include_covariates=True)
    old = existing.transform(query)
    np.testing.assert_allclose(transformed["teacher"], old.to_numpy())
    assert transformed["sex_out_of_support"].tolist() == [True, False]
    np.testing.assert_allclose(transformed["baseline"][0, -2:], 0)
    female = frame.loc[frame.sex == "Female"]
    female_prep = Preprocessor.fit(female, ["PRS_A", "PRS_B"])
    with pytest.raises(ValueError, match="Unsupported"):
        female_prep.transform(query)


def test_bundle_round_trip_and_checksum(data, tmp_path, monkeypatch):
    frame, prep, x = data
    model = model_for(x)
    path = tmp_path / "bundle"
    StudentBundle(model, prep, {"source": "synthetic"}).save(path)
    loaded = load_student_bundle(path)
    for k, v in predict(model, x).items():
        np.testing.assert_allclose(loaded.predict(frame)[k], v, atol=1e-12)
    with (path / "student_state.pt").open("ab") as stream:
        stream.write(b"tampered")
    with pytest.raises(ValueError, match="checksum"):
        load_student_bundle(path)


class FakeTeacher:
    """Deterministic synthetic test oracle, never used in the real-data CLI."""
    def fit(self, x, y):
        self.classes_ = np.array([1, 0])
        self.mean = float(np.mean(y))
        self.mean_x = float(x[:, 0].mean())
        return self

    def predict_proba(self, x):
        p = expit(x[:, 0] * 0.3 + self.mean + self.mean_x)
        return np.column_stack([p, 1 - p])


def teacher_kwargs(tmp_path):
    return {"directory": tmp_path, "contract": {"backend": "SYNTHETIC_TEST_ONLY", "seed": 1},
            "trait": "synthetic", "manifest_hash": "fixture", "factory": FakeTeacher}


def test_positive_mapping_does_not_assume_second_column(data):
    _, _, x = data
    teacher = FakeTeacher().fit(x["teacher"], np.zeros(len(x["z"])))
    probability, mapping = positive_probability(teacher, x["teacher"])
    assert mapping["positive_index"] == 0
    np.testing.assert_allclose(probability, teacher.predict_proba(x["teacher"])[:, 0])


def test_group_boundary_and_fixed_shared_splits():
    d, e = synthetic(), synthetic(40, 1000).drop(columns="y")
    splits, _ = make_split_manifest(d, e, 42)
    assert not splits.sampleid.duplicated().any()
    ds = splits.loc[splits.split == "official_train"]
    assert ds.groupby("kinship_group").final_teacher_fold.nunique().max() == 1
    assert ds.groupby("kinship_group").internal_role.nunique().max() == 1
    collision = e.copy()
    collision.loc[0, "kinship_group"] = d.kinship_group.iloc[0]
    with pytest.raises(ValueError, match="kinship_group"):
        assert_boundary(d, collision)


def test_teacher_oof_scope_h_mutation_cache_and_coverage(tmp_path):
    d, e = synthetic(), synthetic(40, 1000).drop(columns="y")
    splits, _ = make_split_manifest(d, e, 42)
    split = splits.set_index("sampleid").loc[d.sampleid]
    u, h = d.loc[split.internal_role.to_numpy() == "U"], d.loc[split.internal_role.to_numpy() == "H"]
    folds = split.loc[u.sampleid, "u_teacher_fold"].to_numpy()
    common = teacher_kwargs(tmp_path / "oof")
    q = teacher_oof(u, folds, ["PRS_A", "PRS_B"], **common)
    h = h.copy()
    h.y = 1 - h.y  # Fixed split: this cannot enter any U teacher fit/cache key.
    q2 = teacher_oof(u, folds, ["PRS_A", "PRS_B"], **common)
    np.testing.assert_array_equal(q, q2)
    saved = pd.concat([pd.read_parquet(p) for p in (tmp_path / "oof").glob("*/predictions.parquet")])
    assert saved.sampleid.is_unique and set(saved.sampleid) == set(u.sampleid)
    altered = u.copy()
    altered.y = 1 - altered.y
    with pytest.raises(ValueError, match="provenance mismatch"):
        teacher_oof(altered, folds, ["PRS_A", "PRS_B"], **common)
    with pytest.raises(ValueError, match="prediction-only"):
        teacher_predict(u, h, ["PRS_A", "PRS_B"], fold_id="bad", **teacher_kwargs(tmp_path / "bad"))


def test_evaluation_label_mutation_changes_only_metrics(data, tmp_path):
    d, prep, x = data
    e = synthetic(40, 1000)
    q = teacher_predict(d, e.drop(columns="y"), ["PRS_A", "PRS_B"], fold_id="D_to_E",
                        **teacher_kwargs(tmp_path / "teacher"))
    model, _ = fit_student(x, d.y.to_numpy(), None, lam=0, gate_ridge=0.001, seed=17,
                           options=Optimization(), epochs=10)
    before = predict(model, prep.transform(e))["probability"]
    e.y = 1 - e.y
    q2 = teacher_predict(d, e.drop(columns="y"), ["PRS_A", "PRS_B"], fold_id="D_to_E",
                         **teacher_kwargs(tmp_path / "teacher"))
    np.testing.assert_array_equal(q, q2)
    np.testing.assert_array_equal(before, predict(model, prep.transform(e))["probability"])
    assert basic_metrics(e.y, before)["log_loss"] != basic_metrics(1 - e.y, before)["log_loss"]


def test_candidate_selection_averages_seeds_and_pure_uses_fidelity():
    rows = []
    for lam in [0, 0.25, 0.5, 0.75, 1]:
        for ridge in [1e-4, 1e-3, 1e-2]:
            for seed, epoch in zip([17, 29, 43], [20, 40, 60], strict=True):
                rows.append({"lambda": lam, "gate_ridge": ridge, "seed": seed,
                             "selection_metric": 0.1 if lam == 1 else 0.5,
                             "true_label_log_loss": 0.9 if lam == 1 else 0.5, "best_epoch": epoch})
    choices, policy = select_candidates(pd.DataFrame(rows), Optimization())
    assert policy == "S_SUP"  # Pure fidelity CE must not be sorted against true-label CE.
    assert choices["S_KD_MIX"]["lambda"] == 0.25
    assert choices["S_SUP"]["gate_ridge"] == 1e-2
    assert choices["S_SUP"]["refit_epochs"] == 40


def test_metrics_extremes_single_class_and_paired_cluster_bootstrap():
    y = np.tile([0, 1], 10)
    p = np.linspace(0, 1, len(y))
    assert np.isfinite(basic_metrics(y, p)["log_loss"])
    assert np.isfinite(fidelity(p, 1 - p)["bernoulli_kl"])
    assert np.isnan(basic_metrics(np.zeros(20), p)["auroc"])
    assert calibration(np.zeros(20), p)["calibration_status"] == "not_estimable"
    assert np.isnan(fidelity(p, np.full(20, 0.5))["logit_r2"])
    same = {arm: p for arm in ["META", "S_SUP", "S_KD_PURE", "S_KD_MIX", "T_MATCHED"]}
    pairs = paired_bootstrap(y, same, np.arange(20) // 2, 20, 42)
    np.testing.assert_allclose(pairs[["delta", "ci_low", "ci_high"]], 0)


def test_strict_config_unknown_keys_and_real_v1_settings(tmp_path):
    cfg = load_config("configs/prs_distillation_v1.yaml")
    assert cfg.optimization.seeds == [17, 29, 43]
    path = tmp_path / "config.yaml"
    path.write_text("teacher_estimtors: 16\n")
    with pytest.raises(ValueError, match="Unknown"):
        load_config(path)
    path.write_text("optimization:\n  max_epohs: 20\n")
    with pytest.raises(ValueError, match="Unknown"):
        load_config(path)


def test_full_cohort_retains_unresolved_patients_and_only_relaxes_e_boundary(tmp_path):
    d, e = synthetic(120), synthetic(40, 1000).drop(columns="y")
    e.loc[0, "kinship_group"] = d.loc[0, "kinship_group"]
    known = d.iloc[:-2][["sampleid", "kinship_group"]]
    restored = attach_full_sensitivity_groups(d.drop(columns="kinship_group"), known)
    assert restored.sampleid.tolist() == d.sampleid.tolist()
    assert restored.kinship_group.str.startswith("UNRESOLVED_QC::").sum() == 2
    assert restored.iloc[-2:].kinship_group.nunique() == 2
    with pytest.raises(ValueError, match="kinship_group"):
        make_split_manifest(restored, e, 42)
    splits, info = make_split_manifest(restored, e, 42, allow_evaluation_kinship_overlap=True)
    training = splits.loc[splits.split == "official_train"]
    assert training.groupby("kinship_group").final_teacher_fold.nunique().max() == 1
    assert training.groupby("kinship_group").internal_role.nunique().max() == 1
    assert info["evaluation_kinship_overlap_allowed"]
    kwargs = teacher_kwargs(tmp_path / "allowed")
    kwargs["contract"]["cohort_policy"] = "original_full_sensitivity"
    teacher_predict(d, e, ["PRS_A", "PRS_B"], fold_id="D_to_E", **kwargs)
    with pytest.raises(ValueError, match="kinship_group"):
        teacher_predict(d, e, ["PRS_A", "PRS_B"], fold_id="U_to_H", **kwargs)
    duplicate = e.copy()
    duplicate.loc[0, "sampleid"] = d.loc[0, "sampleid"]
    with pytest.raises(ValueError, match="sampleid"):
        teacher_predict(d, duplicate, ["PRS_A", "PRS_B"], fold_id="D_to_E", **kwargs)


def test_historical_teacher_adapter_uses_exact_original_matrix_order(tmp_path):
    import joblib

    from decipehr.tabpfn_ensemble import TabPFNFeatureTransformer

    d, e = synthetic(120), synthetic(40, 1000).drop(columns="y")
    historical = TabPFNFeatureTransformer.fit(
        d, prs_cols=["PRS_A", "PRS_B"], covariates=[AGE, "sex", *PCS], include_covariates=True,
    )
    class CheckingTeacher(FakeTeacher):
        def fit(self, x, y):
            np.testing.assert_array_equal(x, historical.transform(d).to_numpy())
            return super().fit(x, y)

        def predict_proba(self, x):
            np.testing.assert_array_equal(x, historical.transform(e).to_numpy())
            return super().predict_proba(x)

    kwargs = teacher_kwargs(tmp_path / "historic")
    kwargs["contract"]["feature_pipeline"] = "historical_external"
    kwargs["factory"] = CheckingTeacher
    teacher_predict(d, e, ["PRS_A", "PRS_B"], fold_id="D_to_E", **kwargs)
    replay = joblib.load(tmp_path / "historic/historical_teacher_preprocessing.joblib")
    assert replay.output_columns == historical.output_columns
    assert replay.output_columns.index("cov__sex__Female") < replay.output_columns.index("cov__PC1_sd")
    cfg = load_config("configs/prs_distillation_fulltrain.yaml")
    assert cfg.teacher_seed == 42 and cfg.cohort_policy == "original_full_sensitivity"
    assert cfg.teacher_preprocessing == "historical_external"


def test_full_synthetic_phase_a_and_prediction_only_e(tmp_path, monkeypatch):
    """Run all 45 selections, lock, refits and ablations with a clearly synthetic teacher."""
    from decipehr.prs_distillation import experiment

    d, e = synthetic(80), synthetic(40, 1000).drop(columns="y")
    manifest = pd.DataFrame({"column": ["PRS_A", "PRS_B"], "trait": "synthetic"})
    trait_root = tmp_path / "T2D"
    trait_root.mkdir()
    manifest.to_csv(trait_root / "feature_manifest.tsv", sep="\t", index=False)
    splits, _ = make_split_manifest(d, e, 42)
    splits.to_parquet(tmp_path / "split_manifest.parquet", index=False)
    cfg = Config(optimization=Optimization(max_epochs=10, evaluate_every=5, patience=5),
                 evaluation_path=str(tmp_path / "synthetic_evaluation.csv"), bootstrap_replicates=10)
    monkeypatch.setattr(experiment, "verify_run", lambda *_: ({}, {"git_commit": "synthetic", "code_sha256": {}}))
    monkeypatch.setattr(experiment, "load_inputs", lambda *_: (d, e, manifest))
    monkeypatch.setattr(experiment, "teacher_contract", lambda _: {"backend": "SYNTHETIC_TEST_ONLY"})
    monkeypatch.setattr(experiment, "make_teacher", lambda _: FakeTeacher())
    out = experiment.run_core(tmp_path, cfg, "T2D")
    lock = json.loads((out / "selection_lock.json").read_text())
    assert set(lock["arms"]) == {"META", "S_SUP", "S_KD_PURE", "S_KD_MIX"}
    assert len(pd.read_csv(out / "tuning_results.csv")) == 45
    result = pd.read_parquet(out / "direct_predictions.parquet")
    assert "y" not in result and len(result) == len(e) * 13
    experiment.run_ablations(tmp_path, cfg, "T2D")
    assert len(pd.read_parquet(out / "ablation_predictions.parquet")) == len(e) * 6
    # Resuming a complete run only verifies saved files, without refitting.
    monkeypatch.setattr(experiment, "make_teacher", lambda _: pytest.fail("Unexpected training on resume"))
    assert experiment.run_core(tmp_path, cfg, "T2D") == out
    assert not (out / "official_stage2").exists()
    # Report-only can see E.y, but must never train, select, recalibrate or open blind files.
    from decipehr.external_pgs import sha256_file
    from decipehr.prs_distillation.reporting import report

    labels = synthetic(40, 1000)[["sampleid", "y"]].rename(columns={"y": "T2D"})
    labels.to_csv(cfg.evaluation_path, index=False)
    frozen = {p: sha256_file(out / p) for p in ["selection_lock.json", "direct_predictions.parquet",
                                               "ablation_predictions.parquet"]}
    metrics, _ = report(tmp_path, cfg, "T2D")
    assert set(metrics.arm) == {*experiment.CORE_ARMS, *experiment.ABLATIONS, "T_MATCHED"}
    labels.T2D = 1 - labels.T2D
    labels.to_csv(cfg.evaluation_path, index=False)
    mutated_metrics, _ = report(tmp_path, cfg, "T2D")
    assert not np.array_equal(metrics.log_loss, mutated_metrics.log_loss)
    assert frozen == {p: sha256_file(out / p) for p in frozen}
    assert len(list((out / "figures").glob("*.png"))) == 4
