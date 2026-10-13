import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from decipehr.external_pgs_benchmark import PCS, evaluate_official_validation
from decipehr.external_pgs_holdout import predict_holdout_wrapper, prepare_holdout_data


def example():
    rng = np.random.default_rng(21)
    n = 90
    frame = pd.DataFrame({"sampleid": [f"synthetic_{i}" for i in range(n)],
                          "age_sd": rng.normal(size=n), "sex": ["Female", "Male"] * (n // 2),
                          "bc": np.tile([0, 0, 1, 1, 0, 1], n // 6),
                          "PGS000001": rng.normal(size=n), "hard_label": "EUR-like"})
    for pc in PCS:
        frame[pc] = rng.normal(size=n)
    return frame.iloc[:60].copy(), frame.iloc[60:].copy()


def test_unlabelled_wrapper_matches_existing_validation_exactly():
    context, labelled_query = example()
    options = dict(disease="bc", model_ids=["PGS000001"],
                   tabpfn_factory=lambda seed: LogisticRegression(random_state=seed, max_iter=2000))
    expected = evaluate_official_validation(context, labelled_query, **options).predictions
    query = labelled_query.drop(columns="bc")
    actual, oof, fits, _ = predict_holdout_wrapper(context, query, **options)
    np.testing.assert_array_equal(actual.y_prob, expected.y_prob)
    np.testing.assert_array_equal(actual.wrapper_prob, expected.wrapper_prob)
    assert "y_true" not in actual
    assert oof.fold.nunique() == 3 and len(oof) == len(context)
    assert len(fits) == 4
    with pytest.raises(ValueError, match="outcomes"):
        predict_holdout_wrapper(context, labelled_query, **options)
    with pytest.raises(ValueError, match="overlap"):
        predict_holdout_wrapper(context, context.drop(columns="bc"), **options)


def test_holdout_eligibility_missing_scores_and_overlap():
    context, query = example()
    all_rows = pd.concat([context, query])
    scores = all_rows[["sampleid", "PGS000001"]].rename(columns={"PGS000001": "prs"}).assign(
        disease="bc", model="PGS000001")
    options = dict(phenotype=context.drop(columns=["PGS000001", "hard_label"]),
                   holdout_covariates=query.drop(columns=["PGS000001", "hard_label", "bc"]),
                   context_scores=scores, holdout_scores=scores,
                   hard_labels=all_rows[["sampleid", "hard_label"]], disease="bc",
                   model_ids=["PGS000001"], eligibility_rules={"bc": {"column": "sex", "allowed_values": ["Female"]}})
    training, held = prepare_holdout_data(**options)
    assert len(training) == 30 and len(held) == 15
    assert set(held.sex) == {"Female"}
    bad_scores = scores.loc[scores.sampleid.ne(query.sampleid.iloc[0])]
    with pytest.raises(ValueError, match="Missing or nonfinite"):
        prepare_holdout_data(**{**options, "holdout_scores": bad_scores})
    with pytest.raises(ValueError, match="overlap"):
        prepare_holdout_data(**{**options, "holdout_covariates": options["phenotype"].drop(columns="bc")})


def test_variant_axis_ignores_info_but_detects_allele_changes(tmp_path):
    spec = importlib.util.spec_from_file_location("holdout_scoring", Path("scripts/score_external_pgs_holdout.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    a, b = tmp_path / "a.pvar", tmp_path / "b.pvar"
    a.write_text("#CHROM\tPOS\tID\tREF\tALT\n1\t12\trs1\tA\tG\n")
    b.write_text("##INFO=test\n#CHROM\tPOS\tID\tREF\tALT\tINFO\n1\t12\trs1\tA\tG\tAF=0.7\n")
    assert module.variant_axis(a) == module.variant_axis(b)
    b.write_text(b.read_text().replace("A\tG", "G\tA"))
    assert module.variant_axis(a) != module.variant_axis(b)
