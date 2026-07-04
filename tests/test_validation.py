import pandas as pd
import pytest

from decipehr.validation import (
    iter_bootstrap_indices,
    iter_repeated_holdout_indices,
    prepare_official_validation_phenotype,
    summarize_repeated_metrics,
)


def test_prepare_official_validation_phenotype_uses_validation_covariate_order():
    labelled = pd.DataFrame(
        {
            "sampleid": ["T1", "V2", "V1"],
            "cad": [1, 0, 1],
            "bc": [0, 1, 0],
            "age": [10.0, 20.0, 30.0],
        }
    )
    validation_covariates = pd.DataFrame(
        {
            "sampleid": ["V1", "V2"],
            "age": [31.0, 21.0],
        }
    )

    out = prepare_official_validation_phenotype(
        labelled,
        validation_covariates,
        patient_col="sampleid",
        label_cols=["cad", "bc"],
        covariate_cols=["age"],
    )

    assert out.to_dict("records") == [
        {"sampleid": "V1", "cad": 1, "bc": 0, "age": 31.0},
        {"sampleid": "V2", "cad": 0, "bc": 1, "age": 21.0},
    ]


def test_prepare_official_validation_phenotype_errors_on_missing_released_label():
    labelled = pd.DataFrame({"sampleid": ["V1"], "cad": [1]})
    validation_covariates = pd.DataFrame({"sampleid": ["V1", "V2"], "age": [1.0, 2.0]})

    with pytest.raises(ValueError, match="missing released labels"):
        prepare_official_validation_phenotype(
            labelled,
            validation_covariates,
            patient_col="sampleid",
            label_cols=["cad"],
            covariate_cols=["age"],
        )


def test_iter_repeated_holdout_indices_respects_eval_size_and_repeats():
    frame = pd.DataFrame({"cad": [0, 1] * 20})

    splits = list(
        iter_repeated_holdout_indices(
            frame,
            label_col="cad",
            eval_size=10,
            n_repeats=5,
            seed=7,
        )
    )

    assert [repeat for repeat, _, _ in splits] == [0, 1, 2, 3, 4]
    for _, fit_idx, eval_idx in splits:
        assert len(eval_idx) == 10
        assert len(fit_idx) == 30
        assert set(fit_idx).isdisjoint(set(eval_idx))
        assert frame.iloc[eval_idx]["cad"].sum() == 5


def test_iter_bootstrap_indices_resamples_with_replacement():
    samples = list(iter_bootstrap_indices(n=5, n_resamples=3, seed=7))

    assert [repeat for repeat, _ in samples] == [0, 1, 2]
    for _, indices in samples:
        assert len(indices) == 5
        assert indices.min() >= 0
        assert indices.max() < 5


def test_iter_bootstrap_indices_allows_zero_resamples():
    assert list(iter_bootstrap_indices(n=5, n_resamples=0, seed=7)) == []


def test_summarize_repeated_metrics_flattens_repeat_statistics():
    metrics = pd.DataFrame(
        {
            "disease": ["cad", "cad", "cad"],
            "method": ["covariate_only"] * 3,
            "repeat": [0, 1, 2],
            "auc": [0.6, 0.7, 0.8],
            "n": [20, 20, 20],
        }
    )

    summary = summarize_repeated_metrics(metrics, ["disease", "method"])

    row = summary.iloc[0].to_dict()
    assert row["n_repeats"] == 3
    assert row["auc_mean"] == pytest.approx(0.7)
    assert row["auc_median"] == pytest.approx(0.7)
    assert row["n_mean"] == pytest.approx(20.0)
