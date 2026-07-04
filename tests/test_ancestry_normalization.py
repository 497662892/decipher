import pandas as pd

from decipehr.ancestry_normalization import normalize_by_hard_label, normalize_by_soft_weights


def test_normalize_by_hard_label_uses_training_group_stats():
    train = pd.DataFrame(
        {
            "sampleid": ["A1", "A2", "B1", "B2"],
            "prs_cad": [1.0, 3.0, 10.0, 14.0],
        }
    )
    validation = pd.DataFrame({"sampleid": ["A3", "B3"], "prs_cad": [5.0, 18.0]})
    hard = pd.DataFrame(
        {
            "sampleid": ["A1", "A2", "A3", "B1", "B2", "B3"],
            "hard_label": ["EUR-like", "EUR-like", "EUR-like", "AFR-like", "AFR-like", "AFR-like"],
        }
    )

    result = normalize_by_hard_label(
        train_scores=train,
        validation_scores=validation,
        hard_labels=hard,
        patient_col="sampleid",
        score_cols=["prs_cad"],
        min_group_size=1,
    )

    observed = dict(zip(result.validation["sampleid"], result.validation["prs_cad"], strict=True))
    assert observed["A3"] == 3.0
    assert observed["B3"] == 3.0


def test_normalize_by_soft_weights_uses_weighted_group_moments():
    train = pd.DataFrame(
        {
            "sampleid": ["A1", "A2", "B1", "B2"],
            "prs_cad": [1.0, 3.0, 10.0, 14.0],
        }
    )
    validation = pd.DataFrame({"sampleid": ["M1"], "prs_cad": [7.0]})
    weights = pd.DataFrame(
        {
            "sampleid": ["A1", "A2", "B1", "B2", "M1"],
            "w_AFR": [0.0, 0.0, 1.0, 1.0, 0.5],
            "w_EUR": [1.0, 1.0, 0.0, 0.0, 0.5],
        }
    )

    result = normalize_by_soft_weights(
        train_scores=train,
        validation_scores=validation,
        soft_weights=weights,
        patient_col="sampleid",
        score_cols=["prs_cad"],
        groups=("AFR", "EUR"),
        min_effective_weight=1,
    )

    value = float(result.validation.loc[0, "prs_cad"])
    assert abs(value) < 1e-12


def test_normalize_by_soft_weights_ignores_missing_training_scores():
    train = pd.DataFrame(
        {
            "sampleid": ["A1", "A2", "B1", "B2"],
            "prs_bc": [1.0, None, 10.0, 14.0],
        }
    )
    validation = pd.DataFrame({"sampleid": ["A3"], "prs_bc": [3.0]})
    weights = pd.DataFrame(
        {
            "sampleid": ["A1", "A2", "B1", "B2", "A3"],
            "w_AFR": [0.0, 0.0, 1.0, 1.0, 0.0],
            "w_EUR": [1.0, 1.0, 0.0, 0.0, 1.0],
        }
    )

    result = normalize_by_soft_weights(
        train_scores=train,
        validation_scores=validation,
        soft_weights=weights,
        patient_col="sampleid",
        score_cols=["prs_bc"],
        groups=("AFR", "EUR"),
        min_effective_weight=1,
    )

    assert result.parameters.loc[result.parameters["group"].eq("EUR"), "n_effective"].iloc[0] == 1.0
    assert "prs_bc" in result.validation.columns
