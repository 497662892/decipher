import pandas as pd

from decipehr.splits import make_disease_specific_folds, make_patient_folds


def test_make_patient_folds_assigns_each_patient_once():
    df = pd.DataFrame(
        {
            "patient_id": [f"P{i:03d}" for i in range(40)],
            "d1": [0, 1] * 20,
            "d2": [0, 0, 1, 1] * 10,
            "d3": [1, 0, 0, 0] * 10,
        }
    )
    folds = make_patient_folds(df, "patient_id", ["d1", "d2", "d3"], n_folds=5, seed=42)
    assert len(folds) == 40
    assert folds["patient_id"].nunique() == 40
    assert set(folds["fold"]) == {0, 1, 2, 3, 4}


def test_make_disease_specific_folds_balances_each_disease_independently():
    rows = []
    for i in range(180):
        rows.append(
            {
                "patient_id": f"P{i:03d}",
                "cad": i % 2,
                "T2D": (i // 2) % 2,
                "bc": i % 3 == 0,
                "ancestry_cluster": i % 3,
                "age_sd": float(i % 30),
                "sex": "Female" if i % 2 == 0 else "Male",
            }
        )
    df = pd.DataFrame(rows)

    folds = make_disease_specific_folds(
        df,
        patient_col="patient_id",
        label_cols=["cad", "T2D"],
        cluster_col="ancestry_cluster",
        age_col="age_sd",
        sex_col="sex",
        sex_balance_labels=["cad", "T2D"],
        n_folds=5,
        seed=42,
        n_seed_candidates=5,
    )

    assert set(folds["disease"]) == {"cad", "T2D"}
    for disease, frame in folds.groupby("disease"):
        assert disease in {"cad", "T2D"}
        assert frame["patient_id"].nunique() == 180
        assert set(frame["fold"]) == {0, 1, 2, 3, 4}


def test_make_disease_specific_folds_filters_bc_eligibility():
    rows = []
    for i in range(120):
        rows.append(
            {
                "patient_id": f"P{i:03d}",
                "bc": i % 2,
                "ancestry_cluster": i % 3,
                "age_sd": float(i % 20),
                "sex": "Female" if i < 80 else "Male",
            }
        )
    df = pd.DataFrame(rows)

    folds = make_disease_specific_folds(
        df,
        patient_col="patient_id",
        label_cols=["bc"],
        cluster_col="ancestry_cluster",
        age_col="age_sd",
        sex_col="sex",
        sex_balance_labels=[],
        eligibility_rules={"bc": {"column": "sex", "allowed_values": ["Female"]}},
        n_folds=5,
        seed=42,
        n_seed_candidates=5,
    )

    observed_ids = set(folds["patient_id"])
    female_ids = set(df.loc[df["sex"] == "Female", "patient_id"])
    assert observed_ids == female_ids
    assert set(folds["fold"]) == {0, 1, 2, 3, 4}
