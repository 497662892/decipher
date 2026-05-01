import pandas as pd

from decipehr.splits import make_patient_folds


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

