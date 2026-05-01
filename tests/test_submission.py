import pandas as pd

from decipehr.submission import validate_probability_columns, wide_submission


def test_wide_submission():
    pred = pd.DataFrame(
        {
            "patient_id": ["P1", "P1", "P2", "P2"],
            "disease": ["d1", "d2", "d1", "d2"],
            "y_prob": [0.1, 0.2, 0.8, 0.9],
        }
    )
    out = wide_submission(pred, "patient_id")
    validate_probability_columns(out, "patient_id")
    assert list(out.columns) == ["patient_id", "d1", "d2"]
    assert out.shape == (2, 3)

