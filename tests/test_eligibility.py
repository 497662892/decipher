import pandas as pd

from decipehr.eligibility import eligible_counts, filter_eligible, load_eligibility_rules


def test_filter_eligible_keeps_only_configured_sex_for_breast_cancer():
    df = pd.DataFrame(
        {
            "patient_id": ["P1", "P2", "P3"],
            "sex": ["Female", "Male", "F"],
            "breast_cancer": [1, 0, 0],
        }
    )
    rules = {"breast_cancer": {"column": "sex", "allowed_values": ["Female", "F"]}}
    out = filter_eligible(df, "breast_cancer", rules)
    assert out["patient_id"].tolist() == ["P1", "P3"]


def test_eligible_counts_reports_rule_application():
    df = pd.DataFrame(
        {
            "patient_id": ["P1", "P2"],
            "sex": ["F", "M"],
            "breast_cancer": [1, 0],
            "cad": [0, 1],
        }
    )
    rules = {"breast_cancer": {"column": "sex", "allowed_values": ["F"]}}
    counts = eligible_counts(df, ["breast_cancer", "cad"], rules)
    observed = counts.set_index("disease")["n_eligible"].to_dict()
    assert observed == {"breast_cancer": 1, "cad": 2}


def test_filter_eligible_matches_integer_like_numeric_values():
    df = pd.DataFrame(
        {
            "patient_id": ["P1", "P2", "P3"],
            "sex": [2.0, 1.0, 2.0],
            "breast_cancer": [1, 0, 0],
        }
    )
    rules = {"breast_cancer": {"column": "sex", "allowed_values": [2]}}
    out = filter_eligible(df, "breast_cancer", rules)
    assert out["patient_id"].tolist() == ["P1", "P3"]


def test_repository_eligibility_config_filters_actual_bc_label():
    rules = load_eligibility_rules("configs/disease_eligibility.yaml")
    df = pd.DataFrame(
        {
            "sampleid": ["P1", "P2", "P3"],
            "sex": ["Female", "Male", "Unknown"],
            "bc": [1, 0, 0],
        }
    )

    out = filter_eligible(df, "bc", rules)

    assert out["sampleid"].tolist() == ["P1"]
