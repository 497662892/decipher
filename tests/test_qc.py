import pandas as pd

from decipehr.qc import (
    feature_count_report,
    feature_summary,
    filter_info_score_frame,
    parse_maybe_percent,
)


def test_parse_maybe_percent_handles_fraction_and_percent_strings():
    out = parse_maybe_percent(pd.Series(["0.25", "25%", "#NA"]))

    assert out.iloc[0] == 0.25
    assert out.iloc[1] == 0.25
    assert pd.isna(out.iloc[2])


def test_filter_info_score_frame_filters_by_info_and_maf():
    df = pd.DataFrame(
        {
            "col0": ["v1", "v2", "v3", "v4"],
            "MAF": ["1.5%", "0.2%", "20%", "50%"],
            "info_IMPUTE": [0.9, 0.95, 0.2, 0.99],
        }
    )

    out = filter_info_score_frame(df, min_info=0.8, min_maf=0.01)

    assert out.tolist() == ["v1", "v4"]


def test_filter_info_score_frame_computes_maf_from_allele_frequency():
    df = pd.DataFrame(
        {
            "ID": ["v1", "v2", "v3", "v4"],
            "AF": [0.99, 0.995, 0.5, 0.2],
            "info_IMPUTE": [0.9, 0.9, 0.2, 0.95],
        }
    )

    out = filter_info_score_frame(
        df,
        min_info=0.8,
        min_maf=0.01,
        max_af=0.99,
        variant_col="ID",
        freq_col="AF",
        freq_is_maf=False,
    )

    assert out.tolist() == ["v1", "v4"]


def test_feature_summary_marks_roles_and_counts_missing_values():
    df = pd.DataFrame(
        {
            "sampleid": ["s1", "s2", "s3"],
            "cad": [0, 1, 0],
            "age_sd": [0.1, None, 1.2],
            "batch": ["a", "a", "b"],
        }
    )

    out = feature_summary(df, "sampleid", ["cad"], ["age_sd"])

    age = out.loc[out["column"] == "age_sd"].iloc[0]
    batch = out.loc[out["column"] == "batch"].iloc[0]
    assert age["role"] == "covariate"
    assert age["missing"] == 1
    assert age["non_missing"] == 2
    assert batch["role"] == "other"
    assert batch["unique_non_missing"] == 2


def test_feature_count_report_counts_retained_feature_roles():
    df = pd.DataFrame(
        {
            "sampleid": ["s1", "s2"],
            "cad": [0, 1],
            "age_sd": [0.1, 0.2],
            "batch": ["a", "b"],
        }
    )

    out = feature_count_report(df, "sampleid", ["cad"], ["age_sd"])
    counts = dict(zip(out["metric"], out["value"], strict=True))

    assert counts["n_rows"] == 2
    assert counts["n_columns"] == 4
    assert counts["n_label_columns"] == 1
    assert counts["n_covariate_columns"] == 1
    assert counts["n_other_columns"] == 1
    assert counts["n_model_feature_columns"] == 2
