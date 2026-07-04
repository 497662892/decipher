import pandas as pd

from decipehr.pt import (
    build_multithreshold_score_table,
    plink_sscore_columns,
    read_plink_clumps,
)


def test_build_multithreshold_score_table_uses_clumped_leads_and_thresholds():
    harmonized = pd.DataFrame(
        {
            "SNP": ["rs1", "rs2", "rs3"],
            "CHR": [1, 1, 1],
            "BP": [100, 200, 300],
            "A1": ["A", "C", "G"],
            "BETA": [0.2, -0.3, 0.4],
            "P": [0.01, 0.2, 0.001],
        }
    )
    clumps = pd.DataFrame({"SNP": ["rs1", "rs2"]})

    score, summary = build_multithreshold_score_table(
        harmonized,
        clumps,
        p_thresholds=[0.05, 1.0],
    )

    assert score.to_dict("records") == [
        {"SNP": "rs1", "A1": "A", "BETA_p0p05": 0.2, "BETA_p1": 0.2},
        {"SNP": "rs2", "A1": "C", "BETA_p0p05": 0.0, "BETA_p1": -0.3},
    ]
    assert summary["n_clumped_leads"] == 2
    assert summary["n_score_variants"] == 2
    assert summary["n_nonzero_by_threshold"] == {"BETA_p0p05": 1, "BETA_p1": 2}


def test_read_plink_clumps_accepts_plink2_id_column(tmp_path):
    clumps = tmp_path / "trait.clumps"
    clumps.write_text("#CHROM POS ID P TOTAL\n1 100 rs1 0.01 2\n1 200 rs2 0.2 1\n", encoding="utf-8")

    out = read_plink_clumps(clumps)

    assert out.to_dict("records") == [{"SNP": "rs1", "P": 0.01}, {"SNP": "rs2", "P": 0.2}]


def test_plink_sscore_columns_prefers_score_sums():
    cols = ["#IID", "SCORE1_AVG", "SCORE1_SUM", "SCORE2_AVG", "SCORE2_SUM"]

    assert plink_sscore_columns(cols, 2) == ["SCORE1_SUM", "SCORE2_SUM"]


def test_plink_sscore_columns_accepts_score_file_header_names():
    cols = ["#IID", "BETA_p1_AVG", "BETA_p1_SUM", "BETA_p0p05_AVG", "BETA_p0p05_SUM"]

    assert plink_sscore_columns(cols, 2, ["BETA_p1", "BETA_p0p05"]) == [
        "BETA_p1_SUM",
        "BETA_p0p05_SUM",
    ]
