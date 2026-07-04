import pandas as pd

from decipehr.ancestry import (
    ancestry_cluster_report,
    make_ancestry_clusters,
    write_pc_scatter_svg,
)


def test_make_ancestry_clusters_assigns_one_cluster_per_patient(tmp_path):
    df = pd.DataFrame(
        {
            "patient_id": [f"P{i:03d}" for i in range(12)],
            "PC1": [-4, -3, -3, -4, 0, 0, 1, 1, 4, 5, 5, 4],
            "PC2": [0, 0, 1, 1, 4, 5, 5, 4, 0, 0, 1, 1],
            "age": list(range(12)),
            "sex": ["Female", "Male"] * 6,
            "cad": [0, 1] * 6,
        }
    )

    clusters = make_ancestry_clusters(
        df,
        patient_col="patient_id",
        pc_cols=["PC1", "PC2"],
        n_clusters=3,
        seed=42,
        n_init=10,
    )
    report = ancestry_cluster_report(
        df,
        clusters,
        patient_col="patient_id",
        pc_cols=["PC1", "PC2"],
        label_cols=["cad"],
        age_col="age",
        sex_col="sex",
    )
    plot = tmp_path / "pc.svg"
    write_pc_scatter_svg(
        df,
        clusters,
        plot,
        patient_col="patient_id",
        pc1_col="PC1",
        pc2_col="PC2",
    )

    assert len(clusters) == 12
    assert clusters["patient_id"].nunique() == 12
    assert set(clusters["ancestry_cluster"]) == {0, 1, 2}
    assert report["n"].sum() == 12
    assert "PC1_mean" in report.columns
    assert plot.read_text(encoding="utf-8").startswith("<svg")
