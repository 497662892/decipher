import json
import subprocess
import sys

import pandas as pd

from decipehr.ancestry_distance import (
    compute_1000g_distances,
    make_hard_labels,
    make_soft_weights,
)


def _reference_scores() -> pd.DataFrame:
    rows = []
    centers = {
        "AFR": (-4.0, 0.0),
        "AMR": (-2.0, 2.0),
        "EAS": (0.0, 4.0),
        "EUR": (3.0, 0.0),
        "SAS": (1.5, 2.5),
    }
    for group, (pc1, pc2) in centers.items():
        for index, offset in enumerate([-0.08, 0.08]):
            rows.append(
                {
                    "FID": group,
                    "IID": f"{group}{index}",
                    "PC1_AVG": pc1 + offset,
                    "PC2_AVG": pc2 - offset,
                }
            )
    return pd.DataFrame(rows)


def _panel() -> pd.DataFrame:
    ref = _reference_scores()
    panel = ref[["IID"]].copy()
    panel["pop"] = panel["IID"].str[:3]
    panel["super_pop"] = panel["IID"].str[:3]
    return panel


def test_compute_1000g_distances_hard_labels_and_soft_weights():
    ref = _reference_scores()
    target = pd.DataFrame(
        {
            "FID": ["T", "T"],
            "IID": ["target_eur", "target_afr"],
            "PC1_AVG": [3.1, -4.1],
            "PC2_AVG": [0.1, 0.0],
        }
    )

    distances, summary = compute_1000g_distances(
        ref,
        target,
        _panel(),
        pc_cols=["PC1_AVG", "PC2_AVG"],
    )
    hard = make_hard_labels(distances, margin_threshold=0.0)
    soft = make_soft_weights(distances, tau=1.0)

    assert distances["nearest_1000G_superpop"].tolist() == ["EUR", "AFR"]
    assert hard["hard_label"].tolist() == ["EUR-like", "AFR-like"]
    assert soft.loc[0, "w_EUR"] == soft.loc[0, [c for c in soft.columns if c.startswith("w_")]].max()
    assert summary["reference_group_counts"] == {"AFR": 2, "AMR": 2, "EAS": 2, "EUR": 2, "SAS": 2}


def test_compute_ancestry_distances_cli_writes_tables_and_svg(tmp_path):
    ref_path = tmp_path / "1000g.projected.sscore"
    target_path = tmp_path / "target.projected.sscore"
    panel_path = tmp_path / "1000g_panel.tsv"
    _reference_scores().rename(columns={"FID": "#FID"}).to_csv(ref_path, sep="\t", index=False)
    pd.DataFrame(
        {
            "#FID": ["T"],
            "IID": ["target_eur"],
            "PC1_AVG": [3.1],
            "PC2_AVG": [0.1],
        }
    ).to_csv(target_path, sep="\t", index=False)
    _panel().to_csv(panel_path, sep="\t", index=False)
    out_dir = tmp_path / "ancestry"

    subprocess.run(
        [
            sys.executable,
            "scripts/compute_ancestry_distances.py",
            "--ref-sscore",
            str(ref_path),
            "--target-sscore",
            str(target_path),
            "--panel",
            str(panel_path),
            "--pc-cols",
            "PC1_AVG,PC2_AVG",
            "--out-dir",
            str(out_dir),
        ],
        check=True,
    )

    distances = pd.read_csv(out_dir / "target_1000G_distances_pc2.tsv", sep="\t")
    hard = pd.read_csv(out_dir / "target_1000G_hard_labels_pc2_margin0.5.tsv", sep="\t")
    soft = pd.read_csv(out_dir / "target_1000G_soft_weights_pc2_tau1.tsv", sep="\t")
    summary = json.loads((out_dir / "target_1000G_distances_pc2.summary.json").read_text())
    svg = (out_dir / "target_1000G_PC1_PC2_1000G_centers_hard_assignments.svg").read_text(
        encoding="utf-8"
    )

    assert distances.loc[0, "nearest_1000G_superpop"] == "EUR"
    assert hard.loc[0, "hard_label"] == "EUR-like"
    assert soft.loc[0, "soft_nearest_group"] == "EUR-like"
    assert summary["pc_cols"] == ["PC1_AVG", "PC2_AVG"]
    assert "1000G center" in svg
    assert "EUR center" in svg
