import subprocess
import sys

import pandas as pd
import pytest


def _write_summary(path, rows):
    path.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path / "summary_metrics.csv", index=False)


def test_collect_model_performance_manifest_writes_ranked_delta_and_wide_tables(tmp_path):
    old_dir = tmp_path / "tabpfn3_old" / "clip1e-3_ne16" / "official_validation_covariate_adjusted"
    new_dir = tmp_path / "tabpfn3_plus" / "clip1e-3_ne16" / "official_validation_covariate_adjusted"
    _write_summary(
        old_dir,
        [
            {
                "model_name": "tabpfn",
                "disease": "cad",
                "method": "prs_covariate_cad",
                "auc_mean": 0.80,
                "auc_std": 0.01,
                "brier_mean": 0.18,
                "top10_vs_lower90_or_mean": 5.0,
            },
            {
                "model_name": "tabpfn",
                "disease": "bc",
                "method": "prs_covariate_bc",
                "auc_mean": 0.70,
                "auc_std": 0.02,
                "brier_mean": 0.15,
                "top10_vs_lower90_or_mean": 2.5,
            },
        ],
    )
    _write_summary(
        new_dir,
        [
            {
                "model_name": "tabpfn",
                "disease": "cad",
                "method": "prs_covariate_cad",
                "auc_mean": 0.81,
                "auc_std": 0.01,
                "brier_mean": 0.17,
                "top10_vs_lower90_or_mean": 5.5,
            },
            {
                "model_name": "tabpfn",
                "disease": "bc",
                "method": "prs_covariate_bc",
                "auc_mean": 0.69,
                "auc_std": 0.02,
                "brier_mean": 0.151,
                "top10_vs_lower90_or_mean": 2.1,
            },
        ],
    )
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame(
        [
            {
                "model_id": "tabpfn3_old",
                "model_group": "tabpfn",
                "setting": "nonannotated",
                "path": str(old_dir),
            },
            {
                "model_id": "tabpfn3_plus",
                "model_group": "tabpfn",
                "setting": "plus_megaprs_sbayesrc",
                "path": str(new_dir),
            },
        ]
    ).to_csv(manifest, index=False)

    out = tmp_path / "performance.csv"
    wide = tmp_path / "performance.wide.csv"
    subprocess.run(
        [
            sys.executable,
            "scripts/collect_model_performance.py",
            "--manifest",
            str(manifest),
            "--baseline-model-id",
            "tabpfn3_old",
            "--rank-metric",
            "auc",
            "--out",
            str(out),
            "--wide-out",
            str(wide),
        ],
        check=True,
    )

    ranked = pd.read_csv(out)
    best = pd.read_csv(out.with_name("performance.best.csv"))
    wide_frame = pd.read_csv(wide)

    assert set(ranked["model_id"]) == {"tabpfn3_old", "tabpfn3_plus"}
    assert ranked.loc[
        (ranked["model_id"] == "tabpfn3_plus") & (ranked["disease"] == "cad"),
        "delta_auc",
    ].iloc[0] == pytest.approx(0.01)
    assert best.set_index("disease").loc["cad", "model_id"] == "tabpfn3_plus"
    assert best.set_index("disease").loc["bc", "model_id"] == "tabpfn3_old"
    assert {"cad_auc", "bc_auc"}.issubset(set(wide_frame.columns))


def test_collect_model_performance_model_arg_reads_point_metrics(tmp_path):
    validation_dir = tmp_path / "official_validation_pt" / "cad"
    validation_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "model_name": "pt",
                "disease": "cad",
                "method": "covariate_only",
                "auc": 0.79,
            },
            {
                "model_name": "pt",
                "disease": "cad",
                "method": "prs_covariate_cad",
                "auc": 0.795,
                "brier": 0.177,
            },
        ]
    ).to_csv(validation_dir / "point_metrics.csv", index=False)

    out = tmp_path / "pt_performance.csv"
    subprocess.run(
        [
            sys.executable,
            "scripts/collect_model_performance.py",
            "--model",
            f"pt={validation_dir}",
            "--metrics-kind",
            "point",
            "--rank-metric",
            "auc",
            "--out",
            str(out),
        ],
        check=True,
    )

    ranked = pd.read_csv(out)
    assert ranked["model_id"].tolist() == ["pt"]
    assert ranked["method"].tolist() == ["prs_covariate_cad"]
    assert ranked["auc"].tolist() == [0.795]
