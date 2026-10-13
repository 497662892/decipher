import runpy
from pathlib import Path

import numpy as np
import pandas as pd

from decipehr.internal_prs_benchmark import METHODS


def test_identical_predictions_have_zero_cluster_paired_differences():
    script = Path(__file__).resolve().parents[1] / "scripts/report_internal_prs_benchmark.py"
    compare = runpy.run_path(str(script))["cluster_comparisons"]
    base = pd.DataFrame({"sampleid": [f"s{i}" for i in range(30)],
                         "kinship_group": [str(i // 3) for i in range(30)],
                         "label": np.tile([0, 1, 0], 10),
                         "probability": np.linspace(.1, .9, 30)})
    frame = pd.concat([base.assign(method=method) for method in METHODS])
    result = compare(frame, n_bootstraps=9, seed=42)
    assert set(result.n_clusters) == {10}
    assert set(result.n_valid_bootstraps) == {9}
    assert (result[["delta", "ci_low", "ci_high"]] == 0).all().all()
    assert (result.p_centered_bootstrap == 1).all()


def test_replicating_people_within_every_family_does_not_shrink_cluster_ci():
    script = Path(__file__).resolve().parents[1] / "scripts/report_internal_prs_benchmark.py"
    compare = runpy.run_path(str(script))["cluster_comparisons"]
    rng = np.random.default_rng(8)
    base = pd.DataFrame({"sampleid": [f"s{i}" for i in range(30)],
                         "kinship_group": [str(i // 3) for i in range(30)],
                         "label": np.tile([0, 1, 0], 10),
                         "probability": rng.uniform(.1, .8, 30)})
    frame = pd.concat([base.assign(method=method) for method in METHODS], ignore_index=True)
    full = frame.method == "tabpfn_full"
    frame.loc[full, "probability"] = np.clip(frame.loc[full, "probability"] + .15 * frame.loc[full, "label"], .01, .99)
    duplicate = frame.copy()
    duplicate["sampleid"] = "duplicate_" + duplicate.sampleid
    first = compare(frame, n_bootstraps=19, seed=42)
    second = compare(pd.concat([frame, duplicate]), n_bootstraps=19, seed=42)
    np.testing.assert_allclose(first[["delta", "ci_low", "ci_high"]],
                               second[["delta", "ci_low", "ci_high"]], atol=1e-12)
