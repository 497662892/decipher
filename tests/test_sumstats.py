import numpy as np
import pandas as pd

from decipehr.sumstats import standardize_sumstats


def test_standardize_sumstats_converts_or_to_beta():
    raw = pd.DataFrame(
        {
            "rsid": ["rs1", "rs2"],
            "chromosome": [1, 1],
            "position": [100, 200],
            "effect_allele": ["a", "g"],
            "other_allele": ["c", "t"],
            "OR": [2.0, 1.0],
            "p_value": [0.01, 0.5],
        }
    )
    out = standardize_sumstats(raw)
    assert list(out.columns[:7]) == ["SNP", "CHR", "BP", "A1", "A2", "P", "BETA"]
    assert out.loc[out["SNP"] == "rs1", "A1"].item() == "A"
    assert np.isclose(out.loc[out["SNP"] == "rs1", "BETA"].item(), np.log(2.0))

