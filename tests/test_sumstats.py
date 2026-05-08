import numpy as np
import pandas as pd

from decipehr.io import read_table
from decipehr.sumstats import standardize_sumstats, standardize_sumstats_file_streaming


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


def test_standardize_sumstats_file_streaming_writes_chunks(tmp_path):
    raw = pd.DataFrame(
        {
            "rsid": ["rs1", "rs2"],
            "chromosome": [1, 1],
            "position": [100, 200],
            "effect_allele": ["a", "g"],
            "other_allele": ["c", "t"],
            "odds_ratio": [2.0, 1.0],
            "p_value": [0.01, 0.5],
        }
    )
    input_path = tmp_path / "sumstats.tsv.gz"
    output_path = tmp_path / "sumstats.standardized.tsv.gz"
    raw.to_csv(input_path, sep="\t", index=False)

    n_variants = standardize_sumstats_file_streaming(input_path, output_path, chunksize=1)
    out = read_table(output_path)

    assert n_variants == 2
    assert list(out.columns[:7]) == ["SNP", "CHR", "BP", "A1", "A2", "P", "BETA"]
    assert out["SNP"].tolist() == ["rs1", "rs2"]


def test_standardize_sumstats_can_infer_decipehr_id_from_hm_variant_id():
    raw = pd.DataFrame(
        {
            "hm_variant_id": ["1_817186_G_A"],
            "hm_chrom": [1],
            "hm_pos": [817186],
            "hm_effect_allele": ["A"],
            "hm_other_allele": ["G"],
            "hm_beta": [0.2],
            "p_value": [0.01],
        }
    )

    out = standardize_sumstats(raw, variant_id_strategy="decipehr")

    assert out.loc[0, "SNP"] == "c1_817186_snv_G_A"
    assert out.loc[0, "A1"] == "A"


def test_standardize_sumstats_can_infer_decipehr_id_from_alt_column():
    raw = pd.DataFrame(
        {
            "rsid": ["rs1"],
            "chromosome": [1],
            "base_pair_location": [48327],
            "effect_allele": ["C"],
            "other_allele": ["A"],
            "alt": ["A"],
            "odds_ratio": [1.5],
            "p_value": [0.01],
        }
    )

    out = standardize_sumstats(raw, variant_id_strategy="decipehr")

    assert out.loc[0, "SNP"] == "c1_48327_snv_C_A"
    assert out.loc[0, "A1"] == "C"
