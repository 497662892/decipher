import pandas as pd

from decipehr.io import read_table, write_table


def test_read_table_uses_inner_suffix_for_compressed_tsv(tmp_path):
    path = tmp_path / "phenotype.tsv.gz"
    pd.DataFrame({"sampleid": ["P1", "P2"], "note": ["a,b", "c,d"]}).to_csv(
        path, sep="\t", index=False
    )

    out = read_table(path)

    assert list(out.columns) == ["sampleid", "note"]
    assert out["note"].tolist() == ["a,b", "c,d"]


def test_write_table_uses_inner_suffix_for_compressed_tsv(tmp_path):
    path = tmp_path / "phenotype.tsv.gz"
    df = pd.DataFrame({"sampleid": ["P1", "P2"], "cad": [0, 1]})

    write_table(df, path)
    out = read_table(path)

    pd.testing.assert_frame_equal(out, df)
