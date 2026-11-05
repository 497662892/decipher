"""Optional real Hail VDS integration; run in the separate Hail/Spark environment."""

import json
import os
import runpy
import subprocess
import sys

import pytest

hl = pytest.importorskip("hail", reason="Hail/Spark extraction is tested in its separate environment")


def test_reference_blocks_missing_calls_multiallelic_and_roster(tmp_path):
    hl.init(master="local[2]", default_reference="GRCh38", tmp_dir=str(tmp_path / "tmp"),
            log=str(tmp_path / "hail.log"), quiet=True,
            spark_conf={"spark.ui.enabled": "false", "spark.driver.memory": "4g"})
    try:
        ids = hl.literal(["0001", "0002", "0003", "excluded"])
        reference = hl.utils.range_matrix_table(1, 4)
        reference = reference.key_rows_by(locus=hl.locus("chr1", 1))
        reference = reference.key_cols_by(s=ids[reference.col_idx])
        reference = reference.select_rows().select_cols().select_entries(LEN=250)
        reference = reference.filter_entries(reference.s == "0001")
        reference = reference.annotate_globals(ref_block_max_length=250)
        variant = hl.utils.range_matrix_table(3, 4)
        variant = variant.key_rows_by(locus=hl.locus("chr1", (variant.row_idx+1)*100),
            alleles=hl.if_else(variant.row_idx == 1, ["A", "C", "G"], ["A", "C"]))
        variant = variant.key_cols_by(s=ids[variant.col_idx])
        variant = variant.annotate_entries(LGT=hl.if_else((variant.row_idx == 1) & (variant.col_idx == 1), hl.call(1, 1), hl.call(0, 1)),
            LA=hl.if_else((variant.row_idx == 1) & (variant.col_idx == 1), [0, 2], [0, 1]))
        variant = variant.filter_entries((variant.col_idx != 0) & ~((variant.col_idx == 2) & (variant.row_idx == 0)))
        variant = variant.select_rows().select_cols().select_entries("LGT", "LA")
        vds = hl.vds.VariantDataset(reference, variant)
        vds.validate()
        samples = hl.Table.parallelize([hl.Struct(s=f"000{i}") for i in range(1, 4)], key="s")
        loci = hl.Table.parallelize([hl.Struct(locus=hl.Locus("chr1", p, "GRCh38")) for p in [100, 200]], key="locus")
        export = runpy.run_path("scripts/export_aou_prs_inputs.py")["export_chromosome"]
        prefix = str(tmp_path / "export")
        result = export(hl, vds, loci, samples, 1, prefix)
        assert result["n_patients"] == 3 and result["n_exported_variants"] == 3
        observed = hl.import_plink(prefix+".bed", prefix+".bim", prefix+".fam", reference_genome="GRCh38")
        rows = observed.entries().select("GT").collect()
        calls = {(row.locus.position, row.alleles[1], row.s): None if row.GT is None else row.GT.n_alt_alleles() for row in rows}
        assert calls == {(100, "C", "0001"): 0, (100, "C", "0002"): 1, (100, "C", "0003"): None,
                         (200, "C", "0001"): 0, (200, "C", "0002"): 0, (200, "C", "0003"): 1,
                         (200, "G", "0001"): 0, (200, "G", "0002"): 2, (200, "G", "0003"): 0}
        vds.write(str(tmp_path / "source.vds"))
        (tmp_path / "samples.tsv").write_text("sampleid\n0001\n0002\n0003\n")
        (tmp_path / "map.bim").write_text("1\ts1\t0\t100\tC\tA\n1\ts2\t0\t200\tG\tA\n")
        hl.stop()
        command = [sys.executable, "scripts/export_aou_prs_inputs.py", "--vds", str(tmp_path / "source.vds"),
            "--samples", str(tmp_path / "samples.tsv"), "--variant-map", str(tmp_path / "map.bim"),
            "--output-prefix", str(tmp_path / "cli"), "--source-note", "synthetic only",
            "--tmp-dir", str(tmp_path / "tmp"), "--log", str(tmp_path / "cli.log"),
            "--chromosomes", "1", "--master", "local[2]"]
        environment = {k: v for k, v in os.environ.items() if not k.startswith("PYSPARK_GATEWAY_")}
        subprocess.run(command, env=environment, check=True, capture_output=True, timeout=180)
        receipt = json.loads((tmp_path / "cli.chr1.EXPORTED.json").read_text())
        assert receipt["reference_blocks_used"] and receipt["n_exported_variants"] == 3
        before = (tmp_path / "cli.chr1.bed").stat().st_mtime_ns
        subprocess.run(command, env=environment, check=True, capture_output=True, timeout=180)
        assert (tmp_path / "cli.chr1.bed").stat().st_mtime_ns == before
    finally:
        hl.stop()

