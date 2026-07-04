from __future__ import annotations

from pathlib import Path

import pytest

from decipehr.onekg import (
    build_plink_convert_command,
    build_plink_merge_command,
    normalize_1000g_panel,
    one_kg_vcf_name,
    parse_chromosomes,
    pfile_paths,
)


def test_parse_chromosomes_supports_ranges_and_deduplicates() -> None:
    assert parse_chromosomes("1-3,2,5") == [1, 2, 3, 5]


def test_parse_chromosomes_rejects_non_autosomes() -> None:
    with pytest.raises(ValueError, match="Only autosomes"):
        parse_chromosomes("1,23")


def test_normalize_1000g_panel_writes_panel_and_keep(tmp_path: Path) -> None:
    raw = tmp_path / "integrated.panel"
    raw.write_text(
        "sample\tpop\tsuper_pop\tgender\n"
        "HG00001\tGBR\tEUR\tfemale\n"
        "HG00002\tGBR\tEUR\tmale\n"
        "NA00001\tYRI\tAFR\tfemale\n"
        "HG00001\tGBR\tEUR\tfemale\n",
        encoding="utf-8",
    )
    panel_out = tmp_path / "1000g_panel.tsv"
    keep_out = tmp_path / "1000g_panel.keep"

    summary = normalize_1000g_panel(raw, panel_out, keep_out)

    assert summary["input_rows"] == 4
    assert summary["output_rows"] == 3
    assert summary["super_pop_counts"] == {"AFR": 1, "EUR": 2}
    assert panel_out.read_text(encoding="utf-8").splitlines()[0] == "FID\tIID\tpop\tsuper_pop\tgender"
    assert keep_out.read_text(encoding="utf-8").splitlines() == [
        "NA00001\tNA00001",
        "HG00001\tHG00001",
        "HG00002\tHG00002",
    ]


def test_pfile_paths_append_suffixes_to_dotted_prefixes() -> None:
    assert pfile_paths(Path("pgen/chr1.1000g_GRCh38")) == (
        Path("pgen/chr1.1000g_GRCh38.pgen"),
        Path("pgen/chr1.1000g_GRCh38.pvar"),
        Path("pgen/chr1.1000g_GRCh38.psam"),
    )


def test_build_plink_convert_command_uses_stable_variant_ids() -> None:
    command = build_plink_convert_command(
        plink2="plink2",
        vcf=Path("downloads") / one_kg_vcf_name(1),
        out_prefix=Path("pgen/chr1.1000g_GRCh38"),
        threads=4,
        keep_file=Path("1000g_panel.keep"),
    )

    assert command[:3] == ["plink2", "--vcf", f"downloads/{one_kg_vcf_name(1)}"]
    assert ["--keep", "1000g_panel.keep"] == command[4:6]
    assert "--snps-only" in command
    assert "--max-alleles" in command
    assert command[command.index("--set-all-var-ids") + 1] == "@:#:$r:$a"
    assert command[command.index("--threads") + 1] == "4"


def test_build_plink_merge_command() -> None:
    command = build_plink_merge_command(
        plink2="plink2",
        merge_list=Path("pgen/merge.txt"),
        out_prefix=Path("1000g_GRCh38"),
        threads=8,
    )

    assert command == [
        "plink2",
        "--pmerge-list",
        "pgen/merge.txt",
        "--sort-vars",
        "--make-pgen",
        "--threads",
        "8",
        "--out",
        "1000g_GRCh38",
    ]
