from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from decipehr.config import deep_update, load_yaml

DEFAULT_QC_CONFIG: dict[str, Any] = {
    "variant_qc": {
        "min_info": 0.8,
        "min_maf": 0.01,
        "max_af": 0.99,
        "max_variant_missingness": 0.05,
        "genotype_missingness_mode": "dosage",
        "keep_snps_only": True,
        "biallelic_only": True,
        "info_score": {
            "variant_col": "col0",
            "info_col": "info_IMPUTE",
            "freq_col": "MAF",
            "freq_is_maf": True,
        },
        "apply_hwe": False,
        "hwe_p": 1e-6,
        "controls_keep_file": None,
    },
    "sample_qc": {
        "apply_mind": False,
        "max_sample_missingness": 0.02,
        "keep_file": None,
    },
    "prs_alignment": {
        "remove_ambiguous_snps": True,
        "require_allele_match": True,
        "remove_duplicate_gwas_variants": True,
        "keep_only_acgt": True,
        "match_strategy": "id_then_position_alleles",
        "allow_strand_flip": False,
        "strand_flip_requires_frequency_check": True,
        "max_frequency_diff_for_flip": 0.15,
    },
    "paths": {
        "plink2": "plink2",
        "raw_pgen_dir": "data/pgen",
        "raw_pgen_pattern": "chr{chr}",
        "info_score_dir": "data/info_scores",
        "info_score_pattern": "chr{chr}.info_scores.txt",
        "qc_work_dir": "outputs/genotype_qc",
        "gwas_sumstats": None,
        "harmonized_score_dir": "outputs/prs_scores",
    },
}


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "t", "yes", "y"}:
            return True
        if normalized in {"0", "false", "f", "no", "n"}:
            return False
    raise ValueError(f"Expected a boolean-like value, got {value!r}")


def normalize_qc_config(raw: dict[str, Any] | None = None) -> dict[str, Any]:
    raw = raw or {}
    config = deep_update(deepcopy(DEFAULT_QC_CONFIG), raw)

    variant_qc = config["variant_qc"]
    for old_key, new_key in [
        ("min_info", "min_info"),
        ("min_maf", "min_maf"),
        ("max_variant_missingness", "max_variant_missingness"),
    ]:
        if old_key in raw:
            variant_qc[new_key] = raw[old_key]

    old_info = raw.get("info_score", {})
    variant_info = raw.get("variant_qc", {}).get("info_score", {})
    if isinstance(old_info, dict):
        if "variant_col" in old_info:
            variant_qc["info_score"]["variant_col"] = old_info["variant_col"]
        if "info_col" in old_info:
            variant_qc["info_score"]["info_col"] = old_info["info_col"]
        if "maf_col" in old_info and "freq_col" not in old_info:
            variant_qc["info_score"]["freq_col"] = old_info["maf_col"]
    if isinstance(variant_info, dict) and "maf_col" in variant_info and "freq_col" not in variant_info:
        variant_qc["info_score"]["freq_col"] = variant_info["maf_col"]

    variant_qc["info_score"]["freq_is_maf"] = _coerce_bool(
        variant_qc["info_score"].get("freq_is_maf", True)
    )
    config["sample_qc"]["apply_mind"] = _coerce_bool(config["sample_qc"].get("apply_mind", False))
    config["variant_qc"]["keep_snps_only"] = _coerce_bool(
        config["variant_qc"].get("keep_snps_only", True)
    )
    config["variant_qc"]["biallelic_only"] = _coerce_bool(
        config["variant_qc"].get("biallelic_only", True)
    )
    config["variant_qc"]["apply_hwe"] = _coerce_bool(config["variant_qc"].get("apply_hwe", False))
    return config


def load_qc_config(path: str | Path) -> dict[str, Any]:
    return normalize_qc_config(load_yaml(path))
