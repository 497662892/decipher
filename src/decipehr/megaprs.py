from __future__ import annotations

DEFAULT_CHROMOSOMES = ",".join(str(chrom) for chrom in range(1, 23))
DEFAULT_MODEL = "bayesr"
DEFAULT_WINDOW_CM = 1.0
DEFAULT_MIN_MAF = 0.005


def megaprs_model_tag(model: str = DEFAULT_MODEL) -> str:
    return f"megaprs_{model.replace('-', '_')}"
