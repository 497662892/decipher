from __future__ import annotations

DEFAULT_CHROMOSOMES = ",".join(str(chrom) for chrom in range(1, 23))
DEFAULT_CHAIN_LENGTH = 3000
DEFAULT_BURN_IN = 1000
DEFAULT_THIN = 10


def sbayesrc_model_tag() -> str:
    return "sbayesrc"
