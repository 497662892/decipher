from __future__ import annotations

DEFAULT_CHROMOSOMES = ",".join(str(chrom) for chrom in range(1, 23))
DEFAULT_M = 1000
DEFAULT_ITER = 1000
DEFAULT_BURN = 200
DEFAULT_THIN = 5
DEFAULT_R2 = 0.1
DEFAULT_A = 0.1
DEFAULT_C = 1.0
DEFAULT_A0K = 0.5
DEFAULT_B0K = 0.5
DEFAULT_OPT_LLK = 1


def sdpr_model_tag() -> str:
    return "sdpr"
