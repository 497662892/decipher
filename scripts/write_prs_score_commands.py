from __future__ import annotations

import argparse
import shlex
import stat
from pathlib import Path

from decipehr.qc_config import load_qc_config


def q(value: str | Path) -> str:
    return shlex.quote(str(value))


def main() -> None:
    parser = argparse.ArgumentParser(description="Write PLINK2 PRS scoring commands.")
    parser.add_argument("--config", default="configs/qc.yaml")
    parser.add_argument("--score-file", required=True)
    parser.add_argument("--trait-name", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    config = load_qc_config(args.config)
    paths = config["paths"]
    qc_prefix = Path(paths["qc_work_dir"]) / "qc_final" / "all_chromosomes.qc"
    score_dir = Path(paths["harmonized_score_dir"])
    score_file = Path(args.score_file)
    out_prefix = score_dir / f"{args.trait_name}.prs"

    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        f"mkdir -p {q(score_dir)}",
        f"test -f {q(str(qc_prefix) + '.pgen')} || {{ echo 'Missing {qc_prefix}.pgen' >&2; exit 1; }}",
        f"test -f {q(str(qc_prefix) + '.pvar')} || {{ echo 'Missing {qc_prefix}.pvar' >&2; exit 1; }}",
        f"test -f {q(str(qc_prefix) + '.psam')} || {{ echo 'Missing {qc_prefix}.psam' >&2; exit 1; }}",
        f"test -s {q(score_file)} || {{ echo 'Missing or empty score file {score_file}' >&2; exit 1; }}",
        (
            f"{q(paths['plink2'])} --pfile {q(qc_prefix)} "
            f"--score {q(score_file)} 1 2 3 header-read cols=+scoresums "
            f"--out {q(out_prefix)}"
        ),
    ]

    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    print(f"Wrote PRS scoring commands to {path}")


if __name__ == "__main__":
    main()
