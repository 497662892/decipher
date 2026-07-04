from __future__ import annotations

import argparse
import shlex
import stat
from pathlib import Path


def q(value: str | Path) -> str:
    return shlex.quote(str(value))


def parse_chromosomes(value: str) -> list[str]:
    chromosomes: list[str] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            chromosomes.extend(str(chrom) for chrom in range(int(start), int(end) + 1))
        else:
            chromosomes.append(part)
    if not chromosomes:
        raise ValueError("At least one chromosome is required")
    return chromosomes


def _format(pattern: str, chrom: str) -> str:
    return pattern.format(chr=chrom, CHR=chrom)


def _pvar_id_command(pvar: str | Path, out: str | Path) -> str:
    return (
        "awk 'BEGIN{FS=OFS=\"\\t\"} "
        "!/^##/ { if (!seen) {seen=1; next} print $3 }' "
        f"{q(pvar)} > {q(out)}"
    )


def _test_pfile(prefix: str | Path, label: str) -> list[str]:
    return [
        f"test -s {q(str(prefix) + '.pgen')} || {{ echo 'Missing {label} pgen: {prefix}.pgen' >&2; exit 1; }}",
        f"test -s {q(str(prefix) + '.pvar')} || {{ echo 'Missing {label} pvar: {prefix}.pvar' >&2; exit 1; }}",
        f"test -s {q(str(prefix) + '.psam')} || {{ echo 'Missing {label} psam: {prefix}.psam' >&2; exit 1; }}",
    ]


def build_commands(args: argparse.Namespace) -> list[str]:
    chromosomes = parse_chromosomes(args.chromosomes)
    out_dir = Path(args.out_dir)
    list_dir = out_dir / "lists"
    train_dir = out_dir / "train_val_projected"
    test_dir = out_dir / "holdout_test_projected"
    final_dir = out_dir / "qc_final"
    report_dir = out_dir / "reports"
    logs_dir = out_dir / "logs"
    train_chr_list = final_dir / "train_val_common.pmerge_list.txt"
    test_chr_list = final_dir / "holdout_test_common.pmerge_list.txt"
    train_final_prefix = final_dir / args.train_final_prefix_name
    test_final_prefix = final_dir / args.test_final_prefix_name

    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        f"PLINK2=${{PLINK2:-{q(args.plink2)}}}",
        f"THREADS=${{THREADS:-{args.threads}}}",
        f"mkdir -p {q(list_dir)} {q(train_dir)} {q(test_dir)} {q(final_dir)} {q(report_dir)} {q(logs_dir)}",
        f": > {q(train_chr_list)}",
        f": > {q(test_chr_list)}",
        f"test -s {q(args.train_keep)} || {{ echo 'Missing train-val keep file: {args.train_keep}' >&2; exit 1; }}",
        f"test -s {q(args.test_keep)} || {{ echo 'Missing holdout-test keep file: {args.test_keep}' >&2; exit 1; }}",
        "",
    ]

    for chrom in chromosomes:
        train_prefix = _format(args.train_prefix_pattern, chrom)
        holdout_prefix = _format(args.holdout_prefix_pattern, chrom)
        train_ids = list_dir / f"chr{chrom}.train_qc.ids"
        common_ids = list_dir / f"chr{chrom}.common_train_holdout.ids"
        test_projected = test_dir / f"chr{chrom}.holdout_test_common"
        train_projected = train_dir / f"chr{chrom}.train_val_common"
        audit_out = report_dir / f"chr{chrom}.trainval_vs_holdout_pvar_mismatches.csv"
        audit_summary = report_dir / f"chr{chrom}.trainval_vs_holdout_pvar.summary.json"
        lines.extend(
            [
                f"echo '[chr{chrom}] projecting holdout test to train-QC variants at '$(date)",
                *_test_pfile(train_prefix, f"train chr{chrom}"),
                *_test_pfile(holdout_prefix, f"holdout chr{chrom}"),
                _pvar_id_command(str(train_prefix) + ".pvar", train_ids),
                f"test -s {q(train_ids)} || {{ echo 'Empty train variant ID list for chr{chrom}' >&2; exit 1; }}",
                (
                    '"$PLINK2" '
                    f"--pfile {q(holdout_prefix)} --keep {q(args.test_keep)} --extract {q(train_ids)} "
                    f"--make-pgen --sort-vars --threads \"$THREADS\" --out {q(test_projected)}"
                ),
                _pvar_id_command(str(test_projected) + ".pvar", common_ids),
                f"test -s {q(common_ids)} || {{ echo 'Empty common variant ID list for chr{chrom}' >&2; exit 1; }}",
                (
                    '"$PLINK2" '
                    f"--pfile {q(train_prefix)} --keep {q(args.train_keep)} --extract {q(common_ids)} "
                    f"--make-pgen --sort-vars --threads \"$THREADS\" --out {q(train_projected)}"
                ),
                (
                    f"{q(args.python)} scripts/audit_pvar_compatibility.py "
                    f"--left-pvar {q(str(train_projected) + '.pvar')} "
                    f"--right-pvar {q(str(test_projected) + '.pvar')} "
                    "--left-name train_val --right-name holdout_test "
                    f"--out {q(audit_out)} --summary-out {q(audit_summary)} --fail-on-mismatch"
                ),
                f"printf '%s\\n' {q(train_projected)} >> {q(train_chr_list)}",
                f"printf '%s\\n' {q(test_projected)} >> {q(test_chr_list)}",
                "",
            ]
        )

    lines.extend(
        [
            "echo '[final] concatenating train-val chromosomes at '$(date)",
            (
                '"$PLINK2" '
                f"--pmerge-list {q(train_chr_list)} --make-pgen --sort-vars "
                f"--threads \"$THREADS\" --out {q(train_final_prefix)}"
            ),
            "echo '[final] concatenating holdout-test chromosomes at '$(date)",
            (
                '"$PLINK2" '
                f"--pmerge-list {q(test_chr_list)} --make-pgen --sort-vars "
                f"--threads \"$THREADS\" --out {q(test_final_prefix)}"
            ),
            (
                f"{q(args.python)} scripts/audit_pvar_compatibility.py "
                f"--left-pvar {q(str(train_final_prefix) + '.pvar')} "
                f"--right-pvar {q(str(test_final_prefix) + '.pvar')} "
                "--left-name train_val --right-name holdout_test "
                f"--out {q(report_dir / 'trainval_vs_holdout_final_pvar_mismatches.csv')} "
                f"--summary-out {q(report_dir / 'trainval_vs_holdout_final_pvar.summary.json')} "
                "--fail-on-mismatch"
            ),
            (
                '"$PLINK2" '
                f"--pfile {q(train_final_prefix)} --freq --missing --pgen-info "
                f"--threads \"$THREADS\" --out {q(report_dir / (args.train_final_prefix_name + '.summary'))}"
            ),
            (
                '"$PLINK2" '
                f"--pfile {q(test_final_prefix)} --freq --missing --pgen-info "
                f"--threads \"$THREADS\" --out {q(report_dir / (args.test_final_prefix_name + '.summary'))}"
            ),
            "echo 'Final train-val genotype prefix:'",
            f"echo {q(train_final_prefix)}",
            "echo 'Final holdout-test genotype prefix:'",
            f"echo {q(test_final_prefix)}",
            "echo 'Finished final genotype projection at '$(date)",
        ]
    )
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Write PLINK2 commands to project holdout test onto the existing train QC variant set."
    )
    parser.add_argument("--train-prefix-pattern", default="outputs/genotype_qc/qc_final/chr{chr}.qc")
    parser.add_argument(
        "--holdout-prefix-pattern",
        default="data/raw/holdout_phase/genotype_files/plink_files/Holdout_chr{chr}",
    )
    parser.add_argument("--train-keep", default="data/processed/sample_sets/train_val.keep.psam")
    parser.add_argument("--test-keep", default="data/processed/sample_sets/holdout_test.keep.psam")
    parser.add_argument("--out-dir", default="outputs/final_genotype")
    parser.add_argument("--train-final-prefix-name", default="train_val.common")
    parser.add_argument("--test-final-prefix-name", default="holdout_test.common")
    parser.add_argument("--chromosomes", default="1-22")
    parser.add_argument("--plink2", default="plink2")
    parser.add_argument("--python", default=".venv/bin/python")
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    lines = build_commands(args)
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    print(f"Wrote final genotype projection commands to {path}")


if __name__ == "__main__":
    main()
