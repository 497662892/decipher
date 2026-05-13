from __future__ import annotations

import argparse
import shlex
import stat
from pathlib import Path

import pandas as pd

from decipehr.config import load_yaml
from decipehr.io import write_table
from decipehr.megaprs import (
    DEFAULT_CHROMOSOMES,
    DEFAULT_MIN_MAF,
    DEFAULT_MODEL,
    DEFAULT_WINDOW_CM,
    megaprs_model_tag,
)


def q(value: str | Path) -> str:
    return shlex.quote(str(value))


def _chmod_executable(path: Path) -> None:
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _manifest_entry(path: str | Path, disease: str) -> dict[str, object]:
    manifest = load_yaml(path).get("summary_statistics", {})
    if disease not in manifest:
        raise ValueError(f"Disease {disease!r} is not present in {path}")
    entry = manifest[disease]
    if not isinstance(entry, dict):
        raise ValueError(f"Manifest entry for {disease} must be a mapping")
    return entry


def _optional_module_lines(module_load: str | None) -> list[str]:
    if not module_load:
        return []
    return [f"if command -v module >/dev/null 2>&1; then module load {q(module_load)} || true; fi"]


def _command_test(command: str | Path, label: str) -> str:
    command_str = str(command)
    if "/" in command_str or command_str.startswith("."):
        return f"test -x {q(command_str)} || {{ echo 'Missing executable {label}: {command_str}' >&2; exit 1; }}"
    return (
        f"command -v {q(command_str)} >/dev/null 2>&1 || "
        f"{{ echo 'Cannot find executable {label}: {command_str}' >&2; exit 1; }}"
    )


def _parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "t", "yes", "y"}:
        return True
    if normalized in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Cannot parse boolean value: {value!r}")


def _parse_csv_list(value: str) -> list[str]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise ValueError("Expected at least one comma-separated value")
    return items


def _frequency_script(
    *,
    disease: str,
    plink2: str,
    module_load: str | None,
    genotype_prefix: Path,
    train_keep: Path,
    variant_extract: Path | None,
    freq_prefix: Path,
    threads: int,
) -> str:
    tokens = [q(plink2), "--pfile", q(genotype_prefix), "--keep", q(train_keep)]
    if variant_extract is not None:
        tokens.extend(["--extract", q(variant_extract)])
    tokens.extend(
        [
            "--freq",
            "cols=+alt1freq",
            "--threads",
            f'"${{SLURM_CPUS_PER_TASK:-{threads}}}"',
            "--out",
            q(freq_prefix),
        ]
    )
    return "\n".join(
        [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            "",
            *_optional_module_lines(module_load),
            f"echo 'Started MegaPRS training allele frequency disease={disease} at '$(date)",
            _command_test(plink2, "PLINK2"),
            f"test -f {q(str(genotype_prefix) + '.pgen')} || {{ echo 'Missing {genotype_prefix}.pgen' >&2; exit 1; }}",
            f"test -f {q(str(genotype_prefix) + '.pvar')} || {{ echo 'Missing {genotype_prefix}.pvar' >&2; exit 1; }}",
            f"test -f {q(str(genotype_prefix) + '.psam')} || {{ echo 'Missing {genotype_prefix}.psam' >&2; exit 1; }}",
            f"test -s {q(train_keep)} || {{ echo 'Missing training keep file: {train_keep}' >&2; exit 1; }}",
            *(
                []
                if variant_extract is None
                else [
                    f"test -s {q(variant_extract)} || {{ echo 'Missing variant extract file: {variant_extract}' >&2; exit 1; }}"
                ]
            ),
            f"mkdir -p {q(freq_prefix.parent)}",
            f"if [[ ! -s {q(str(freq_prefix) + '.afreq')} || \"${{REUSE_MEGAPRS_FREQ:-1}}\" != \"1\" ]]; then",
            "  " + (" \\\n  ".join(tokens)),
            "else",
            f"  echo 'Reusing MegaPRS frequency file: {freq_prefix}.afreq'",
            "fi",
            f"test -s {q(str(freq_prefix) + '.afreq')} || {{ echo 'Missing frequency output: {freq_prefix}.afreq' >&2; exit 1; }}",
            f"echo 'Finished MegaPRS training allele frequency disease={disease} at '$(date)",
            "",
        ]
    )


def _make_reference_script(
    *,
    disease: str,
    plink2: str,
    module_load: str | None,
    genotype_prefix: Path,
    train_keep: Path,
    reference_extract: Path | None,
    chromosomes: str,
    bed_prefix: Path,
    threads: int,
) -> str:
    tokens = [
        q(plink2),
        "--pfile",
        q(genotype_prefix),
        "--keep",
        q(train_keep),
        "--chr",
        q(chromosomes),
    ]
    if reference_extract is not None:
        tokens.extend(["--extract", q(reference_extract)])
    tokens.extend(
        [
            "--make-bed",
            "--threads",
            f'"${{SLURM_CPUS_PER_TASK:-{threads}}}"',
            "--out",
            q(bed_prefix),
        ]
    )
    return "\n".join(
        [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            "",
            *_optional_module_lines(module_load),
            f"echo 'Started MegaPRS PLINK bed reference disease={disease} at '$(date)",
            _command_test(plink2, "PLINK2"),
            f"test -f {q(str(genotype_prefix) + '.pgen')} || {{ echo 'Missing {genotype_prefix}.pgen' >&2; exit 1; }}",
            f"test -s {q(train_keep)} || {{ echo 'Missing training keep file: {train_keep}' >&2; exit 1; }}",
            *(
                []
                if reference_extract is None
                else [
                    f"test -s {q(reference_extract)} || {{ echo 'Missing MegaPRS reference extract file: {reference_extract}' >&2; exit 1; }}"
                ]
            ),
            f"mkdir -p {q(bed_prefix.parent)}",
            f"if [[ ! -s {q(str(bed_prefix) + '.bed')} || \"${{REUSE_MEGAPRS_REFERENCE:-1}}\" != \"1\" ]]; then",
            "  " + (" \\\n  ".join(tokens)),
            "else",
            f"  echo 'Reusing MegaPRS PLINK bed reference: {bed_prefix}.bed'",
            "fi",
            f"test -s {q(str(bed_prefix) + '.bed')} || {{ echo 'Missing MegaPRS reference bed: {bed_prefix}.bed' >&2; exit 1; }}",
            f"test -s {q(str(bed_prefix) + '.bim')} || {{ echo 'Missing MegaPRS reference bim: {bed_prefix}.bim' >&2; exit 1; }}",
            f"test -s {q(str(bed_prefix) + '.fam')} || {{ echo 'Missing MegaPRS reference fam: {bed_prefix}.fam' >&2; exit 1; }}",
            f"echo 'Finished MegaPRS PLINK bed reference disease={disease} at '$(date)",
            "",
        ]
    )


def _prepare_script(
    *,
    disease: str,
    python_bin: str,
    harmonized: Path,
    freq_file: Path,
    sumstats_file: Path,
    predictors_file: Path,
    summary_out: Path,
    chromosomes: str,
    default_n: float,
    chunksize: int,
    drop_ambiguous: bool,
) -> str:
    drop_ambiguous_arg = " --drop-ambiguous" if drop_ambiguous else ""
    return "\n".join(
        [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            "",
            f"echo 'Started MegaPRS summary-stat preparation disease={disease} at '$(date)",
            f"mkdir -p {q(sumstats_file.parent)} {q(summary_out.parent)}",
            f"test -s {q(harmonized)} || {{ echo 'Missing harmonized sumstats: {harmonized}' >&2; exit 1; }}",
            f"test -s {q(freq_file)} || {{ echo 'Missing PLINK2 frequency file: {freq_file}' >&2; exit 1; }}",
            (
                f"{q(python_bin)} -u scripts/prepare_megaprs_inputs.py "
                f"--harmonized {q(harmonized)} "
                f"--freq {q(freq_file)} "
                f"--out {q(sumstats_file)} "
                f"--predictors-out {q(predictors_file)} "
                f"--summary-out {q(summary_out)} "
                f"--chromosomes {q(chromosomes)} "
                f"--default-n {default_n:g} "
                f"--chunksize {chunksize}"
                f"{drop_ambiguous_arg}"
            ),
            f"echo 'Finished MegaPRS summary-stat preparation disease={disease} at '$(date)",
            "",
        ]
    )


def _calc_cors_script(
    *,
    disease: str,
    chrom: str | None,
    ldak: str,
    module_load: str | None,
    bed_prefix: Path,
    cors_prefix: Path,
    break_points: Path | None,
    window_cm: float,
    min_maf: float,
    max_cor: float | None,
    threads: int,
) -> str:
    tokens = [
        q(ldak),
        "--calc-cors",
        q(cors_prefix),
        "--bfile",
        q(bed_prefix),
        "--min-maf",
        f"{min_maf:g}",
        "--max-threads",
        f'"${{SLURM_CPUS_PER_TASK:-{threads}}}"',
    ]
    if max_cor is not None:
        tokens.extend(["--max-cor", f"{max_cor:g}"])
    if chrom is not None:
        tokens.extend(["--chr", q(chrom)])
    if break_points is not None:
        tokens.extend(["--break-points", q(break_points)])
    else:
        tokens.extend(["--window-cm", f"{window_cm:g}"])
    label = f" chr={chrom}" if chrom is not None else ""
    return "\n".join(
        [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            "",
            *_optional_module_lines(module_load),
            f"echo 'Started MegaPRS LDAK correlations disease={disease}{label} at '$(date)",
            _command_test(ldak, "LDAK"),
            f"test -s {q(str(bed_prefix) + '.bed')} || {{ echo 'Missing reference bed: {bed_prefix}.bed' >&2; exit 1; }}",
            f"test -s {q(str(bed_prefix) + '.bim')} || {{ echo 'Missing reference bim: {bed_prefix}.bim' >&2; exit 1; }}",
            f"test -s {q(str(bed_prefix) + '.fam')} || {{ echo 'Missing reference fam: {bed_prefix}.fam' >&2; exit 1; }}",
            *(
                []
                if break_points is None
                else [
                    f"test -s {q(break_points)} || {{ echo 'Missing MegaPRS break-points file: {break_points}' >&2; exit 1; }}"
                ]
            ),
            f"mkdir -p {q(cors_prefix.parent)}",
            f"if [[ ! -s {q(str(cors_prefix) + '.cors.root')} || \"${{REUSE_MEGAPRS_CORS:-1}}\" != \"1\" ]]; then",
            "  " + (" \\\n  ".join(tokens)),
            "else",
            f"  echo 'Reusing MegaPRS correlations: {cors_prefix}.cors.root'",
            "fi",
            f"test -s {q(str(cors_prefix) + '.cors.root')} || {{ echo 'Missing MegaPRS correlations output: {cors_prefix}.cors.root' >&2; exit 1; }}",
            f"echo 'Finished MegaPRS LDAK correlations disease={disease}{label} at '$(date)",
            "",
        ]
    )


def _array_runner_script(*, task_list: Path, label: str) -> str:
    return "\n".join(
        [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            "",
            f"TASK_LIST={q(task_list)}",
            'ARRAY_INDEX="${SLURM_ARRAY_TASK_ID:-}"',
            'if [[ -z "$ARRAY_INDEX" ]]; then',
            f"  echo 'SLURM_ARRAY_TASK_ID is required for {label}' >&2",
            "  exit 1",
            "fi",
            'TASK_SCRIPT="$(sed -n "${ARRAY_INDEX}p" "$TASK_LIST")"',
            'if [[ -z "$TASK_SCRIPT" ]]; then',
            '  echo "No task script for array index $ARRAY_INDEX in $TASK_LIST" >&2',
            "  exit 1",
            "fi",
            'bash "$TASK_SCRIPT"',
            "",
        ]
    )


def _join_cors_script(
    *,
    disease: str,
    ldak: str,
    module_load: str | None,
    cors_prefixes: list[Path],
    corslist_file: Path,
    joined_cors_prefix: Path,
) -> str:
    if not cors_prefixes:
        raise ValueError("Expected at least one chromosome correlation prefix")
    tokens = [
        q(ldak),
        "--join-cors",
        q(joined_cors_prefix),
        "--corslist",
        q(corslist_file),
    ]
    list_entries = " ".join(q(prefix) for prefix in cors_prefixes)
    return "\n".join(
        [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            "",
            *_optional_module_lines(module_load),
            f"echo 'Started MegaPRS LDAK join-cors disease={disease} at '$(date)",
            _command_test(ldak, "LDAK"),
            f"mkdir -p {q(joined_cors_prefix.parent)} {q(corslist_file.parent)}",
            f"if [[ ! -s {q(str(joined_cors_prefix) + '.cors.root')} || \"${{REUSE_MEGAPRS_CORS:-1}}\" != \"1\" ]]; then",
            f"  for CORS_PREFIX in {list_entries}; do",
            '    test -s "${CORS_PREFIX}.cors.root" || { echo "Missing chromosome correlations: ${CORS_PREFIX}.cors.root" >&2; exit 1; }',
            "  done",
            f"  printf '%s\\n' {list_entries} > {q(corslist_file)}",
            "  " + (" \\\n  ".join(tokens)),
            "else",
            f"  echo 'Reusing joined MegaPRS correlations: {joined_cors_prefix}.cors.root'",
            "fi",
            f"test -s {q(str(joined_cors_prefix) + '.cors.root')} || {{ echo 'Missing joined MegaPRS correlations output: {joined_cors_prefix}.cors.root' >&2; exit 1; }}",
            f"echo 'Finished MegaPRS LDAK join-cors disease={disease} at '$(date)",
            "",
        ]
    )


def _pre_cors_script(*, frequency_script: Path, prepare_script: Path, reference_script: Path) -> str:
    return "\n".join(
        [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            "",
            f"bash {q(frequency_script)}",
            f"bash {q(prepare_script)}",
            f"bash {q(reference_script)}",
            "",
        ]
    )


def _megaprs_script(
    *,
    disease: str,
    ldak: str,
    module_load: str | None,
    sumstats_file: Path,
    cors_prefix: Path,
    effect_prefix: Path,
    effects_file: Path,
    model: str,
    allow_ambiguous: bool,
    annotation_number: int | None,
    annotation_prefix: Path | None,
    check_mcmc: bool | None,
    mcmc_solve: bool | None,
    threads: int,
) -> str:
    tokens = [
        q(ldak),
        "--mega-prs",
        q(effect_prefix),
        "--model",
        q(model),
        "--summary",
        q(sumstats_file),
        "--cors",
        q(cors_prefix),
        "--allow-ambiguous",
        "YES" if allow_ambiguous else "NO",
        "--max-threads",
        f'"${{SLURM_CPUS_PER_TASK:-{threads}}}"',
    ]
    if annotation_number is not None:
        tokens.extend(["--annotation-number", str(annotation_number)])
    if annotation_prefix is not None:
        tokens.extend(["--annotation-prefix", q(annotation_prefix)])
    if check_mcmc is not None:
        tokens.extend(["--check-MCMC", "YES" if check_mcmc else "NO"])
    if mcmc_solve is not None:
        tokens.extend(["--MCMC-solve", "YES" if mcmc_solve else "NO"])
    return "\n".join(
        [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            "",
            *_optional_module_lines(module_load),
            f"echo 'Started MegaPRS disease={disease} model={model} at '$(date)",
            _command_test(ldak, "LDAK"),
            f"test -s {q(sumstats_file)} || {{ echo 'Missing MegaPRS sumstats: {sumstats_file}' >&2; exit 1; }}",
            f"test -s {q(str(cors_prefix) + '.cors.root')} || {{ echo 'Missing MegaPRS correlations: {cors_prefix}.cors.root' >&2; exit 1; }}",
            *(
                []
                if annotation_prefix is None
                else [
                    f"test -n {q(str(annotation_number or ''))} || {{ echo 'MEGAPRS_ANNOTATION_NUMBER is required with annotation prefix' >&2; exit 1; }}",
                    f"test -s {q(str(annotation_prefix) + '1')} || {{ echo 'Missing MegaPRS annotation file: {annotation_prefix}1' >&2; exit 1; }}",
                ]
            ),
            f"mkdir -p {q(effect_prefix.parent)}",
            f"if [[ ! -s {q(effects_file)} || \"${{REUSE_MEGAPRS_EFFECTS:-1}}\" != \"1\" ]]; then",
            "  " + (" \\\n  ".join(tokens)),
            "else",
            f"  echo 'Reusing MegaPRS effects: {effects_file}'",
            "fi",
            f"test -s {q(effects_file)} || {{ echo 'Missing MegaPRS effects output: {effects_file}' >&2; exit 1; }}",
            f"echo 'Finished MegaPRS disease={disease} model={model} at '$(date)",
            "",
        ]
    )


def _score_script(
    *,
    disease: str,
    python_bin: str,
    plink2: str,
    module_load: str | None,
    effects_file: Path,
    score_file: Path,
    score_summary: Path,
    genotype_prefix: Path,
    score_keep: Path,
    fit_score_keep: Path,
    score_prefix: Path,
    fit_score_prefix: Path,
    threads: int,
) -> str:
    return "\n".join(
        [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            "",
            *_optional_module_lines(module_load),
            f"echo 'Started MegaPRS scoring disease={disease} at '$(date)",
            _command_test(plink2, "PLINK2"),
            f"test -s {q(effects_file)} || {{ echo 'Missing MegaPRS effects: {effects_file}' >&2; exit 1; }}",
            f"test -f {q(str(genotype_prefix) + '.pgen')} || {{ echo 'Missing {genotype_prefix}.pgen' >&2; exit 1; }}",
            f"test -f {q(str(genotype_prefix) + '.pvar')} || {{ echo 'Missing {genotype_prefix}.pvar' >&2; exit 1; }}",
            f"test -f {q(str(genotype_prefix) + '.psam')} || {{ echo 'Missing {genotype_prefix}.psam' >&2; exit 1; }}",
            f"test -s {q(score_keep)} || {{ echo 'Missing scoring keep file: {score_keep}' >&2; exit 1; }}",
            f"test -s {q(fit_score_keep)} || {{ echo 'Missing model-fit scoring keep file: {fit_score_keep}' >&2; exit 1; }}",
            f"mkdir -p {q(score_file.parent)} {q(score_prefix.parent)}",
            (
                f"{q(python_bin)} -u scripts/build_megaprs_scorefile.py "
                f"--effects {q(effects_file)} "
                f"--out {q(score_file)} "
                f"--summary-out {q(score_summary)}"
            ),
            (
                f"{q(plink2)} --pfile {q(genotype_prefix)} --keep {q(score_keep)} "
                f"--score {q(score_file)} 1 2 3 header-read cols=+scoresums "
                f"--threads \"${{SLURM_CPUS_PER_TASK:-{threads}}}\" --out {q(score_prefix)}"
            ),
            (
                f"{q(plink2)} --pfile {q(genotype_prefix)} --keep {q(fit_score_keep)} "
                f"--score {q(score_file)} 1 2 3 header-read cols=+scoresums "
                f"--threads \"${{SLURM_CPUS_PER_TASK:-{threads}}}\" --out {q(fit_score_prefix)}"
            ),
            f"echo 'Finished MegaPRS scoring disease={disease} at '$(date)",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Write full-training LDAK MegaPRS commands for one disease.")
    parser.add_argument("--disease", required=True)
    parser.add_argument("--manifest", default="configs/official_sumstats.yaml")
    parser.add_argument("--genotype-prefix", default="outputs/genotype_qc/qc_final/all_chromosomes.qc")
    parser.add_argument("--harmonized-dir", default="data/processed/sumstats/harmonized")
    parser.add_argument("--train-keep", default="data/processed/sample_sets/labeled_train.keep.psam")
    parser.add_argument("--score-keep", default="data/processed/sample_sets/unlabelled_validation.keep.psam")
    parser.add_argument("--fit-score-keep", default=None)
    parser.add_argument("--variant-extract", default=None)
    parser.add_argument("--out-dir", default="outputs/megaprs_full_train")
    parser.add_argument("--reference-dir", default=None)
    parser.add_argument("--ldak", default="ldak")
    parser.add_argument("--python", default="python")
    parser.add_argument("--plink2", default="plink2")
    parser.add_argument("--ldak-module", default=None)
    parser.add_argument("--plink-module", default=None)
    parser.add_argument("--n-gwas", type=float, default=None)
    parser.add_argument("--chromosomes", default=DEFAULT_CHROMOSOMES)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--break-points", default=None)
    parser.add_argument("--window-cm", type=float, default=DEFAULT_WINDOW_CM)
    parser.add_argument("--min-maf", type=float, default=DEFAULT_MIN_MAF)
    parser.add_argument("--max-cor", type=float, default=None)
    parser.add_argument("--annotation-number", type=int, default=None)
    parser.add_argument("--annotation-prefix", default=None)
    parser.add_argument("--check-mcmc", type=_parse_bool, default=None)
    parser.add_argument("--mcmc-solve", type=_parse_bool, default=None)
    parser.add_argument("--allow-ambiguous", type=_parse_bool, default=True)
    parser.add_argument("--drop-ambiguous-sumstats", action="store_true")
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--chunksize", type=int, default=250_000)
    args = parser.parse_args()

    entry = _manifest_entry(args.manifest, args.disease)
    accession = entry["accession"]
    n_gwas = args.n_gwas if args.n_gwas is not None else entry.get("default_n")
    if n_gwas is None:
        raise ValueError(
            f"No --n-gwas was supplied and manifest entry {args.disease!r} has no default_n"
        )
    n_gwas = float(n_gwas)
    chromosomes = _parse_csv_list(args.chromosomes)

    harmonized = Path(args.harmonized_dir) / f"{args.disease}.{accession}.harmonized.tsv.gz"
    out_dir = Path(args.out_dir) / args.disease
    input_dir = out_dir / "inputs"
    task_dir = out_dir / "tasks"
    effect_dir = out_dir / "effects"
    scorefile_dir = out_dir / "scorefiles"
    score_dir = out_dir / "scores"
    command_dir = out_dir / "commands"
    report_dir = out_dir / "reports"
    reference_dir = Path(args.reference_dir) if args.reference_dir else out_dir / "reference"
    for path in [
        input_dir,
        task_dir,
        effect_dir,
        scorefile_dir,
        score_dir,
        command_dir,
        report_dir,
        reference_dir,
    ]:
        path.mkdir(parents=True, exist_ok=True)

    freq_prefix = input_dir / f"{args.disease}.megaprs.training"
    freq_file = Path(f"{freq_prefix}.afreq")
    sumstats_file = input_dir / f"{args.disease}.megaprs.summaries"
    predictors_file = input_dir / f"{args.disease}.megaprs.predictors"
    ref_bed_prefix = reference_dir / f"{args.disease}.megaprs.reference"
    cors_by_chr_dir = reference_dir / "cors_by_chr"
    cors_prefix = reference_dir / f"{args.disease}.megaprs.cors"
    corslist_file = command_dir / f"{args.disease}.megaprs.corslist.txt"
    effect_prefix = effect_dir / f"{args.disease}.{megaprs_model_tag(args.model)}"
    effects_file = Path(f"{effect_prefix}.effects")
    score_file = scorefile_dir / f"{args.disease}.{megaprs_model_tag(args.model)}.score.tsv"
    score_summary = scorefile_dir / f"{args.disease}.{megaprs_model_tag(args.model)}.score.summary.json"
    score_prefix = score_dir / f"{args.disease}.{megaprs_model_tag(args.model)}"
    fit_score_prefix = score_dir / f"{args.disease}.{megaprs_model_tag(args.model)}.fit"
    workflow_script = command_dir / f"{args.disease}.megaprs_full_train.commands.sh"
    genotype_prefix = Path(args.genotype_prefix)
    score_keep = Path(args.score_keep)
    fit_score_keep = Path(args.fit_score_keep) if args.fit_score_keep else Path(args.train_keep)
    variant_extract = Path(args.variant_extract) if args.variant_extract else None
    break_points = Path(args.break_points) if args.break_points else None
    annotation_prefix = Path(args.annotation_prefix) if args.annotation_prefix else None
    if annotation_prefix is not None and args.annotation_number is None:
        raise ValueError("--annotation-number is required with --annotation-prefix")
    reference_extract = variant_extract if variant_extract is not None else predictors_file

    freq_script = command_dir / f"{args.disease}.megaprs.frequency.sh"
    freq_script.write_text(
        _frequency_script(
            disease=args.disease,
            plink2=args.plink2,
            module_load=args.plink_module,
            genotype_prefix=genotype_prefix,
            train_keep=Path(args.train_keep),
            variant_extract=variant_extract,
            freq_prefix=freq_prefix,
            threads=args.threads,
        ),
        encoding="utf-8",
    )
    _chmod_executable(freq_script)

    prepare_script = command_dir / f"{args.disease}.megaprs.prepare_sumstats.sh"
    prepare_script.write_text(
        _prepare_script(
            disease=args.disease,
            python_bin=args.python,
            harmonized=harmonized,
            freq_file=freq_file,
            sumstats_file=sumstats_file,
            predictors_file=predictors_file,
            summary_out=report_dir / f"{args.disease}.megaprs_input.summary.json",
            chromosomes=args.chromosomes,
            default_n=n_gwas,
            chunksize=args.chunksize,
            drop_ambiguous=args.drop_ambiguous_sumstats,
        ),
        encoding="utf-8",
    )
    _chmod_executable(prepare_script)

    reference_script = command_dir / f"{args.disease}.megaprs.make_reference.sh"
    reference_script.write_text(
        _make_reference_script(
            disease=args.disease,
            plink2=args.plink2,
            module_load=args.plink_module,
            genotype_prefix=genotype_prefix,
            train_keep=Path(args.train_keep),
            reference_extract=reference_extract,
            chromosomes=args.chromosomes,
            bed_prefix=ref_bed_prefix,
            threads=args.threads,
        ),
        encoding="utf-8",
    )
    _chmod_executable(reference_script)

    pre_cors_script = command_dir / f"{args.disease}.megaprs.pre_cors.sh"
    pre_cors_script.write_text(
        _pre_cors_script(
            frequency_script=freq_script,
            prepare_script=prepare_script,
            reference_script=reference_script,
        ),
        encoding="utf-8",
    )
    _chmod_executable(pre_cors_script)

    cors_task_scripts: list[Path] = []
    cors_prefixes: list[Path] = []
    for chrom in chromosomes:
        chrom_cors_prefix = cors_by_chr_dir / f"{args.disease}.megaprs.chr{chrom}.cors"
        cors_prefixes.append(chrom_cors_prefix)
        cors_task_script = task_dir / f"{args.disease}.megaprs.calc_cors.chr{chrom}.sh"
        cors_task_script.write_text(
            _calc_cors_script(
                disease=args.disease,
                chrom=chrom,
                ldak=args.ldak,
                module_load=args.ldak_module,
                bed_prefix=ref_bed_prefix,
                cors_prefix=chrom_cors_prefix,
                break_points=break_points,
                window_cm=args.window_cm,
                min_maf=args.min_maf,
                max_cor=args.max_cor,
                threads=args.threads,
            ),
            encoding="utf-8",
        )
        _chmod_executable(cors_task_script)
        cors_task_scripts.append(cors_task_script)

    cors_task_list = command_dir / f"{args.disease}.megaprs.calc_cors.array_tasks.txt"
    cors_task_list.write_text(
        "\n".join(str(path) for path in cors_task_scripts) + "\n",
        encoding="utf-8",
    )

    cors_script = command_dir / f"{args.disease}.megaprs.calc_cors.array.sh"
    cors_script.write_text(
        _array_runner_script(task_list=cors_task_list, label=f"MegaPRS calc-cors disease={args.disease}"),
        encoding="utf-8",
    )
    _chmod_executable(cors_script)

    join_cors_script = command_dir / f"{args.disease}.megaprs.join_cors.sh"
    join_cors_script.write_text(
        _join_cors_script(
            disease=args.disease,
            ldak=args.ldak,
            module_load=args.ldak_module,
            cors_prefixes=cors_prefixes,
            corslist_file=corslist_file,
            joined_cors_prefix=cors_prefix,
        ),
        encoding="utf-8",
    )
    _chmod_executable(join_cors_script)

    run_script = task_dir / f"{args.disease}.megaprs.sh"
    run_script.write_text(
        _megaprs_script(
            disease=args.disease,
            ldak=args.ldak,
            module_load=args.ldak_module,
            sumstats_file=sumstats_file,
            cors_prefix=cors_prefix,
            effect_prefix=effect_prefix,
            effects_file=effects_file,
            model=args.model,
            allow_ambiguous=args.allow_ambiguous,
            annotation_number=args.annotation_number,
            annotation_prefix=annotation_prefix,
            check_mcmc=args.check_mcmc,
            mcmc_solve=args.mcmc_solve,
            threads=args.threads,
        ),
        encoding="utf-8",
    )
    _chmod_executable(run_script)

    score_script = command_dir / f"{args.disease}.megaprs.score.sh"
    score_script.write_text(
        _score_script(
            disease=args.disease,
            python_bin=args.python,
            plink2=args.plink2,
            module_load=args.plink_module,
            effects_file=effects_file,
            score_file=score_file,
            score_summary=score_summary,
            genotype_prefix=genotype_prefix,
            score_keep=score_keep,
            fit_score_keep=fit_score_keep,
            score_prefix=score_prefix,
            fit_score_prefix=fit_score_prefix,
            threads=args.threads,
        ),
        encoding="utf-8",
    )
    _chmod_executable(score_script)

    workflow_script.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                f"bash {q(pre_cors_script)}",
                f"while IFS= read -r TASK_SCRIPT; do bash \"$TASK_SCRIPT\"; done < {q(cors_task_list)}",
                f"bash {q(join_cors_script)}",
                f"bash {q(run_script)}",
                f"bash {q(score_script)}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    _chmod_executable(workflow_script)

    score_list = command_dir / f"{args.disease}.megaprs_full_train.score_scripts.txt"
    score_list.write_text(str(score_script) + "\n", encoding="utf-8")
    manifest_out = command_dir / f"{args.disease}.megaprs_full_train.tasks.csv"
    write_table(
        pd.DataFrame(
            [
                {
                    "disease": args.disease,
                    "model_scope": "full_train",
                    "model": megaprs_model_tag(args.model),
                    "megaprs_model": args.model,
                    "n_gwas": n_gwas,
                    "chromosomes": args.chromosomes,
                    "workflow_script": str(workflow_script),
                    "frequency_script": str(freq_script),
                    "pre_cors_script": str(pre_cors_script),
                    "reference_script": str(reference_script),
                    "prepare_script": str(prepare_script),
                    "cors_script": str(cors_script),
                    "cors_task_list": str(cors_task_list),
                    "n_cors_tasks": len(cors_task_scripts),
                    "join_cors_script": str(join_cors_script),
                    "run_script": str(run_script),
                    "score_script": str(score_script),
                    "sumstats_file": str(sumstats_file),
                    "predictors_file": str(predictors_file),
                    "reference_extract": str(reference_extract),
                    "reference_bed": str(ref_bed_prefix) + ".bed",
                    "cors_by_chr_dir": str(cors_by_chr_dir),
                    "cors_prefix": str(cors_prefix),
                    "corslist_file": str(corslist_file),
                    "effects_file": str(effects_file),
                    "score_file": str(score_file),
                    "sscore": str(score_prefix) + ".sscore",
                    "fit_sscore": str(fit_score_prefix) + ".sscore",
                    "ldak": args.ldak,
                    "break_points": None if break_points is None else str(break_points),
                    "window_cm": args.window_cm,
                    "min_maf": args.min_maf,
                    "annotation_number": args.annotation_number,
                    "annotation_prefix": None if annotation_prefix is None else str(annotation_prefix),
                    "check_mcmc": args.check_mcmc,
                    "mcmc_solve": args.mcmc_solve,
                }
            ]
        ),
        manifest_out,
    )

    print(f"Wrote MegaPRS workflow script to {workflow_script}")
    print(f"Wrote {len(cors_task_scripts)} MegaPRS chromosome correlation scripts to {cors_task_list}")
    print(f"Wrote MegaPRS join-cors script to {join_cors_script}")
    print(f"Wrote MegaPRS score script list to {score_list}")
    print(f"Wrote MegaPRS task manifest to {manifest_out}")


if __name__ == "__main__":
    main()
