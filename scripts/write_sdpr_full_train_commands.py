from __future__ import annotations

import argparse
import shlex
import stat
from pathlib import Path

import pandas as pd

from decipehr.config import load_yaml
from decipehr.io import read_table, write_table
from decipehr.schema import parse_csv_list
from decipehr.sdpr import (
    DEFAULT_A,
    DEFAULT_A0K,
    DEFAULT_B0K,
    DEFAULT_BURN,
    DEFAULT_C,
    DEFAULT_CHROMOSOMES,
    DEFAULT_ITER,
    DEFAULT_M,
    DEFAULT_OPT_LLK,
    DEFAULT_R2,
    DEFAULT_THIN,
    sdpr_model_tag,
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


def _resolve_sample_size(cli_n: int | None, entry: dict[str, object]) -> int:
    value: object = cli_n
    if value in (None, ""):
        value = entry.get("default_n")
    if value in (None, ""):
        raise ValueError("SDPR requires --n-gwas or manifest default_n")
    n_gwas = int(round(float(value)))
    if n_gwas <= 0:
        raise ValueError("SDPR GWAS sample size must be positive")
    return n_gwas


def _resolve_default_n(cli_default_n: float | None, entry: dict[str, object], n_gwas: int) -> float:
    if cli_default_n is not None:
        default_n = cli_default_n
    else:
        manifest_default = entry.get("default_n")
        default_n = n_gwas if manifest_default in (None, "") else float(manifest_default)
    if default_n <= 0:
        raise ValueError("SDPR default N must be positive")
    return float(default_n)


def _chromosomes(value: str) -> list[str]:
    chromosomes: list[str] = []
    seen: set[str] = set()
    for item in parse_csv_list(value):
        if "-" in item:
            start, end = item.split("-", 1)
            expanded = [str(chrom) for chrom in range(int(start), int(end) + 1)]
        else:
            expanded = [str(int(item))]
        for chrom in expanded:
            number = int(chrom)
            if number < 1 or number > 22:
                raise ValueError(f"SDPR only supports autosomes 1-22, got {chrom!r}")
            if chrom not in seen:
                seen.add(chrom)
                chromosomes.append(chrom)
    if not chromosomes:
        raise ValueError("At least one chromosome is required")
    return chromosomes


def _render_command(tokens: list[str]) -> str:
    return (" \\" + "\n  ").join(tokens)


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


def _keep_samples(path: Path) -> pd.Series:
    keep = read_table(path)
    for column in ["#IID", "IID", "sampleid", "patient_id", "sample_id"]:
        if column in keep.columns:
            values = keep[column]
            break
    else:
        values = keep.iloc[:, 0]
    values = values.astype("string").dropna()
    values = values.loc[~values.isin(["", ".", "nan", "<NA>"])]
    return values.drop_duplicates()


def _write_union_keep(paths: list[Path], out: Path) -> None:
    values = pd.concat([_keep_samples(path) for path in paths], ignore_index=True)
    values = values.drop_duplicates().sort_values(kind="mergesort")
    write_table(pd.DataFrame({"#IID": values.to_numpy()}), out)


def _plink_make_bed_tokens(
    *,
    plink2: str,
    genotype_prefix: Path,
    keep: Path,
    chromosomes: str,
    variant_extract: Path | None,
    out_prefix: Path,
    threads: int,
) -> list[str]:
    tokens = [
        q(plink2),
        "--pfile",
        q(genotype_prefix),
        "--keep",
        q(keep),
        "--chr",
        q(chromosomes),
    ]
    if variant_extract is not None:
        tokens.extend(["--extract", q(variant_extract)])
    tokens.extend(
        [
            "--make-bed",
            "--threads",
            f'"${{SLURM_CPUS_PER_TASK:-{threads}}}"',
            "--out",
            q(out_prefix),
        ]
    )
    return tokens


def _reference_check_lines(ref_dir: Path, chromosomes: list[str]) -> list[str]:
    lines: list[str] = []
    for chrom in chromosomes:
        dat = ref_dir / f"chr{chrom}.dat"
        snpinfo = ref_dir / f"chr{chrom}.snpInfo"
        lines.extend(
            [
                f"test -s {q(dat)} || {{ echo 'Missing SDPR reference LD file: {dat}' >&2; exit 1; }}",
                f"test -s {q(snpinfo)} || {{ echo 'Missing SDPR SNP info file: {snpinfo}' >&2; exit 1; }}",
            ]
        )
    return lines


def _prepare_reference_script(
    *,
    disease: str,
    ref_mode: str,
    raw_sumstats: Path | None,
    python_bin: str,
    plink2: str,
    sdpr_bin: str,
    module_load: str | None,
    genotype_prefix: Path,
    harmonized: Path,
    train_keep: Path,
    score_keep: Path,
    fit_score_keep: Path,
    all_keep: Path,
    ref_bed_prefix: Path,
    score_bed_prefix: Path,
    variant_extract: Path | None,
    sdpr_sumstats: Path,
    target_id_map: Path | None,
    valid_bim: Path,
    summary_out: Path,
    ref_dir: Path,
    chromosomes: list[str],
    default_n: float,
    chunksize: int,
    r2: float,
    threads: int,
) -> str:
    chromosome_arg = ",".join(chromosomes)
    ref_bed_tokens: list[str] = []
    if ref_mode == "target":
        ref_bed_tokens = _plink_make_bed_tokens(
            plink2=plink2,
            genotype_prefix=genotype_prefix,
            keep=train_keep,
            chromosomes=chromosome_arg,
            variant_extract=variant_extract,
            out_prefix=ref_bed_prefix,
            threads=threads,
        )
    score_bed_tokens = _plink_make_bed_tokens(
        plink2=plink2,
        genotype_prefix=genotype_prefix,
        keep=all_keep,
        chromosomes=chromosome_arg,
        variant_extract=variant_extract,
        out_prefix=score_bed_prefix,
        threads=threads,
    )
    sumstats_tokens = [
        q(python_bin),
        "-u",
        "scripts/prepare_sdpr_inputs.py",
        "--harmonized",
        q(harmonized),
        "--out",
        q(sdpr_sumstats),
        "--summary-out",
        q(summary_out),
        "--chromosomes",
        q(chromosome_arg),
        "--default-n",
        f"{default_n:g}",
        "--chunksize",
        str(chunksize),
    ]
    if raw_sumstats is not None and target_id_map is not None:
        sumstats_tokens.extend(
            [
                "--raw-sumstats",
                q(raw_sumstats),
                "--target-id-map-out",
                q(target_id_map),
                "--target-bim",
                q(str(score_bed_prefix) + ".bim"),
                "--valid-bim-out",
                q(valid_bim),
            ]
        )

    ref_lines = []
    if ref_mode == "prebuilt":
        ref_lines = [
            "",
            f"echo 'Using prebuilt SDPR LD reference: {ref_dir}'",
            *_reference_check_lines(ref_dir, chromosomes),
        ]
    else:
        for chrom in chromosomes:
            dat = ref_dir / f"chr{chrom}.dat"
            snpinfo = ref_dir / f"chr{chrom}.snpInfo"
            sentinel = ref_dir / f"chr{chrom}.make_ref.done"
            ref_lines.extend(
                [
                    "",
                    (
                        f"if [[ -s {q(dat)} && -s {q(snpinfo)} && -s {q(sentinel)} "
                        '&& "${REUSE_SDPR_REF:-1}" == "1" ]]; then'
                    ),
                    f"  echo 'Reusing SDPR LD reference for chr{chrom}: {ref_dir}'",
                    "else",
                    (
                        f"  {q(sdpr_bin)} -make_ref -ref_prefix {q(ref_bed_prefix)} "
                        f"-chr {chrom} -ref_dir {q(ref_dir)} "
                        f"-n_threads \"${{SLURM_CPUS_PER_TASK:-{threads}}}\" -r2 {r2:g}"
                    ),
                    f"  test -s {q(dat)} || {{ echo 'Missing SDPR reference LD file: {dat}' >&2; exit 1; }}",
                    f"  test -s {q(snpinfo)} || {{ echo 'Missing SDPR SNP info file: {snpinfo}' >&2; exit 1; }}",
                    f"  touch {q(sentinel)}",
                    "fi",
                ]
            )

    make_reference_bed_lines: list[str] = []
    if ref_mode == "target":
        make_reference_bed_lines = [
            "",
            f"if [[ ! -s {q(str(ref_bed_prefix) + '.bed')} || \"${{REUSE_SDPR_BED:-1}}\" != \"1\" ]]; then",
            f"  {_render_command(ref_bed_tokens)}",
            "else",
            f"  echo 'Reusing SDPR reference PLINK bed files: {ref_bed_prefix}'",
            "fi",
            f"test -s {q(str(ref_bed_prefix) + '.bed')} || {{ echo 'Missing {ref_bed_prefix}.bed' >&2; exit 1; }}",
            f"test -s {q(str(ref_bed_prefix) + '.bim')} || {{ echo 'Missing {ref_bed_prefix}.bim' >&2; exit 1; }}",
            f"test -s {q(str(ref_bed_prefix) + '.fam')} || {{ echo 'Missing {ref_bed_prefix}.fam' >&2; exit 1; }}",
        ]

    reuse_sumstats_conditions = [f"-s {q(sdpr_sumstats)}", f"-s {q(summary_out)}"]
    if raw_sumstats is not None and target_id_map is not None:
        reuse_sumstats_conditions.extend([f"-s {q(target_id_map)}", f"-s {q(valid_bim)}"])

    return "\n".join(
        [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            "",
            *_optional_module_lines(module_load),
            f"echo 'Started SDPR reference preparation disease={disease} at '$(date)",
            f"export OMP_NUM_THREADS=${{SLURM_CPUS_PER_TASK:-{threads}}}",
            f"export MKL_NUM_THREADS=${{SLURM_CPUS_PER_TASK:-{threads}}}",
            f"export OPENBLAS_NUM_THREADS=${{SLURM_CPUS_PER_TASK:-{threads}}}",
            (
                f"mkdir -p {q(ref_bed_prefix.parent)} {q(score_bed_prefix.parent)} "
                f"{q(sdpr_sumstats.parent)} {q(summary_out.parent)} {q(ref_dir)}"
            ),
            _command_test(python_bin, "python"),
            _command_test(plink2, "plink2"),
            _command_test(sdpr_bin, "SDPR"),
            *(
                [f"test -s {q(raw_sumstats)} || {{ echo 'Missing raw sumstats: {raw_sumstats}' >&2; exit 1; }}"]
                if raw_sumstats is not None
                else []
            ),
            f"test -f {q(str(genotype_prefix) + '.pgen')} || {{ echo 'Missing {genotype_prefix}.pgen' >&2; exit 1; }}",
            f"test -f {q(str(genotype_prefix) + '.pvar')} || {{ echo 'Missing {genotype_prefix}.pvar' >&2; exit 1; }}",
            f"test -f {q(str(genotype_prefix) + '.psam')} || {{ echo 'Missing {genotype_prefix}.psam' >&2; exit 1; }}",
            f"test -s {q(harmonized)} || {{ echo 'Missing harmonized sumstats: {harmonized}' >&2; exit 1; }}",
            f"test -s {q(train_keep)} || {{ echo 'Missing full training keep file: {train_keep}' >&2; exit 1; }}",
            f"test -s {q(score_keep)} || {{ echo 'Missing scoring keep file: {score_keep}' >&2; exit 1; }}",
            f"test -s {q(fit_score_keep)} || {{ echo 'Missing model-fit scoring keep file: {fit_score_keep}' >&2; exit 1; }}",
            f"test -s {q(all_keep)} || {{ echo 'Missing all-samples keep file: {all_keep}' >&2; exit 1; }}",
            *(
                [
                    f"test -s {q(variant_extract)} || {{ echo 'Missing variant extract file: {variant_extract}' >&2; exit 1; }}"
                ]
                if variant_extract is not None
                else []
            ),
            *make_reference_bed_lines,
            "",
            f"if [[ ! -s {q(str(score_bed_prefix) + '.bed')} || \"${{REUSE_SDPR_BED:-1}}\" != \"1\" ]]; then",
            f"  {_render_command(score_bed_tokens)}",
            "else",
            f"  echo 'Reusing SDPR scoring PLINK bed files: {score_bed_prefix}'",
            "fi",
            f"test -s {q(str(score_bed_prefix) + '.bed')} || {{ echo 'Missing {score_bed_prefix}.bed' >&2; exit 1; }}",
            f"test -s {q(str(score_bed_prefix) + '.bim')} || {{ echo 'Missing {score_bed_prefix}.bim' >&2; exit 1; }}",
            f"test -s {q(str(score_bed_prefix) + '.fam')} || {{ echo 'Missing {score_bed_prefix}.fam' >&2; exit 1; }}",
            "",
            (
                f"if [[ {' && '.join(reuse_sumstats_conditions)} "
                '&& "${REUSE_SDPR_SUMSTATS:-1}" == "1" ]]; then'
            ),
            f"  echo 'Reusing SDPR summary statistics: {sdpr_sumstats}'",
            "else",
            f"  {_render_command(sumstats_tokens)}",
            "fi",
            f"test -s {q(sdpr_sumstats)} || {{ echo 'Missing SDPR summary statistics: {sdpr_sumstats}' >&2; exit 1; }}",
            f"test -s {q(valid_bim)} || {{ echo 'Missing SDPR validation BIM: {valid_bim}' >&2; exit 1; }}",
            *ref_lines,
            f"echo 'Finished SDPR reference preparation disease={disease} at '$(date)",
            "",
        ]
    )


def _task_script(
    *,
    disease: str,
    chrom: str,
    sdpr_bin: str,
    module_load: str | None,
    ref_dir: Path,
    sdpr_sumstats: Path,
    valid_bim: Path,
    effect_file: Path,
    n_gwas: int,
    m_components: int,
    n_iter: int,
    n_burn: int,
    thin: int,
    a: float,
    c: float,
    a0k: float,
    b0k: float,
    opt_llk: int,
    threads: int,
) -> str:
    dat = ref_dir / f"chr{chrom}.dat"
    snpinfo = ref_dir / f"chr{chrom}.snpInfo"
    mcmc_tokens = [
        q(sdpr_bin),
        "-mcmc",
        "-ref_dir",
        q(ref_dir),
        "-ss",
        q(sdpr_sumstats),
        "-N",
        str(n_gwas),
        "-chr",
        chrom,
        "-out",
        q(effect_file),
        "-valid",
        q(valid_bim),
        "-n_threads",
        f'"${{SLURM_CPUS_PER_TASK:-{threads}}}"',
        "-M",
        str(m_components),
        "-iter",
        str(n_iter),
        "-burn",
        str(n_burn),
        "-thin",
        str(thin),
        "-a",
        f"{a:g}",
        "-c",
        f"{c:g}",
        "-a0k",
        f"{a0k:g}",
        "-b0k",
        f"{b0k:g}",
        "-opt_llk",
        str(opt_llk),
    ]
    return "\n".join(
        [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            "",
            *_optional_module_lines(module_load),
            f"echo 'Started SDPR disease={disease} chr={chrom} at '$(date)",
            f"export OMP_NUM_THREADS=${{SLURM_CPUS_PER_TASK:-{threads}}}",
            f"export MKL_NUM_THREADS=${{SLURM_CPUS_PER_TASK:-{threads}}}",
            f"export OPENBLAS_NUM_THREADS=${{SLURM_CPUS_PER_TASK:-{threads}}}",
            f"mkdir -p {q(effect_file.parent)}",
            _command_test(sdpr_bin, "SDPR"),
            f"test -s {q(sdpr_sumstats)} || {{ echo 'Missing SDPR summary statistics: {sdpr_sumstats}' >&2; exit 1; }}",
            f"test -s {q(dat)} || {{ echo 'Missing SDPR reference LD file: {dat}' >&2; exit 1; }}",
            f"test -s {q(snpinfo)} || {{ echo 'Missing SDPR SNP info file: {snpinfo}' >&2; exit 1; }}",
            f"test -s {q(valid_bim)} || {{ echo 'Missing SDPR validation BIM: {valid_bim}' >&2; exit 1; }}",
            "",
            f"if [[ ! -s {q(effect_file)} || \"${{REUSE_SDPR_EFFECTS:-1}}\" != \"1\" ]]; then",
            f"  {_render_command(mcmc_tokens)}",
            "else",
            f"  echo 'Reusing SDPR effect file: {effect_file}'",
            "fi",
            f"test -s {q(effect_file)} || {{ echo 'Missing SDPR effect file after MCMC: {effect_file}' >&2; exit 1; }}",
            f"echo 'Finished SDPR disease={disease} chr={chrom} at '$(date)",
            "",
        ]
    )


def _score_script(
    *,
    disease: str,
    effect_files: list[Path],
    target_id_map: Path | None,
    score_file: Path,
    score_summary: Path,
    genotype_prefix: Path,
    score_genotype_prefix: Path,
    fit_score_genotype_prefix: Path,
    score_keep: Path,
    fit_score_keep: Path,
    score_prefix: Path,
    fit_score_prefix: Path,
    python_bin: str,
    plink2: str,
    module_load: str | None,
    threads: int,
) -> str:
    effect_args = " ".join(q(path) for path in effect_files)
    target_id_map_args = f"--target-id-map {q(target_id_map)} " if target_id_map is not None else ""
    return "\n".join(
        [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            "",
            *_optional_module_lines(module_load),
            f"echo 'Started SDPR scoring disease={disease} at '$(date)",
            f"mkdir -p {q(score_file.parent)} {q(score_prefix.parent)}",
            _command_test(python_bin, "python"),
            _command_test(plink2, "plink2"),
            *[
                f"test -s {q(path)} || {{ echo 'Missing SDPR effect file: {path}' >&2; exit 1; }}"
                for path in effect_files
            ],
            f"test -f {q(str(genotype_prefix) + '.pgen')} || {{ echo 'Missing {genotype_prefix}.pgen' >&2; exit 1; }}",
            f"test -f {q(str(genotype_prefix) + '.pvar')} || {{ echo 'Missing {genotype_prefix}.pvar' >&2; exit 1; }}",
            f"test -f {q(str(genotype_prefix) + '.psam')} || {{ echo 'Missing {genotype_prefix}.psam' >&2; exit 1; }}",
            f"test -f {q(str(score_genotype_prefix) + '.pgen')} || {{ echo 'Missing {score_genotype_prefix}.pgen' >&2; exit 1; }}",
            f"test -f {q(str(score_genotype_prefix) + '.pvar')} || {{ echo 'Missing {score_genotype_prefix}.pvar' >&2; exit 1; }}",
            f"test -f {q(str(score_genotype_prefix) + '.psam')} || {{ echo 'Missing {score_genotype_prefix}.psam' >&2; exit 1; }}",
            f"test -f {q(str(fit_score_genotype_prefix) + '.pgen')} || {{ echo 'Missing {fit_score_genotype_prefix}.pgen' >&2; exit 1; }}",
            f"test -f {q(str(fit_score_genotype_prefix) + '.pvar')} || {{ echo 'Missing {fit_score_genotype_prefix}.pvar' >&2; exit 1; }}",
            f"test -f {q(str(fit_score_genotype_prefix) + '.psam')} || {{ echo 'Missing {fit_score_genotype_prefix}.psam' >&2; exit 1; }}",
            f"test -s {q(score_keep)} || {{ echo 'Missing scoring keep file: {score_keep}' >&2; exit 1; }}",
            f"test -s {q(fit_score_keep)} || {{ echo 'Missing model-fit scoring keep file: {fit_score_keep}' >&2; exit 1; }}",
            "",
            (
                f"if [[ -s {q(str(score_prefix) + '.sscore')} && -s {q(str(fit_score_prefix) + '.sscore')} "
                '&& "${REUSE_SDPR_SCORES:-0}" == "1" ]]; then'
            ),
            f"  echo 'Reusing SDPR PLINK2 scores: {score_prefix}.sscore and {fit_score_prefix}.sscore'",
            "  exit 0",
            "fi",
            "",
            (
                f"{q(python_bin)} -u scripts/build_sdpr_scorefile.py "
                f"--effect-files {effect_args} "
                f"{target_id_map_args}"
                f"--out {q(score_file)} "
                f"--summary-out {q(score_summary)}"
            ),
            (
                f"{q(plink2)} --pfile {q(score_genotype_prefix)} --keep {q(score_keep)} "
                f"--score {q(score_file)} 1 2 3 header-read cols=+scoresums "
                f"--threads \"${{SLURM_CPUS_PER_TASK:-{threads}}}\" --out {q(score_prefix)}"
            ),
            (
                f"{q(plink2)} --pfile {q(fit_score_genotype_prefix)} --keep {q(fit_score_keep)} "
                f"--score {q(score_file)} 1 2 3 header-read cols=+scoresums "
                f"--threads \"${{SLURM_CPUS_PER_TASK:-{threads}}}\" --out {q(fit_score_prefix)}"
            ),
            f"echo 'Finished SDPR scoring disease={disease} at '$(date)",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Write full-training SDPR commands for one disease.")
    parser.add_argument("--disease", required=True)
    parser.add_argument("--manifest", default="configs/official_sumstats.yaml")
    parser.add_argument(
        "--sumstats-root",
        default="data/raw/training_phase/supplementary_files/summary_statistics",
    )
    parser.add_argument("--genotype-prefix", default="outputs/genotype_qc/qc_final/all_chromosomes.qc")
    parser.add_argument(
        "--score-genotype-prefix",
        default=None,
        help="PGEN prefix used for final scoring. Defaults to --genotype-prefix.",
    )
    parser.add_argument(
        "--fit-score-genotype-prefix",
        default=None,
        help="PGEN prefix used for fitting/calibrating the validation model. Defaults to --genotype-prefix.",
    )
    parser.add_argument("--harmonized-dir", default="data/processed/sumstats/harmonized")
    parser.add_argument("--train-keep", default="data/processed/sample_sets/labeled_train.keep.psam")
    parser.add_argument("--score-keep", default="data/processed/sample_sets/unlabelled_validation.keep.psam")
    parser.add_argument("--fit-score-keep", default=None)
    parser.add_argument("--out-dir", default="outputs/sdpr_full_train")
    parser.add_argument("--variant-extract", default=None)
    parser.add_argument("--sdpr-bin", default="data/resources/sdpr/bin/SDPR")
    parser.add_argument("--ref-mode", choices=["target", "prebuilt"], default="target")
    parser.add_argument("--prebuilt-ref-dir", default=None)
    parser.add_argument("--python", default="python")
    parser.add_argument("--plink2", default="plink2")
    parser.add_argument("--module-load", default=None)
    parser.add_argument("--n-gwas", type=int, default=None)
    parser.add_argument("--default-n", type=float, default=None)
    parser.add_argument("--chromosomes", default=DEFAULT_CHROMOSOMES)
    parser.add_argument("--chunksize", type=int, default=250_000)
    parser.add_argument("--m-components", type=int, default=DEFAULT_M)
    parser.add_argument("--n-iter", type=int, default=DEFAULT_ITER)
    parser.add_argument("--n-burn", type=int, default=DEFAULT_BURN)
    parser.add_argument("--thin", type=int, default=DEFAULT_THIN)
    parser.add_argument("--r2", type=float, default=DEFAULT_R2)
    parser.add_argument("--a", type=float, default=DEFAULT_A)
    parser.add_argument("--c", type=float, default=DEFAULT_C)
    parser.add_argument("--a0k", type=float, default=DEFAULT_A0K)
    parser.add_argument("--b0k", type=float, default=DEFAULT_B0K)
    parser.add_argument("--opt-llk", type=int, choices=[1, 2], default=DEFAULT_OPT_LLK)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--commands-out", default=None)
    parser.add_argument("--task-list-out", default=None)
    parser.add_argument("--score-script-list-out", default=None)
    parser.add_argument("--task-manifest-out", default=None)
    args = parser.parse_args()

    entry = _manifest_entry(args.manifest, args.disease)
    accession = entry["accession"]
    raw_sumstats = Path(args.sumstats_root) / str(entry["file"])
    n_gwas = _resolve_sample_size(args.n_gwas, entry)
    default_n = _resolve_default_n(args.default_n, entry, n_gwas)
    chromosomes = _chromosomes(args.chromosomes)
    if args.ref_mode == "prebuilt" and not args.prebuilt_ref_dir:
        raise ValueError("--prebuilt-ref-dir is required when --ref-mode prebuilt")

    harmonized = Path(args.harmonized_dir) / f"{args.disease}.{accession}.harmonized.tsv.gz"
    out_dir = Path(args.out_dir) / args.disease
    input_dir = out_dir / "inputs"
    keep_dir = out_dir / "keep"
    task_dir = out_dir / "tasks"
    bed_dir = out_dir / "bed"
    ref_dir = Path(args.prebuilt_ref_dir) if args.ref_mode == "prebuilt" else out_dir / "ref"
    effect_dir = out_dir / "effects"
    scorefile_dir = out_dir / "scorefiles"
    score_dir = out_dir / "scores"
    command_dir = out_dir / "commands"
    report_dir = out_dir / "reports"
    for path in [
        input_dir,
        keep_dir,
        task_dir,
        bed_dir,
        ref_dir,
        effect_dir,
        scorefile_dir,
        score_dir,
        command_dir,
        report_dir,
    ]:
        path.mkdir(parents=True, exist_ok=True)

    train_keep = Path(args.train_keep)
    score_keep = Path(args.score_keep)
    fit_score_keep = Path(args.fit_score_keep) if args.fit_score_keep else train_keep
    genotype_prefix = Path(args.genotype_prefix)
    score_genotype_prefix = (
        Path(args.score_genotype_prefix) if args.score_genotype_prefix else genotype_prefix
    )
    fit_score_genotype_prefix = (
        Path(args.fit_score_genotype_prefix) if args.fit_score_genotype_prefix else genotype_prefix
    )
    all_keep = keep_dir / f"{args.disease}.full_train.all.keep.psam"
    _write_union_keep([train_keep, score_keep, fit_score_keep], all_keep)

    ref_bed_prefix = bed_dir / f"{args.disease}.full_train.reference_samples"
    score_bed_prefix = bed_dir / f"{args.disease}.full_train.score_samples"
    sdpr_sumstats = input_dir / f"{args.disease}.sdpr.ss"
    target_id_map = input_dir / f"{args.disease}.sdpr.target_id_map.tsv" if args.ref_mode == "prebuilt" else None
    valid_bim = (
        input_dir / f"{args.disease}.sdpr.valid.bim"
        if args.ref_mode == "prebuilt"
        else Path(str(score_bed_prefix) + ".bim")
    )
    input_summary = report_dir / f"{args.disease}.sdpr_input.summary.json"
    score_file = scorefile_dir / f"{args.disease}.sdpr.score.tsv"
    score_summary = scorefile_dir / f"{args.disease}.sdpr.score.summary.json"
    score_prefix = score_dir / f"{args.disease}.sdpr"
    fit_score_prefix = score_dir / f"{args.disease}.sdpr.fit"
    variant_extract = Path(args.variant_extract) if args.variant_extract else None

    prepare_script = command_dir / f"{args.disease}.sdpr_prepare_reference.sh"
    prepare_script.write_text(
        _prepare_reference_script(
            disease=args.disease,
            ref_mode=args.ref_mode,
            raw_sumstats=raw_sumstats if args.ref_mode == "prebuilt" else None,
            python_bin=args.python,
            plink2=args.plink2,
            sdpr_bin=args.sdpr_bin,
            module_load=args.module_load,
            genotype_prefix=genotype_prefix,
            harmonized=harmonized,
            train_keep=train_keep,
            score_keep=score_keep,
            fit_score_keep=fit_score_keep,
            all_keep=all_keep,
            ref_bed_prefix=ref_bed_prefix,
            score_bed_prefix=score_bed_prefix,
            variant_extract=variant_extract,
            sdpr_sumstats=sdpr_sumstats,
            target_id_map=target_id_map,
            valid_bim=valid_bim,
            summary_out=input_summary,
            ref_dir=ref_dir,
            chromosomes=chromosomes,
            default_n=default_n,
            chunksize=args.chunksize,
            r2=args.r2,
            threads=args.threads,
        ),
        encoding="utf-8",
    )
    _chmod_executable(prepare_script)

    task_scripts = []
    effect_files = []
    for chrom in chromosomes:
        effect_file = effect_dir / f"{args.disease}.chr{chrom}.sdpr.txt"
        task_script = task_dir / f"{args.disease}.chr{chrom}.sdpr.sh"
        task_script.write_text(
            _task_script(
                disease=args.disease,
                chrom=chrom,
                sdpr_bin=args.sdpr_bin,
                module_load=args.module_load,
                ref_dir=ref_dir,
                sdpr_sumstats=sdpr_sumstats,
                valid_bim=valid_bim,
                effect_file=effect_file,
                n_gwas=n_gwas,
                m_components=args.m_components,
                n_iter=args.n_iter,
                n_burn=args.n_burn,
                thin=args.thin,
                a=args.a,
                c=args.c,
                a0k=args.a0k,
                b0k=args.b0k,
                opt_llk=args.opt_llk,
                threads=args.threads,
            ),
            encoding="utf-8",
        )
        _chmod_executable(task_script)
        task_scripts.append(task_script)
        effect_files.append(effect_file)

    score_script = command_dir / f"{args.disease}.sdpr.score.sh"
    score_script.write_text(
        _score_script(
            disease=args.disease,
            effect_files=effect_files,
            target_id_map=target_id_map,
            score_file=score_file,
            score_summary=score_summary,
            genotype_prefix=genotype_prefix,
            score_genotype_prefix=score_genotype_prefix,
            fit_score_genotype_prefix=fit_score_genotype_prefix,
            score_keep=score_keep,
            fit_score_keep=fit_score_keep,
            score_prefix=score_prefix,
            fit_score_prefix=fit_score_prefix,
            python_bin=args.python,
            plink2=args.plink2,
            module_load=args.module_load,
            threads=args.threads,
        ),
        encoding="utf-8",
    )
    _chmod_executable(score_script)

    commands_out = (
        Path(args.commands_out)
        if args.commands_out
        else command_dir / f"{args.disease}.sdpr_full_train.commands.sh"
    )
    task_list_out = (
        Path(args.task_list_out)
        if args.task_list_out
        else command_dir / f"{args.disease}.sdpr_full_train.array_tasks.txt"
    )
    score_script_list_out = (
        Path(args.score_script_list_out)
        if args.score_script_list_out
        else command_dir / f"{args.disease}.sdpr_full_train.score_scripts.txt"
    )
    task_manifest_out = (
        Path(args.task_manifest_out)
        if args.task_manifest_out
        else command_dir / f"{args.disease}.sdpr_full_train.tasks.csv"
    )

    task_list_out.write_text("\n".join(str(path) for path in task_scripts) + "\n", encoding="utf-8")
    score_script_list_out.write_text(str(score_script) + "\n", encoding="utf-8")
    commands_out.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                "",
                f"bash {q(prepare_script)}",
                f"while IFS= read -r TASK_SCRIPT; do bash \"$TASK_SCRIPT\"; done < {q(task_list_out)}",
                f"bash {q(score_script)}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    _chmod_executable(commands_out)

    manifest_rows = [
        {
            "disease": args.disease,
            "model_scope": "full_train",
            "model": sdpr_model_tag(),
            "ref_mode": args.ref_mode,
            "n_gwas": n_gwas,
            "default_n": default_n,
            "chromosomes": ",".join(chromosomes),
            "m_components": args.m_components,
            "n_iter": args.n_iter,
            "n_burn": args.n_burn,
            "thin": args.thin,
            "r2": args.r2,
            "a": args.a,
            "c": args.c,
            "a0k": args.a0k,
            "b0k": args.b0k,
            "opt_llk": args.opt_llk,
            "sdpr_bin": args.sdpr_bin,
            "prepare_script": str(prepare_script),
            "score_script": str(score_script),
            "genotype_prefix": str(genotype_prefix),
            "score_genotype_prefix": str(score_genotype_prefix),
            "fit_score_genotype_prefix": str(fit_score_genotype_prefix),
            "train_keep": str(train_keep),
            "score_keep": str(score_keep),
            "fit_score_keep": str(fit_score_keep),
            "all_keep": str(all_keep),
            "ref_bed_prefix": str(ref_bed_prefix),
            "score_bed_prefix": str(score_bed_prefix),
            "variant_extract": str(variant_extract) if variant_extract is not None else "",
            "harmonized": str(harmonized),
            "sdpr_sumstats": str(sdpr_sumstats),
            "raw_sumstats": str(raw_sumstats) if args.ref_mode == "prebuilt" else "",
            "target_id_map": str(target_id_map) if target_id_map is not None else "",
            "valid_bim": str(valid_bim),
            "ref_dir": str(ref_dir),
            "effect_files": ",".join(str(path) for path in effect_files),
            "score_file": str(score_file),
            "sscore": str(score_prefix) + ".sscore",
            "fit_sscore": str(fit_score_prefix) + ".sscore",
        }
    ]
    write_table(pd.DataFrame(manifest_rows), task_manifest_out)

    print(f"Wrote SDPR reference preparation script to {prepare_script}")
    print(f"Wrote {len(task_scripts)} SDPR chromosome task scripts to {task_list_out}")
    print(f"Wrote SDPR scoring script to {score_script_list_out}")
    print(f"Wrote sequential command script to {commands_out}")
    print(f"Wrote SDPR task manifest to {task_manifest_out}")


if __name__ == "__main__":
    main()
