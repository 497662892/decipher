from __future__ import annotations

import argparse
import shlex
import stat
from pathlib import Path

import pandas as pd

from decipehr.config import load_yaml
from decipehr.io import write_table
from decipehr.sbayesrc import (
    DEFAULT_BURN_IN,
    DEFAULT_CHAIN_LENGTH,
    DEFAULT_CHROMOSOMES,
    DEFAULT_THIN,
    sbayesrc_model_tag,
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
    tokens = [
        q(plink2),
        "--pfile",
        q(genotype_prefix),
        "--keep",
        q(train_keep),
    ]
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
            f"echo 'Started sBayesRC training allele frequency disease={disease} at '$(date)",
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
            f"if [[ ! -s {q(str(freq_prefix) + '.afreq')} || \"${{REUSE_SBAYESRC_FREQ:-1}}\" != \"1\" ]]; then",
            "  " + (" \\\n  ".join(tokens)),
            "else",
            f"  echo 'Reusing sBayesRC frequency file: {freq_prefix}.afreq'",
            "fi",
            f"test -s {q(str(freq_prefix) + '.afreq')} || {{ echo 'Missing frequency output: {freq_prefix}.afreq' >&2; exit 1; }}",
            f"echo 'Finished sBayesRC training allele frequency disease={disease} at '$(date)",
            "",
        ]
    )


def _prepare_script(
    *,
    disease: str,
    python_bin: str,
    raw_sumstats: Path,
    harmonized: Path,
    freq_file: Path,
    ma_file: Path,
    target_id_map: Path,
    summary_out: Path,
    chromosomes: str,
    default_n: float,
    chunksize: int,
) -> str:
    return "\n".join(
        [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            "",
            f"echo 'Started sBayesRC .ma preparation disease={disease} at '$(date)",
            f"mkdir -p {q(ma_file.parent)} {q(summary_out.parent)}",
            f"test -s {q(raw_sumstats)} || {{ echo 'Missing raw sumstats: {raw_sumstats}' >&2; exit 1; }}",
            f"test -s {q(harmonized)} || {{ echo 'Missing harmonized sumstats: {harmonized}' >&2; exit 1; }}",
            f"test -s {q(freq_file)} || {{ echo 'Missing PLINK2 frequency file: {freq_file}' >&2; exit 1; }}",
            (
                f"{q(python_bin)} -u scripts/prepare_sbayesrc_inputs.py "
                f"--raw-sumstats {q(raw_sumstats)} "
                f"--harmonized {q(harmonized)} "
                f"--freq {q(freq_file)} "
                f"--out {q(ma_file)} "
                f"--target-id-map-out {q(target_id_map)} "
                f"--summary-out {q(summary_out)} "
                f"--chromosomes {q(chromosomes)} "
                f"--default-n {default_n:g} "
                f"--chunksize {chunksize}"
            ),
            f"echo 'Finished sBayesRC .ma preparation disease={disease} at '$(date)",
            "",
        ]
    )


def _impute_script(
    *,
    disease: str,
    gctb: str,
    module_load: str | None,
    ldm_eigen_dir: Path,
    ma_file: Path,
    impute_out_prefix: Path,
    imputed_ma: Path,
    threads: int,
) -> str:
    return "\n".join(
        [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            "",
            *_optional_module_lines(module_load),
            f"echo 'Started sBayesRC summary-stat imputation disease={disease} at '$(date)",
            _command_test(gctb, "GCTB"),
            f"test -d {q(ldm_eigen_dir)} || {{ echo 'Missing sBayesRC LD eigen directory: {ldm_eigen_dir}' >&2; exit 1; }}",
            f"test -s {q(ma_file)} || {{ echo 'Missing sBayesRC .ma file: {ma_file}' >&2; exit 1; }}",
            f"mkdir -p {q(impute_out_prefix.parent)}",
            f"if [[ ! -s {q(imputed_ma)} || \"${{REUSE_SBAYESRC_IMPUTED:-1}}\" != \"1\" ]]; then",
            (
                f"  {q(gctb)} --ldm-eigen {q(ldm_eigen_dir)} "
                f"--gwas-summary {q(ma_file)} --impute-summary "
                f"--thread \"${{SLURM_CPUS_PER_TASK:-{threads}}}\" "
                f"--out {q(impute_out_prefix)}"
            ),
            "else",
            f"  echo 'Reusing sBayesRC imputed summary statistics: {imputed_ma}'",
            "fi",
            f"test -s {q(imputed_ma)} || {{ echo 'Missing imputed summary statistics: {imputed_ma}' >&2; exit 1; }}",
            f"echo 'Finished sBayesRC summary-stat imputation disease={disease} at '$(date)",
            "",
        ]
    )


def _sbayesrc_script(
    *,
    disease: str,
    gctb: str,
    module_load: str | None,
    ldm_eigen_dir: Path,
    annot_file: Path,
    imputed_ma: Path,
    effect_prefix: Path,
    snpres: Path,
    chain_length: int,
    burn_in: int,
    thin: int,
    seed: int,
    threads: int,
    ldm_eigen_cutoff: float | None,
    no_mcmc_bin: bool,
) -> str:
    optional = " --no-mcmc-bin" if no_mcmc_bin else ""
    cutoff_arg = f" --ldm-eigen-cutoff {ldm_eigen_cutoff:g}" if ldm_eigen_cutoff else ""
    return "\n".join(
        [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            "",
            *_optional_module_lines(module_load),
            f"echo 'Started sBayesRC disease={disease} at '$(date)",
            _command_test(gctb, "GCTB"),
            f"test -d {q(ldm_eigen_dir)} || {{ echo 'Missing sBayesRC LD eigen directory: {ldm_eigen_dir}' >&2; exit 1; }}",
            f"test -s {q(annot_file)} || {{ echo 'Missing sBayesRC annotation file: {annot_file}' >&2; exit 1; }}",
            f"test -s {q(imputed_ma)} || {{ echo 'Missing imputed sBayesRC .ma file: {imputed_ma}' >&2; exit 1; }}",
            f"mkdir -p {q(effect_prefix.parent)}",
            f"if [[ ! -s {q(snpres)} || \"${{REUSE_SBAYESRC_EFFECTS:-1}}\" != \"1\" ]]; then",
            (
                f"  {q(gctb)} --ldm-eigen {q(ldm_eigen_dir)} "
                f"--gwas-summary {q(imputed_ma)} --sbayes RC "
                f"--annot {q(annot_file)} "
                f"--chain-length {chain_length} --burn-in {burn_in} --thin {thin} "
                f"--seed {seed} --thread \"${{SLURM_CPUS_PER_TASK:-{threads}}}\""
                f"{cutoff_arg}{optional} --out {q(effect_prefix)}"
            ),
            "else",
            f"  echo 'Reusing sBayesRC effects: {snpres}'",
            "fi",
            f"test -s {q(snpres)} || {{ echo 'Missing sBayesRC snpRes output: {snpres}' >&2; exit 1; }}",
            f"echo 'Finished sBayesRC disease={disease} at '$(date)",
            "",
        ]
    )


def _score_script(
    *,
    disease: str,
    python_bin: str,
    plink2: str,
    module_load: str | None,
    snpres: Path,
    target_id_map: Path,
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
            f"echo 'Started sBayesRC scoring disease={disease} at '$(date)",
            _command_test(plink2, "PLINK2"),
            f"test -s {q(snpres)} || {{ echo 'Missing sBayesRC snpRes: {snpres}' >&2; exit 1; }}",
            f"test -s {q(target_id_map)} || {{ echo 'Missing sBayesRC target ID map: {target_id_map}' >&2; exit 1; }}",
            f"test -f {q(str(genotype_prefix) + '.pgen')} || {{ echo 'Missing {genotype_prefix}.pgen' >&2; exit 1; }}",
            f"test -f {q(str(genotype_prefix) + '.pvar')} || {{ echo 'Missing {genotype_prefix}.pvar' >&2; exit 1; }}",
            f"test -f {q(str(genotype_prefix) + '.psam')} || {{ echo 'Missing {genotype_prefix}.psam' >&2; exit 1; }}",
            f"test -s {q(score_keep)} || {{ echo 'Missing scoring keep file: {score_keep}' >&2; exit 1; }}",
            f"test -s {q(fit_score_keep)} || {{ echo 'Missing model-fit scoring keep file: {fit_score_keep}' >&2; exit 1; }}",
            f"mkdir -p {q(score_file.parent)} {q(score_prefix.parent)}",
            (
                f"{q(python_bin)} -u scripts/build_sbayesrc_scorefile.py "
                f"--snpres {q(snpres)} "
                f"--target-id-map {q(target_id_map)} "
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
            f"echo 'Finished sBayesRC scoring disease={disease} at '$(date)",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Write full-training sBayesRC commands for one disease.")
    parser.add_argument("--disease", required=True)
    parser.add_argument("--manifest", default="configs/official_sumstats.yaml")
    parser.add_argument("--sumstats-root", required=True)
    parser.add_argument("--genotype-prefix", default="outputs/genotype_qc/qc_final/all_chromosomes.qc")
    parser.add_argument("--harmonized-dir", default="data/processed/sumstats/harmonized")
    parser.add_argument("--train-keep", default="data/processed/sample_sets/labeled_train.keep.psam")
    parser.add_argument("--score-keep", default="data/processed/sample_sets/unlabelled_validation.keep.psam")
    parser.add_argument("--fit-score-keep", default=None)
    parser.add_argument("--variant-extract", default=None)
    parser.add_argument("--out-dir", default="outputs/sbayesrc_full_train")
    parser.add_argument("--ldm-eigen-dir", required=True)
    parser.add_argument("--annot-file", required=True)
    parser.add_argument("--gctb", default="gctb")
    parser.add_argument("--python", default="python")
    parser.add_argument("--plink2", default="plink2")
    parser.add_argument("--gctb-module", default=None)
    parser.add_argument("--plink-module", default=None)
    parser.add_argument("--n-gwas", type=float, default=None)
    parser.add_argument("--chromosomes", default=DEFAULT_CHROMOSOMES)
    parser.add_argument("--chain-length", type=int, default=DEFAULT_CHAIN_LENGTH)
    parser.add_argument("--burn-in", type=int, default=DEFAULT_BURN_IN)
    parser.add_argument("--thin", type=int, default=DEFAULT_THIN)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--ldm-eigen-cutoff", type=float, default=None)
    parser.add_argument("--chunksize", type=int, default=250_000)
    parser.add_argument("--no-mcmc-bin", action="store_true")
    args = parser.parse_args()

    entry = _manifest_entry(args.manifest, args.disease)
    accession = entry["accession"]
    n_gwas = args.n_gwas if args.n_gwas is not None else entry.get("default_n")
    if n_gwas is None:
        raise ValueError(
            f"No --n-gwas was supplied and manifest entry {args.disease!r} has no default_n"
        )
    n_gwas = float(n_gwas)

    raw_sumstats = Path(args.sumstats_root) / str(entry["file"])
    harmonized = Path(args.harmonized_dir) / f"{args.disease}.{accession}.harmonized.tsv.gz"
    out_dir = Path(args.out_dir) / args.disease
    input_dir = out_dir / "inputs"
    task_dir = out_dir / "tasks"
    effect_dir = out_dir / "effects"
    scorefile_dir = out_dir / "scorefiles"
    score_dir = out_dir / "scores"
    command_dir = out_dir / "commands"
    report_dir = out_dir / "reports"
    for path in [input_dir, task_dir, effect_dir, scorefile_dir, score_dir, command_dir, report_dir]:
        path.mkdir(parents=True, exist_ok=True)

    freq_prefix = input_dir / f"{args.disease}.sbayesrc.training"
    freq_file = Path(f"{freq_prefix}.afreq")
    ma_file = input_dir / f"{args.disease}.sbayesrc.ma"
    impute_prefix = input_dir / f"{args.disease}.sbayesrc"
    imputed_ma = input_dir / f"{args.disease}.sbayesrc.imputed.ma"
    target_id_map = input_dir / f"{args.disease}.sbayesrc.target_id_map.tsv"
    effect_prefix = effect_dir / f"{args.disease}.sbayesrc"
    snpres = effect_dir / f"{args.disease}.sbayesrc.snpRes"
    score_file = scorefile_dir / f"{args.disease}.sbayesrc.score.tsv"
    score_summary = scorefile_dir / f"{args.disease}.sbayesrc.score.summary.json"
    score_prefix = score_dir / f"{args.disease}.sbayesrc"
    fit_score_prefix = score_dir / f"{args.disease}.sbayesrc.fit"
    genotype_prefix = Path(args.genotype_prefix)
    score_keep = Path(args.score_keep)
    fit_score_keep = Path(args.fit_score_keep) if args.fit_score_keep else Path(args.train_keep)
    variant_extract = Path(args.variant_extract) if args.variant_extract else None

    freq_script = command_dir / f"{args.disease}.sbayesrc.frequency.sh"
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

    prepare_script = command_dir / f"{args.disease}.sbayesrc.prepare_ma.sh"
    prepare_script.write_text(
        _prepare_script(
            disease=args.disease,
            python_bin=args.python,
            raw_sumstats=raw_sumstats,
            harmonized=harmonized,
            freq_file=freq_file,
            ma_file=ma_file,
            target_id_map=target_id_map,
            summary_out=report_dir / f"{args.disease}.sbayesrc_input.summary.json",
            chromosomes=args.chromosomes,
            default_n=n_gwas,
            chunksize=args.chunksize,
        ),
        encoding="utf-8",
    )
    _chmod_executable(prepare_script)

    impute_script = command_dir / f"{args.disease}.sbayesrc.impute_summary.sh"
    impute_script.write_text(
        _impute_script(
            disease=args.disease,
            gctb=args.gctb,
            module_load=args.gctb_module,
            ldm_eigen_dir=Path(args.ldm_eigen_dir),
            ma_file=ma_file,
            impute_out_prefix=impute_prefix,
            imputed_ma=imputed_ma,
            threads=args.threads,
        ),
        encoding="utf-8",
    )
    _chmod_executable(impute_script)

    run_script = task_dir / f"{args.disease}.sbayesrc.sh"
    run_script.write_text(
        _sbayesrc_script(
            disease=args.disease,
            gctb=args.gctb,
            module_load=args.gctb_module,
            ldm_eigen_dir=Path(args.ldm_eigen_dir),
            annot_file=Path(args.annot_file),
            imputed_ma=imputed_ma,
            effect_prefix=effect_prefix,
            snpres=snpres,
            chain_length=args.chain_length,
            burn_in=args.burn_in,
            thin=args.thin,
            seed=args.seed,
            threads=args.threads,
            ldm_eigen_cutoff=args.ldm_eigen_cutoff,
            no_mcmc_bin=args.no_mcmc_bin,
        ),
        encoding="utf-8",
    )
    _chmod_executable(run_script)

    score_script = command_dir / f"{args.disease}.sbayesrc.score.sh"
    score_script.write_text(
        _score_script(
            disease=args.disease,
            python_bin=args.python,
            plink2=args.plink2,
            module_load=args.plink_module,
            snpres=snpres,
            target_id_map=target_id_map,
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

    workflow_script = command_dir / f"{args.disease}.sbayesrc_full_train.commands.sh"
    workflow_script.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                f"bash {q(freq_script)}",
                f"bash {q(prepare_script)}",
                f"bash {q(impute_script)}",
                f"bash {q(run_script)}",
                f"bash {q(score_script)}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    _chmod_executable(workflow_script)

    score_list = command_dir / f"{args.disease}.sbayesrc_full_train.score_scripts.txt"
    score_list.write_text(str(score_script) + "\n", encoding="utf-8")
    manifest_out = command_dir / f"{args.disease}.sbayesrc_full_train.tasks.csv"
    write_table(
        pd.DataFrame(
            [
                {
                    "disease": args.disease,
                    "model_scope": "full_train",
                    "model": sbayesrc_model_tag(),
                    "n_gwas": n_gwas,
                    "chain_length": args.chain_length,
                    "burn_in": args.burn_in,
                    "thin": args.thin,
                    "seed": args.seed,
                    "workflow_script": str(workflow_script),
                    "frequency_script": str(freq_script),
                    "prepare_script": str(prepare_script),
                    "impute_script": str(impute_script),
                    "run_script": str(run_script),
                    "score_script": str(score_script),
                    "ma_file": str(ma_file),
                    "imputed_ma": str(imputed_ma),
                    "target_id_map": str(target_id_map),
                    "snpres": str(snpres),
                    "score_file": str(score_file),
                    "sscore": str(score_prefix) + ".sscore",
                    "fit_sscore": str(fit_score_prefix) + ".sscore",
                    "ldm_eigen_dir": str(args.ldm_eigen_dir),
                    "ldm_eigen_cutoff": args.ldm_eigen_cutoff,
                    "annot_file": str(args.annot_file),
                    "gctb": args.gctb,
                }
            ]
        ),
        manifest_out,
    )

    print(f"Wrote sBayesRC workflow script to {workflow_script}")
    print(f"Wrote sBayesRC score script list to {score_list}")
    print(f"Wrote sBayesRC task manifest to {manifest_out}")


if __name__ == "__main__":
    main()
