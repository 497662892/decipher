from __future__ import annotations

import argparse
import shlex
import stat
from pathlib import Path

import pandas as pd

from decipehr.config import load_yaml
from decipehr.io import read_table, write_table
from decipehr.lassosum2 import (
    DEFAULT_DELTA_GRID,
    DEFAULT_DFMAX,
    DEFAULT_LAMBDA_MIN_RATIO,
    DEFAULT_MAXITER,
    DEFAULT_NLAMBDA,
    DEFAULT_TOL,
    lassosum2_parameter_grid,
)
from decipehr.schema import parse_csv_list


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


def _float_list(values: str) -> list[float]:
    parsed = [float(value) for value in parse_csv_list(values)]
    if not parsed:
        raise ValueError("Grid must contain at least one numeric value")
    return parsed


def _csv_float(values: list[float]) -> str:
    return ",".join(f"{value:g}" for value in values)


def _bool_arg(value: bool) -> str:
    return "true" if value else "false"


def _parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    lowered = value.lower()
    if lowered in {"1", "true", "t", "yes", "y"}:
        return True
    if lowered in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected true/false, got {value!r}")


def _resolve_default_n(cli_default_n: float | None, entry: dict[str, object]) -> float | None:
    if cli_default_n is not None:
        return cli_default_n
    manifest_default = entry.get("default_n")
    if manifest_default in (None, ""):
        return None
    value = float(manifest_default)
    if value <= 0:
        raise ValueError("Manifest default_n must be positive")
    return value


def _render_command(tokens: list[str]) -> str:
    return " \\\n  ".join(tokens)


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


def _task_script(
    *,
    disease: str,
    plink2: str,
    rscript: str,
    lassosum2_script: Path,
    genotype_prefix: Path,
    score_genotype_prefix: Path,
    fit_score_genotype_prefix: Path,
    harmonized: Path,
    train_keep: Path,
    score_keep: Path,
    fit_score_keep: Path,
    all_keep: Path,
    bed_prefix: Path,
    score_bed_prefix: Path,
    fit_score_bed_prefix: Path,
    variant_extract: Path | None,
    score_file: Path,
    fit_score_file: Path,
    summary_file: Path,
    model_file: Path,
    ld_backing_prefix: Path,
    tmp_dir: Path,
    patient_col: str,
    delta_grid: list[float],
    nlambda: int,
    lambda_min_ratio: float,
    dfmax: int,
    maxiter: int,
    tol: float,
    chromosomes: str,
    threads: int,
    default_n: float | None,
    join_by_pos: bool,
    save_beta_grid: bool,
    ld_reference_mode: str,
    ld_ref_map: Path,
    ld_ref_dir: Path,
    ld_ref_pattern: str,
) -> str:
    def plink_make_bed_tokens(pfile_prefix: Path, keep_path: Path, out_prefix: Path) -> list[str]:
        tokens = [
            q(plink2),
            "--pfile",
            q(pfile_prefix),
            "--keep",
            q(keep_path),
        ]
        if chromosomes:
            tokens.extend(["--chr", q(chromosomes)])
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

    train_plink_tokens = plink_make_bed_tokens(genotype_prefix, train_keep, bed_prefix)
    score_plink_tokens = plink_make_bed_tokens(score_genotype_prefix, score_keep, score_bed_prefix)
    fit_score_plink_tokens = plink_make_bed_tokens(
        fit_score_genotype_prefix,
        fit_score_keep,
        fit_score_bed_prefix,
    )

    r_tokens = [
        q(rscript),
        q(lassosum2_script),
        "--disease",
        q(disease),
        "--fold",
        "0",
        "--model-scope",
        "full_train",
        "--bed-prefix",
        q(bed_prefix),
        "--score-bed-prefix",
        q(score_bed_prefix),
        "--fit-score-bed-prefix",
        q(fit_score_bed_prefix),
        "--harmonized",
        q(harmonized),
        "--train-keep",
        q(train_keep),
        "--score-keep",
        q(score_keep),
        "--fit-score-keep",
        q(fit_score_keep),
        "--patient-col",
        q(patient_col),
        "--ld-reference-mode",
        q(ld_reference_mode),
        "--ld-ref-map",
        q(ld_ref_map),
        "--ld-ref-dir",
        q(ld_ref_dir),
        "--ld-ref-pattern",
        q(ld_ref_pattern),
        "--score-out",
        q(score_file),
        "--fit-score-out",
        q(fit_score_file),
        "--summary-out",
        q(summary_file),
        "--model-out",
        q(model_file),
        "--ld-backing-prefix",
        q(ld_backing_prefix),
        "--delta-grid",
        q(_csv_float(delta_grid)),
        "--nlambda",
        str(nlambda),
        "--lambda-min-ratio",
        f"{lambda_min_ratio:g}",
        "--dfmax",
        str(dfmax),
        "--maxiter",
        str(maxiter),
        "--tol",
        f"{tol:g}",
        "--threads",
        f'"${{SLURM_CPUS_PER_TASK:-{threads}}}"',
        "--join-by-pos",
        _bool_arg(join_by_pos),
        "--save-beta-grid",
        _bool_arg(save_beta_grid),
        "--reuse-ld-corr",
        '"${REUSE_LASSOSUM2_LD:-1}"',
    ]
    if default_n is not None:
        r_tokens.extend(["--default-n", f"{default_n:g}"])

    bed_setup_lines = [
        f"if [[ ! -s {q(str(bed_prefix) + '.bed')} || \"${{REUSE_LASSOSUM2_BED:-1}}\" != \"1\" ]]; then",
        f"  {_render_command(train_plink_tokens)}",
        "else",
        f"  echo 'Reusing existing PLINK bed files: {bed_prefix}'",
        "fi",
    ]
    if score_bed_prefix != bed_prefix:
        bed_setup_lines.extend(
            [
                "",
                f"if [[ ! -s {q(str(score_bed_prefix) + '.bed')} || \"${{REUSE_LASSOSUM2_BED:-1}}\" != \"1\" ]]; then",
                f"  {_render_command(score_plink_tokens)}",
                "else",
                f"  echo 'Reusing existing scoring PLINK bed files: {score_bed_prefix}'",
                "fi",
            ]
        )
    if fit_score_bed_prefix not in {bed_prefix, score_bed_prefix}:
        bed_setup_lines.extend(
            [
                "",
                f"if [[ ! -s {q(str(fit_score_bed_prefix) + '.bed')} || \"${{REUSE_LASSOSUM2_BED:-1}}\" != \"1\" ]]; then",
                f"  {_render_command(fit_score_plink_tokens)}",
                "else",
                f"  echo 'Reusing existing model-fit scoring PLINK bed files: {fit_score_bed_prefix}'",
                "fi",
            ]
        )

    bed_check_lines = []
    for label, prefix in [
        ("training", bed_prefix),
        ("scoring", score_bed_prefix),
        ("model-fit scoring", fit_score_bed_prefix),
    ]:
        bed_check_lines.extend(
            [
                f"test -s {q(str(prefix) + '.bed')} || {{ echo 'Missing {label} {prefix}.bed' >&2; exit 1; }}",
                f"test -s {q(str(prefix) + '.bim')} || {{ echo 'Missing {label} {prefix}.bim' >&2; exit 1; }}",
                f"test -s {q(str(prefix) + '.fam')} || {{ echo 'Missing {label} {prefix}.fam' >&2; exit 1; }}",
            ]
        )

    pfile_check_lines = []
    for label, prefix in [
        ("training", genotype_prefix),
        ("scoring", score_genotype_prefix),
        ("model-fit scoring", fit_score_genotype_prefix),
    ]:
        pfile_check_lines.extend(
            [
                f"test -f {q(str(prefix) + '.pgen')} || {{ echo 'Missing {label} {prefix}.pgen' >&2; exit 1; }}",
                f"test -f {q(str(prefix) + '.pvar')} || {{ echo 'Missing {label} {prefix}.pvar' >&2; exit 1; }}",
                f"test -f {q(str(prefix) + '.psam')} || {{ echo 'Missing {label} {prefix}.psam' >&2; exit 1; }}",
            ]
        )

    return "\n".join(
        [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            "",
            f"echo 'Started full-train lassosum2 task disease={disease} at '$(date)",
            (
                f"mkdir -p {q(bed_prefix.parent)} {q(score_file.parent)} {q(summary_file.parent)} "
                f"{q(model_file.parent)} {q(ld_backing_prefix.parent)} {q(tmp_dir)}"
            ),
            *pfile_check_lines,
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
            f"test -s {q(ld_ref_map)} || {{ echo 'Missing LD reference map: {ld_ref_map}' >&2; exit 1; }}",
            f"test -d {q(ld_ref_dir)} || {{ echo 'Missing LD reference directory: {ld_ref_dir}' >&2; exit 1; }}",
            (
                f"test -s {q(ld_ref_dir / ld_ref_pattern.replace('{chr}', '22'))} || "
                f"{{ echo 'Missing LD reference chr22 matrix under {ld_ref_dir}' >&2; exit 1; }}"
            ),
            "",
            (
                f"if [[ -s {q(score_file)} && -s {q(fit_score_file)} "
                "&& \"${REUSE_LASSOSUM2_SCORES:-0}\" == \"1\" ]]; then"
            ),
            f"  echo 'Reusing existing lassosum2 scores: {score_file} and {fit_score_file}'",
            "  exit 0",
            "fi",
            "",
            *bed_setup_lines,
            "",
            *bed_check_lines,
            "",
            _render_command(r_tokens),
            f"echo 'Finished full-train lassosum2 task disease={disease} at '$(date)",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Write full-training lassosum2 commands for one disease.")
    parser.add_argument("--disease", required=True)
    parser.add_argument("--manifest", default="configs/official_sumstats.yaml")
    parser.add_argument("--patient-col", default="sampleid")
    parser.add_argument("--genotype-prefix", default="outputs/genotype_qc/qc_final/all_chromosomes.qc")
    parser.add_argument("--score-genotype-prefix", default=None)
    parser.add_argument("--fit-score-genotype-prefix", default=None)
    parser.add_argument("--harmonized-dir", default="data/processed/sumstats/harmonized")
    parser.add_argument("--train-keep", default="data/processed/sample_sets/labeled_train.keep.psam")
    parser.add_argument("--score-keep", default="data/processed/sample_sets/unlabelled_validation.keep.psam")
    parser.add_argument("--fit-score-keep", default=None)
    parser.add_argument("--out-dir", default="outputs/lassosum2_ukbb_full_train")
    parser.add_argument("--plink2", default="plink2")
    parser.add_argument("--rscript", default="Rscript")
    parser.add_argument("--lassosum2-script", default="scripts/run_lassosum2_task.R")
    parser.add_argument("--delta-grid", default=_csv_float(DEFAULT_DELTA_GRID))
    parser.add_argument("--nlambda", type=int, default=DEFAULT_NLAMBDA)
    parser.add_argument("--lambda-min-ratio", type=float, default=DEFAULT_LAMBDA_MIN_RATIO)
    parser.add_argument("--dfmax", type=int, default=DEFAULT_DFMAX)
    parser.add_argument("--maxiter", type=int, default=DEFAULT_MAXITER)
    parser.add_argument("--tol", type=float, default=DEFAULT_TOL)
    parser.add_argument(
        "--ld-reference-mode",
        choices=["ukb_hm3_plus", "external"],
        default="ukb_hm3_plus",
    )
    parser.add_argument("--ld-ref-map", required=True)
    parser.add_argument("--ld-ref-dir", required=True)
    parser.add_argument("--ld-ref-pattern", default="LD_with_blocks_chr{chr}.rds")
    parser.add_argument("--variant-extract", default=None)
    parser.add_argument("--ld-cache-dir", default=None)
    parser.add_argument("--chromosomes", default="1-22")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--default-n", type=float, default=None)
    parser.add_argument("--match-by", choices=["id", "position"], default="position")
    parser.add_argument("--save-beta-grid", action="store_true")
    parser.add_argument("--commands-out", default=None)
    parser.add_argument("--task-list-out", default=None)
    parser.add_argument("--task-manifest-out", default=None)
    args = parser.parse_args()

    entry = _manifest_entry(args.manifest, args.disease)
    accession = entry["accession"]
    default_n = _resolve_default_n(args.default_n, entry)
    harmonized = Path(args.harmonized_dir) / f"{args.disease}.{accession}.harmonized.tsv.gz"

    delta_grid = _float_list(args.delta_grid)
    n_models = len(lassosum2_parameter_grid(delta_grid=delta_grid, nlambda=args.nlambda))

    out_dir = Path(args.out_dir) / args.disease
    keep_dir = out_dir / "keep"
    task_dir = out_dir / "tasks"
    bed_dir = out_dir / "bed"
    score_dir = out_dir / "scores"
    report_dir = out_dir / "reports"
    model_dir = out_dir / "models"
    ld_dir = Path(args.ld_cache_dir) / args.disease if args.ld_cache_dir else out_dir / "ld"
    tmp_dir = out_dir / "tmp"
    command_dir = out_dir / "commands"
    for path in [keep_dir, task_dir, bed_dir, score_dir, report_dir, model_dir, ld_dir, tmp_dir, command_dir]:
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

    tag = "full_train"
    bed_prefix = bed_dir / f"{args.disease}.{tag}.score_samples"
    score_bed_prefix = (
        bed_prefix
        if score_genotype_prefix == genotype_prefix and score_keep == train_keep
        else bed_dir / f"{args.disease}.{tag}.holdout_score_samples"
    )
    fit_score_bed_prefix = (
        bed_prefix
        if fit_score_genotype_prefix == genotype_prefix and fit_score_keep == train_keep
        else bed_dir / f"{args.disease}.{tag}.fit_score_samples"
    )
    score_file = score_dir / f"{args.disease}.{tag}.lassosum2_scores_long.csv"
    fit_score_file = score_dir / f"{args.disease}.{tag}.lassosum2_training_scores_long.csv"
    summary_file = report_dir / f"{args.disease}.{tag}.lassosum2_summary.csv"
    model_file = model_dir / f"{args.disease}.{tag}.lassosum2_model.rds"
    ld_backing_prefix = ld_dir / f"{args.disease}.{tag}.lassosum2_corr"
    task_script = task_dir / f"{args.disease}.{tag}.lassosum2.sh"
    task_script.write_text(
        _task_script(
            disease=args.disease,
            plink2=args.plink2,
            rscript=args.rscript,
            lassosum2_script=Path(args.lassosum2_script),
            genotype_prefix=genotype_prefix,
            score_genotype_prefix=score_genotype_prefix,
            fit_score_genotype_prefix=fit_score_genotype_prefix,
            harmonized=harmonized,
            train_keep=train_keep,
            score_keep=score_keep,
            fit_score_keep=fit_score_keep,
            all_keep=all_keep,
            bed_prefix=bed_prefix,
            score_bed_prefix=score_bed_prefix,
            fit_score_bed_prefix=fit_score_bed_prefix,
            variant_extract=Path(args.variant_extract) if args.variant_extract else None,
            score_file=score_file,
            fit_score_file=fit_score_file,
            summary_file=summary_file,
            model_file=model_file,
            ld_backing_prefix=ld_backing_prefix,
            tmp_dir=tmp_dir,
            patient_col=args.patient_col,
            delta_grid=delta_grid,
            nlambda=args.nlambda,
            lambda_min_ratio=args.lambda_min_ratio,
            dfmax=args.dfmax,
            maxiter=args.maxiter,
            tol=args.tol,
            chromosomes=args.chromosomes,
            threads=args.threads,
            default_n=default_n,
            join_by_pos=args.match_by == "position",
            save_beta_grid=args.save_beta_grid,
            ld_reference_mode=args.ld_reference_mode,
            ld_ref_map=Path(args.ld_ref_map),
            ld_ref_dir=Path(args.ld_ref_dir),
            ld_ref_pattern=args.ld_ref_pattern,
        ),
        encoding="utf-8",
    )
    _chmod_executable(task_script)

    manifest_rows = [
        {
            "disease": args.disease,
            "model_scope": "full_train",
            "ld_reference_mode": args.ld_reference_mode,
            "delta_grid": _csv_float(delta_grid),
            "nlambda": args.nlambda,
            "lambda_min_ratio": args.lambda_min_ratio,
            "dfmax": args.dfmax,
            "maxiter": args.maxiter,
            "tol": args.tol,
            "default_n": default_n if default_n is not None else "",
            "n_models": n_models,
            "task_script": str(task_script),
            "train_keep": str(train_keep),
            "score_keep": str(score_keep),
            "fit_score_keep": str(fit_score_keep),
            "all_keep": str(all_keep),
            "genotype_prefix": str(genotype_prefix),
            "score_genotype_prefix": str(score_genotype_prefix),
            "fit_score_genotype_prefix": str(fit_score_genotype_prefix),
            "bed_prefix": str(bed_prefix),
            "score_bed_prefix": str(score_bed_prefix),
            "fit_score_bed_prefix": str(fit_score_bed_prefix),
            "variant_extract": args.variant_extract or "",
            "harmonized": str(harmonized),
            "score_file": str(score_file),
            "fit_score_file": str(fit_score_file),
            "summary_file": str(summary_file),
            "model_file": str(model_file),
            "ld_backing_prefix": str(ld_backing_prefix),
            "ld_cache_dir": str(ld_dir),
            "ld_ref_map": args.ld_ref_map,
            "ld_ref_dir": args.ld_ref_dir,
            "ld_ref_pattern": args.ld_ref_pattern,
        }
    ]

    commands_out = (
        Path(args.commands_out)
        if args.commands_out
        else command_dir / f"{args.disease}.lassosum2_full_train.commands.sh"
    )
    task_list_out = (
        Path(args.task_list_out)
        if args.task_list_out
        else command_dir / f"{args.disease}.lassosum2_full_train.array_tasks.txt"
    )
    task_manifest_out = (
        Path(args.task_manifest_out)
        if args.task_manifest_out
        else command_dir / f"{args.disease}.lassosum2_full_train.tasks.csv"
    )
    commands_out.write_text(
        "\n".join(["#!/usr/bin/env bash", "set -euo pipefail", "", f"bash {q(task_script)}", ""]),
        encoding="utf-8",
    )
    _chmod_executable(commands_out)
    task_list_out.write_text(str(task_script) + "\n", encoding="utf-8")
    write_table(pd.DataFrame(manifest_rows), task_manifest_out)

    print(f"Wrote full-train lassosum2 task script for {args.disease} to {task_dir}")
    print(f"Wrote {n_models} lassosum2 models")
    print(f"Wrote sequential command script to {commands_out}")
    print(f"Wrote Slurm array task list to {task_list_out}")
    print(f"Wrote task manifest to {task_manifest_out}")


if __name__ == "__main__":
    main()
