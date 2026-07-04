from __future__ import annotations

import argparse
import shlex
import stat
from pathlib import Path

import pandas as pd

from decipehr.config import load_yaml
from decipehr.io import read_table, write_table
from decipehr.ldpred2 import (
    DEFAULT_H2_GRID,
    DEFAULT_P_GRID,
    DEFAULT_SPARSE_GRID,
    ldpred2_parameter_grid,
)
from decipehr.schema import parse_csv_list, require_columns


def q(value: str | Path) -> str:
    return shlex.quote(str(value))


def _manifest_entry(path: str | Path, disease: str) -> dict[str, object]:
    manifest = load_yaml(path).get("summary_statistics", {})
    if disease not in manifest:
        raise ValueError(f"Disease {disease!r} is not present in {path}")
    entry = manifest[disease]
    if not isinstance(entry, dict):
        raise ValueError(f"Manifest entry for {disease} must be a mapping")
    return entry


def _write_keep_file(samples: pd.Series, path: Path) -> None:
    values = samples.astype("string").dropna().drop_duplicates().sort_values(kind="mergesort")
    write_table(pd.DataFrame({"#IID": values.to_numpy()}), path)


def _chmod_executable(path: Path) -> None:
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _float_list(values: str) -> list[float]:
    parsed = [float(value) for value in parse_csv_list(values)]
    if not parsed:
        raise ValueError("Grid must contain at least one numeric value")
    return parsed


def _bool_list(values: str) -> list[bool]:
    parsed: list[bool] = []
    for value in parse_csv_list(values):
        lowered = value.lower()
        if lowered in {"1", "true", "t", "yes", "y", "sparse"}:
            parsed.append(True)
        elif lowered in {"0", "false", "f", "no", "n", "dense"}:
            parsed.append(False)
        else:
            raise ValueError(f"Cannot parse sparse grid value {value!r}")
    if not parsed:
        raise ValueError("Sparse grid must contain at least one value")
    return parsed


def _csv_float(values: list[float]) -> str:
    return ",".join(f"{value:g}" for value in values)


def _csv_bool(values: list[bool]) -> str:
    return ",".join("1" if value else "0" for value in values)


def _bool_arg(value: bool) -> str:
    return "true" if value else "false"


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


def _task_script(
    *,
    disease: str,
    fold: int,
    plink2: str,
    rscript: str,
    ldpred2_script: Path,
    genotype_prefix: Path,
    harmonized: Path,
    train_keep: Path,
    all_keep: Path,
    bed_prefix: Path,
    variant_extract: Path | None,
    score_file: Path,
    summary_file: Path,
    model_file: Path,
    ld_backing_prefix: Path,
    tmp_dir: Path,
    patient_col: str,
    h2_grid: list[float],
    p_grid: list[float],
    sparse_grid: list[bool],
    ld_window_cm: float,
    ld_window_mb: float,
    genetic_map_dir: Path | None,
    genetic_map_type: str,
    allow_mb_fallback: bool,
    chromosomes: str,
    threads: int,
    default_n: float | None,
    join_by_pos: bool,
    save_beta_grid: bool,
    ld_reference_mode: str,
    ld_ref_map: Path | None,
    ld_ref_dir: Path | None,
    ld_ref_pattern: str,
) -> str:
    plink_tokens = [
        q(plink2),
        "--pfile",
        q(genotype_prefix),
        "--keep",
        q(all_keep),
    ]
    if chromosomes:
        plink_tokens.extend(["--chr", q(chromosomes)])
    if variant_extract is not None:
        plink_tokens.extend(["--extract", q(variant_extract)])
    plink_tokens.extend(
        [
            "--make-bed",
            "--threads",
            f'"${{SLURM_CPUS_PER_TASK:-{threads}}}"',
            "--out",
            q(bed_prefix),
        ]
    )

    r_tokens = [
        q(rscript),
        q(ldpred2_script),
        "--disease",
        q(disease),
        "--fold",
        str(fold),
        "--bed-prefix",
        q(bed_prefix),
        "--harmonized",
        q(harmonized),
        "--train-keep",
        q(train_keep),
        "--score-keep",
        q(all_keep),
        "--patient-col",
        q(patient_col),
        "--ld-reference-mode",
        q(ld_reference_mode),
        "--score-out",
        q(score_file),
        "--summary-out",
        q(summary_file),
        "--model-out",
        q(model_file),
        "--ld-backing-prefix",
        q(ld_backing_prefix),
        "--h2-grid",
        q(_csv_float(h2_grid)),
        "--p-grid",
        q(_csv_float(p_grid)),
        "--sparse-grid",
        q(_csv_bool(sparse_grid)),
        "--ld-window-cm",
        f"{ld_window_cm:g}",
        "--ld-window-mb",
        f"{ld_window_mb:g}",
        "--genetic-map-type",
        q(genetic_map_type),
        "--threads",
        f'"${{SLURM_CPUS_PER_TASK:-{threads}}}"',
        "--join-by-pos",
        _bool_arg(join_by_pos),
        "--save-beta-grid",
        _bool_arg(save_beta_grid),
        "--reuse-ld-corr",
        '"${REUSE_LDPRED2_LD:-1}"',
    ]
    if genetic_map_dir is not None:
        r_tokens.extend(["--genetic-map-dir", q(genetic_map_dir)])
    if allow_mb_fallback:
        r_tokens.extend(["--allow-mb-fallback", "true"])
    if default_n is not None:
        r_tokens.extend(["--default-n", f"{default_n:g}"])
    if ld_ref_map is not None:
        r_tokens.extend(["--ld-ref-map", q(ld_ref_map)])
    if ld_ref_dir is not None:
        r_tokens.extend(["--ld-ref-dir", q(ld_ref_dir)])
    if ld_ref_pattern:
        r_tokens.extend(["--ld-ref-pattern", q(ld_ref_pattern)])

    return "\n".join(
        [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            "",
            f"echo 'Started LDpred2 task disease={disease} fold={fold} at '$(date)",
            (
                f"mkdir -p {q(bed_prefix.parent)} {q(score_file.parent)} {q(summary_file.parent)} "
                f"{q(model_file.parent)} {q(ld_backing_prefix.parent)} {q(tmp_dir)}"
            ),
            f"test -f {q(str(genotype_prefix) + '.pgen')} || {{ echo 'Missing {genotype_prefix}.pgen' >&2; exit 1; }}",
            f"test -f {q(str(genotype_prefix) + '.pvar')} || {{ echo 'Missing {genotype_prefix}.pvar' >&2; exit 1; }}",
            f"test -f {q(str(genotype_prefix) + '.psam')} || {{ echo 'Missing {genotype_prefix}.psam' >&2; exit 1; }}",
            f"test -s {q(harmonized)} || {{ echo 'Missing harmonized sumstats: {harmonized}' >&2; exit 1; }}",
            f"test -s {q(train_keep)} || {{ echo 'Missing training keep file: {train_keep}' >&2; exit 1; }}",
            f"test -s {q(all_keep)} || {{ echo 'Missing scoring keep file: {all_keep}' >&2; exit 1; }}",
            *(
                [
                    f"test -s {q(variant_extract)} || {{ echo 'Missing variant extract file: {variant_extract}' >&2; exit 1; }}"
                ]
                if variant_extract is not None
                else []
            ),
            *(
                [
                    f"test -s {q(ld_ref_map)} || {{ echo 'Missing LD reference map: {ld_ref_map}' >&2; exit 1; }}",
                    f"test -d {q(ld_ref_dir)} || {{ echo 'Missing LD reference directory: {ld_ref_dir}' >&2; exit 1; }}",
                    (
                        f"test -s {q(ld_ref_dir / ld_ref_pattern.replace('{chr}', '22'))} || "
                        f"{{ echo 'Missing LD reference chr22 matrix under {ld_ref_dir}' >&2; exit 1; }}"
                    ),
                ]
                if ld_reference_mode != "target" and ld_ref_map is not None and ld_ref_dir is not None
                else []
            ),
            "",
            f"if [[ -s {q(score_file)} && \"${{REUSE_LDPRED2_SCORES:-0}}\" == \"1\" ]]; then",
            f"  echo 'Reusing existing LDpred2 scores: {score_file}'",
            "  exit 0",
            "fi",
            "",
            f"if [[ ! -s {q(str(bed_prefix) + '.bed')} || \"${{REUSE_LDPRED2_BED:-1}}\" != \"1\" ]]; then",
            f"  {_render_command(plink_tokens)}",
            "else",
            f"  echo 'Reusing existing PLINK bed files: {bed_prefix}'",
            "fi",
            "",
            f"test -s {q(str(bed_prefix) + '.bed')} || {{ echo 'Missing {bed_prefix}.bed' >&2; exit 1; }}",
            f"test -s {q(str(bed_prefix) + '.bim')} || {{ echo 'Missing {bed_prefix}.bim' >&2; exit 1; }}",
            f"test -s {q(str(bed_prefix) + '.fam')} || {{ echo 'Missing {bed_prefix}.fam' >&2; exit 1; }}",
            "",
            _render_command(r_tokens),
            f"echo 'Finished LDpred2 task disease={disease} fold={fold} at '$(date)",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Write fold-aware LDpred2 commands for one disease.")
    parser.add_argument("--disease", required=True)
    parser.add_argument("--manifest", default="configs/official_sumstats.yaml")
    parser.add_argument("--splits", default="data/processed/splits/disease_specific_split_seed42.csv")
    parser.add_argument("--patient-col", default="sampleid")
    parser.add_argument("--genotype-prefix", default="outputs/genotype_qc/qc_final/all_chromosomes.qc")
    parser.add_argument("--harmonized-dir", default="data/processed/sumstats/harmonized")
    parser.add_argument("--out-dir", default="outputs/ldpred2_cv")
    parser.add_argument("--plink2", default="plink2")
    parser.add_argument("--rscript", default="Rscript")
    parser.add_argument("--ldpred2-script", default="scripts/run_ldpred2_task.R")
    parser.add_argument("--h2-grid", default=_csv_float(DEFAULT_H2_GRID))
    parser.add_argument("--p-grid", default=_csv_float(DEFAULT_P_GRID))
    parser.add_argument("--sparse-grid", default=_csv_bool(DEFAULT_SPARSE_GRID))
    parser.add_argument("--ld-window-cm", type=float, default=3.0)
    parser.add_argument("--ld-window-mb", type=float, default=3.0)
    parser.add_argument(
        "--ld-reference-mode",
        choices=["target", "ukb_hm3_plus", "external"],
        default="target",
    )
    parser.add_argument("--ld-ref-map", default=None)
    parser.add_argument("--ld-ref-dir", default=None)
    parser.add_argument("--ld-ref-pattern", default="LD_with_blocks_chr{chr}.rds")
    parser.add_argument("--variant-extract", default=None)
    parser.add_argument(
        "--ld-cache-dir",
        default=None,
        help="Optional shared directory for reusable LD correlation SFBM caches.",
    )
    parser.add_argument("--genetic-map-dir", default="data/resources/genetic_maps")
    parser.add_argument("--genetic-map-type", default="hg38_price")
    parser.add_argument("--allow-mb-fallback", action="store_true")
    parser.add_argument("--chromosomes", default="1-22")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--default-n", type=float, default=None)
    parser.add_argument("--match-by", choices=["id", "position"], default="id")
    parser.add_argument("--save-beta-grid", action="store_true")
    parser.add_argument("--commands-out", default=None)
    parser.add_argument("--task-list-out", default=None)
    parser.add_argument("--task-manifest-out", default=None)
    args = parser.parse_args()

    entry = _manifest_entry(args.manifest, args.disease)
    accession = entry["accession"]
    default_n = _resolve_default_n(args.default_n, entry)
    harmonized = Path(args.harmonized_dir) / f"{args.disease}.{accession}.harmonized.tsv.gz"

    h2_grid = _float_list(args.h2_grid)
    p_grid = _float_list(args.p_grid)
    sparse_grid = _bool_list(args.sparse_grid)
    grid = ldpred2_parameter_grid(h2_grid=h2_grid, p_grid=p_grid, sparse_grid=sparse_grid)

    splits = read_table(args.splits)
    require_columns(splits, [args.patient_col, "disease", "fold"], "splits")
    disease_splits = splits.loc[splits["disease"] == args.disease, [args.patient_col, "fold"]].copy()
    if disease_splits.empty:
        raise ValueError(f"No split rows found for disease {args.disease}")
    if disease_splits[args.patient_col].duplicated().any():
        duplicated = disease_splits.loc[
            disease_splits[args.patient_col].duplicated(), args.patient_col
        ].head(5)
        raise ValueError(f"Split file has duplicate patients for {args.disease}: {duplicated.tolist()}")
    if args.ld_reference_mode != "target" and (not args.ld_ref_map or not args.ld_ref_dir):
        raise ValueError("--ld-ref-map and --ld-ref-dir are required when --ld-reference-mode is not 'target'.")

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

    all_keep = keep_dir / f"{args.disease}.all.keep.psam"
    _write_keep_file(disease_splits[args.patient_col], all_keep)

    task_scripts: list[Path] = []
    manifest_rows = []
    genotype_prefix = Path(args.genotype_prefix)
    ldpred2_script = Path(args.ldpred2_script)
    variant_extract = Path(args.variant_extract) if args.variant_extract else None
    genetic_map_dir = Path(args.genetic_map_dir) if args.genetic_map_dir else None
    ld_ref_map = Path(args.ld_ref_map) if args.ld_ref_map else None
    ld_ref_dir = Path(args.ld_ref_dir) if args.ld_ref_dir else None
    for fold in sorted(disease_splits["fold"].dropna().astype(int).unique()):
        train_keep = keep_dir / f"{args.disease}.fold{fold}.train.keep.psam"
        val_keep = keep_dir / f"{args.disease}.fold{fold}.val.keep.psam"
        _write_keep_file(disease_splits.loc[disease_splits["fold"].astype(int) != fold, args.patient_col], train_keep)
        _write_keep_file(disease_splits.loc[disease_splits["fold"].astype(int) == fold, args.patient_col], val_keep)

        tag = f"fold{fold}"
        bed_prefix = bed_dir / f"{args.disease}.{tag}.score_samples"
        score_file = score_dir / f"{args.disease}.{tag}.ldpred2_scores_long.csv"
        summary_file = report_dir / f"{args.disease}.{tag}.ldpred2_summary.csv"
        model_file = model_dir / f"{args.disease}.{tag}.ldpred2_model.rds"
        ld_backing_prefix = ld_dir / f"{args.disease}.{tag}.ldpred2_corr"
        task_script = task_dir / f"{args.disease}.{tag}.ldpred2.sh"
        task_script.write_text(
            _task_script(
                disease=args.disease,
                fold=fold,
                plink2=args.plink2,
                rscript=args.rscript,
                ldpred2_script=ldpred2_script,
                genotype_prefix=genotype_prefix,
                harmonized=harmonized,
                train_keep=train_keep,
                all_keep=all_keep,
                bed_prefix=bed_prefix,
                variant_extract=variant_extract,
                score_file=score_file,
                summary_file=summary_file,
                model_file=model_file,
                ld_backing_prefix=ld_backing_prefix,
                tmp_dir=tmp_dir,
                patient_col=args.patient_col,
                h2_grid=h2_grid,
                p_grid=p_grid,
                sparse_grid=sparse_grid,
                ld_window_cm=args.ld_window_cm,
                ld_window_mb=args.ld_window_mb,
                genetic_map_dir=genetic_map_dir,
                genetic_map_type=args.genetic_map_type,
                allow_mb_fallback=args.allow_mb_fallback,
                chromosomes=args.chromosomes,
                threads=args.threads,
                default_n=default_n,
                join_by_pos=args.match_by == "position",
                save_beta_grid=args.save_beta_grid,
                ld_reference_mode=args.ld_reference_mode,
                ld_ref_map=ld_ref_map,
                ld_ref_dir=ld_ref_dir,
                ld_ref_pattern=args.ld_ref_pattern,
            ),
            encoding="utf-8",
        )
        _chmod_executable(task_script)
        task_scripts.append(task_script)
        manifest_rows.append(
            {
                "disease": args.disease,
                "fold": fold,
                "h2_grid": _csv_float(h2_grid),
                "p_grid": _csv_float(p_grid),
                "sparse_grid": _csv_bool(sparse_grid),
                "n_models": len(grid),
                "task_script": str(task_script),
                "train_keep": str(train_keep),
                "all_keep": str(all_keep),
                "val_keep": str(val_keep),
                "bed_prefix": str(bed_prefix),
                "variant_extract": str(variant_extract) if variant_extract is not None else "",
                "harmonized": str(harmonized),
                "default_n": default_n if default_n is not None else "",
                "score_file": str(score_file),
                "summary_file": str(summary_file),
                "model_file": str(model_file),
                "ld_backing_prefix": str(ld_backing_prefix),
                "ld_cache_dir": str(ld_dir),
                "ld_reference_mode": args.ld_reference_mode,
                "ld_ref_map": str(ld_ref_map) if ld_ref_map is not None else "",
                "ld_ref_dir": str(ld_ref_dir) if ld_ref_dir is not None else "",
                "ld_ref_pattern": args.ld_ref_pattern,
            }
        )

    commands_out = (
        Path(args.commands_out) if args.commands_out else command_dir / f"{args.disease}.ldpred2_cv.commands.sh"
    )
    task_list_out = (
        Path(args.task_list_out)
        if args.task_list_out
        else command_dir / f"{args.disease}.ldpred2_cv.array_tasks.txt"
    )
    task_manifest_out = (
        Path(args.task_manifest_out)
        if args.task_manifest_out
        else command_dir / f"{args.disease}.ldpred2_cv.tasks.csv"
    )
    commands_out.parent.mkdir(parents=True, exist_ok=True)
    task_list_out.parent.mkdir(parents=True, exist_ok=True)
    task_manifest_out.parent.mkdir(parents=True, exist_ok=True)

    commands_out.write_text(
        "\n".join(["#!/usr/bin/env bash", "set -euo pipefail", "", *[f"bash {q(path)}" for path in task_scripts], ""]),
        encoding="utf-8",
    )
    _chmod_executable(commands_out)
    task_list_out.write_text("\n".join(str(path) for path in task_scripts) + "\n", encoding="utf-8")
    write_table(pd.DataFrame(manifest_rows), task_manifest_out)

    print(f"Wrote {len(task_scripts)} LDpred2 fold task scripts for {args.disease} to {task_dir}")
    print(f"Wrote {len(grid)} LDpred2 grid models per fold")
    print(f"Wrote sequential command script to {commands_out}")
    print(f"Wrote Slurm array task list to {task_list_out}")
    print(f"Wrote task manifest to {task_manifest_out}")


if __name__ == "__main__":
    main()
