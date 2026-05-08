from __future__ import annotations

import argparse
import shlex
import stat
from pathlib import Path

import pandas as pd

from decipehr.config import load_yaml
from decipehr.io import write_table
from decipehr.prs import pt_parameter_grid
from decipehr.pt import plink_score_columns, r2_tag
from decipehr.schema import parse_csv_list

DEFAULT_P_THRESHOLDS = "1,0.5,0.05,0.0005,0.000005,0.00000005"
DEFAULT_R2_THRESHOLDS = "0.2,0.4,0.6,0.8"


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


def _chmod_executable(path: Path) -> None:
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _task_script(
    *,
    disease: str,
    r2_threshold: float,
    p_thresholds: list[float],
    plink2: str,
    genotype_prefix: Path,
    harmonized: Path,
    train_keep: Path,
    score_keep: Path,
    clump_prefix: Path,
    score_file: Path,
    score_summary: Path,
    score_prefix: Path,
    window_kb: int,
    threads: int,
) -> str:
    score_cols = " ".join(str(col) for col in plink_score_columns(len(p_thresholds)))
    p_threshold_arg = ",".join(f"{threshold:g}" for threshold in p_thresholds)
    return "\n".join(
        [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            "",
            f"echo 'Started full-train P+T task disease={disease} r2={r2_threshold:g} at '$(date)",
            f"mkdir -p {q(clump_prefix.parent)} {q(score_file.parent)} {q(score_prefix.parent)}",
            f"test -f {q(str(genotype_prefix) + '.pgen')} || {{ echo 'Missing {genotype_prefix}.pgen' >&2; exit 1; }}",
            f"test -f {q(str(genotype_prefix) + '.pvar')} || {{ echo 'Missing {genotype_prefix}.pvar' >&2; exit 1; }}",
            f"test -f {q(str(genotype_prefix) + '.psam')} || {{ echo 'Missing {genotype_prefix}.psam' >&2; exit 1; }}",
            f"test -s {q(harmonized)} || {{ echo 'Missing harmonized sumstats: {harmonized}' >&2; exit 1; }}",
            f"test -s {q(train_keep)} || {{ echo 'Missing full training keep file: {train_keep}' >&2; exit 1; }}",
            f"test -s {q(score_keep)} || {{ echo 'Missing scoring keep file: {score_keep}' >&2; exit 1; }}",
            "",
            f"if [[ ! -s {q(str(clump_prefix) + '.clumps')} || \"${{REUSE_CLUMPS:-0}}\" != \"1\" ]]; then",
            (
                f"  {q(plink2)} --pfile {q(genotype_prefix)} --keep {q(train_keep)} "
                f"--clump {q(harmonized)} --clump-id-field SNP --clump-p-field P "
                f"--clump-p1 1 --clump-p2 1 --clump-r2 {r2_threshold:g} "
                f"--clump-kb {window_kb} --threads \"${{SLURM_CPUS_PER_TASK:-{threads}}}\" "
                f"--out {q(clump_prefix)}"
            ),
            "else",
            f"  echo 'Reusing existing clumps: {clump_prefix}.clumps'",
            "fi",
            "",
            (
                "python -u scripts/build_pt_scorefile.py "
                f"--harmonized {q(harmonized)} "
                f"--clumps {q(str(clump_prefix) + '.clumps')} "
                f"--p-thresholds {q(p_threshold_arg)} "
                f"--out {q(score_file)} "
                f"--summary {q(score_summary)}"
            ),
            (
                f"{q(plink2)} --pfile {q(genotype_prefix)} --keep {q(score_keep)} "
                f"--score {q(score_file)} 1 2 header-read cols=+scoresums "
                f"--score-col-nums {score_cols} "
                f"--threads \"${{SLURM_CPUS_PER_TASK:-{threads}}}\" --out {q(score_prefix)}"
            ),
            f"echo 'Finished full-train P+T task disease={disease} r2={r2_threshold:g} at '$(date)",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Write full-training PLINK2 P+T commands for one disease."
    )
    parser.add_argument("--disease", required=True)
    parser.add_argument("--manifest", default="configs/official_sumstats.yaml")
    parser.add_argument("--patient-col", default="sampleid")
    parser.add_argument("--genotype-prefix", default="outputs/genotype_qc/qc_final/all_chromosomes.qc")
    parser.add_argument("--harmonized-dir", default="data/processed/sumstats/harmonized")
    parser.add_argument("--train-keep", default="data/processed/sample_sets/labeled_train.keep.psam")
    parser.add_argument("--score-keep", default="data/processed/sample_sets/unlabelled_validation.keep.psam")
    parser.add_argument("--out-dir", default="outputs/pt_full_train")
    parser.add_argument("--plink2", default="plink2")
    parser.add_argument("--p-thresholds", default=DEFAULT_P_THRESHOLDS)
    parser.add_argument("--r2-thresholds", default=DEFAULT_R2_THRESHOLDS)
    parser.add_argument("--window-kb", type=int, default=250)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--commands-out", default=None)
    parser.add_argument("--task-list-out", default=None)
    parser.add_argument("--task-manifest-out", default=None)
    args = parser.parse_args()

    entry = _manifest_entry(args.manifest, args.disease)
    accession = entry["accession"]
    harmonized = Path(args.harmonized_dir) / f"{args.disease}.{accession}.harmonized.tsv.gz"

    p_thresholds = [float(value) for value in parse_csv_list(args.p_thresholds)]
    r2_thresholds = [float(value) for value in parse_csv_list(args.r2_thresholds)]
    grid = pt_parameter_grid(pvalue_thresholds=[1.0], r2_thresholds=r2_thresholds)

    out_dir = Path(args.out_dir) / args.disease
    task_dir = out_dir / "tasks"
    clump_dir = out_dir / "clumps"
    scorefile_dir = out_dir / "scorefiles"
    score_dir = out_dir / "scores"
    command_dir = out_dir / "commands"
    for path in [task_dir, clump_dir, scorefile_dir, score_dir, command_dir]:
        path.mkdir(parents=True, exist_ok=True)

    task_scripts: list[Path] = []
    manifest_rows = []
    genotype_prefix = Path(args.genotype_prefix)
    train_keep = Path(args.train_keep)
    score_keep = Path(args.score_keep)
    for _, row in grid.iterrows():
        r2_threshold = float(row.clump_r2)
        tag = r2_tag(r2_threshold)
        clump_prefix = clump_dir / f"{args.disease}.full_train.{tag}"
        score_file = scorefile_dir / f"{args.disease}.full_train.{tag}.score.tsv"
        score_summary = scorefile_dir / f"{args.disease}.full_train.{tag}.score.summary.json"
        score_prefix = score_dir / f"{args.disease}.full_train.{tag}"
        task_script = task_dir / f"{args.disease}.full_train.{tag}.sh"
        task_script.write_text(
            _task_script(
                disease=args.disease,
                r2_threshold=r2_threshold,
                p_thresholds=p_thresholds,
                plink2=args.plink2,
                genotype_prefix=genotype_prefix,
                harmonized=harmonized,
                train_keep=train_keep,
                score_keep=score_keep,
                clump_prefix=clump_prefix,
                score_file=score_file,
                score_summary=score_summary,
                score_prefix=score_prefix,
                window_kb=args.window_kb,
                threads=args.threads,
            ),
            encoding="utf-8",
        )
        _chmod_executable(task_script)
        task_scripts.append(task_script)
        manifest_rows.append(
            {
                "disease": args.disease,
                "model_scope": "full_train",
                "clump_r2": r2_threshold,
                "r2_tag": tag,
                "p_thresholds": ",".join(f"{threshold:g}" for threshold in p_thresholds),
                "task_script": str(task_script),
                "train_keep": str(train_keep),
                "score_keep": str(score_keep),
                "clumps": str(clump_prefix) + ".clumps",
                "score_file": str(score_file),
                "score_summary": str(score_summary),
                "sscore": str(score_prefix) + ".sscore",
            }
        )

    commands_out = (
        Path(args.commands_out) if args.commands_out else command_dir / f"{args.disease}.pt_full_train.commands.sh"
    )
    task_list_out = (
        Path(args.task_list_out)
        if args.task_list_out
        else command_dir / f"{args.disease}.pt_full_train.array_tasks.txt"
    )
    task_manifest_out = (
        Path(args.task_manifest_out)
        if args.task_manifest_out
        else command_dir / f"{args.disease}.pt_full_train.tasks.csv"
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

    print(f"Wrote {len(task_scripts)} full-train P+T task scripts for {args.disease} to {task_dir}")
    print(f"Wrote sequential command script to {commands_out}")
    print(f"Wrote Slurm array task list to {task_list_out}")
    print(f"Wrote task manifest to {task_manifest_out}")


if __name__ == "__main__":
    main()
