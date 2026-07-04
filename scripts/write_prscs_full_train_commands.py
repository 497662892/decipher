from __future__ import annotations

import argparse
import shlex
import stat
from pathlib import Path

import pandas as pd

from decipehr.config import load_yaml
from decipehr.io import write_table
from decipehr.schema import parse_csv_list

DEFAULT_CHROMOSOMES = ",".join(str(chrom) for chrom in range(1, 23))


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


def _append_suffix(path: str | Path, suffix: str) -> Path:
    return Path(f"{path}{suffix}")


def _phi_tag(phi: str) -> str:
    if phi.lower() == "auto":
        return "phiauto"
    value = float(phi)
    return f"phi{value:.0e}".replace("+", "")


def _prscs_effect_file(effect_prefix: Path, phi: str, chrom: str) -> Path:
    if phi.lower() == "auto":
        return Path(f"{effect_prefix}_pst_eff_a1_b0.5_phiauto_chr{chrom}.txt")
    return Path(f"{effect_prefix}_pst_eff_a1_b0.5_phi{float(phi):1.0e}_chr{chrom}.txt")


def _reference_tests(ref_dir: Path, chrom: str) -> list[str]:
    panel = ref_dir.name
    if "1kg" in panel:
        snpinfo = ref_dir / "snpinfo_1kg_hm3"
        ldblk = ref_dir / f"ldblk_1kg_chr{chrom}.hdf5"
    elif "ukbb" in panel:
        snpinfo = ref_dir / "snpinfo_ukbb_hm3"
        ldblk = ref_dir / f"ldblk_ukbb_chr{chrom}.hdf5"
    else:
        snpinfo = ref_dir / "snpinfo_1kg_hm3"
        ldblk = ref_dir / f"ldblk_1kg_chr{chrom}.hdf5"
    return [
        f"test -s {q(snpinfo)} || {{ echo 'Missing PRS-CS SNP info file: {snpinfo}' >&2; exit 1; }}",
        f"test -s {q(ldblk)} || {{ echo 'Missing PRS-CS LD block file: {ldblk}' >&2; exit 1; }}",
    ]


def _optional_module_line(module_load: str | None) -> list[str]:
    if not module_load:
        return []
    return [f"module load {module_load}"]


def _prepare_script(
    *,
    raw_sumstats: Path,
    harmonized: Path,
    input_prefix: Path,
    summary: Path,
    chromosomes: list[str],
    mode: str,
    chunksize: int,
    python_bin: str,
) -> str:
    return "\n".join(
        [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            "",
            f"mkdir -p {q(input_prefix.parent)}",
            f"test -s {q(raw_sumstats)} || {{ echo 'Missing raw sumstats: {raw_sumstats}' >&2; exit 1; }}",
            f"test -s {q(harmonized)} || {{ echo 'Missing harmonized sumstats: {harmonized}' >&2; exit 1; }}",
            (
                f"{q(python_bin)} -u scripts/prepare_prscs_inputs.py "
                f"--raw-sumstats {q(raw_sumstats)} "
                f"--harmonized {q(harmonized)} "
                f"--out-prefix {q(input_prefix)} "
                f"--summary-out {q(summary)} "
                f"--chromosomes {q(','.join(chromosomes))} "
                f"--mode {q(mode)} "
                f"--chunksize {chunksize}"
            ),
            "",
        ]
    )


def _task_script(
    *,
    disease: str,
    chrom: str,
    phi: str,
    python_bin: str,
    prscs_script: Path,
    ref_dir: Path,
    bim_prefix: Path,
    sst_file: Path,
    n_gwas: int,
    effect_prefix: Path,
    n_iter: int,
    n_burnin: int,
    thin: int,
    threads: int,
) -> str:
    phi_arg = "" if phi.lower() == "auto" else f" --phi={phi}"
    effect_file = _prscs_effect_file(effect_prefix, phi, chrom)
    return "\n".join(
        [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            "",
            f"echo 'Started PRS-CS disease={disease} phi={phi} chr={chrom} at '$(date)",
            f"export MKL_NUM_THREADS=${{SLURM_CPUS_PER_TASK:-{threads}}}",
            f"export NUMEXPR_NUM_THREADS=${{SLURM_CPUS_PER_TASK:-{threads}}}",
            f"export OMP_NUM_THREADS=${{SLURM_CPUS_PER_TASK:-{threads}}}",
            f"mkdir -p {q(effect_prefix.parent)}",
            f"test -s {q(prscs_script)} || {{ echo 'Missing PRS-CS script: {prscs_script}' >&2; exit 1; }}",
            f"test -s {q(sst_file)} || {{ echo 'Missing PRS-CS sumstats: {sst_file}' >&2; exit 1; }}",
            f"test -s {q(str(bim_prefix) + '.bim')} || {{ echo 'Missing PRS-CS BIM: {bim_prefix}.bim' >&2; exit 1; }}",
            *_reference_tests(ref_dir, chrom),
            "",
            f"if [[ ! -s {q(effect_file)} || \"${{REUSE_PRSCS_EFFECTS:-0}}\" != \"1\" ]]; then",
            (
                f"  {q(python_bin)} {q(prscs_script)} "
                f"--ref_dir={q(ref_dir)} "
                f"--bim_prefix={q(bim_prefix)} "
                f"--sst_file={q(sst_file)} "
                f"--n_gwas={n_gwas} "
                f"--out_dir={q(effect_prefix)} "
                f"--n_iter={n_iter} --n_burnin={n_burnin} --thin={thin} "
                f"--chrom={chrom}{phi_arg}"
            ),
            "else",
            f"  echo 'Reusing existing PRS-CS effects: {effect_file}'",
            "fi",
            f"echo 'Finished PRS-CS disease={disease} phi={phi} chr={chrom} at '$(date)",
            "",
        ]
    )


def _score_script(
    *,
    disease: str,
    phi: str,
    effect_files: list[Path],
    target_id_map: Path,
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
    threads: int,
    module_load: str | None,
) -> str:
    effect_args = " ".join(q(path) for path in effect_files)
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        *_optional_module_line(module_load),
        f"echo 'Started PRS-CS scoring disease={disease} phi={phi} at '$(date)",
        f"mkdir -p {q(score_file.parent)} {q(score_prefix.parent)}",
        *[
            f"test -s {q(path)} || {{ echo 'Missing PRS-CS effect file: {path}' >&2; exit 1; }}"
            for path in effect_files
        ],
        f"test -s {q(target_id_map)} || {{ echo 'Missing PRS-CS target ID map: {target_id_map}' >&2; exit 1; }}",
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
            f"{q(python_bin)} -u scripts/build_prscs_scorefile.py "
            f"--effect-files {effect_args} "
            f"--target-id-map {q(target_id_map)} "
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
        f"echo 'Finished PRS-CS scoring disease={disease} phi={phi} at '$(date)",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Write full-training PRS-CS commands for one disease."
    )
    parser.add_argument("--disease", required=True)
    parser.add_argument("--manifest", default="configs/official_sumstats.yaml")
    parser.add_argument("--sumstats-root", required=True)
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
    parser.add_argument("--out-dir", default="outputs/prscs_full_train")
    parser.add_argument("--ref-dir", default="data/resources/prscs/ldblk_1kg_eur")
    parser.add_argument("--prscs-script", default="data/resources/prscs/PRScs/PRScs.py")
    parser.add_argument("--python", default="python")
    parser.add_argument("--plink2", default="plink2")
    parser.add_argument("--module-load", default=None)
    parser.add_argument("--n-gwas", type=int, default=None)
    parser.add_argument("--phi-grid", default="1e-2")
    parser.add_argument("--chromosomes", default=DEFAULT_CHROMOSOMES)
    parser.add_argument("--n-iter", type=int, default=1000)
    parser.add_argument("--n-burnin", type=int, default=500)
    parser.add_argument("--thin", type=int, default=5)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--sumstats-mode", choices=["auto", "beta-se", "beta-p"], default="auto")
    parser.add_argument("--chunksize", type=int, default=250_000)
    args = parser.parse_args()

    entry = _manifest_entry(args.manifest, args.disease)
    accession = entry["accession"]
    n_gwas = args.n_gwas or entry.get("default_n")
    if n_gwas is None:
        raise ValueError(
            f"No --n-gwas was supplied and manifest entry {args.disease!r} has no default_n"
        )
    n_gwas = int(n_gwas)

    chromosomes = parse_csv_list(args.chromosomes)
    phi_values = parse_csv_list(args.phi_grid)
    raw_sumstats = Path(args.sumstats_root) / str(entry["file"])
    harmonized = Path(args.harmonized_dir) / f"{args.disease}.{accession}.harmonized.tsv.gz"
    out_dir = Path(args.out_dir) / args.disease
    input_dir = out_dir / "inputs"
    task_dir = out_dir / "tasks"
    effect_dir = out_dir / "effects"
    scorefile_dir = out_dir / "scorefiles"
    score_dir = out_dir / "scores"
    command_dir = out_dir / "commands"
    for path in [input_dir, task_dir, effect_dir, scorefile_dir, score_dir, command_dir]:
        path.mkdir(parents=True, exist_ok=True)

    input_prefix = input_dir / f"{args.disease}.prscs"
    prepare = command_dir / f"{args.disease}.prscs_prepare_inputs.sh"
    prepare.write_text(
        _prepare_script(
            raw_sumstats=raw_sumstats,
            harmonized=harmonized,
            input_prefix=input_prefix,
            summary=input_dir / f"{args.disease}.prscs_input.summary.json",
            chromosomes=chromosomes,
            mode=args.sumstats_mode,
            chunksize=args.chunksize,
            python_bin=args.python,
        ),
        encoding="utf-8",
    )
    _chmod_executable(prepare)

    task_scripts = []
    manifest_rows = []
    score_scripts = []
    genotype_prefix = Path(args.genotype_prefix)
    score_genotype_prefix = (
        Path(args.score_genotype_prefix) if args.score_genotype_prefix else genotype_prefix
    )
    fit_score_genotype_prefix = (
        Path(args.fit_score_genotype_prefix) if args.fit_score_genotype_prefix else genotype_prefix
    )
    score_keep = Path(args.score_keep)
    fit_score_keep = Path(args.fit_score_keep) if args.fit_score_keep else Path(args.train_keep)
    for phi in phi_values:
        tag = _phi_tag(phi)
        effect_prefix = effect_dir / f"{args.disease}.{tag}"
        effect_files = [_prscs_effect_file(effect_prefix, phi, chrom) for chrom in chromosomes]
        for chrom in chromosomes:
            task_script = task_dir / f"{args.disease}.{tag}.chr{chrom}.sh"
            task_script.write_text(
                _task_script(
                    disease=args.disease,
                    chrom=chrom,
                    phi=phi,
                    python_bin=args.python,
                    prscs_script=Path(args.prscs_script),
                    ref_dir=Path(args.ref_dir),
                    bim_prefix=input_prefix,
                    sst_file=_append_suffix(input_prefix, ".sst"),
                    n_gwas=n_gwas,
                    effect_prefix=effect_prefix,
                    n_iter=args.n_iter,
                    n_burnin=args.n_burnin,
                    thin=args.thin,
                    threads=args.threads,
                ),
                encoding="utf-8",
            )
            _chmod_executable(task_script)
            task_scripts.append(task_script)

        score_file = scorefile_dir / f"{args.disease}.{tag}.score.tsv"
        score_prefix = score_dir / f"{args.disease}.{tag}"
        fit_score_prefix = score_dir / f"{args.disease}.{tag}.fit"
        score_script = command_dir / f"{args.disease}.{tag}.score.sh"
        score_script.write_text(
            _score_script(
                disease=args.disease,
                phi=phi,
                effect_files=effect_files,
                target_id_map=_append_suffix(input_prefix, ".target_id_map.tsv"),
                score_file=score_file,
                score_summary=scorefile_dir / f"{args.disease}.{tag}.score.summary.json",
                genotype_prefix=genotype_prefix,
                score_genotype_prefix=score_genotype_prefix,
                fit_score_genotype_prefix=fit_score_genotype_prefix,
                score_keep=score_keep,
                fit_score_keep=fit_score_keep,
                score_prefix=score_prefix,
                fit_score_prefix=fit_score_prefix,
                python_bin=args.python,
                plink2=args.plink2,
                threads=args.threads,
                module_load=args.module_load,
            ),
            encoding="utf-8",
        )
        _chmod_executable(score_script)
        score_scripts.append(score_script)
        manifest_rows.append(
            {
                "disease": args.disease,
                "model_scope": "full_train",
                "phi": phi,
                "phi_tag": tag,
                "n_gwas": n_gwas,
                "chromosomes": ",".join(chromosomes),
                "prepare_script": str(prepare),
                "score_script": str(score_script),
                "genotype_prefix": str(genotype_prefix),
                "score_genotype_prefix": str(score_genotype_prefix),
                "fit_score_genotype_prefix": str(fit_score_genotype_prefix),
                "target_id_map": str(_append_suffix(input_prefix, ".target_id_map.tsv")),
                "score_file": str(score_file),
                "sscore": str(score_prefix) + ".sscore",
                "fit_sscore": str(fit_score_prefix) + ".sscore",
            }
        )

    task_list = command_dir / f"{args.disease}.prscs_full_train.array_tasks.txt"
    task_list.write_text("\n".join(str(path) for path in task_scripts) + "\n", encoding="utf-8")
    score_list = command_dir / f"{args.disease}.prscs_full_train.score_scripts.txt"
    score_list.write_text("\n".join(str(path) for path in score_scripts) + "\n", encoding="utf-8")
    manifest_out = command_dir / f"{args.disease}.prscs_full_train.tasks.csv"
    write_table(pd.DataFrame(manifest_rows), manifest_out)

    print(f"Wrote PRS-CS prepare script to {prepare}")
    print(f"Wrote {len(task_scripts)} PRS-CS chromosome task scripts to {task_list}")
    print(f"Wrote PRS-CS scoring scripts to {score_list}")
    print(f"Wrote PRS-CS task manifest to {manifest_out}")


if __name__ == "__main__":
    main()
