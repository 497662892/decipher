"""Prepare frozen PRS and reference PCs from an external cohort with PLINK2."""

from __future__ import annotations

import csv
import fcntl
import gzip
import hashlib
import json
import shutil
import subprocess
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from decipehr.external_bundle import contained, load_bundle, read_json
from decipehr.external_pgs import collect_score_parts, plink_score_command, sha256_file, write_json
from decipehr.io import read_table


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def file_identity(path):
    """Large genotype streams use explicit file identity, not a claimed full hash."""
    path = Path(path)
    stat = path.stat()
    value = {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    if stat.st_size <= 128 * 1024 * 1024:
        value["sha256"] = sha256_file(path)
    if path.is_symlink():
        value["symlink_target"] = str(path.resolve())
    return value


def stage(directory, fingerprint, action):
    """Resume verified stages; retain interrupted work before retrying that stage."""
    directory = Path(directory)
    receipt = directory / "COMPLETE.json"
    if receipt.exists():
        saved = read_json(receipt)
        if saved["fingerprint"] != fingerprint:
            raise ValueError(f"Inputs changed for completed stage {directory}; use a new run directory")
        for name, identity in saved["outputs"].items():
            if file_identity(directory / name) != identity:
                raise ValueError(f"Completed stage output changed: {directory / name}")
        print(f"Reuse {directory.name}", flush=True)
        return saved["result"]
    if directory.exists():
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
        directory.rename(directory.with_name(directory.name + ".incomplete-" + stamp))
    directory.mkdir(parents=True)
    started = time.monotonic()
    result = action(directory)
    outputs = {str(p.relative_to(directory)): file_identity(p)
               for p in sorted(directory.rglob("*")) if p.is_file()}
    write_json(receipt, {"status": "complete", "fingerprint": fingerprint, "outputs": outputs, "result": result,
                         "seconds": time.monotonic() - started})
    return result


@contextmanager
def run_lock(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".run.lock").open("w") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another process is using this run directory") from exc
        yield


def load_cohort_config(path):
    path = Path(path).resolve()
    config = yaml.safe_load(path.read_text())
    allowed = {"schema_version", "cohort", "bundle", "run_dir", "genome_build", "genotypes", "covariates",
               "columns", "sex_mapping", "age", "coverage", "qc", "runtime", "inference"}
    if not isinstance(config, dict) or config.get("schema_version") != 1 or set(config) - allowed:
        raise ValueError("Unknown cohort configuration key or schema version")
    for name in ["cohort", "bundle", "run_dir", "genome_build", "genotypes", "covariates"]:
        if not config.get(name):
            raise ValueError(f"Missing cohort configuration: {name}")
    if config["genome_build"] != "GRCh38":
        raise ValueError("These frozen models require GRCh38; liftover is a separate upstream operation")
    for name in ["bundle", "run_dir", "covariates"]:
        config[name] = str((path.parent / config[name]).resolve())
    if not isinstance(config["genotypes"], list) or not config["genotypes"]:
        raise ValueError("genotypes must be a nonempty list of filesets")
    for source in config["genotypes"]:
        if set(source) - {"format", "path", "sample", "ref_mode", "dosage_field"}:
            raise ValueError("Unknown genotype-source option")
        if source.get("format") not in {"pfile", "bfile", "vcf", "bgen"}:
            raise ValueError("Genotypes support pfile, bfile, vcf, and bgen")
        for name in ["path", "sample"]:
            if name in source:
                source[name] = str((path.parent / source[name]).resolve())
        if source["format"] == "bgen" and (not source.get("sample") or source.get("ref_mode") not in {"ref-first", "ref-last"}):
            raise ValueError("BGEN needs a sample file and an explicit ref-first/ref-last convention")
    config.setdefault("columns", {"sampleid": "sampleid", "sex": "sex", "age": "age_sd"})
    if set(config["columns"]) != {"sampleid", "sex", "age"}:
        raise ValueError("columns must name sampleid, sex, and age")
    config.setdefault("sex_mapping", {"Female": "Female", "Male": "Male", "Unknown": "Unknown"})
    if set(config["sex_mapping"].values()) - {"Female", "Male", "Unknown"}:
        raise ValueError("Sex mapping values must be Female, Male, or Unknown")
    config.setdefault("age", {"mode": "deferred"})
    if config["age"].get("mode") not in {"deferred", "training_scale"}:
        raise ValueError("Age mode is deferred or training_scale; raw years are not a compatible input")
    config.setdefault("coverage", {})
    if set(config["coverage"]) - {"missing_variants", "min_prs_fraction", "min_pc_fraction"}:
        raise ValueError("Unknown coverage option")
    config["coverage"].setdefault("missing_variants", "error")
    config["coverage"].setdefault("min_prs_fraction", .99)
    config["coverage"].setdefault("min_pc_fraction", .99)
    if config["coverage"]["missing_variants"] not in {"error", "reference_mean"}:
        raise ValueError("missing_variants is error or reference_mean")
    if any(not 0 < config["coverage"][k] <= 1 for k in ["min_prs_fraction", "min_pc_fraction"]):
        raise ValueError("Coverage fractions must be in (0, 1]")
    config.setdefault("qc", {})
    if set(config["qc"]) - {"max_sample_missingness"}:
        raise ValueError("Unknown QC option")
    config["qc"].setdefault("max_sample_missingness", .05)
    if not 0 <= config["qc"]["max_sample_missingness"] <= 1:
        raise ValueError("max_sample_missingness must be in [0, 1]")
    config.setdefault("runtime", {})
    if set(config["runtime"]) - {"plink2", "threads", "memory_mb"}:
        raise ValueError("Unknown runtime option")
    config["runtime"].setdefault("plink2", "plink2")
    config["runtime"].setdefault("threads", 4)
    config["runtime"].setdefault("memory_mb", 16000)
    if config["runtime"]["threads"] < 1 or config["runtime"]["memory_mb"] < 1024:
        raise ValueError("Positive threads and at least 1024 MB PLINK memory required")
    config.setdefault("inference", {})
    return config


def read_covariates(config):
    columns = config["columns"]
    options = {} if Path(config["covariates"]).suffix == ".parquet" else {"dtype": {columns["sampleid"]: str}}
    raw = read_table(config["covariates"], **options)
    if raw.columns.duplicated().any() or set(raw) & {"T2D", "cad", "bc", "y", "y_true"}:
        raise ValueError("Covariates need unique columns and no outcome columns; keep outcomes separate")
    if not {columns["sampleid"], columns["sex"]}.issubset(raw):
        raise ValueError("Covariates must contain the configured sampleid and sex columns")
    ids = raw[columns["sampleid"]]
    if ids.isna().any() or ids.astype(str).str.contains(r"\s").any() or ids.astype(str).eq("").any() or ids.astype(str).duplicated().any() or raw.empty:
        raise ValueError("Patient identifiers must be unique, nonempty strings without whitespace")
    sex = raw[columns["sex"]].astype("string").map(config["sex_mapping"])
    if sex.isna().any():
        raise ValueError("Unmapped/missing sex values; declare the mapping explicitly in the cohort config")
    result = pd.DataFrame({"sampleid": ids.astype(str), "sex": sex.astype(str)})
    if config["age"]["mode"] == "training_scale":
        if columns["age"] not in raw:
            raise ValueError("The compatible training-scale age column is missing")
        result["age_sd"] = pd.to_numeric(raw[columns["age"]], errors="raise")
        if not np.isfinite(result.age_sd).all():
            raise ValueError("Age must be finite; do not impute it from the external cohort")
    return result


def genotype_arguments(source):
    kind, path = source["format"], source["path"]
    if kind == "pfile":
        compressed = not Path(path + ".pvar").exists() and Path(path + ".pvar.zst").exists()
        return ["--pfile", path, *(["vzs"] if compressed else [])]
    if kind == "bfile":
        return ["--bfile", path]
    if kind == "vcf":
        return ["--vcf", path, *([f"dosage={source['dosage_field']}"] if source.get("dosage_field") else [])]
    return ["--bgen", path, source["ref_mode"], "--sample", source["sample"]]


def genotype_files(source):
    kind, path = source["format"], source["path"]
    if kind in {"pfile", "bfile"}:
        suffixes = [".pgen", ".pvar", ".psam"] if kind == "pfile" else [".bed", ".bim", ".fam"]
        paths = [Path(path + s) for s in suffixes]
        if kind == "pfile" and not paths[1].exists():
            paths[1] = Path(path + ".pvar.zst")
        if kind == "pfile" and Path(path + ".pgen.pgi").exists():
            paths.append(Path(path + ".pgen.pgi"))
        return paths
    return [Path(path), *([Path(source["sample"])] if kind == "bgen" else [])]


def _plink(config, args, directory, name):
    executable = shutil.which(config["runtime"]["plink2"])
    if executable is None:
        raise FileNotFoundError("PLINK2 unavailable; install it or set runtime.plink2")
    command = [executable, *map(str, args), "--threads", str(config["runtime"]["threads"]),
               "--memory", str(config["runtime"]["memory_mb"]), "--out", str(directory / name)]
    write_json(directory / f"{name}.command.json", command)
    print(f"PLINK {directory.name}/{name}", flush=True)
    with (directory / f"{name}.stdout.log").open("w") as log:
        subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)


def _sample_ids(path):
    frame = pd.read_csv(path, sep=r"\s+", dtype=str)
    name = "#IID" if "#IID" in frame else "IID"
    if name not in frame or frame[name].isna().any() or not frame[name].is_unique:
        raise ValueError("Genotype IIDs must uniquely identify samples, including across FIDs")
    return frame[name].tolist()


def _pvar_ids(path):
    with Path(path).open() as stream:
        first = next(line for line in stream if not line.startswith("##"))
        if first.split()[:5] != ["#CHROM", "POS", "ID", "REF", "ALT"]:
            raise ValueError("Unexpected PLINK PVAR schema")
        for line in stream:
            yield line.split("\t", 4)[2]


def align_source(config, bundle, source, directory, sample_ids):
    """PLINK matches positions/alleles; original genotypes are never rewritten."""
    root = Path(config["bundle"])
    args = [*genotype_arguments(source), "--set-all-var-ids", "UNMATCHED:@:#:$r:$a",
            "--new-id-max-allele-len", "100", "missing", "--recover-var-ids",
            contained(root, bundle["variant_map"]), "partial"]
    if source["format"] == "pfile":
        # No filtering or reordering here: the new PVAR retains the binary row axis.
        _plink(config, [*args, "--make-just-pvar", "cols=-xheader,-maybequal,-maybefilter,-maybeinfo,-maybecm"], directory, "aligned")
        Path(directory / "aligned.pgen").symlink_to(source["path"] + ".pgen")
        if Path(source["path"] + ".pgen.pgi").exists():
            Path(directory / "aligned.pgen.pgi").symlink_to(source["path"] + ".pgen.pgi")
        ids = _sample_ids(source["path"] + ".psam")
    else:
        # Conversion retains dosage when present; filtering here is handled by PLINK.
        pd.DataFrame({"#IID": sample_ids}).to_csv(directory / "keep.tsv", sep="\t", index=False)
        try:
            _plink(config, [*args, "--extract", contained(root, bundle["required_variants"]),
                           "--keep", directory / "keep.tsv", "--make-pgen"], directory, "aligned")
        except subprocess.CalledProcessError:
            if "No variants remaining after" in (directory / "aligned.stdout.log").read_text():
                return {"empty": True, "format": source["format"]}
            raise
        ids = _sample_ids(directory / "aligned.psam")
    if not set(sample_ids).issubset(ids):
        raise ValueError("A genotype shard is missing requested patients")
    # IID is the declared patient key. Preserve sample order, omit FID and phenotypes.
    pd.DataFrame({"#IID": ids}).to_csv(directory / "aligned.psam", sep="\t", index=False)
    return {"empty": False, "format": source["format"], "n_binary_samples": len(ids)}


def _coverage(bundle_root, bundle, present, directory):
    index = pd.Index(list(present))
    counts = {}
    for part in bundle["score_parts"]:
        selected = pd.Index((bundle_root / part["variants"]).read_text().splitlines())
        for chunk in pd.read_csv(bundle_root / part["weights"], sep="\t", chunksize=100000, dtype={"ID": str}):
            keep = selected.get_indexer(chunk.iloc[:, 0]) >= 0
            chunk = chunk.loc[keep]
            matched = index.get_indexer(chunk.iloc[:, 0]) >= 0
            for number in part["score_columns"]:
                name = chunk.columns[number-1]
                model = bundle["score_aliases"][name]
                values = chunk[name].to_numpy(float)
                nonzero = values != 0
                values = np.abs(values)
                row = counts.setdefault(model, np.zeros(4))
                row += [nonzero.sum(), (nonzero & matched).sum(), values.sum(), values[matched].sum()]
    rows = []
    for model, (n, observed, weight, observed_weight) in counts.items():
        if n <= 0 or weight <= 0:
            raise ValueError("A frozen PRS has no nonzero coefficients")
        trait = next(trait for trait, ids in bundle["prs"].items() if model in ids)
        rows.append({"trait": trait, "model": model, "n_terms": int(n), "n_present_terms": int(observed),
                     "fraction": observed/n, "absolute_weight_fraction": observed_weight/weight})
    table = pd.DataFrame(rows)
    table.to_csv(directory / "prs_coverage.csv", index=False)
    pc_ids = set((bundle_root / bundle["pc"]["variants"]).read_text().splitlines())
    return table, {"n_expected": len(pc_ids), "n_present": len(pc_ids & present),
                   "fraction": len(pc_ids & present)/len(pc_ids)}


def prepare_cohort(config, bundle=None):
    bundle_root, root = Path(config["bundle"]), Path(config["run_dir"])
    bundle = bundle or load_bundle(bundle_root)
    covariates = read_covariates(config)
    ids = covariates.sampleid.tolist()
    required = set((bundle_root / bundle["required_variants"]).read_text().splitlines())
    inputs = {str(p): file_identity(p) for source in config["genotypes"] for p in genotype_files(source)}
    executable = shutil.which(config["runtime"]["plink2"])
    if executable is None:
        raise FileNotFoundError("PLINK2 unavailable; install it or set runtime.plink2")
    engine = {"plink2_sha256": sha256_file(executable), "preparation_code_sha256": sha256_file(__file__)}
    fingerprint = digest({"bundle": sha256_file(bundle_root / "BUNDLE.json"), "source": inputs,
        "genotypes": config["genotypes"], "sample_ids": ids, "coverage": config["coverage"], "qc": config["qc"], "engine": engine})
    base = root / "preparation"
    # Metadata/age updates do not invalidate genetic preparation when patient IDs are unchanged.
    if (base / "PREPARED.json").exists():
        saved = read_json(base / "PREPARED.json")
        if saved["fingerprint"] != fingerprint:
            raise ValueError("Genetic preparation inputs changed; use a new run_dir")
        for name, expected in saved["output_sha256"].items():
            if sha256_file(base / name) != expected:
                raise ValueError(f"Prepared input changed: {name}")
        print("Reuse completed genetic preparation", flush=True)
        return saved
    base.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"#IID": ids}).to_csv(base / "keep.tsv", sep="\t", index=False)
    aligned, present = [], set()
    for number, source in enumerate(config["genotypes"]):
        directory = base / f"source{number}"
        result = stage(directory, digest({"source": source, "inputs": inputs, "bundle": sha256_file(bundle_root / "BUNDLE.json"), "ids": ids, "engine": engine}),
                       lambda dest, src=source: align_source(config, bundle, src, dest, ids))
        if result["empty"]:
            continue
        found = [key for key in _pvar_ids(directory / "aligned.pvar") if key in required]
        if len(found) != len(set(found)) or present.intersection(found):
            raise ValueError("Duplicate matched variants within/across genotype shards; resolve upstream")
        present.update(found)
        if found:
            aligned.append(directory / "aligned")
    missing = required - present
    def coverage_action(dest):
        table, pc = _coverage(bundle_root, bundle, present, dest)
        (dest / "missing.snplist").write_text("".join(x + "\n" for x in sorted(missing)))
        return {"n_required": len(required), "n_present": len(present), "n_missing": len(missing), "pc": pc,
                "min_prs_fraction": float(table.fraction.min()), "missing_policy": config["coverage"]["missing_variants"]}
    coverage = stage(base / "coverage", fingerprint, coverage_action)
    if not aligned or coverage["min_prs_fraction"] < config["coverage"]["min_prs_fraction"] or coverage["pc"]["fraction"] < config["coverage"]["min_pc_fraction"]:
        raise ValueError("Genetic coverage is below the configured threshold; inspect preparation/coverage before proceeding")
    if missing and config["coverage"]["missing_variants"] == "error":
        raise ValueError(f"{len(missing)} required SNPs are absent. Inspect coverage; reference_mean is an explicit alternative to upstream recovery")
    missing_genotype = None
    if missing:
        def pad_action(dest):
            # Every entry is missing. PLINK, using frozen AF, performs imputation later.
            _plink(config, ["--dummy", len(ids), len(missing), 1, 1, "acgt", "pheno-ct=0", "--make-pgen", "--seed", 17], dest, "missing")
            pd.DataFrame({"#IID": ids}).to_csv(dest / "missing.psam", sep="\t", index=False)
            with (bundle_root / bundle["variant_map"]).open() as source, (dest / "missing.pvar").open("w") as output:
                output.write("#CHROM\tPOS\tID\tREF\tALT\n")
                for chrom, key, _, pos, alt, ref in csv.reader(source, delimiter="\t"):
                    if key in missing:
                        output.write(f"{chrom}\t{pos}\t{key}\t{ref}\t{alt}\n")
            return {"n_variants": len(missing), "all_genotypes_missing": True, "generated_phenotypes_used": False}
        stage(base / "absent_variants", fingerprint, pad_action)
        missing_genotype = base / "absent_variants/missing"
    if len(aligned) > 1:
        def merge_action(dest):
            (dest / "inputs.txt").write_text("".join(str(p.resolve()) + "\n" for p in aligned))
            _plink(config, ["--pmerge-list", dest / "inputs.txt", "pfile", "--extract",
                           bundle_root / bundle["required_variants"], "--keep", base / "keep.tsv",
                           "--make-pgen", "--sort-vars"], dest, "cohort")
            return {"n_inputs": len(aligned)}
        stage(base / "merged", fingerprint, merge_action)
        genotype = base / "merged/cohort"
    else:
        genotype = aligned[0]
    def qc_action(dest):
        _plink(config, ["--pfile", genotype, "--extract", bundle_root / bundle["required_variants"] ,
                       "--keep", base / "keep.tsv", "--missing", "sample-only", "scols=+nmissdosage,+fmissdosage"], dest, "missingness")
        table = pd.read_csv(dest / "missingness.smiss", sep=r"\s+", dtype={"#IID": str, "IID": str})
        # Absent SNPs are an additional missing call for every patient. Keep this
        # separate from native dosage streams; PLINK2 cannot interleave PGENs yet.
        rates = (table["MISSING_DOSAGE_CT"].to_numpy(float) + len(missing)) / (table["OBS_CT"].to_numpy(float) + len(missing))
        if not np.isfinite(rates).all():
            raise ValueError("Undefined genotype missingness")
        return {"maximum_missing_dosage_fraction": float(rates.max()), "median_missing_dosage_fraction": float(np.median(rates)),
                "patients_over_threshold": int((rates > config["qc"]["max_sample_missingness"]).sum()), "n_patients": len(rates)}
    qc = stage(base / "qc", fingerprint, qc_action)
    if qc["patients_over_threshold"]:
        raise ValueError("Patients exceed the declared dosage-missingness threshold; inspect preparation/qc; no patients were silently dropped")
    score_frames = []
    genotype_groups = [(genotype, present)]
    if missing_genotype is not None:
        genotype_groups.append((missing_genotype, missing))
    for number, part in enumerate(bundle["score_parts"]):
        expected = set((bundle_root / part["variants"]).read_text().splitlines())
        scored_paths = []
        for group_number, (group_genotype, group_ids) in enumerate(genotype_groups):
            wanted = expected & group_ids
            if not wanted:
                continue
            def score_action(dest, item=part, part_number=number, pfile=group_genotype, expected_ids=wanted):
                command = plink_score_command(pfile=str(pfile), scorefile=str(bundle_root / item["weights"]),
                    frequency_file=str(bundle_root / bundle["reference_frequencies"]), out=str(dest / "scores"),
                    plink2=shutil.which(config["runtime"]["plink2"]), threads=config["runtime"]["threads"],
                    memory_mb=config["runtime"]["memory_mb"], extract_file=str(bundle_root / item["variants"]),
                    score_columns=item["score_columns"])
                command += ["--keep", str(base / "keep.tsv")]
                write_json(dest / "command.json", command)
                print(f"Score frozen matrix {part_number+1}/{len(bundle['score_parts'])}", flush=True)
                with (dest / "stdout.log").open("w") as log:
                    subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
                used = (dest / "scores.sscore.vars").read_text().splitlines()
                if len(used) != len(expected_ids) or set(used) != expected_ids:
                    raise ValueError("PLINK did not score exactly the frozen matrix variant partition")
                return {"n_scored_variants": len(used)}
            directory = base / (f"score{number}" if group_number == 0 else f"score{number}_missing")
            stage(directory, fingerprint, score_action)
            scored_paths.append(directory / "scores.sscore")
        with gzip.open(bundle_root / part["weights"], "rt") as stream:
            columns = stream.readline().rstrip().split("\t")
        for index in part["score_columns"]:
            alias = columns[index-1]
            model = bundle["score_aliases"][alias]
            trait = next(trait for trait, names in bundle["prs"].items() if model in names)
            score_frame = collect_score_parts(scored_paths, model, trait, tool_accession=alias)
            if not score_frame.sampleid.is_unique or set(score_frame.sampleid) != set(ids):
                raise ValueError("A PRS matrix has incomplete patient coverage")
            score_frames.append(score_frame)
    scores = pd.concat(score_frames, ignore_index=True).groupby(["sampleid", "disease", "model"], sort=False, as_index=False).prs.sum()
    if len(scores) != len(ids) * sum(map(len, bundle["prs"].values())) or set(scores.sampleid) != set(ids):
        raise ValueError("Raw PRS coverage differs from the declared cohort and panels")
    scores.to_csv(base / "scores.long.csv", index=False)
    pc_frames = []
    expected = set((bundle_root / bundle["pc"]["variants"]).read_text().splitlines())
    for group_number, (group_genotype, group_ids) in enumerate(genotype_groups):
        wanted = expected & group_ids
        if not wanted:
            continue
        def pc_action(dest, pfile=group_genotype, expected_ids=wanted):
            _plink(config, ["--pfile", pfile, "--extract", bundle_root / bundle["pc"]["variants"],
                "--keep", base / "keep.tsv", "--read-freq", bundle_root / bundle["pc"]["frequencies"], "--error-on-freq-calc",
                "--score", bundle_root / bundle["pc"]["weights"], 1, 2, "header-read", "variance-standardize", "cols=+scoresums", "list-variants",
                "--score-col-nums", "3-12"], dest, "pcs")
            used = (dest / "pcs.sscore.vars").read_text().splitlines()
            if len(used) != len(expected_ids) or set(used) != expected_ids:
                raise ValueError("Reference PCA used a different marker partition")
            return {"n_variants": len(used), "pca_refitted": False, "oadp": False}
        directory = base / ("pc" if group_number == 0 else "pc_missing")
        stage(directory, fingerprint, pc_action)
        pc_frame = pd.read_csv(directory / "pcs.sscore", sep=r"\s+", dtype={"#IID": str, "IID": str}).rename(columns={"#IID": "sampleid", "IID": "sampleid"})
        if not pc_frame.sampleid.is_unique or set(pc_frame.sampleid) != set(ids):
            raise ValueError("A PC partition has incomplete patient coverage")
        pc_frames.append(pc_frame)
    pcs = pd.concat(pc_frames).groupby("sampleid", sort=False, as_index=False)[[f"PC{i}_SUM" for i in range(1, 11)]].sum()
    pcs = pcs[["sampleid", *[f"PC{i}_SUM" for i in range(1, 11)]]].rename(columns={f"PC{i}_SUM": f"PC{i}_sd" for i in range(1, 11)})
    if not pcs.sampleid.is_unique or set(pcs.sampleid) != set(ids):
        raise ValueError("Reference PC patient coverage differs")
    wide = scores.pivot(index="sampleid", columns="model", values="prs").reindex(ids)
    features = wide.reset_index().merge(pcs, on="sampleid", validate="one_to_one")
    if not np.isfinite(features.drop(columns="sampleid").to_numpy()).all():
        raise ValueError("Genetic preparation produced missing/nonfinite features")
    features.to_parquet(base / "genetic_features.parquet", index=False)
    pcs.to_csv(base / "reference_pcs.tsv", sep="\t", index=False)
    receipt = {"status": "genetic_preparation_complete", "fingerprint": fingerprint, "n_patients": len(ids),
        "bundle_sha256": sha256_file(bundle_root / "BUNDLE.json"), "input_genotype_identity": inputs, "engine": engine,
        "large_genotype_identity_policy": "size and modification time; SHA256 additionally recorded for files <=128 MiB",
        "sample_ids_sha256": digest(ids), "coverage": coverage, "qc": qc,
        "age_used": False, "outcomes_used": False, "new_snp_weights_fitted": False,
        "output_sha256": {name: sha256_file(base / name) for name in
                          ["scores.long.csv", "reference_pcs.tsv", "genetic_features.parquet"]}}
    write_json(base / "PREPARED.json", receipt)
    return receipt
