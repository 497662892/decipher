"""Inventory and apply published PGS weights without fitting SNP effects."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import os
import re
import shutil
import subprocess
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from decipehr.io import write_table

PGS_ID = re.compile(r"PGS\d{6}\Z")
LOCAL_SCORE_ID = re.compile(r"LOCAL_[A-Za-z0-9_]+\Z")
BUILDS = ("GRCh37", "GRCh38")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: str | Path, value: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_candidates(path: str | Path, diseases: list[str] | None = None) -> list[dict]:
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if config.get("schema_version") != 1:
        raise ValueError("Unsupported external PGS manifest version")
    candidates = []
    for group in config["score_sets"]:
        if diseases is not None and group["disease"] not in diseases:
            continue
        if group["disease"] not in {"cad", "bc", "T2D"}:
            raise ValueError("External PGS disease must be cad, bc, or T2D")
        for accession in group["ids"]:
            if not PGS_ID.fullmatch(accession):
                raise ValueError(f"Invalid PGS accession: {accession}")
            candidates.append({
                **{key: value for key, value in group.items() if key != "ids"},
                "id": accession,
                "catalog_url": f"https://www.pgscatalog.org/score/{accession}/",
            })
    ids = [item["id"] for item in candidates]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("Manifest must contain a nonempty, unique set of PGS IDs")
    return candidates


def inventory_catalog(
    manifest: str | Path, out_dir: str | Path, *, build: str,
    diseases: list[str] | None = None,
) -> dict:
    if build not in BUILDS:
        raise ValueError(f"Unsupported genome build: {build}")
    out = Path(out_dir)
    if (out / "inventory.json").exists() or (out / "review.yaml").exists():
        raise FileExistsError("Use a new inventory directory to preserve the existing review")
    out.mkdir(parents=True, exist_ok=True)
    resources = []
    for candidate in load_candidates(manifest, diseases):
        accession = candidate["id"]
        api_url = f"https://www.pgscatalog.org/rest/score/{accession}/"
        with urllib.request.urlopen(api_url, timeout=90) as response:
            metadata = json.load(response)
        if metadata.get("id") != accession:
            raise ValueError(f"Catalog returned a different accession for {accession}")
        weight_url = metadata.get("ftp_harmonized_scoring_files", {}).get(build, {}).get("positions")
        resources.append({**candidate, "metadata": metadata, "weight_url": weight_url})
    inventory = {
        "schema_version": 1, "build": build,
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "manifest_sha256": sha256_file(manifest), "resources": resources,
    }
    write_json(out / "inventory.json", inventory)
    review = {
        "target_cohort": "", "use_context": "research",
        "scores": {
            item["id"]: {
                "include": False, "trait_match": "unknown", "allowed_use": "unknown",
                "sample_overlap": "unknown", "evidence": [],
            } for item in resources
        },
    }
    (out / "review.yaml").write_text(yaml.safe_dump(review, sort_keys=False), encoding="utf-8")
    return inventory


def reviewed_resources(inventory: dict, review: dict) -> list[dict]:
    if inventory.get("build") not in BUILDS:
        raise ValueError("Inventory genome build is missing or invalid")
    if not str(review.get("target_cohort", "")).strip():
        raise ValueError("Review must identify the target cohort for sample-overlap checks")
    if review.get("use_context") not in {"research", "competition"}:
        raise ValueError("Review use_context must be research or competition")
    resources = {item["id"]: item for item in inventory["resources"]}
    if set(review["scores"]) - set(resources):
        raise ValueError("Review contains accessions absent from the inventory")
    selected = []
    for accession, decision in review["scores"].items():
        if decision.get("include") is not True:
            continue
        expected = {"trait_match": "confirmed", "allowed_use": "allowed", "sample_overlap": "excluded"}
        unresolved = [key for key, value in expected.items() if decision.get(key) != value]
        if unresolved or not decision.get("evidence"):
            raise ValueError(f"{accession}: incomplete resource review ({', '.join(unresolved)})")
        item = resources[accession]
        if not item.get("weight_url"):
            raise ValueError(f"{accession}: no harmonized weights for {inventory['build']}")
        selected.append({**item, "review": decision})
    if not selected:
        raise ValueError("No reviewed external scores selected; unknown overlap is not independent")
    return selected


def read_weight_header(path: str | Path, *, accession: str, build: str) -> dict[str, str]:
    header = {}
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            if not line.startswith("#"):
                break
            key, separator, value = line.lstrip("#").strip().partition("=")
            if separator:
                header[key.lower()] = value
    id_field = "score_id" if LOCAL_SCORE_ID.fullmatch(accession) else "pgs_id"
    if header.get(id_field) != accession or header.get("hmpos_build") != build:
        raise ValueError(f"{accession}: scoring-file accession or harmonized genome build mismatch")
    if int(header.get("variants_number", "0")) < 1:
        raise ValueError(f"{accession}: scoring-file variant count is missing")
    return header


def fetch_reviewed_weights(
    inventory_path: str | Path, review_path: str | Path, out_dir: str | Path,
) -> dict:
    inventory = json.loads(Path(inventory_path).read_text(encoding="utf-8"))
    review = yaml.safe_load(Path(review_path).read_text(encoding="utf-8"))
    selected = reviewed_resources(inventory, review)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "weights.lock.json").exists():
        raise FileExistsError("Use a new directory; do not overwrite a frozen weight lock")
    locked = []
    for item in selected:
        accession = item["id"]
        path = out / f"{accession}_hmPOS_{inventory['build']}.txt.gz"
        if not path.exists():
            temporary = path.with_name(path.name + ".part")
            with urllib.request.urlopen(item["weight_url"], timeout=180) as response:
                with temporary.open("wb") as stream:
                    shutil.copyfileobj(response, stream)
            temporary.replace(path)
        header = read_weight_header(path, accession=accession, build=inventory["build"])
        locked.append({
            **item, "path": str(path.resolve()), "sha256": sha256_file(path),
            "header": header,
        })
    lock = {
        "schema_version": 1, "build": inventory["build"], "target_cohort": review["target_cohort"],
        "use_context": review["use_context"], "inventory_sha256": sha256_file(inventory_path),
        "review_sha256": sha256_file(review_path), "resources": locked,
    }
    write_json(out / "weights.lock.json", lock)
    return lock


def load_weight_lock(path: str | Path, *, verify_files: bool = True) -> dict:
    lock = json.loads(Path(path).read_text(encoding="utf-8"))
    if lock.get("schema_version") != 1 or lock.get("build") not in BUILDS:
        raise ValueError("Invalid frozen-weight lock")
    ids = [item["id"] for item in lock["resources"]]
    if not ids or len(ids) != len(set(ids)) or any(
        not (PGS_ID.fullmatch(x) or LOCAL_SCORE_ID.fullmatch(x)) for x in ids
    ):
        raise ValueError("Frozen-weight lock has empty, invalid, or duplicate accessions")
    reviewed_resources(lock, {
        "target_cohort": lock["target_cohort"], "use_context": lock["use_context"],
        "scores": {item["id"]: item["review"] for item in lock["resources"]},
    })
    for item in lock["resources"]:
        if LOCAL_SCORE_ID.fullmatch(item["id"]):
            if item.get("source_type") != "publication_supplement":
                raise ValueError("Local scores must identify their published supplement source")
            for prefix in ("source", "chain"):
                if not item.get(f"{prefix}_path") or not re.fullmatch(
                    r"[0-9a-f]{64}", item.get(f"{prefix}_sha256", ""),
                ):
                    raise ValueError(f"Local score lacks frozen {prefix} provenance")
                if verify_files and sha256_file(item[f"{prefix}_path"]) != item[f"{prefix}_sha256"]:
                    raise ValueError(f"Local score {prefix} has changed: {item['id']}")
    if verify_files:
        for item in lock["resources"]:
            read_weight_header(item["path"], accession=item["id"], build=lock["build"])
            if sha256_file(item["path"]) != item["sha256"]:
                raise ValueError(f"Frozen weights have changed: {item['id']}")
    return lock


def plink_score_command(
    *, pfile: str, scorefile: str, frequency_file: str, out: str,
    plink2: str = "plink2", threads: int = 4, extract_file: str | None = None,
    score_columns: list[int] | None = None, memory_mb: int = 16000,
) -> list[str]:
    if not frequency_file:
        raise ValueError("External reference frequencies are required for genotype imputation")
    command = [
        plink2, "--pfile", pfile, "--read-freq", frequency_file,
        "--error-on-freq-calc",
        "--score", scorefile, "1", "2", *(["3"] if score_columns is None else []),
        "header-read", "list-variants", "cols=+scoresums", "--threads", str(threads),
        "--memory", str(memory_mb), "--out", out,
    ]
    if score_columns is not None:
        if not score_columns or min(score_columns) < 3:
            raise ValueError("Scoring columns must be nonempty and start after ID/allele")
        command.extend(["--score-col-nums", ",".join(map(str, score_columns))])
    if extract_file is not None:
        command.extend(["--extract", extract_file])
    return command


def collect_score_parts(
    paths: list[Path], accession: str, disease: str, *, tool_accession: str | None = None,
) -> pd.DataFrame:
    """Sum matcher-generated allele partitions, never PLINK score averages."""
    if not paths:
        raise ValueError("No PLINK score parts to collect")
    total = None
    for path in paths:
        frame = pd.read_csv(path, sep=r"\s+", dtype={"#IID": str, "IID": str})
        frame = frame.rename(columns={"#IID": "IID"})
        score_col = f"{tool_accession or accession}_SUM"
        if "IID" not in frame or score_col not in frame:
            raise ValueError(f"Expected IID and {score_col} in {path.name}")
        if frame["IID"].isna().any() or frame["IID"].duplicated().any():
            raise ValueError("PLINK IID must uniquely identify each patient, including across FIDs")
        values = frame.set_index("IID")[score_col].astype(float)
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite external PRS")
        if total is None:
            total = values
        else:
            if set(total.index) != set(values.index):
                raise ValueError("PLINK score partitions have different patient sets")
            total = total + values.reindex(total.index)
    assert total is not None
    return pd.DataFrame({
        "sampleid": total.index, "disease": disease, "model": accession, "prs": total.to_numpy(),
    })


def write_matching_pvar(
    source: str | Path, destination: str | Path,
    reference_alleles: dict[str, tuple[str, str]] | None = None,
) -> dict[str, int]:
    """Project PVAR columns, optionally retaining only reference-compatible variants.

    pgscatalog.match 0.4.0's Polars reader can misparse metadata plus optional
    columns. Only the matcher's input copy is projected; PLINK uses the original.
    """
    columns = ["#CHROM", "POS", "ID", "REF", "ALT"]
    counts = {"n_input": 0, "n_written": 0, "n_missing_reference": 0, "n_allele_mismatch": 0}
    with Path(source).open(encoding="utf-8") as stream:
        header = next((line for line in stream if not line.startswith("##")), "")
        names = header.rstrip("\n").split("\t")
        if not set(columns).issubset(names):
            raise ValueError("Target PVAR must have the five standard tab-separated columns")
        indices = [names.index(column) for column in columns]
        with Path(destination).open("w", encoding="utf-8", newline="") as output:
            writer = csv.writer(output, delimiter="\t", lineterminator="\n")
            writer.writerow(columns)
            for row in csv.reader(stream, delimiter="\t"):
                if len(row) <= max(indices):
                    raise ValueError("Malformed target PVAR row")
                selected = [row[index] for index in indices]
                counts["n_input"] += 1
                if reference_alleles is not None:
                    alleles = reference_alleles.get(selected[2])
                    if alleles is None:
                        counts["n_missing_reference"] += 1
                        continue
                    if set(alleles) != set(selected[3:5]):
                        counts["n_allele_mismatch"] += 1
                        continue
                writer.writerow(selected)
                counts["n_written"] += 1
    return counts


def prepare_reference_frequencies(
    *, target_pvar: str, reference_pfile: str, target_build: str, reference_build: str,
    reference_provenance: str, out_dir: str | Path, plink2: str = "plink2",
    threads: int = 4, memory_mb: int = 16000,
) -> dict:
    """Reuse conservative variant matching; estimate frequencies in the reference only."""
    from decipehr.ancestry_harmonization import match_target_to_reference, read_variant_table

    if target_build not in BUILDS or target_build != reference_build:
        raise ValueError("Target and reference genome builds must match")
    if not reference_provenance.strip() or threads < 1 or memory_mb < 640:
        raise ValueError("Reference provenance and positive resource limits are required")
    for suffix in (".pgen", ".pvar", ".psam"):
        if not Path(reference_pfile + suffix).is_file():
            raise FileNotFoundError(reference_pfile + suffix)
    out = Path(out_dir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Use an empty reference-frequency output directory")
    out.mkdir(parents=True, exist_ok=True)
    matched, matching_summary = match_target_to_reference(
        read_variant_table(target_pvar), read_variant_table(reference_pfile + ".pvar"),
    )
    if matched.empty:
        raise ValueError("No unambiguous target/reference SNPs matched")
    write_json(out / "matching_summary.json", matching_summary)
    write_table(matched, out / "matched_variants.tsv")
    extract = out / "reference.snplist"
    matched["ref_id"].to_csv(extract, header=False, index=False)
    prefix = out / "reference"
    command = [
        plink2, "--pfile", reference_pfile, "--extract", str(extract), "--freq",
        "--threads", str(threads), "--memory", str(memory_mb), "--out", str(prefix),
    ]
    write_json(out / "commands.json", [command])
    with (out / "reference.stdout.log").open("w") as log:
        subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
    frequencies = pd.read_csv(str(prefix) + ".afreq", sep=r"\s+", dtype=str)
    if frequencies["ID"].duplicated().any() or set(frequencies["ID"]) != set(matched["ref_id"]):
        raise ValueError("PLINK reference frequencies differ from the matched variant set")
    values = pd.to_numeric(frequencies["ALT_FREQS"], errors="coerce")
    if not values.between(0, 1).all():
        raise ValueError("Reference has undefined or multiallelic frequencies")
    frequencies["ID"] = frequencies["ID"].map(matched.set_index("ref_id")["target_id"])
    # Only IDs change. REF/ALT and ALT_FREQS retain the reference orientation;
    # PLINK's --read-freq resolves any target REF/ALT swap by allele labels.
    frequency_path = out / "reference.target_ids.afreq"
    frequencies.to_csv(frequency_path, sep="\t", index=False)
    with Path(reference_pfile + ".psam").open() as stream:
        n_reference_samples = sum(bool(line.strip()) and not line.startswith("#") for line in stream)
    result = {
        "schema_version": 1, "build": target_build, "n_variants": len(frequencies),
        "n_reference_samples": n_reference_samples,
        "reference_pfile": str(Path(reference_pfile).resolve()),
        "target_pvar": str(Path(target_pvar).resolve()),
        "reference_provenance": reference_provenance,
        "frequency_file": str(frequency_path.resolve()),
        "frequency_sha256": sha256_file(frequency_path),
        "input_sha256": {str(path): sha256_file(path) for path in (
            Path(target_pvar), *(Path(reference_pfile + suffix) for suffix in (".pgen", ".pvar", ".psam")),
        )},
    }
    write_json(out / "reference_run.json", result)
    return result


def score_frozen_weights(
    lock_path: str | Path, *, pfile: str, target_build: str, reference_freq: str,
    reference_provenance: str, out_dir: str | Path, plink2: str = "plink2", threads: int = 4,
    min_overlap: float = 0.75, memory_mb: int = 16000,
) -> dict:
    """Delegate all variant matching and scoring to PGS Catalog tools and PLINK2."""
    lock = load_weight_lock(lock_path)
    if lock["build"] != target_build:
        raise ValueError("Target genotype and harmonized weight genome builds must match")
    if not reference_provenance.strip() or not 0 < min_overlap <= 1 or threads < 1 or memory_mb < 640:
        raise ValueError("Reference provenance, valid min_overlap, and positive threads are required")
    for suffix in (".pgen", ".pvar", ".psam"):
        if not Path(pfile + suffix).is_file():
            raise FileNotFoundError(pfile + suffix)
    out = Path(out_dir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Use an empty scoring output directory to avoid mixing runs")
    out.mkdir(parents=True, exist_ok=True)
    matching_pvar = out / "target.match.pvar"
    frequencies = pd.read_csv(reference_freq, sep=r"\s+", dtype=str)
    required = {"ID", "REF", "ALT", "ALT_FREQS"}
    if not required.issubset(frequencies.columns) or frequencies["ID"].duplicated().any():
        raise ValueError("Reference must be a PLINK2 .afreq with unique variant IDs")
    frequencies = frequencies.set_index("ID")
    numeric_freq = pd.to_numeric(frequencies["ALT_FREQS"], errors="coerce")
    if not numeric_freq.between(0, 1).all():
        raise ValueError("External reference frequencies must be finite and biallelic")
    reference_alleles = dict(zip(
        frequencies.index, zip(frequencies["REF"], frequencies["ALT"], strict=True), strict=True,
    ))
    target_reference_qc = write_matching_pvar(pfile + ".pvar", matching_pvar, reference_alleles)
    del reference_alleles
    write_json(out / "target_reference_qc.json", target_reference_qc)
    if not target_reference_qc["n_written"]:
        raise ValueError("No target variants have compatible external reference frequencies")
    frequency_sha = sha256_file(reference_freq)
    command_log = []

    def run(command: list[str], log_path: Path) -> None:
        print(f"Running {log_path.name}", flush=True)
        command_log.append(command)
        write_json(out / "commands.json", command_log)
        env = {**os.environ, "POLARS_MAX_THREADS": str(threads)}
        with log_path.open("w", encoding="utf-8") as log:
            subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT, env=env)

    run([plink2, "--version"], out / "plink2.version.txt")
    formatted, matched = out / "formatted", out / "matched"
    formatted.mkdir()
    matched.mkdir()
    run([
        "uv", "tool", "run", "--python", "3.12", "--from", "pgscatalog.core==1.1.1",
        "pgscatalog-format", "-s", *(item["path"] for item in lock["resources"]),
        "-t", target_build, "-o", str(formatted), "--threads", str(threads),
    ], out / "format.log")
    normalised = [str(formatted / ("normalised_" + Path(item["path"]).name)) for item in lock["resources"]]
    run([
        "uv", "tool", "run", "--python", "3.12", "--from", "pgscatalog.match==0.4.0",
        "--with", "pgscatalog.core==1.1.1", "pgscatalog-match",
        "--dataset", "external", "--scorefiles", *normalised,
        "--target", str(matching_pvar), "--outdir", str(matched), "--combined",
        "--min_overlap", str(min_overlap),
    ], out / "match.log")
    # Coverage decisions use only the official match log and published variant
    # counts. No phenotype, association, or validation-performance data is read.
    summary = pd.read_csv(matched / "external_summary.csv")
    # Catalog-format may append the harmonized build to its output accession.
    # Preserve its coefficient columns and map only the model identifier back to
    # the checked header/lock; a different accession or build is never accepted.
    aliases = {
        name: item["id"] for item in lock["resources"]
        for name in (item["id"], f"{item['id']}_hmPOS_{target_build}")
    }
    if set(summary["accession"]) - set(aliases):
        raise ValueError("Matcher summary contains an unknown accession or genome build")
    summary["model"] = summary["accession"].map(aliases)
    qc_rows, included = [], []
    tool_accessions = {}
    for item in lock["resources"]:
        accession = item["id"]
        rows = summary.loc[summary["model"] == accession]
        if rows.empty or rows["score_pass"].nunique() != 1 or rows["accession"].nunique() != 1:
            raise ValueError(f"{accession}: incomplete official matching summary")
        tool_accessions[accession] = str(rows["accession"].iloc[0])
        matched_count = int(rows.loc[rows["match_status"] == "matched", "count"].sum())
        original_count = int(item["header"]["variants_number"])
        coverage = matched_count / original_count
        matcher_pass = str(rows["score_pass"].iloc[0]).lower() == "true"
        keep = matcher_pass and min_overlap <= coverage <= 1
        if keep:
            included.append(item)
        qc_rows.append({
            "model": accession, "disease": item["disease"], "weight_sha256": item["sha256"],
            "matcher_accession": tool_accessions[accession],
            "n_published_variants": original_count, "n_formatted_variants": int(rows["count"].sum()),
            "n_matched_variants": matched_count, "n_unmatched_variants": original_count - matched_count,
            "coverage": coverage, "included": keep,
            "exclusion_reason": "" if keep else "below_predeclared_variant_coverage",
        })
    write_table(pd.DataFrame(qc_rows), out / "scoring_qc.csv")
    if not included:
        raise ValueError("No external PGS passed the predeclared variant-coverage threshold")
    included_ids = {item["id"] for item in included}
    all_ids = {item["id"] for item in lock["resources"]}
    scorefiles = sorted(matched.glob("*.scorefile.gz"))
    if not scorefiles or any("_additive_" not in path.name for path in scorefiles):
        raise ValueError("No additive score files, or unsupported nonadditive external model")
    parts_by_score = {accession: [] for accession in included_ids}
    for part, scorefile in enumerate(scorefiles):
        weights = pd.read_csv(scorefile, sep="\t", dtype=str)
        raw_ids = list(weights.columns[2:])
        if set(raw_ids) - set(aliases):
            raise ValueError("Matcher emitted an accession outside the frozen weight lock")
        score_ids = [aliases[x] for x in raw_ids]
        if len(score_ids) != len(set(score_ids)) or any(
            raw != tool_accessions[canonical] for raw, canonical in zip(raw_ids, score_ids, strict=True)
        ):
            raise ValueError("Matched coefficient columns disagree with the official summary")
        if list(weights.columns[:2]) != ["ID", "effect_allele"] and list(weights.columns[:2]) != ["ID", "EA"]:
            raise ValueError("Unexpected matched variant/allele columns")
        if set(score_ids) - all_ids:
            raise ValueError("Matcher emitted an accession outside the frozen weight lock")
        selected_columns = [i + 3 for i, accession in enumerate(score_ids) if accession in included_ids]
        if not selected_columns:
            continue
        ids, alleles = weights.iloc[:, 0], weights.iloc[:, 1]
        if ids.duplicated().any() or not ids.isin(frequencies.index).all():
            raise ValueError("Duplicate score IDs or missing external reference frequencies")
        freq = frequencies.loc[ids].reset_index(drop=True)
        if not ((alleles == freq["REF"]) | (alleles == freq["ALT"])).all():
            raise ValueError("External reference allele pair is incompatible with matched scoring alleles")
        prefix = out / f"part{part}"
        extract_path = Path(str(prefix) + ".snplist")
        ids.to_csv(extract_path, index=False, header=False)
        run(plink_score_command(
            pfile=pfile, scorefile=str(scorefile), frequency_file=reference_freq,
            out=str(prefix), plink2=plink2, threads=threads, extract_file=str(extract_path),
            score_columns=selected_columns, memory_mb=memory_mb,
        ), out / f"part{part}.stdout.log")
        used = Path(str(prefix) + ".sscore.vars").read_text(encoding="utf-8").splitlines()
        if len(used) != len(ids) or set(used) != set(ids):
            raise ValueError("PLINK scored a different variant set than the audited matcher output")
        for accession in set(score_ids) & included_ids:
            parts_by_score[accession].append(Path(str(prefix) + ".sscore"))
    frames = [collect_score_parts(
        parts_by_score[item["id"]], item["id"], item["disease"], tool_accession=tool_accessions[item["id"]],
    ) for item in included]
    scores = pd.concat(frames, ignore_index=True)
    patient_sets = scores.groupby("model")["sampleid"].agg(set)
    if any(value != patient_sets.iloc[0] for value in patient_sets):
        raise ValueError("External models have different patient sets")
    for row in qc_rows:
        row["n_score_parts"] = len(parts_by_score.get(row["model"], []))
    write_table(scores, out / "scores.long.csv")
    write_table(pd.DataFrame(qc_rows), out / "scoring_qc.csv")
    scoring_lock = out / "weights.scored.lock.json"
    write_json(scoring_lock, {
        **lock, "resources": included, "source_weight_lock_sha256": sha256_file(lock_path),
        "source_weight_lock": str(Path(lock_path).resolve()), "min_overlap": min_overlap,
        "excluded_by_variant_qc": [row for row in qc_rows if not row["included"]],
    })
    result = {
        "schema_version": 1, "weight_lock_sha256": sha256_file(scoring_lock),
        "scored_weight_lock": str(scoring_lock.resolve()),
        "source_weight_lock_sha256": sha256_file(lock_path),
        "build": target_build, "pfile": str(Path(pfile).resolve()),
        "reference_frequency_sha256": frequency_sha,
        "reference_provenance": reference_provenance, "min_overlap": min_overlap,
        "target_reference_qc": target_reference_qc,
        "scores_sha256": sha256_file(out / "scores.long.csv"),
        "qc_sha256": sha256_file(out / "scoring_qc.csv"),
        "normalizer": "pgscatalog.core==1.1.1", "matcher": "pgscatalog.match==0.4.0",
        "batch_scoring": True, "n_requested_scores": len(lock["resources"]), "n_scored": len(included),
    }
    write_json(out / "scoring_run.json", result)
    return result
