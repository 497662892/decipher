"""Assemble the frozen 3-versus-5 BC panels after both Ho components pass scoring QC."""

import argparse
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from decipehr.external_pgs import load_weight_lock, sha256_file, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    root = Path(args.run_dir)
    protocol = json.loads((root / "PROTOCOL.json").read_text())
    for name, path in protocol["source_paths"].items():
        if sha256_file(path) != protocol["source_sha256"][name]:
            raise ValueError(f"Frozen preparation input changed: {name}")
    locks, runs, frames, qcs = {}, {}, {}, {}
    directories = {"baseline": root / "banks/baseline", "new": root / "scoring_new"}
    for name, directory in directories.items():
        locks[name] = load_weight_lock(directory / "weights.scored.lock.json")
        runs[name] = json.loads((directory / "scoring_run.json").read_text())
        for key, filename in [("weight_lock_sha256", "weights.scored.lock.json"),
                              ("scores_sha256", "scores.long.csv"), ("qc_sha256", "scoring_qc.csv")]:
            if runs[name][key] != sha256_file(directory / filename):
                raise ValueError(f"Changed {name} scoring artifact: {filename}")
        frames[name] = pd.read_csv(directory / "scores.long.csv", dtype={"sampleid": str},
                                   float_precision="round_trip")
        qcs[name] = pd.read_csv(directory / "scoring_qc.csv")
    if [r["id"] for r in locks["new"]["resources"]] != protocol["new_candidates"]:
        raise ValueError("Both Ho components must pass the fixed 75% gate for the planned 5-score panel")
    if [r["id"] for r in locks["baseline"]["resources"]] != protocol["baseline_ids"]:
        raise ValueError("Baseline score identities changed")
    for key in ("pfile", "reference_frequency_sha256", "min_overlap"):
        if runs["baseline"][key] != runs["new"][key]:
            raise ValueError(f"Baseline and new scoring differ in {key}")
    scores = pd.concat([frames["baseline"], frames["new"]], ignore_index=True)
    if (scores.duplicated(["sampleid", "disease", "model"]).any()
            or not scores.disease.eq("bc").all() or not np.isfinite(scores.prs).all()):
        raise ValueError("Duplicate, non-BC, or nonfinite scores")
    patient_sets = scores.groupby("model").sampleid.agg(set)
    if any(s != patient_sets.iloc[0] for s in patient_sets):
        raise ValueError("The five scores do not cover the same patients")
    out = root / "banks/expanded"
    out.mkdir(parents=True, exist_ok=False)
    scores.to_csv(out / "scores.long.csv", index=False)
    os.chmod(out / "scores.long.csv", 0o600)
    qc = pd.concat([qcs["baseline"], qcs["new"]], ignore_index=True)
    qc.to_csv(out / "scoring_qc.csv", index=False)
    resources = locks["baseline"]["resources"] + locks["new"]["resources"]
    write_json(out / "weights.scored.lock.json", {
        **locks["baseline"], "resources": resources,
        "assembly_sources": [str(p) for p in directories.values()],
        "excluded_by_variant_qc": qc.loc[~qc.included].to_dict("records"),
    })
    write_json(out / "scoring_run.json", {
        **runs["baseline"], "assembly_only": True, "n_scored": len(resources),
        "n_requested_scores": len(qc), "batch_scoring": False,
        "scored_weight_lock": str((out / "weights.scored.lock.json").resolve()),
        "weight_lock_sha256": sha256_file(out / "weights.scored.lock.json"),
        "scores_sha256": sha256_file(out / "scores.long.csv"),
        "qc_sha256": sha256_file(out / "scoring_qc.csv"),
        "source_scoring_runs": {str(p / "scoring_run.json"): sha256_file(p / "scoring_run.json")
                                for p in directories.values()},
    })
    reread = pd.read_csv(out / "scores.long.csv", dtype={"sampleid": str}, float_precision="round_trip")
    retained = reread.loc[reread.model.isin(protocol["baseline_ids"])].reset_index(drop=True)
    if not retained.equals(frames["baseline"].reset_index(drop=True)):
        raise ValueError("Baseline score values changed during assembly")
    for scope in ("development", "oof"):
        script = ("scripts/evaluate_external_pgs_validation.py" if scope == "development"
                  else "scripts/evaluate_external_pgs_cv.py")
        command = [
            "uv", "run", "--no-sync", "python", script,
            "--weight-lock", str(out / "weights.scored.lock.json"),
            "--scores", str(out / "scores.long.csv"), "--scoring-run", str(out / "scoring_run.json"),
            "--ancestry-labels", protocol["source_paths"]["ancestry"],
            "--eligibility-config", protocol["source_paths"]["eligibility"],
            "--diseases", "bc", "--methods", ",".join(protocol["methods"]),
            "--preflight-only", "--out-dir", str(root / "preflight" / scope / "expanded"),
        ]
        if scope == "oof":
            command += ["--splits", str(root / "shared_folds.csv")]
        subprocess.run(command, check=True)
    write_json(root / "ASSEMBLED.json", {
        "panels": {"baseline": protocol["baseline_ids"], "expanded": [r["id"] for r in resources]},
        "baseline_scores_preserved_exactly": True,
        "n_scored_before_female_eligibility": len(patient_sets.iloc[0]),
        "expanded_preflights_passed": 2,
    })


if __name__ == "__main__":
    main()
