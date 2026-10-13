"""Compare original and QC/family-aware fits on identical development patients."""

import json
import os
from pathlib import Path

import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from decipehr.ancestry_performance import prediction_metrics
from decipehr.eligibility import filter_eligible, load_eligibility_rules
from decipehr.external_pgs import sha256_file, write_json
from decipehr.internal_prs_benchmark import DISEASES, METHODS


def metrics(frame):
    result = prediction_metrics(frame.label.to_numpy(), frame.probability.to_numpy())
    result.update(auc=roc_auc_score(frame.label, frame.probability),
                  log_loss=log_loss(frame.label, frame.probability),
                  brier=brier_score_loss(frame.label, frame.probability))
    return result


def main():
    os.umask(0o077)
    root = Path("outputs/internal_prs/nyu_20261011")
    manifest = json.loads((root / "prepared/manifest.json").read_text())
    rules = load_eligibility_rules(manifest["source_paths"]["eligibility"])
    phenotype = pd.read_csv(manifest["source_paths"]["development"], dtype={"sampleid": str})
    rows, deltas = [], []
    for disease in DISEASES:
        path = root / "original_cohort_sensitivity" / disease
        receipt = json.loads((path / "COMPLETE.json").read_text())
        if receipt["status"] != "complete" or receipt["prepared_manifest_sha256"] != sha256_file(root / "prepared/manifest.json"):
            raise ValueError("Incomplete sensitivity run")
        for name, digest in receipt["output_sha256"].items():
            if sha256_file(path / name) != digest:
                raise ValueError("Sensitivity output changed")
        original = pd.read_csv(path / "predictions.csv", dtype={"sampleid": str}, float_precision="round_trip")
        main_path = root / "partitions" / disease / "development"
        main_receipt = json.loads((main_path / "COMPLETE.json").read_text())
        if sha256_file(main_path / "predictions.csv") != main_receipt["output_sha256"]["predictions.csv"]:
            raise ValueError("Main development prediction changed")
        primary = pd.read_csv(main_path / "predictions.csv", dtype={"sampleid": str}, float_precision="round_trip")
        if original.duplicated(["sampleid", "method"]).any() or set(original.method) != set(METHODS):
            raise ValueError("Invalid sensitivity coverage")
        expected = filter_eligible(phenotype, disease, rules).set_index("sampleid")[disease].sort_index()
        common = original.loc[original.sampleid.isin(primary.sampleid)].copy()
        for method in METHODS:
            current = original.loc[original.method == method].set_index("sampleid").sort_index()
            if not current.index.equals(expected.index) or not (current.label == expected).all():
                raise ValueError("Sensitivity labels or patients changed")
            estimates = {}
            for cohort, frame in (("original_all", original), ("original_common_patients", common),
                                  ("qc_family_common_patients", primary)):
                selected = frame.loc[frame.method == method]
                estimates[cohort] = metrics(selected)
                rows.append({"disease": disease, "method": method, "cohort": cohort, **estimates[cohort]})
            delta = {metric: estimates["qc_family_common_patients"][metric] - estimates["original_common_patients"][metric]
                     for metric in ("auc", "log_loss", "brier")}
            deltas.append({"disease": disease, "method": method, "n_common": len(common.loc[common.method == method]),
                           **{f"qc_fit_minus_original_fit_{key}": value for key, value in delta.items()}})
    out = root / "original_cohort_sensitivity/report"
    out.mkdir(exist_ok=False)
    pd.DataFrame(rows).to_csv(out / "metrics.csv", index=False)
    pd.DataFrame(deltas).to_csv(out / "common_patient_context_changes.csv", index=False)
    write_json(out / "VERIFIED.json", {
        "status": "complete", "diseases": list(DISEASES), "methods": list(METHODS),
        "inference": "descriptive sensitivity; original patient-only fits do not resolve relatedness",
        "script_sha256": sha256_file(__file__),
        "output_sha256": {p.name: sha256_file(p) for p in out.iterdir() if p.is_file()},
    })
    print(f"Verified sensitivity report: {out}")


if __name__ == "__main__":
    main()
