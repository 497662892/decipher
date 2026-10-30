"""Predict the previous QC validation subset with the historically aligned QC Teacher.

This small control isolates the training-cohort change from seed/column-order changes.
It never loads validation outcomes or fits a student.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import replace
from pathlib import Path

import pandas as pd

from decipehr.external_pgs import sha256_file, write_json
from decipehr.prs_distillation.config import load_config
from decipehr.prs_distillation.experiment import load_inputs, verify_run
from decipehr.prs_distillation.teacher import make_teacher, teacher_contract, teacher_predict


def main():
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--reference-run", required=True)
    args = parser.parse_args()
    root, reference = Path(args.run_dir), Path(args.reference_run)
    cfg = load_config(args.config)
    verify_run(root, cfg)
    if cfg.cohort_policy != "original_full_sensitivity" or cfg.teacher_preprocessing != "historical_external":
        raise ValueError("This control requires the declared full-cohort historical Teacher configuration")
    previous = json.loads((reference / "audit_report.json").read_text())
    if previous["cohort_policy"] != "audited_qc":
        raise ValueError("Reference must be the previous audited QC cohort")
    control_cfg = replace(cfg, cohort_policy="audited_qc")
    contract = teacher_contract(control_cfg)
    for trait in cfg.traits:
        full_d, full_e, full_manifest = load_inputs(root, trait)
        d, e, manifest = load_inputs(reference, trait)
        pd.testing.assert_frame_equal(manifest, full_manifest)
        for subset, full in [(d, full_d), (e, full_e)]:
            matched = full.set_index("sampleid", drop=False).loc[subset.sampleid].reset_index(drop=True)
            pd.testing.assert_frame_equal(subset, matched)
        directory = root / "aligned_qc_teacher" / trait
        teacher_predict(
            d, e, manifest.column.tolist(), directory=directory, contract=contract, trait=trait,
            manifest_hash=sha256_file(reference / trait / "feature_manifest.tsv"), fold_id="D_to_E",
            factory=lambda: make_teacher(control_cfg),
        )
        write_json(directory / "CONTROL.json", {
            "status": "predictions_complete_not_evaluated", "reference_run": str(reference),
            "n_training": len(d), "n_evaluation": len(e), "teacher": contract,
            "validation_labels_read": False, "student_fitted": False,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "script_sha256": sha256_file(__file__),
            "reference_input_receipt_sha256": sha256_file(reference / trait / "input_receipt.json"),
            "cache_sha256": sha256_file(directory / "cache.json"),
        })


if __name__ == "__main__":
    main()
