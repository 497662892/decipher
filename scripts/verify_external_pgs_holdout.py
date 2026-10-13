"""Independently reproduce saved wrapper probabilities and audit holdout completeness."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from decipehr.eligibility import filter_eligible, load_eligibility_rules
from decipehr.external_pgs import sha256_file, write_json
from decipehr.external_pgs_benchmark import DEMOGRAPHICS, PCS, _ids_hash, _inner_splits
from decipehr.models import fit_logistic_predict_proba
from decipehr.tabpfn_ensemble import logit_from_probability


def main() -> None:
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default="outputs/external_pgs/holdout_wrapper_20261012")
    args = parser.parse_args()
    root = Path(args.run_dir)
    target = root / "FINAL_VERIFIED.json"
    if target.exists():
        raise FileExistsError("Holdout verification already completed")
    selection = json.loads((root / "SELECTION.json").read_text())
    raw = pd.read_csv("data/processed/final/holdout_test_covariates.csv", dtype={"sampleid": str})
    phenotype = pd.read_csv("data/processed/final/train_val_phenotype.csv", dtype={"sampleid": str})
    rules = load_eligibility_rules("configs/disease_eligibility.yaml")
    parts, counts, sources = [], [], {}
    for disease in ["T2D", "cad", "bc"]:
        directory = root / "predictions" / disease
        receipt = json.loads((directory / "VERIFIED.json").read_text())
        for name, digest in receipt["output_sha256"].items():
            if sha256_file(directory / name) != digest:
                raise ValueError("Changed saved prediction output")
        for name, digest in receipt["input_sha256"].items():
            if sha256_file(receipt["input_paths"][name]) != digest:
                raise ValueError("Changed model input")
        pred = pd.read_csv(directory / "holdout_predictions.csv", dtype={"sampleid": str, "y_prob": np.float32}, float_precision="round_trip")
        oof = pd.read_csv(directory / "wrapper_training_oof.csv", dtype={"sampleid": str, "y_prob": np.float32})
        fits = pd.read_csv(directory / "fit_audit.csv")
        query = filter_eligible(raw, disease, rules).reset_index(drop=True)
        # The benchmark prepares context in stable patient-ID order before CV.
        context = filter_eligible(phenotype, disease, rules).sort_values(
            "sampleid", kind="stable",
        ).reset_index(drop=True)
        if pred.sampleid.duplicated().any() or oof.sampleid.duplicated().any():
            raise ValueError("Duplicated prediction IDs")
        if set(pred.sampleid) != set(query.sampleid) or set(oof.sampleid) != set(context.sampleid):
            raise ValueError("Prediction coverage differs from eligible cohort")
        pred = pred.set_index("sampleid").loc[query.sampleid].reset_index()
        oof = oof.set_index("sampleid").loc[context.sampleid].reset_index()
        if not np.array_equal(context[disease].to_numpy(), oof.y_true.to_numpy()):
            raise ValueError("Wrapper training outcomes differ")
        probabilities = pred[["y_prob", "wrapper_prob"]].to_numpy()
        if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
            raise ValueError("Invalid holdout probabilities")
        if "y_true" in pred or set(context.sampleid) & set(pred.sampleid):
            raise ValueError("Holdout label or patient leakage")
        for fold, (train_idx, val_idx) in enumerate(_inner_splits(context, disease, 42)):
            row = fits.loc[fits.fit_scope.eq(f"wrapper_inner_{fold}")].iloc[0]
            if row.context_ids_sha256 != _ids_hash(context.iloc[train_idx]) or row.query_ids_sha256 != _ids_hash(context.iloc[val_idx]):
                raise ValueError("Wrapper OOF fit hash mismatch")
            if not oof.iloc[val_idx].fold.eq(fold).all():
                raise ValueError("OOF folds differ from frozen implementation")
        full = fits.loc[fits.fit_scope.eq("full_trainval_context")].iloc[0]
        if full.context_ids_sha256 != _ids_hash(context) or full.query_ids_sha256 != _ids_hash(query):
            raise ValueError("Final context/query hash mismatch")
        expected = fit_logistic_predict_proba(
            train=context.assign(external_logit=logit_from_probability(oof.y_prob.to_numpy(), eps=0.001)),
            val=query.assign(external_logit=logit_from_probability(pred.y_prob.to_numpy(), eps=0.001)),
            label_col=disease, feature_cols=[*DEMOGRAPHICS, *PCS, "external_logit"],
            patient_col="sampleid", class_weight=None,
        )
        error = float(np.max(np.abs(expected - pred.wrapper_prob.to_numpy())))
        if error > 1e-10:
            raise ValueError(f"Wrapper replay mismatch: {disease} {error}")
        counts.append({"disease": disease, "n_context": len(context), "context_cases": int(context[disease].sum()),
                       "n_holdout": len(pred), "n_scores": len(selection["diseases"][disease]["model_ids"]),
                       "wrapper_replay_max_abs_error": error, "metrics_available": False})
        parts.append(pred)
        sources[str(directory / "VERIFIED.json")] = sha256_file(directory / "VERIFIED.json")
    combined = pd.concat(parts, ignore_index=True)
    combined.to_csv(root / "holdout_predictions.long.csv", index=False)
    wide = combined.pivot(index="sampleid", columns="disease", values="wrapper_prob").reindex(raw.sampleid)
    wide.reset_index().to_csv(root / "holdout_wrapper_probabilities.csv", index=False)
    pd.DataFrame(counts).to_csv(root / "cohort_audit.csv", index=False)
    write_json(target, {
        "status": "complete", "holdout_labels_available": False, "metrics_available": False,
        "user_confirmed_no_labels": True, "n_unique_holdout": len(raw), "n_patient_disease_predictions": len(combined),
        "cohorts": counts, "source_receipts_sha256": sources,
        "selection_sha256": sha256_file(root / "SELECTION.json"),
        "output_sha256": {name: sha256_file(root / name) for name in ["holdout_predictions.long.csv", "holdout_wrapper_probabilities.csv", "cohort_audit.csv"]},
        "verification": ["Complete female eligibility", "No context-query overlap", "Every wrapper training score is OOF", "Independent logistic wrapper replay", "Frozen inputs unchanged"],
        "wide_bc_missingness": "Blank BC for noneligible patients is intentional; no predictions fabricated",
    })
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
