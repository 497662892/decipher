"""Freeze panel choices from completed development results, never holdout outcomes."""

from __future__ import annotations

import datetime
import json
from pathlib import Path

import pandas as pd

from decipehr.external_pgs import sha256_file, write_json


def main() -> None:
    root = Path("outputs/external_pgs/holdout_wrapper_20261012")
    if (root / "SELECTION.json").exists():
        raise FileExistsError("Panel selection is already frozen")
    original = Path("outputs/external_pgs/run_20261010")
    choices = {"T2D": [("original26", original / "scoring_v2", original / "official_validation/T2D") ]}
    for disease, directory in [("cad", "cad_expansion_20261012"), ("bc", "bc_ho2022_preparation_20261012")]:
        base = Path("outputs/external_pgs") / directory
        choices[disease] = [(panel, base / "banks" / panel, base / "evaluation/development" / panel)
                            for panel in ["baseline", "expanded"]]
    hashes, selected, rows = {}, {}, []
    for disease, candidates in choices.items():
        ranked = []
        for panel, bank, evaluation in candidates:
            run = json.loads((evaluation / "run.json").read_text())
            if run["status"] != "complete":
                raise ValueError("Incomplete development evaluation")
            metrics_path = evaluation / "summary_metrics.csv"
            metrics = pd.read_csv(metrics_path)
            metric = metrics.loc[metrics.method.eq("tabpfn_prs_pc") & metrics.ancestry_group.eq("ALL")
                                 & metrics.prediction_scale.eq("official_wrapper")]
            if len(metric) != 1:
                raise ValueError("Missing unique development wrapper metric")
            result = metric.iloc[0]
            lock_path = bank / "weights.scored.lock.json"
            lock = json.loads(lock_path.read_text())
            ids = [r["id"] for r in lock["resources"] if r["disease"] == disease]
            if ids != run["score_banks"][disease]:
                raise ValueError("Score bank differs from development evaluation")
            row = {"disease": disease, "panel": panel, "bank": str(bank), "model_ids": ids,
                   "n_scores": len(ids), "development_wrapper_auc": float(result.auc),
                   "development_wrapper_log_loss": float(result.log_loss),
                   "development_wrapper_r2": float(result.nagelkerke_r2)}
            rows.append(row)
            ranked.append(row)
            for path in [metrics_path, evaluation / "run.json", lock_path, bank / "scores.long.csv", bank / "scoring_run.json"]:
                hashes[str(path)] = sha256_file(path)
        selected[disease] = sorted(ranked, key=lambda r: (-r["development_wrapper_auc"], r["development_wrapper_log_loss"]))[0]
    write_json(root / "SELECTION.json", {
        "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "method": "tabpfn_prs_pc", "prediction_scale": "official_wrapper",
        "interpretation": "Best observed development wrapper AUC among full-panel alternatives of the current fixed primary method; not a global optimum or proof of a significant gain",
        "selection_rule": json.loads((root / "PLAN.json").read_text())["panel_selection"],
        "holdout_outcomes_used": False, "diseases": selected, "candidates": rows, "input_sha256": hashes,
    })
    for disease, row in selected.items():
        print(disease, row["panel"], row["n_scores"], row["development_wrapper_auc"])


if __name__ == "__main__":
    main()
