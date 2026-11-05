"""Evaluate prediction-sealed OADP models against both frozen PC baselines."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from report_ancestry_transfer import ARMS, METRICS, metrics
from report_prs_distillation_bio import paired_bootstrap
from run_ancestry_oadp import BASELINE, MODE
from run_ancestry_transfer import FILTERS, SOURCE, TRAITS, hashes, read

from decipehr.external_pgs import sha256_file, write_json
from decipehr.prs_distillation.data import verify_files
from decipehr.prs_distillation.reporting import basic_metrics


def evaluate(root, trait):
    verify_files(Path.cwd(), read(root / "PREPARED.json")["input_sha256"])
    # Assessment labels are opened only after all three traits have sealed their predictions.
    for name in TRAITS:
        for parent in [root, BASELINE]:
            directory = parent / "private" / name
            receipt = read(directory / "COMPLETE.json")
            if receipt["validation_labels_read"]:
                raise ValueError("Training read validation outcomes")
            verify_files(directory, {n: receipt["output_sha256"][n] for n in ["predictions.parquet", "selections.csv"]})
    out = root / "report" / trait
    if (out / "COMPLETE.json").exists():
        verify_files(out, read(out / "COMPLETE.json")["output_sha256"])
        return
    out.mkdir(parents=True, exist_ok=True)
    frame = pd.concat([pd.read_parquet(parent / "private" / trait / "predictions.parquet")
                       for parent in [BASELINE, root]], ignore_index=True)
    audit = read(SOURCE / "audit_report.json")
    label_path = Path(audit["paths"]["evaluation"])
    verify_files(Path.cwd(), {str(label_path): audit["source_sha256"][str(label_path)]})
    labels = pd.read_csv(label_path, dtype={"sampleid": str}, usecols=["sampleid", trait]).set_index("sampleid")[trait]
    membership_path = "private/cohort_membership.parquet"
    verify_files(FILTERS, {membership_path: read(FILTERS / "VERIFIED.json")["output_sha256"][membership_path]})
    membership = pd.read_parquet(FILTERS / membership_path)
    original_metrics = pd.read_csv(BASELINE / "report/metrics.csv").set_index(["trait", "cohort", "mode", "arm"])
    original_counts = pd.read_csv(BASELINE / "report/cohort_counts.csv").set_index(["trait", "cohort"])
    rows, differences, counts, subgroup = [], [], [], []
    for scope in ["original_val", "nested_oof"]:
        selected = frame.loc[frame.fold.eq(-1) if scope == "original_val" else frame.fold.ge(0)].copy()
        selected["key"] = selected["mode"] + "/" + selected.arm
        wide = selected.pivot(index="sampleid", columns="key", values="probability")
        info = selected[["sampleid", "kinship_group", "hard_label"]].drop_duplicates().set_index("sampleid").loc[wide.index]
        if not info.index.is_unique or wide.isna().any().any() or len(wide.columns) != 18:
            raise ValueError("Matched patient/model coverage failed")
        if scope == "original_val":
            y = labels.loc[wide.index].to_numpy(int)
            member = membership.loc[membership.trait.eq(trait)].set_index("sampleid")
            if set(member.index) != set(wide.index):
                raise ValueError("Validation population changed")
            member = member.loc[wide.index]
            np.testing.assert_array_equal(member.kinship_group, info.kinship_group)
            masks = {"original_val": np.ones(len(wide), bool),
                     "exclude_shared_group": ~member.remove_rule1.to_numpy(),
                     "exclude_direct_0177": ~member.remove_rule2.to_numpy()}
        else:
            training = pd.read_parquet(SOURCE / trait / "training.parquet").set_index("sampleid")
            if set(training.index) != set(wide.index):
                raise ValueError("OOF population changed")
            y = training.loc[wide.index, "y"].to_numpy(int)
            masks = {"nested_oof": np.ones(len(wide), bool)}
        for cohort, mask in masks.items():
            count = {"trait": trait, "cohort": cohort, "n": int(mask.sum()), "events": int(y[mask].sum()),
                     "n_kinship_groups": info.loc[mask, "kinship_group"].nunique()}
            for key in ["n", "events", "n_kinship_groups"]:
                if count[key] != original_counts.loc[(trait, cohort), key]:
                    raise ValueError("Cohort count changed")
            counts.append(count)
            for key in wide:
                mode, arm = key.split("/")
                measured = metrics(y[mask], wide[key].to_numpy()[mask])
                if mode != MODE:
                    np.testing.assert_allclose([measured[m] for m in METRICS],
                        original_metrics.loc[(trait, cohort, mode, arm), METRICS].to_numpy(float), atol=1e-12, rtol=0)
                rows.append({"trait": trait, "cohort": cohort, "mode": mode, "arm": arm, **measured})
            pd.DataFrame(rows).to_csv(out / "metrics.csv", index=False)
            for arm in ARMS:
                # Reuse the established bootstrap through explicit comparison slots.
                # The same seed/groups give identical resamples for all models.
                slots = {"H_SELECTED": MODE, "M0_KD": "reference_pc", "M0_SUP": "official_pc"}
                arrays = {slot: wide[f"{mode}/{arm}"].to_numpy()[mask] for slot, mode in slots.items()}
                paired = paired_bootstrap(y[mask], arrays, info.loc[mask, "kinship_group"].to_numpy(),
                                          "nested", replicates=1000)
                paired = paired.loc[paired.arm.eq("H_SELECTED")].copy()
                paired["arm"] = f"{MODE}/{arm}"
                paired["reference"] = paired.reference.map({slot: f"{mode}/{arm}" for slot, mode in slots.items()})
                differences.append(paired.assign(trait=trait, cohort=cohort))
            pd.concat(differences, ignore_index=True).to_csv(out / "paired_differences.csv", index=False)
            print(f"Evaluated {trait}/{cohort}: n={mask.sum()}, three PC modes, six models", flush=True)
        if scope == "original_val":
            for group in sorted(info.hard_label.unique()):
                mask = info.hard_label.eq(group).to_numpy()
                for key in wide:
                    mode, arm = key.split("/")
                    values = basic_metrics(y[mask], wide[key].to_numpy()[mask])
                    subgroup.append({"trait": trait, "group": group, "mode": mode, "arm": arm, **values,
                        "unstable": values["n"] < 100 or min(values["events"], values["nonevents"]) < 20,
                        "stratum_source": "preexisting_1000G_hard_label_held_fixed"})
    pd.DataFrame(counts).to_csv(out / "cohort_counts.csv", index=False)
    pd.DataFrame(subgroup).to_csv(out / "subgroup_metrics.csv", index=False)
    selections = pd.concat([pd.read_csv(parent / "private" / trait / "selections.csv") for parent in [BASELINE, root]])
    selections.to_csv(out / "single_prs_selections.csv", index=False)
    write_json(out / "COMPLETE.json", {"status": "complete", "trait": trait,
        "original_metrics_reproduced": True, "matched_patients": True,
        "bootstrap_replicates": 1000, "bootstrap_seed": 20261028,
        "bootstrap_unit": "available kinship group", "multiplicity_adjusted": False,
        "models_refitted_within_bootstrap": False,
        "implementation_sha256": {str(p): sha256_file(p) for p in [Path(__file__),
            Path("scripts/report_prs_distillation_bio.py"), Path("scripts/report_ancestry_transfer.py")]},
        "output_sha256": hashes(out)})


def main():
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--trait", choices=TRAITS, required=True)
    args = parser.parse_args()
    evaluate(args.out_dir, args.trait)


if __name__ == "__main__":
    main()
