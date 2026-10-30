"""Compare frozen v1 mixed-KD and current M0_KD predictions on identical val patients."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from report_prs_distillation_bio import paired_bootstrap

from decipehr.external_pgs import sha256_file, write_json
from decipehr.metrics import nagelkerke_r2
from decipehr.prs_distillation.reporting import basic_metrics

SOURCE = Path("outputs/prs_distillation_v1/phase_a_fulltrain_20261026_3d4334d2_0161ad36d950")
BIO = Path("outputs/prs_distillation_bio/exploration_20261026_3569648f99a8")
OUT = Path("outputs/prs_distillation_bio/original_proposal_comparison_20261027")


def read(p):
    return json.loads(p.read_text())


def main():
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=OUT)
    parser.add_argument("--kinship-dir", type=Path, help="Sealed validation-only kinship-filter analysis")
    args = parser.parse_args()
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=False)
    used = {}

    def checked(p, expected):
        actual = sha256_file(p)
        assert actual == expected, p
        used[str(p)] = actual

    old_seal, bio_seal = read(SOURCE / "COMPLETE.json"), read(BIO / "COMPLETE.json")
    audit = read(SOURCE / "audit_report.json")
    labels_path = Path(audit["paths"]["evaluation"])
    checked(labels_path, audit["source_sha256"][str(labels_path)])
    labels = pd.read_csv(labels_path, dtype={"sampleid": str}, usecols=["sampleid", "T2D", "cad", "bc"]).set_index("sampleid")
    assert labels.index.is_unique
    expected_path = BIO / "report/val/metrics.csv"
    checked(expected_path, bio_seal["aggregate_output_sha256"]["report/val/metrics.csv"])
    expected = pd.read_csv(expected_path).set_index(["trait", "arm"])
    membership = counts = filtered_expected = None
    if args.kinship_dir:
        kinship = args.kinship_dir
        kinship_seal = read(kinship / "VERIFIED.json")
        assert kinship_seal["status"] == "completed_and_verified"
        assert kinship_seal["train_unchanged"] and not kinship_seal["refitted"]
        for name in ["ANALYSIS_LOCK.json", "private/cohort_membership.parquet", "report/cohort_counts.csv", "report/metrics.csv"]:
            checked(kinship / name, kinship_seal["output_sha256"][name])
        lock = read(kinship / "ANALYSIS_LOCK.json")
        assert not lock["evaluation_labels_used_for_filtering"] and not lock["reselected"]
        assert lock["unknown_kinship"] == "retained in both filters"
        checked(BIO / "COMPLETE.json", lock["input_sha256"][str(BIO / "COMPLETE.json")])
        membership = pd.read_parquet(kinship / "private/cohort_membership.parquet")
        counts = pd.read_csv(kinship / "report/cohort_counts.csv").set_index(["trait", "cohort"])
        filtered_expected = pd.read_csv(kinship / "report/metrics.csv").set_index(["trait", "cohort", "arm"])
        used[str(kinship / "VERIFIED.json")] = sha256_file(kinship / "VERIFIED.json")
    metrics, intervals, seed_deltas = [], [], []
    for trait in ["T2D", "cad", "bc"]:
        frames = {}
        for name, root, arm in [("V1_KD_MIX", SOURCE, "S_KD_MIX"), ("M0_KD", BIO, "M0_KD")]:
            core = root / "core" / trait
            receipt_path = core / "PREDICTIONS_COMPLETE.json"
            seal = read(receipt_path)
            if root == BIO:
                checked(receipt_path, bio_seal["upstream_receipt_sha256"][str(receipt_path)])
            else:
                checked(receipt_path, old_seal["output_sha256"][str(receipt_path.relative_to(SOURCE))])
            assert not seal["smoke"] and not seal["evaluation_labels_read"]
            checked(core / "direct_predictions.parquet", seal["output_sha256"]["direct_predictions.parquet"])
            frames[name] = pd.read_parquet(core / "direct_predictions.parquet",
                columns=["sampleid", "kinship_group", "arm", "seed", "probability", "p_teacher"], filters=[("arm", "=", arm)])
        eligible_path = SOURCE / trait / "evaluation_features.parquet"
        checked(eligible_path, old_seal["output_sha256"][str(eligible_path.relative_to(SOURCE))])
        eligible = pd.read_parquet(eligible_path, columns=["sampleid", "sex"])
        if trait == "bc":
            assert eligible.sex.eq("Female").all()
        base = frames["V1_KD_MIX"].loc[frames["V1_KD_MIX"].seed.eq(17)].set_index("sampleid")
        ids = base.index.tolist()
        assert base.index.is_unique and set(ids) == set(eligible.sampleid)
        y, groups = labels.loc[ids, trait].to_numpy(int), base.kinship_group.to_numpy()
        assert np.isin(y, [0, 1]).all()
        masks = {"original_val": np.ones(len(ids), dtype=bool)}
        if membership is not None:
            members = membership.loc[membership.trait.eq(trait)].set_index("sampleid")
            assert members.index.is_unique and set(members.index) == set(ids)
            members = members.loc[ids]
            np.testing.assert_array_equal(members.kinship_group, groups)
            assert not (members.remove_rule2 & ~members.remove_rule1).any()
            masks.update(exclude_shared_group=~members.remove_rule1.to_numpy(),
                         exclude_direct_0177=~members.remove_rule2.to_numpy())
            for assessment, mask in masks.items():
                prior_counts = counts.loc[(trait, assessment)]
                assert int(mask.sum()) == prior_counts["n"]
                assert int(y[mask].sum()) == prior_counts["events"]
                assert mask[members.unresolved.to_numpy()].all()
        for seed in [17, 29, 43]:
            arrays = {}
            for name, frame in frames.items():
                part = frame.loc[frame.seed.eq(seed)].set_index("sampleid")
                assert part.index.is_unique and set(part.index) == set(ids)
                part = part.loc[ids]
                np.testing.assert_array_equal(part.kinship_group, groups)
                np.testing.assert_array_equal(part.p_teacher, base.p_teacher)
                arrays[name] = part.probability.to_numpy()
            for assessment, mask in masks.items():
                points = {}
                for name, p in arrays.items():
                    points[name] = {**basic_metrics(y[mask], p[mask]), "nagelkerke_r2": nagelkerke_r2(y[mask], p[mask])}
                    if seed == 17:
                        for metric in ["auroc", "log_loss", "brier", "nagelkerke_r2"]:
                            if assessment == "original_val":
                                np.testing.assert_allclose(points[name][metric], expected.loc[(trait, name), metric], atol=1e-12, rtol=0)
                            if filtered_expected is not None and name == "M0_KD":
                                np.testing.assert_allclose(points[name][metric], filtered_expected.loc[(trait, assessment, name), metric], atol=1e-12, rtol=0)
                    metrics.append({"trait": trait, "assessment": assessment, "arm": name, "seed": seed, **points[name]})
                seed_deltas.append({"trait": trait, "assessment": assessment, "seed": seed,
                    **{f"delta_{m}": points["M0_KD"][m]-points["V1_KD_MIX"][m] for m in ["auroc", "log_loss", "nagelkerke_r2"]}})
                if seed == 17:
                    # The paired engine exposes H_SELECTED-versus-V1; this slot contains only M0_KD.
                    paired = paired_bootstrap(y[mask], {"H_SELECTED": arrays["M0_KD"][mask], "V1_KD_MIX": arrays["V1_KD_MIX"][mask]},
                                              groups[mask], "val", replicates=1000)
                    intervals.append(paired.assign(arm="M0_KD", trait=trait, seed=17, assessment=assessment))
        print(f"Verified {trait}: N={len(ids)}, identical teacher probabilities and patient groups", flush=True)
    for name, frame in [("metrics", pd.DataFrame(metrics)), ("paired_differences", pd.concat(intervals, ignore_index=True)),
                        ("seed_differences", pd.DataFrame(seed_deltas))]:
        frame.to_csv(out / f"{name}.csv", index=False)
    if counts is not None:
        counts.reset_index().to_csv(out / "cohort_counts.csv", index=False)
    for p in [SOURCE / "COMPLETE.json", BIO / "COMPLETE.json", SOURCE / "audit_report.json",
              Path("docs/1026/PRS_Prototype_Distillation_Codex_Plan_v1.md"), Path(__file__), Path("scripts/report_prs_distillation_bio.py")]:
        used[str(p)] = sha256_file(p)
    (out / "source_snapshot.py").write_bytes(Path(__file__).read_bytes())
    write_json(out / "VERIFIED.json", {"status": "completed_and_verified", "new_fits": 0, "primary_seed": 17,
        "comparison": "Current BIO M0_KD minus original v1 S_KD_MIX; matched full-train models and validation patients",
        "assessments": list(masks), "train_unchanged": True, "refitted": False, "reselected": False,
        "kinship_filters_reused_from_sealed_membership": membership is not None,
        "filtered_m0_metrics_reproduced": filtered_expected is not None,
        "unresolved_kinship_retained": True,
        "same_patients_and_teacher_predictions": True, "eligible_female_bc_only": True,
        "original_primary_metrics_reproduced": True, "bootstrap_replicates": 1000,
        "bootstrap_unit": "available_kinship_group", "bootstrap_seed": 20261028,
        "conditional_on_fixed_predictions": True, "multiplicity_adjusted": False, "developmental_posthoc_comparison": True,
        "input_sha256": used, "output_sha256": {p.name: sha256_file(p) for p in out.iterdir() if p.is_file()}})


if __name__ == "__main__":
    main()
