from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from decipehr.ancestry import DEFAULT_PC_COLS, write_pc_scatter_svg
from decipehr.io import read_table, write_table
from decipehr.qc import feature_count_report, feature_summary, label_balance, missingness_report
from decipehr.schema import parse_csv_list, require_columns, validate_binary_labels, validate_unique

DEFAULT_TRAIN_VAL_PHENOTYPE = (
    "data/raw/holdout_phase/phenotype_files/"
    "Training_Phase_Cov_file_with_Phenotype.txt"
)
DEFAULT_HOLDOUT_COVARIATES = (
    "data/raw/holdout_phase/phenotype_files/Holdout_Cov_file.txt"
)
DEFAULT_TRAIN_QC_PSAM = "outputs/genotype_qc/qc_final/all_chromosomes.qc.psam"
DEFAULT_HOLDOUT_PSAM = (
    "data/raw/holdout_phase/genotype_files/plink_files/"
    "Holdout_chr1.psam"
)


def _sample_col(frame: pd.DataFrame, requested: str | None, context: str) -> str:
    if requested:
        require_columns(frame, [requested], context)
        return requested
    for column in ["#IID", "IID", "sampleid", "patient_id"]:
        if column in frame.columns:
            return column
    raise ValueError(f"{context} is missing a recognizable sample ID column")


def _ordered_ids(frame: pd.DataFrame, column: str, context: str) -> list[str]:
    require_columns(frame, [column], context)
    values = frame[column].astype("string").str.strip()
    bad = values.isna() | (values == "")
    if bad.any():
        raise ValueError(f"{context} has blank sample IDs in {column}")
    duplicated = values.duplicated()
    if duplicated.any():
        examples = values.loc[duplicated].head(5).tolist()
        raise ValueError(f"{context} has duplicate sample IDs in {column}: {examples}")
    return values.tolist()


def _read_psam_ids(path: str | Path, sample_col: str | None, context: str) -> list[str]:
    psam = read_table(path)
    column = _sample_col(psam, sample_col, context)
    return _ordered_ids(psam, column, context)


def _write_keep(ids: list[str], path: str | Path) -> None:
    write_table(pd.DataFrame({"#IID": ids}), path)


def _write_id_list(ids: list[str], path: str | Path, patient_col: str) -> None:
    write_table(pd.DataFrame({patient_col: ids}), path)


def _alignment_rows(
    *,
    train_val_ids: list[str],
    train_qc_ids: list[str],
    test_ids: list[str],
    holdout_psam_ids: list[str],
) -> pd.DataFrame:
    train_val_set = set(train_val_ids)
    train_qc_set = set(train_qc_ids)
    test_set = set(test_ids)
    holdout_set = set(holdout_psam_ids)
    rows = [
        {"metric": "train_val_phenotype", "n": len(train_val_ids)},
        {"metric": "train_qc_psam", "n": len(train_qc_ids)},
        {"metric": "train_val_in_train_qc", "n": len(train_val_set & train_qc_set)},
        {"metric": "train_val_without_train_qc", "n": len(train_val_set - train_qc_set)},
        {"metric": "train_qc_not_train_val", "n": len(train_qc_set - train_val_set)},
        {"metric": "holdout_covariates", "n": len(test_ids)},
        {"metric": "holdout_psam", "n": len(holdout_psam_ids)},
        {"metric": "holdout_covariates_in_holdout_psam", "n": len(test_set & holdout_set)},
        {"metric": "holdout_covariates_without_holdout_psam", "n": len(test_set - holdout_set)},
        {"metric": "holdout_psam_not_covariates", "n": len(holdout_set - test_set)},
    ]
    return pd.DataFrame(rows)


def _relabel_by_train_size(labels: np.ndarray) -> tuple[np.ndarray, dict[int, int]]:
    values, counts = np.unique(labels, return_counts=True)
    order = sorted(zip(values.tolist(), counts.tolist(), strict=True), key=lambda item: (-item[1], item[0]))
    mapping = {int(old): new for new, (old, _) in enumerate(order)}
    relabeled = np.array([mapping[int(label)] for label in labels], dtype=int)
    return relabeled, mapping


def _cluster_quality(
    train_val: pd.DataFrame,
    pc_cols: list[str],
    *,
    k_min: int,
    k_max: int,
    seed: int,
    n_init: int,
) -> pd.DataFrame:
    pcs = train_val[pc_cols].apply(pd.to_numeric, errors="coerce")
    scaled = StandardScaler().fit_transform(pcs)
    rows = []
    for k in range(k_min, k_max + 1):
        model = KMeans(n_clusters=k, random_state=seed, n_init=n_init).fit(scaled)
        sample_size = 5000 if len(scaled) > 5000 else None
        rows.append(
            {
                "k": int(k),
                "inertia": float(model.inertia_),
                "silhouette": float(
                    silhouette_score(scaled, model.labels_, sample_size=sample_size, random_state=seed)
                ),
            }
        )
    return pd.DataFrame(rows)


def _fit_assign_pc_clusters(
    train_val: pd.DataFrame,
    test: pd.DataFrame,
    *,
    patient_col: str,
    pc_cols: list[str],
    n_clusters: int,
    seed: int,
    n_init: int,
) -> pd.DataFrame:
    require_columns(train_val, [patient_col, *pc_cols], "train-val phenotype")
    require_columns(test, [patient_col, *pc_cols], "holdout covariates")
    train_pcs = train_val[pc_cols].apply(pd.to_numeric, errors="coerce")
    test_pcs = test[pc_cols].apply(pd.to_numeric, errors="coerce")
    if train_pcs.isna().any().any() or test_pcs.isna().any().any():
        raise ValueError("PC columns must be numeric and complete in train-val and holdout test")

    scaler = StandardScaler().fit(train_pcs)
    train_scaled = scaler.transform(train_pcs)
    test_scaled = scaler.transform(test_pcs)
    model = KMeans(n_clusters=n_clusters, random_state=seed, n_init=n_init).fit(train_scaled)

    train_labels, mapping = _relabel_by_train_size(model.labels_)
    test_old = model.predict(test_scaled)
    test_labels = np.array([mapping[int(label)] for label in test_old], dtype=int)
    old_by_new = {new: old for old, new in mapping.items()}
    ordered_old_labels = [old_by_new[index] for index in range(n_clusters)]

    train_distances = model.transform(train_scaled)[:, ordered_old_labels]
    test_distances = model.transform(test_scaled)[:, ordered_old_labels]

    train_out = _cluster_frame(
        train_val,
        train_labels,
        train_distances,
        split="train_val",
        patient_col=patient_col,
        pc_cols=pc_cols,
    )
    test_out = _cluster_frame(
        test,
        test_labels,
        test_distances,
        split="holdout_test",
        patient_col=patient_col,
        pc_cols=pc_cols,
    )
    return pd.concat([train_out, test_out], ignore_index=True)


def _cluster_frame(
    frame: pd.DataFrame,
    labels: np.ndarray,
    distances: np.ndarray,
    *,
    split: str,
    patient_col: str,
    pc_cols: list[str],
) -> pd.DataFrame:
    sorted_distances = np.sort(distances, axis=1)
    out = frame[[patient_col, *pc_cols]].copy()
    out.insert(1, "split", split)
    out.insert(2, "ancestry_cluster", labels.astype(int))
    out.insert(3, "ancestry_cluster_distance", distances[np.arange(len(labels)), labels])
    out.insert(4, "ancestry_cluster_distance_margin", sorted_distances[:, 1] - sorted_distances[:, 0])
    for cluster_index in range(distances.shape[1]):
        out[f"d_ancestry_cluster_{cluster_index}"] = distances[:, cluster_index]
    return out


def _cluster_summary(assignments: pd.DataFrame, pc_cols: list[str]) -> pd.DataFrame:
    rows = []
    for (split, cluster), frame in assignments.groupby(["split", "ancestry_cluster"], sort=True):
        row: dict[str, int | float | str] = {
            "split": str(split),
            "ancestry_cluster": int(cluster),
            "n": int(len(frame)),
            "distance_mean": float(frame["ancestry_cluster_distance"].mean()),
            "distance_margin_mean": float(frame["ancestry_cluster_distance_margin"].mean()),
        }
        for pc_col in pc_cols:
            row[f"{pc_col}_mean"] = float(pd.to_numeric(frame[pc_col], errors="coerce").mean())
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare final train-val and holdout-test tables, keep files, and PC-based ancestry assignments."
    )
    parser.add_argument("--train-val-phenotype", default=DEFAULT_TRAIN_VAL_PHENOTYPE)
    parser.add_argument("--holdout-covariates", default=DEFAULT_HOLDOUT_COVARIATES)
    parser.add_argument("--train-qc-psam", default=DEFAULT_TRAIN_QC_PSAM)
    parser.add_argument("--holdout-psam", default=DEFAULT_HOLDOUT_PSAM)
    parser.add_argument("--patient-col", default="sampleid")
    parser.add_argument("--train-psam-col", default=None)
    parser.add_argument("--holdout-psam-col", default=None)
    parser.add_argument("--labels", default="cad,bc,T2D")
    parser.add_argument(
        "--covariates",
        default="sex,age_sd,PC1_sd,PC2_sd,PC3_sd,PC4_sd,PC5_sd,PC6_sd,PC7_sd,PC8_sd,PC9_sd,PC10_sd",
    )
    parser.add_argument("--pc-cols", default=",".join(DEFAULT_PC_COLS))
    parser.add_argument("--out-dir", default="data/processed/final")
    parser.add_argument("--sample-set-dir", default="data/processed/sample_sets")
    parser.add_argument("--ancestry-dir", default="outputs/final_ancestry")
    parser.add_argument("--n-clusters", type=int, default=10)
    parser.add_argument("--quality-k-min", type=int, default=2)
    parser.add_argument("--quality-k-max", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-init", type=int, default=50)
    args = parser.parse_args()

    patient_col = args.patient_col
    labels = parse_csv_list(args.labels)
    covariates = parse_csv_list(args.covariates)
    pc_cols = parse_csv_list(args.pc_cols)

    train_val = read_table(args.train_val_phenotype)
    test = read_table(args.holdout_covariates)
    require_columns(train_val, [patient_col, *labels, *covariates], "train-val phenotype")
    require_columns(test, [patient_col, *covariates], "holdout covariates")
    validate_unique(train_val, patient_col, "train-val phenotype")
    validate_unique(test, patient_col, "holdout covariates")
    validate_binary_labels(train_val, labels)

    train_val_ids = _ordered_ids(train_val, patient_col, "train-val phenotype")
    test_ids = _ordered_ids(test, patient_col, "holdout covariates")
    train_qc_ids = _read_psam_ids(args.train_qc_psam, args.train_psam_col, "train QC psam")
    holdout_psam_ids = _read_psam_ids(args.holdout_psam, args.holdout_psam_col, "holdout psam")

    train_qc_set = set(train_qc_ids)
    holdout_set = set(holdout_psam_ids)
    train_keep_ids = [sample_id for sample_id in train_val_ids if sample_id in train_qc_set]
    test_keep_ids = [sample_id for sample_id in test_ids if sample_id in holdout_set]
    missing_train = [sample_id for sample_id in train_val_ids if sample_id not in train_qc_set]
    extra_train_qc = [sample_id for sample_id in train_qc_ids if sample_id not in set(train_val_ids)]
    missing_test = [sample_id for sample_id in test_ids if sample_id not in holdout_set]
    extra_holdout = [sample_id for sample_id in holdout_psam_ids if sample_id not in set(test_ids)]

    out_dir = Path(args.out_dir)
    sample_set_dir = Path(args.sample_set_dir)
    ancestry_dir = Path(args.ancestry_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sample_set_dir.mkdir(parents=True, exist_ok=True)
    ancestry_dir.mkdir(parents=True, exist_ok=True)

    train_val_out = out_dir / "train_val_phenotype.csv"
    test_out = out_dir / "holdout_test_covariates.csv"
    write_table(train_val, train_val_out)
    write_table(test, test_out)

    reports_dir = out_dir / "reports"
    write_table(missingness_report(train_val), reports_dir / "train_val_missingness.csv")
    write_table(label_balance(train_val, labels), reports_dir / "train_val_label_balance.csv")
    write_table(
        feature_summary(train_val, patient_col, labels, covariates),
        reports_dir / "train_val_feature_summary.csv",
    )
    write_table(
        feature_count_report(train_val, patient_col, labels, covariates),
        reports_dir / "train_val_feature_counts.csv",
    )
    write_table(missingness_report(test), reports_dir / "holdout_test_missingness.csv")
    write_table(
        feature_summary(test, patient_col, [], covariates),
        reports_dir / "holdout_test_feature_summary.csv",
    )
    write_table(
        feature_count_report(test, patient_col, [], covariates),
        reports_dir / "holdout_test_feature_counts.csv",
    )

    write_table(
        _alignment_rows(
            train_val_ids=train_val_ids,
            train_qc_ids=train_qc_ids,
            test_ids=test_ids,
            holdout_psam_ids=holdout_psam_ids,
        ),
        reports_dir / "sample_alignment_counts.csv",
    )
    _write_id_list(missing_train, reports_dir / "train_val_without_train_qc_samples.csv", patient_col)
    _write_id_list(extra_train_qc, reports_dir / "train_qc_not_train_val_samples.csv", patient_col)
    _write_id_list(missing_test, reports_dir / "holdout_test_without_holdout_psam_samples.csv", patient_col)
    _write_id_list(extra_holdout, reports_dir / "holdout_psam_not_covariates_samples.csv", patient_col)

    train_keep = sample_set_dir / "train_val.keep.psam"
    test_keep = sample_set_dir / "holdout_test.keep.psam"
    _write_keep(train_keep_ids, train_keep)
    _write_keep(test_keep_ids, test_keep)
    _write_id_list(train_keep_ids, sample_set_dir / "train_val_samples.txt", patient_col)
    _write_id_list(test_keep_ids, sample_set_dir / "holdout_test_samples.txt", patient_col)

    assignments = _fit_assign_pc_clusters(
        train_val,
        test,
        patient_col=patient_col,
        pc_cols=pc_cols,
        n_clusters=args.n_clusters,
        seed=args.seed,
        n_init=args.n_init,
    )
    assignments_out = ancestry_dir / "pc_ancestry_assignments.csv"
    write_table(assignments, assignments_out)
    write_table(_cluster_summary(assignments, pc_cols), ancestry_dir / "pc_ancestry_cluster_summary.csv")
    write_table(
        _cluster_quality(
            train_val,
            pc_cols,
            k_min=args.quality_k_min,
            k_max=args.quality_k_max,
            seed=args.seed,
            n_init=args.n_init,
        ),
        ancestry_dir / "pc_ancestry_cluster_quality.csv",
    )

    train_with_ancestry = train_val.merge(
        assignments[[patient_col, "ancestry_cluster", "ancestry_cluster_distance", "ancestry_cluster_distance_margin"]],
        on=patient_col,
        how="left",
        validate="one_to_one",
    )
    test_with_ancestry = test.merge(
        assignments[[patient_col, "ancestry_cluster", "ancestry_cluster_distance", "ancestry_cluster_distance_margin"]],
        on=patient_col,
        how="left",
        validate="one_to_one",
    )
    write_table(train_with_ancestry, out_dir / "train_val_phenotype_with_pc_ancestry.csv")
    write_table(test_with_ancestry, out_dir / "holdout_test_covariates_with_pc_ancestry.csv")
    write_pc_scatter_svg(
        pd.concat([train_val[[patient_col, *pc_cols]], test[[patient_col, *pc_cols]]], ignore_index=True),
        assignments[[patient_col, "ancestry_cluster"]],
        ancestry_dir / "pc1_pc2_ancestry_clusters.svg",
        patient_col=patient_col,
        cluster_col="ancestry_cluster",
        pc1_col=pc_cols[0],
        pc2_col=pc_cols[1],
    )

    summary = {
        "train_val_phenotype": str(train_val_out),
        "holdout_test_covariates": str(test_out),
        "train_keep": str(train_keep),
        "holdout_test_keep": str(test_keep),
        "pc_ancestry_assignments": str(assignments_out),
        "n_train_val": len(train_val_ids),
        "n_train_val_keep": len(train_keep_ids),
        "n_holdout_test": len(test_ids),
        "n_holdout_test_keep": len(test_keep_ids),
        "n_train_qc_extra": len(extra_train_qc),
        "n_holdout_psam_extra": len(extra_holdout),
        "n_pc_ancestry_clusters": args.n_clusters,
        "pc_cols": pc_cols,
    }
    (out_dir / "final_preprocessing_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
