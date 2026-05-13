from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from decipehr.io import write_table
from decipehr.schema import require_columns

DEFAULT_GENOTYPE_SAMPLE_COLUMNS = ("#IID", "IID", "sampleid", "patient_id")
LABELED_TRAIN = "labeled_train"
UNLABELLED_VALIDATION = "unlabelled_validation"
UNUSED_QC = "unused_qc"


@dataclass(frozen=True)
class SampleSetTables:
    manifest: pd.DataFrame
    counts: pd.DataFrame
    all_qc_samples: pd.DataFrame
    labeled_train_samples: pd.DataFrame
    unlabelled_validation_samples: pd.DataFrame
    phenotype_without_qc_samples: pd.DataFrame
    validation_without_qc_samples: pd.DataFrame


def resolve_sample_column(
    frame: pd.DataFrame,
    requested: str | None,
    context: str,
    candidates: tuple[str, ...] = DEFAULT_GENOTYPE_SAMPLE_COLUMNS,
) -> str:
    if requested:
        require_columns(frame, [requested], context)
        return requested
    for column in candidates:
        if column in frame.columns:
            return column
    raise ValueError(f"{context} is missing a sample ID column; tried {list(candidates)}")


def _ordered_unique_sample_ids(frame: pd.DataFrame, column: str, context: str) -> list[str]:
    require_columns(frame, [column], context)
    values = frame[column].astype("string").str.strip()
    if values.isna().any() or (values == "").any():
        raise ValueError(f"{context} has missing or blank sample IDs in {column}")
    duplicated = values.duplicated()
    if duplicated.any():
        examples = values.loc[duplicated].head(5).tolist()
        raise ValueError(f"{context} has duplicate sample IDs in {column}, examples={examples}")
    return values.tolist()


def build_sample_set_tables(
    genotype_samples: pd.DataFrame,
    phenotype: pd.DataFrame,
    patient_col: str,
    genotype_sample_col: str | None = None,
    validation_samples: pd.DataFrame | None = None,
    validation_patient_col: str | None = None,
) -> SampleSetTables:
    """Split QC genotype samples into labeled training and unlabelled validation sets."""
    sample_col = resolve_sample_column(genotype_samples, genotype_sample_col, "genotype_samples")
    genotype_ids = _ordered_unique_sample_ids(genotype_samples, sample_col, "genotype_samples")
    phenotype_ids = _ordered_unique_sample_ids(phenotype, patient_col, "phenotype")
    phenotype_set = set(phenotype_ids)
    genotype_set = set(genotype_ids)
    validation_ids: list[str] = []
    validation_set: set[str] = set()
    if validation_samples is not None:
        validation_col = validation_patient_col or patient_col
        validation_ids = _ordered_unique_sample_ids(validation_samples, validation_col, "validation_samples")
        validation_set = set(validation_ids)

    manifest_rows = []
    for order, sample_id in enumerate(genotype_ids):
        has_phenotype = sample_id in phenotype_set
        in_validation = sample_id in validation_set
        if has_phenotype:
            sample_set = LABELED_TRAIN
        elif validation_samples is None or in_validation:
            sample_set = UNLABELLED_VALIDATION
        else:
            sample_set = UNUSED_QC
        manifest_rows.append(
            {
                patient_col: sample_id,
                "genotype_order": order,
                "has_phenotype": has_phenotype,
                "in_validation_covariates": in_validation,
                "sample_set": sample_set,
            }
        )
    manifest = pd.DataFrame(manifest_rows)

    phenotype_without_qc = [sample_id for sample_id in phenotype_ids if sample_id not in genotype_set]
    labeled = manifest.loc[manifest["sample_set"] == LABELED_TRAIN, [patient_col]].reset_index(drop=True)
    unlabelled = manifest.loc[
        manifest["sample_set"] == UNLABELLED_VALIDATION, [patient_col]
    ].reset_index(drop=True)
    all_qc = manifest[[patient_col]].copy()
    phenotype_only = pd.DataFrame({patient_col: phenotype_without_qc})
    validation_without_qc = pd.DataFrame({patient_col: [sample_id for sample_id in validation_ids if sample_id not in genotype_set]})

    counts = pd.DataFrame(
        [
            {"sample_set": "all_qc", "n": len(all_qc)},
            {"sample_set": "phenotype", "n": len(phenotype_ids)},
            {"sample_set": "validation_covariates", "n": len(validation_ids)},
            {"sample_set": LABELED_TRAIN, "n": len(labeled)},
            {"sample_set": UNLABELLED_VALIDATION, "n": len(unlabelled)},
            {"sample_set": UNUSED_QC, "n": int((manifest["sample_set"] == UNUSED_QC).sum())},
            {"sample_set": "phenotype_without_qc", "n": len(phenotype_only)},
            {"sample_set": "validation_without_qc", "n": len(validation_without_qc)},
        ]
    )
    return SampleSetTables(
        manifest=manifest,
        counts=counts,
        all_qc_samples=all_qc,
        labeled_train_samples=labeled,
        unlabelled_validation_samples=unlabelled,
        phenotype_without_qc_samples=phenotype_only,
        validation_without_qc_samples=validation_without_qc,
    )


def _write_keep_file(samples: pd.DataFrame, patient_col: str, path: str | Path) -> None:
    keep = samples.rename(columns={patient_col: "#IID"})
    write_table(keep[["#IID"]], path)


def write_sample_set_tables(
    tables: SampleSetTables,
    out_dir: str | Path,
    patient_col: str,
) -> dict[str, Path]:
    """Write manifest, count report, simple sample lists, and PLINK2 keep files."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "manifest": out_dir / "sample_sets.csv",
        "counts": out_dir / "sample_set_counts.csv",
        "all_qc_samples": out_dir / "all_qc_samples.txt",
        "labeled_train_samples": out_dir / "labeled_train_samples.txt",
        "unlabelled_validation_samples": out_dir / "unlabelled_validation_samples.txt",
        "phenotype_without_qc_samples": out_dir / "phenotype_without_qc_samples.txt",
        "validation_without_qc_samples": out_dir / "validation_without_qc_samples.txt",
        "labeled_train_keep": out_dir / "labeled_train.keep.psam",
        "unlabelled_validation_keep": out_dir / "unlabelled_validation.keep.psam",
    }
    write_table(tables.manifest, paths["manifest"])
    write_table(tables.counts, paths["counts"])
    write_table(tables.all_qc_samples, paths["all_qc_samples"])
    write_table(tables.labeled_train_samples, paths["labeled_train_samples"])
    write_table(tables.unlabelled_validation_samples, paths["unlabelled_validation_samples"])
    write_table(tables.phenotype_without_qc_samples, paths["phenotype_without_qc_samples"])
    write_table(tables.validation_without_qc_samples, paths["validation_without_qc_samples"])
    _write_keep_file(tables.labeled_train_samples, patient_col, paths["labeled_train_keep"])
    _write_keep_file(
        tables.unlabelled_validation_samples,
        patient_col,
        paths["unlabelled_validation_keep"],
    )
    return paths
