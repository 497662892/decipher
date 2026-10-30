"""Fit-scope-only scaling and the prescribed centered age basis."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import SplineTransformer, StandardScaler

from decipehr.prs_distillation.config import digest

AGE = "age_sd"
PCS = [f"PC{i}_sd" for i in range(1, 11)]


def frame_hash(frame: pd.DataFrame) -> str:
    """Hash ordered values, schema and dtypes, without serializing IDs to reports."""
    import hashlib

    h = hashlib.sha256()
    h.update(pd.util.hash_pandas_object(frame, index=False).to_numpy().tobytes())
    h.update(digest([(c, str(frame[c].dtype)) for c in frame]).encode())
    return h.hexdigest()


def id_hash(frame: pd.DataFrame) -> str:
    return digest(sorted(frame.sampleid.astype(str).tolist()))


def assert_boundary(context: pd.DataFrame, query: pd.DataFrame, *, require_group_disjoint=True):
    for key in ("sampleid", "kinship_group"):
        if context[key].isna().any() or query[key].isna().any():
            raise ValueError(f"Missing {key}")
        if (key == "sampleid" or require_group_disjoint) and set(context[key]) & set(query[key]):
            raise ValueError(f"Context/query overlap in {key}")


@dataclass
class Preprocessor:
    prs: list[str]
    source_columns: list[str]
    kept_prs: list[str]
    dropped_prs: dict[str, str]
    categories: list[str]
    scaler: StandardScaler
    spline: SplineTransformer | None
    basis_mean: np.ndarray
    age_mean: float
    age_sd: float
    age_min: float
    age_max: float
    reference_age: float
    age_quartiles: list[float]
    fit_n: int
    fit_ids_hash: str
    fit_features_hash: str

    @classmethod
    def fit(cls, frame: pd.DataFrame, prs: list[str]):
        columns = [*prs, AGE, *PCS, "sex"]
        if not prs or len(set(prs)) != len(prs):
            raise ValueError("Nonempty unique PRS columns required")
        cls.validate_frame(frame, columns)
        if set(frame.sex.astype(str)) - {"Female", "Male", "Unknown"}:
            raise ValueError("Unsupported provided sex category")
        if "sampleid" not in frame or not frame.sampleid.is_unique or frame.sampleid.isna().any():
            raise ValueError("Preprocessor requires unique, nonmissing fit IDs")
        kept, dropped = [], {}
        for name in prs:
            if frame[name].nunique() == 1:
                dropped[name] = "constant_in_fit_scope"
            else:
                identical = next((k for k in kept if np.array_equal(frame[name], frame[k])), None)
                if identical is None:
                    kept.append(name)
                else:
                    dropped[name] = f"exact_duplicate_of:{identical}"
        if not kept:
            raise ValueError("No nonconstant, distinct PRS remain")
        numeric = [*kept, AGE, *PCS]
        scaler = StandardScaler().fit(frame[numeric])
        ages = frame[[AGE]].to_numpy(dtype=float)
        knots = np.quantile(ages[:, 0], [0, 0.5, 1])
        spline = None
        age_mean, age_sd = float(ages.mean()), float(ages.std())
        if len(np.unique(knots)) == 3 and frame[AGE].nunique() >= 3:
            spline = SplineTransformer(n_knots=3, degree=2, knots="quantile",
                                      include_bias=False, extrapolation="constant").fit(ages)
            basis = spline.transform(ages)
        else:
            basis = (ages - age_mean) / (age_sd or 1.0)
        return cls(prs, columns, kept, dropped, sorted(frame.sex.astype(str).unique()),
                   scaler, spline, basis.mean(axis=0), age_mean, age_sd or 1.0,
                   float(ages.min()), float(ages.max()), float(np.median(ages)),
                   np.quantile(ages, [0.25, 0.5, 0.75]).tolist(), len(frame), id_hash(frame),
                   frame_hash(frame[["sampleid", *columns]]))

    @staticmethod
    def validate_frame(frame, columns):
        if frame.columns.duplicated().any() or any(c not in frame for c in columns):
            raise ValueError("Missing or duplicated input feature columns")
        # Extra identifiers/outcomes are ignored, but feature order is an explicit contract.
        if [c for c in frame if c in columns] != columns:
            raise ValueError("Input feature order differs from the bundle contract")
        if frame[columns].isna().any().any():
            raise ValueError("Missing features: the v1 primary analysis uses complete inputs")
        if not np.isfinite(frame[[c for c in columns if c != "sex"]].to_numpy(float)).all():
            raise ValueError("Nonfinite input features")

    def age_basis(self, age):
        age = np.asarray(age, dtype=float).reshape(-1, 1)
        if self.spline is not None:
            return self.spline.transform(age) - self.basis_mean
        clipped = np.clip(age, self.age_min, self.age_max)
        return (clipped - self.age_mean) / self.age_sd - self.basis_mean

    def transform(self, frame):
        self.validate_frame(frame, self.source_columns)
        unseen = set(frame.sex.astype(str)) - set(self.categories)
        # The provided categorical Unknown value is distinct from a missing value.
        # Match the existing TabPFNFeatureTransformer's all-zero one-hot encoding
        # when this declared category is absent in a context. Never map it to sex.
        if unseen and (unseen != {"Unknown"} or len(self.categories) == 1):
            raise ValueError("Unsupported sex category in query")
        numeric = self.scaler.transform(frame[[*self.kept_prs, AGE, *PCS]])
        k = len(self.kept_prs)
        z, a = numeric[:, :k], numeric[:, k + 1:]
        c = np.empty((len(frame), 0))
        if len(self.categories) > 1:
            c = np.column_stack([(frame.sex.astype(str) == cat).astype(float)
                                 for cat in self.categories])
        phi = self.age_basis(frame[AGE])
        baseline = np.column_stack([np.ones(len(frame)), phi, a, c])
        return {"z": z, "a": a, "phi": phi, "baseline": baseline,
                "teacher": np.column_stack([z, numeric[:, k:k + 1], a, c]),
                "sex_out_of_support": ~frame.sex.astype(str).isin(self.categories).to_numpy(),
                "age_out_of_support": ((frame[AGE] < self.age_min) |
                                       (frame[AGE] > self.age_max)).to_numpy()}

    def metadata(self):
        return {
            "source_columns": self.source_columns, "prs_columns": self.kept_prs,
            "dropped_prs": self.dropped_prs, "sex_categories": self.categories,
            "constant_covariates_removed": ["sex"] if len(self.categories) == 1 else [],
            "scaler_mean": self.scaler.mean_.tolist(), "scaler_scale": self.scaler.scale_.tolist(),
            "basis": "spline_3knots_degree2" if self.spline is not None else "logged_linear_fallback",
            "basis_mean": self.basis_mean.tolist(),
            "spline_knots": self.spline.bsplines_[0].t.tolist() if self.spline is not None else [],
            "age_units": "provided standardized age (age_sd); not years",
            "age_support": [self.age_min, self.age_max], "reference_age": self.reference_age,
            "age_quartiles": self.age_quartiles, "fit_n": self.fit_n,
            "fit_ids_hash": self.fit_ids_hash, "fit_features_hash": self.fit_features_hash,
            "prs_policy": "existing context mean/SD ddof=0; no ancestry residualization",
            "sex_unknown_policy": "declared Unknown absent from fit uses existing all-zero one-hot encoding and is flagged; other unseen categories fail",
        }
