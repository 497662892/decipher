from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from decipehr.schema import require_columns

TabPFNStrategy = Literal["prs_only", "covariate_adjusted"]


@dataclass(frozen=True)
class NumericFeature:
    source: str
    output: str
    mean: float
    sd: float


@dataclass(frozen=True)
class CategoricalFeature:
    source: str
    output_prefix: str
    categories: tuple[str, ...]


@dataclass(frozen=True)
class TabPFNFeatureTransformer:
    numeric_features: tuple[NumericFeature, ...]
    categorical_features: tuple[CategoricalFeature, ...]
    output_columns: tuple[str, ...]

    @classmethod
    def fit(
        cls,
        frame: pd.DataFrame,
        *,
        prs_cols: list[str],
        covariates: list[str],
        include_covariates: bool,
    ) -> TabPFNFeatureTransformer:
        require_columns(frame, prs_cols, "TabPFN PRS feature frame")
        if include_covariates:
            require_columns(frame, covariates, "TabPFN covariate feature frame")

        numeric_features: list[NumericFeature] = []
        categorical_features: list[CategoricalFeature] = []
        output_columns: list[str] = []

        for column in prs_cols:
            feature = _fit_numeric_feature(frame[column], source=column, output=f"prs__{column}")
            if feature is not None:
                numeric_features.append(feature)
                output_columns.append(feature.output)

        if include_covariates:
            for column in covariates:
                numeric = pd.to_numeric(frame[column], errors="coerce")
                if pd.api.types.is_numeric_dtype(frame[column]) or numeric.notna().mean() >= 0.95:
                    feature = _fit_numeric_feature(
                        numeric,
                        source=column,
                        output=f"cov__{column}",
                    )
                    if feature is not None:
                        numeric_features.append(feature)
                        output_columns.append(feature.output)
                    continue

                values = _categorical_values(frame[column])
                categories = tuple(sorted(values.dropna().unique().tolist()))
                if len(categories) <= 1:
                    continue
                output_prefix = f"cov__{column}"
                categorical_features.append(
                    CategoricalFeature(source=column, output_prefix=output_prefix, categories=categories)
                )
                output_columns.extend([f"{output_prefix}__{category}" for category in categories])

        if not output_columns:
            raise ValueError("No usable TabPFN ensemble features remain after preprocessing")
        return cls(
            numeric_features=tuple(numeric_features),
            categorical_features=tuple(categorical_features),
            output_columns=tuple(output_columns),
        )

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        parts: list[pd.DataFrame] = []
        for feature in self.numeric_features:
            values = pd.to_numeric(frame[feature.source], errors="coerce").fillna(feature.mean)
            parts.append(((values - feature.mean) / feature.sd).rename(feature.output).to_frame())
        for feature in self.categorical_features:
            values = _categorical_values(frame[feature.source])
            encoded = pd.DataFrame(index=frame.index)
            for category in feature.categories:
                encoded[f"{feature.output_prefix}__{category}"] = (values == category).astype(float)
            parts.append(encoded)
        matrix = pd.concat(parts, axis=1) if parts else pd.DataFrame(index=frame.index)
        return matrix.reindex(columns=list(self.output_columns), fill_value=0.0).astype(float)

    def summary(self) -> pd.DataFrame:
        rows = []
        for feature in self.numeric_features:
            feature_type = "prs" if feature.output.startswith("prs__") else "covariate"
            rows.append(
                {
                    "source": feature.source,
                    "feature": feature.output,
                    "feature_type": feature_type,
                    "encoding": "standardized_numeric",
                    "mean": feature.mean,
                    "sd": feature.sd,
                    "category": "",
                }
            )
        for feature in self.categorical_features:
            for category in feature.categories:
                rows.append(
                    {
                        "source": feature.source,
                        "feature": f"{feature.output_prefix}__{category}",
                        "feature_type": "covariate",
                        "encoding": "one_hot",
                        "mean": np.nan,
                        "sd": np.nan,
                        "category": category,
                    }
                )
        return pd.DataFrame(rows)


def _fit_numeric_feature(values: pd.Series, *, source: str, output: str) -> NumericFeature | None:
    numeric = pd.to_numeric(values, errors="coerce")
    mean = float(numeric.mean())
    if not np.isfinite(mean):
        return None
    sd = float(numeric.std(ddof=0))
    if not np.isfinite(sd) or sd <= 0:
        return None
    return NumericFeature(source=source, output=output, mean=mean, sd=sd)


def _categorical_values(values: pd.Series) -> pd.Series:
    return values.astype("string").str.strip().fillna("__missing__")


def logit_from_probability(probability: np.ndarray, *, eps: float = 1e-5) -> np.ndarray:
    probs = np.asarray(probability, dtype=float)
    clipped = np.clip(probs, eps, 1.0 - eps)
    return np.log(clipped / (1.0 - clipped))


def scale_validation_logits(
    *,
    train_logits: np.ndarray,
    validation_logits: np.ndarray,
    mode: str = "train_sd",
) -> tuple[np.ndarray, np.ndarray, float, float]:
    train = np.asarray(train_logits, dtype=float)
    validation = np.asarray(validation_logits, dtype=float)
    if mode == "none":
        return train, validation, 0.0, 1.0
    if mode != "train_sd":
        raise ValueError(f"Unsupported score scaling mode: {mode}")
    mean = float(np.nanmean(train))
    sd = float(np.nanstd(train))
    if not np.isfinite(mean):
        mean = 0.0
    if not np.isfinite(sd) or sd <= 0:
        sd = 1.0
    return (train - mean) / sd, (validation - mean) / sd, mean, sd


def stratified_oof_splits(y: pd.Series, *, requested_splits: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    labels = y.astype(int).to_numpy()
    counts = pd.Series(labels).value_counts()
    if len(counts) < 2:
        raise ValueError("TabPFN context labels must contain both classes")
    n_splits = min(int(requested_splits), int(counts.min()))
    if n_splits < 2:
        raise ValueError("At least two cases per class are required for OOF TabPFN scoring")
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(splitter.split(np.zeros(len(labels)), labels))
