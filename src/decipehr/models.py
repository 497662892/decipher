from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from decipehr.schema import require_columns


@dataclass
class FoldPrediction:
    fold: int
    disease: str
    method: str
    predictions: pd.DataFrame


def _feature_pipeline(frame: pd.DataFrame, feature_cols: list[str]) -> ColumnTransformer:
    numeric_cols = [col for col in feature_cols if pd.api.types.is_numeric_dtype(frame[col])]
    categorical_cols = [col for col in feature_cols if col not in numeric_cols]
    transformers = []
    if numeric_cols:
        transformers.append(
            ("num", Pipeline([("imputer", SimpleImputer()), ("scaler", StandardScaler())]), numeric_cols)
        )
    if categorical_cols:
        transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_cols,
            )
        )
    return ColumnTransformer(transformers)


def fit_predict_logistic(
    train: pd.DataFrame,
    val: pd.DataFrame,
    label_col: str,
    feature_cols: list[str],
    patient_col: str = "patient_id",
    method: str = "logistic",
    fold: int = 0,
) -> FoldPrediction:
    require_columns(train, [patient_col, label_col, *feature_cols], "train")
    require_columns(val, [patient_col, label_col, *feature_cols], "validation")
    if train[label_col].nunique() < 2:
        raise ValueError(f"Training fold for {label_col} has fewer than two classes")
    pipe = Pipeline(
        [
            ("features", _feature_pipeline(train, feature_cols)),
            (
                "model",
                LogisticRegression(max_iter=2000, solver="lbfgs", class_weight="balanced"),
            ),
        ]
    )
    pipe.fit(train[feature_cols], train[label_col].astype(int))
    prob = pipe.predict_proba(val[feature_cols])[:, 1]
    pred = pd.DataFrame(
        {
            patient_col: val[patient_col].to_numpy(),
            "fold": fold,
            "disease": label_col,
            "method": method,
            "y_true": val[label_col].astype(int).to_numpy(),
            "y_prob": np.clip(prob, 0.0, 1.0),
        }
    )
    return FoldPrediction(fold, label_col, method, pred)

