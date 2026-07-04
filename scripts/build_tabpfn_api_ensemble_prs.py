#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from decipehr.config import load_config
from decipehr.eligibility import filter_eligible, load_eligibility_rules
from decipehr.ensemble import pivot_long_prs_scores
from decipehr.io import read_table, write_table
from decipehr.schema import parse_csv_list, require_columns, validate_unique
from decipehr.tabpfn_ensemble import (
    TabPFNFeatureTransformer,
    TabPFNStrategy,
    logit_from_probability,
    scale_validation_logits,
    stratified_oof_splits,
)

DEFAULT_CONFIG = "configs/default.yaml"
DEFAULT_ELIGIBILITY_CONFIG = "configs/disease_eligibility.yaml"
DEFAULT_TRAIN_PHENOTYPE = (
    "data/raw/training_phase/phenotype_files/"
    "Training_Cov_file.txt"
)
DEFAULT_VALIDATION_COVARIATES = (
    "data/raw/training_phase/phenotype_files/"
    "Testing_Cov_file.txt"
)
DEFAULT_BASE_URL = "https://api.priorlabs.ai"


def _require_mapping(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key, {})
    if not isinstance(value, dict):
        raise ValueError(f"Config key {key!r} must be a mapping")
    return value


def _required_value(mapping: dict[str, Any], key: str, context: str) -> str:
    value = mapping.get(key)
    if value in (None, ""):
        raise ValueError(f"Missing required config value {context}.{key}")
    return str(value)


def _csv_from_config(value: object, context: str) -> str:
    if isinstance(value, str):
        parsed = parse_csv_list(value)
    elif isinstance(value, list):
        parsed = [str(item) for item in value]
    else:
        raise ValueError(f"Config value {context} must be a string or list")
    if not parsed:
        raise ValueError(f"Config value {context} must not be empty")
    return ",".join(parsed)


def _labels(args: argparse.Namespace, columns: dict[str, Any]) -> list[str]:
    if args.labels and args.disease:
        raise ValueError("Use either --labels or --disease, not both")
    if args.disease:
        return [args.disease]
    if args.labels:
        return parse_csv_list(args.labels)
    return parse_csv_list(_csv_from_config(columns.get("labels"), "columns.labels"))


def _strategies(value: str) -> list[TabPFNStrategy]:
    allowed = {"prs_only", "covariate_adjusted"}
    parsed = []
    for item in parse_csv_list(value):
        if item not in allowed:
            raise ValueError(f"Unsupported TabPFN strategy {item!r}; expected one of {sorted(allowed)}")
        parsed.append(item)  # type: ignore[arg-type]
    if not parsed:
        raise ValueError("At least one TabPFN strategy is required")
    return parsed


def _load_httpx():
    try:
        import httpx
    except ImportError as exc:
        raise SystemExit(
            "The 'httpx' package is required for Prior Labs REST API calls. "
            "Install it in the active environment or run with the project uv environment."
        ) from exc
    return httpx


def _csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False).encode("utf-8")


def _positive_probabilities_from_api_prediction(prediction: Any) -> np.ndarray:
    proba = np.asarray(prediction, dtype=float)
    if proba.ndim == 1:
        return proba
    if proba.ndim != 2 or proba.shape[1] < 2:
        raise ValueError(f"Unexpected classification probability shape from Prior Labs API: {proba.shape}")
    return proba[:, 1]


class PriorLabsTabPFNClient:
    def __init__(
        self,
        *,
        token: str,
        base_url: str = DEFAULT_BASE_URL,
        timeout_s: float = 900.0,
        conflict_retries: int = 5,
        conflict_sleep_s: float = 60.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.headers = {"Authorization": f"Bearer {token}"}
        self.conflict_retries = conflict_retries
        self.conflict_sleep_s = conflict_sleep_s
        httpx = _load_httpx()
        self.client = httpx.Client(timeout=timeout_s)

    def close(self) -> None:
        self.client.close()

    def _post_json(self, path: str, payload: dict[str, Any]):
        for attempt in range(self.conflict_retries + 1):
            response = self.client.post(f"{self.base_url}{path}", headers=self.headers, json=payload)
            if path == "/tabpfn/fit" and response.status_code == 409 and attempt < self.conflict_retries:
                time.sleep(self.conflict_sleep_s * (attempt + 1))
                continue
            break
        try:
            response.raise_for_status()
        except Exception as exc:
            detail = response.text.strip()
            if len(detail) > 2000:
                detail = detail[:2000] + "...[truncated]"
            raise RuntimeError(
                f"Prior Labs API request to {path} failed with HTTP {response.status_code}: {detail}"
            ) from exc
        return response.json()

    def prepare_train_upload(self) -> dict[str, Any]:
        return self._post_json(
            "/tabpfn/prepare_train_set_upload",
            {"x_train_info": {"format": "csv"}, "y_train_info": {"format": "csv"}},
        )

    def prepare_test_upload(self, fitted_train_set_id: str) -> dict[str, Any]:
        return self._post_json(
            "/tabpfn/prepare_test_set_upload",
            {"fitted_train_set_id": fitted_train_set_id, "x_test_info": {"format": "csv"}},
        )

    def upload_bytes(self, info: dict[str, Any], payload: bytes) -> None:
        urls = info.get("signed_urls")
        if not urls:
            raise ValueError("Prior Labs upload response did not include signed_urls")
        headers = info.get("required_headers") or {}
        response = self.client.put(urls[0], content=payload, headers=headers)
        response.raise_for_status()

    def fit(
        self,
        *,
        train_set_upload_id: str,
        tabpfn_config: dict[str, Any] | None,
        tabpfn_systems: list[str],
        thinking_effort: str,
        thinking_timeout_s: float | None,
        thinking_metric: str | None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "train_set_upload_id": train_set_upload_id,
            "task": "classification",
            "tabpfn_systems": tabpfn_systems,
            "thinking_effort": thinking_effort,
        }
        if tabpfn_config:
            payload["tabpfn_config"] = tabpfn_config
        if thinking_timeout_s is not None:
            payload["thinking_timeout_s"] = thinking_timeout_s
        if thinking_metric:
            payload["thinking_effort_metric"] = thinking_metric
        return self._post_json("/tabpfn/fit", payload)

    def predict(self, *, test_set_upload_id: str, fitted_train_set_id: str) -> dict[str, Any]:
        return self._post_json(
            "/tabpfn/predict",
            {
                "test_set_upload_id": test_set_upload_id,
                "fitted_train_set_id": fitted_train_set_id,
                "task_config": {
                    "task": "classification",
                    "predict_params": {"output_type": "probas"},
                },
            },
        )


def _api_fit_predict(
    *,
    api: PriorLabsTabPFNClient,
    context: pd.DataFrame,
    y_context: pd.Series,
    eval_frame: pd.DataFrame,
    prs_cols: list[str],
    covariates: list[str],
    include_covariates: bool,
    thinking_effort: str,
    thinking_timeout_s: float | None,
    thinking_metric: str | None,
    tabpfn_config: dict[str, Any] | None,
    tabpfn_systems: list[str],
    disease: str,
    strategy: str,
    fit_scope: str,
) -> tuple[np.ndarray, pd.DataFrame, dict[str, Any]]:
    transformer = TabPFNFeatureTransformer.fit(
        context,
        prs_cols=prs_cols,
        covariates=covariates,
        include_covariates=include_covariates,
    )
    x_context = transformer.transform(context)
    x_eval = transformer.transform(eval_frame)
    y_payload = pd.DataFrame({"target": y_context.astype(int).to_numpy()})

    prep = api.prepare_train_upload()
    api.upload_bytes(prep["x_train_info"], _csv_bytes(x_context))
    api.upload_bytes(prep["y_train_info"], _csv_bytes(y_payload))
    fit_response = api.fit(
        train_set_upload_id=prep["train_set_upload_id"],
        tabpfn_config=tabpfn_config,
        tabpfn_systems=tabpfn_systems,
        thinking_effort=thinking_effort,
        thinking_timeout_s=thinking_timeout_s,
        thinking_metric=thinking_metric,
    )
    fitted_train_set_id = fit_response["fitted_train_set_id"]

    test_prep = api.prepare_test_upload(fitted_train_set_id)
    api.upload_bytes(test_prep["x_test_info"], _csv_bytes(x_eval))
    predict_response = api.predict(
        test_set_upload_id=test_prep["test_set_upload_id"],
        fitted_train_set_id=fitted_train_set_id,
    )
    probs = _positive_probabilities_from_api_prediction(predict_response["prediction"])
    if len(probs) != len(eval_frame):
        raise ValueError(
            f"Prior Labs API returned {len(probs)} predictions for {len(eval_frame)} eval rows"
        )

    api_summary = {
        "disease": disease,
        "strategy": strategy,
        "fit_scope": fit_scope,
        "n_context": int(len(context)),
        "n_eval": int(len(eval_frame)),
        "n_features": int(x_context.shape[1]),
        "tabpfn_config": tabpfn_config or {},
        "tabpfn_systems": tabpfn_systems,
        "train_set_upload_id": prep.get("train_set_upload_id"),
        "test_set_upload_id": test_prep.get("test_set_upload_id"),
        "fitted_train_set_id": fitted_train_set_id,
        "fit_response": fit_response,
        "predict_metadata": predict_response.get("metadata", {}),
    }
    return probs, transformer.summary(), api_summary


def _oof_probabilities(
    *,
    api: PriorLabsTabPFNClient,
    context: pd.DataFrame,
    y_context: pd.Series,
    prs_cols: list[str],
    covariates: list[str],
    include_covariates: bool,
    n_folds: int,
    seed: int,
    thinking_effort: str,
    thinking_timeout_s: float | None,
    thinking_metric: str | None,
    tabpfn_config: dict[str, Any] | None,
    tabpfn_systems: list[str],
    disease: str,
    strategy: str,
) -> tuple[np.ndarray, pd.DataFrame, list[dict[str, Any]], int]:
    probs = np.full(len(context), np.nan, dtype=float)
    feature_summaries = []
    api_summaries = []
    splits = stratified_oof_splits(y_context, requested_splits=n_folds, seed=seed)
    for fold, (fit_idx, holdout_idx) in enumerate(splits):
        fold_probs, feature_summary, api_summary = _api_fit_predict(
            api=api,
            context=context.iloc[fit_idx],
            y_context=y_context.iloc[fit_idx],
            eval_frame=context.iloc[holdout_idx],
            prs_cols=prs_cols,
            covariates=covariates,
            include_covariates=include_covariates,
            thinking_effort=thinking_effort,
            thinking_timeout_s=thinking_timeout_s,
            thinking_metric=thinking_metric,
            tabpfn_config=tabpfn_config,
            tabpfn_systems=tabpfn_systems,
            disease=disease,
            strategy=strategy,
            fit_scope=f"train_oof_fold_{fold}",
        )
        probs[holdout_idx] = fold_probs
        feature_summaries.append(feature_summary.assign(oof_fold=fold))
        api_summaries.append(api_summary)
    if np.isnan(probs).any():
        raise ValueError("TabPFN API OOF prediction did not produce scores for all context rows")
    return probs, pd.concat(feature_summaries, ignore_index=True), api_summaries, len(splits)


def _build_strategy_scores(
    *,
    api: PriorLabsTabPFNClient,
    disease: str,
    strategy: TabPFNStrategy,
    train_base: pd.DataFrame,
    validation_base: pd.DataFrame,
    prs_cols: list[str],
    covariates: list[str],
    eligibility_rules: dict[str, dict[str, Any]],
    n_folds: int,
    seed: int,
    thinking_effort: str,
    thinking_timeout_s: float | None,
    thinking_metric: str | None,
    tabpfn_config: dict[str, Any] | None,
    tabpfn_systems: list[str],
    train_prediction_mode: str,
    clip_eps: float,
    score_scale: str,
) -> tuple[pd.Series, pd.Series, pd.DataFrame, dict[str, Any], list[dict[str, Any]]]:
    include_covariates = strategy == "covariate_adjusted"
    context = filter_eligible(train_base, disease, eligibility_rules).copy()
    validation_eval = filter_eligible(validation_base, disease, eligibility_rules).copy()
    if context.empty:
        raise ValueError(f"No eligible TabPFN API context rows remain for {disease}")
    if validation_eval.empty:
        raise ValueError(f"No eligible TabPFN API validation rows remain for {disease}")

    y_context = context[disease].astype(int)
    if y_context.nunique() < 2:
        raise ValueError(f"TabPFN API context labels for {disease} have fewer than two classes")

    if train_prediction_mode == "oof":
        train_probs, oof_features, api_summaries, actual_folds = _oof_probabilities(
            api=api,
            context=context,
            y_context=y_context,
            prs_cols=prs_cols,
            covariates=covariates,
            include_covariates=include_covariates,
            n_folds=n_folds,
            seed=seed,
            thinking_effort=thinking_effort,
            thinking_timeout_s=thinking_timeout_s,
            thinking_metric=thinking_metric,
            tabpfn_config=tabpfn_config,
            tabpfn_systems=tabpfn_systems,
            disease=disease,
            strategy=strategy,
        )
    elif train_prediction_mode == "in_context":
        train_probs, feature_summary, api_summary = _api_fit_predict(
            api=api,
            context=context,
            y_context=y_context,
            eval_frame=context,
            prs_cols=prs_cols,
            covariates=covariates,
            include_covariates=include_covariates,
            thinking_effort=thinking_effort,
            thinking_timeout_s=thinking_timeout_s,
            thinking_metric=thinking_metric,
            tabpfn_config=tabpfn_config,
            tabpfn_systems=tabpfn_systems,
            disease=disease,
            strategy=strategy,
            fit_scope="train_in_context",
        )
        oof_features = feature_summary.assign(oof_fold=-1)
        api_summaries = [api_summary]
        actual_folds = 1
    else:
        raise ValueError(f"Unsupported train prediction mode: {train_prediction_mode}")

    validation_probs, validation_features, validation_api_summary = _api_fit_predict(
        api=api,
        context=context,
        y_context=y_context,
        eval_frame=validation_eval,
        prs_cols=prs_cols,
        covariates=covariates,
        include_covariates=include_covariates,
        thinking_effort=thinking_effort,
        thinking_timeout_s=thinking_timeout_s,
        thinking_metric=thinking_metric,
        tabpfn_config=tabpfn_config,
        tabpfn_systems=tabpfn_systems,
        disease=disease,
        strategy=strategy,
        fit_scope="validation_full_context",
    )
    api_summaries.append(validation_api_summary)

    train_logits = logit_from_probability(train_probs, eps=clip_eps)
    validation_logits = logit_from_probability(validation_probs, eps=clip_eps)
    train_scaled, validation_scaled, logit_mean, logit_sd = scale_validation_logits(
        train_logits=train_logits,
        validation_logits=validation_logits,
        mode=score_scale,
    )

    train_out = pd.Series(np.nan, index=train_base.index, dtype=float)
    validation_out = pd.Series(np.nan, index=validation_base.index, dtype=float)
    train_out.loc[context.index] = train_scaled
    validation_out.loc[validation_eval.index] = validation_scaled

    feature_summary = pd.concat(
        [
            oof_features.assign(feature_fit_scope="train_oof"),
            validation_features.assign(oof_fold=-1, feature_fit_scope="validation_full_context"),
        ],
        ignore_index=True,
    )
    feature_summary.insert(0, "strategy", strategy)
    feature_summary.insert(0, "disease", disease)

    summary = {
        "disease": disease,
        "strategy": strategy,
        "n_context": int(len(context)),
        "context_cases": int(y_context.sum()),
        "n_validation_scored": int(len(validation_eval)),
        "n_prs_features": int(len(prs_cols)),
        "n_covariates_requested": int(len(covariates) if include_covariates else 0),
        "train_prediction_mode": train_prediction_mode,
        "n_oof_folds": int(actual_folds),
        "n_api_fits": int(len(api_summaries)),
        "probability_min": float(np.nanmin(train_probs)),
        "probability_max": float(np.nanmax(train_probs)),
        "validation_probability_min": float(np.nanmin(validation_probs)),
        "validation_probability_max": float(np.nanmax(validation_probs)),
        "clip_eps": float(clip_eps),
        "score_scale": score_scale,
        "thinking_effort": thinking_effort,
        "thinking_timeout_s": thinking_timeout_s if thinking_timeout_s is not None else math.nan,
        "thinking_metric": thinking_metric or "",
        "tabpfn_config": json.dumps(tabpfn_config or {}, sort_keys=True),
        "tabpfn_systems": ",".join(tabpfn_systems),
        "logit_train_mean": float(logit_mean),
        "logit_train_sd": float(logit_sd),
    }
    return train_out, validation_out, feature_summary, summary, api_summaries


def _read_long_scores(path_template: str, disease: str, cache: dict[str, pd.DataFrame]) -> pd.DataFrame:
    path = path_template.format(disease=disease)
    if path not in cache:
        cache[path] = read_table(path)
    return cache[path]


def _reindex_scores(scores: pd.DataFrame, patient_order: pd.DataFrame, patient_col: str) -> pd.DataFrame:
    return patient_order.merge(scores, on=patient_col, how="left", validate="one_to_one")


def _write_validation_commands(
    *,
    out_dir: Path,
    strategies: list[TabPFNStrategy],
    args: argparse.Namespace,
    labels_arg: str,
    covariates_arg: str,
    patient_col: str,
) -> None:
    commands = ["#!/usr/bin/env bash", "set -euo pipefail", ""]
    for strategy in strategies:
        strategy_dir = out_dir / strategy
        model_name = f"{args.model_name}_{strategy}" if args.model_name else f"tabpfn_api_{strategy}"
        command = [
            "uv",
            "run",
            "python",
            "scripts/validate_official_model.py",
            "--config",
            args.config,
            "--model-name",
            model_name,
            "--patient-col",
            patient_col,
            "--labels",
            labels_arg,
            "--covariates",
            covariates_arg,
            "--train-prs-scores",
            str(strategy_dir / "train_prs_scores.csv"),
            "--validation-prs-scores",
            str(strategy_dir / "validation_prs_scores.csv"),
            "--eligibility-config",
            args.eligibility_config,
            "--out-dir",
            str(out_dir / f"official_validation_{strategy}"),
        ]
        if args.config_override:
            command.extend(["--config-override", args.config_override])
        commands.append(shlex.join(command))
    (out_dir / "validate_tabpfn_api_ensemble_commands.sh").write_text(
        "\n\n".join(commands) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Use Prior Labs TabPFN API thinking mode as a nonlinear PRS ensemble/meta-model. "
            "Feature preprocessing and output scaling mirror the local TabPFN ensemble script."
        )
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--config-override", default=None)
    parser.add_argument("--model-name", default="tabpfn_api_thinking")
    parser.add_argument("--disease", default=None)
    parser.add_argument("--labels", default=None)
    parser.add_argument("--covariates", default=None)
    parser.add_argument("--patient-col", default=None)
    parser.add_argument("--train-phenotype", default=None)
    parser.add_argument("--validation-covariates", default=None)
    parser.add_argument("--eligibility-config", default=DEFAULT_ELIGIBILITY_CONFIG)
    parser.add_argument("--train-prs-long-scores", required=True)
    parser.add_argument("--validation-prs-long-scores", required=True)
    parser.add_argument("--disease-col", default="disease")
    parser.add_argument("--model-col", default="model")
    parser.add_argument("--score-col", default="prs")
    parser.add_argument("--strategies", default="covariate_adjusted")
    parser.add_argument("--n-folds", type=int, default=None)
    parser.add_argument("--train-prediction-mode", choices=["oof", "in_context"], default="oof")
    parser.add_argument("--clip-eps", type=float, default=1e-5)
    parser.add_argument("--score-scale", choices=["train_sd", "none"], default="train_sd")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--api-token-env", default="PRIORLABS_API_KEY")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--http-timeout-s", type=float, default=1800.0)
    parser.add_argument("--api-conflict-retries", type=int, default=5)
    parser.add_argument("--api-conflict-sleep-s", type=float, default=60.0)
    parser.add_argument(
        "--model-version",
        default=None,
        help=(
            "Deprecated guard only. The current Prior Labs JSON API does not accept "
            "model_version in tabpfn_config; thinking mode selects the TabPFN-3-Plus path."
        ),
    )
    parser.add_argument("--tabpfn-systems", default="preprocessing,thinking")
    parser.add_argument("--thinking-effort", choices=["medium", "high"], default="high")
    parser.add_argument("--thinking-timeout-s", type=float, default=300.0)
    parser.add_argument("--thinking-metric", default="roc_auc")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--no-validation-commands", action="store_true")
    args = parser.parse_args()

    if args.clip_eps <= 0 or args.clip_eps >= 0.5:
        raise ValueError("--clip-eps must be between 0 and 0.5")
    token = os.environ.get(args.api_token_env) or os.environ.get("TABPFN_TOKEN")
    if not token:
        raise SystemExit(
            f"Missing Prior Labs API token. Set {args.api_token_env} or TABPFN_TOKEN in the environment."
        )

    config = load_config(args.config, args.config_override)
    paths = _require_mapping(config, "paths")
    columns = _require_mapping(config, "columns")
    labels = _labels(args, columns)
    labels_arg = ",".join(labels)
    covariates = parse_csv_list(args.covariates or _csv_from_config(columns.get("covariates"), "columns.covariates"))
    covariates_arg = ",".join(covariates)
    patient_col = args.patient_col or _required_value(columns, "patient_id", "columns")
    seed = args.seed if args.seed is not None else int(config.get("project", {}).get("seed", 42))
    n_folds = args.n_folds if args.n_folds is not None else int(config.get("project", {}).get("n_folds", 5))
    strategies = _strategies(args.strategies)
    tabpfn_systems = parse_csv_list(args.tabpfn_systems)
    if "thinking" not in tabpfn_systems:
        raise ValueError("--tabpfn-systems must include 'thinking' for this API thinking runner")
    if args.model_version:
        raise ValueError(
            "--model-version is not supported by the current Prior Labs JSON API. "
            "Use --tabpfn-systems preprocessing,thinking to request TabPFN-3-Plus thinking mode."
        )
    tabpfn_config = None

    train_phenotype_path = args.train_phenotype or str(
        paths.get("official_train_phenotype", DEFAULT_TRAIN_PHENOTYPE)
    )
    validation_covariates_path = args.validation_covariates or str(
        paths.get("official_validation_covariates", DEFAULT_VALIDATION_COVARIATES)
    )
    train_phenotype = read_table(train_phenotype_path)
    validation_covariates = read_table(validation_covariates_path)
    require_columns(train_phenotype, [patient_col, *labels, *covariates], "training phenotype")
    require_columns(validation_covariates, [patient_col, *covariates], "validation covariates")
    validate_unique(train_phenotype, patient_col, "training phenotype")
    validate_unique(validation_covariates, patient_col, "validation covariates")

    eligibility_rules = load_eligibility_rules(args.eligibility_config)
    long_cache: dict[str, pd.DataFrame] = {}
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_feature_summaries = []
    all_fit_summaries = []
    all_api_summaries = []
    strategy_train_scores = {strategy: train_phenotype[[patient_col]].copy() for strategy in strategies}
    strategy_validation_scores = {
        strategy: validation_covariates[[patient_col]].copy()
        for strategy in strategies
    }

    api = PriorLabsTabPFNClient(
        token=token,
        base_url=args.base_url,
        timeout_s=args.http_timeout_s,
        conflict_retries=args.api_conflict_retries,
        conflict_sleep_s=args.api_conflict_sleep_s,
    )
    try:
        for disease in labels:
            train_long = _read_long_scores(args.train_prs_long_scores, disease, long_cache)
            validation_long = _read_long_scores(args.validation_prs_long_scores, disease, long_cache)
            train_prs = pivot_long_prs_scores(
                train_long,
                disease,
                patient_col=patient_col,
                disease_col=args.disease_col,
                model_col=args.model_col,
                score_col=args.score_col,
                context="training long PRS scores",
            )
            validation_prs = pivot_long_prs_scores(
                validation_long,
                disease,
                patient_col=patient_col,
                disease_col=args.disease_col,
                model_col=args.model_col,
                score_col=args.score_col,
                context="validation long PRS scores",
            )
            prs_cols = [col for col in train_prs.columns if col != patient_col]
            missing_validation_cols = [col for col in prs_cols if col not in validation_prs.columns]
            if missing_validation_cols:
                raise ValueError(f"Validation PRS scores are missing columns: {missing_validation_cols}")

            train_base = train_phenotype.merge(train_prs, on=patient_col, how="inner", validate="one_to_one")
            validation_base = validation_covariates.merge(
                validation_prs,
                on=patient_col,
                how="inner",
                validate="one_to_one",
            )
            if train_base.empty or validation_base.empty:
                raise ValueError(f"No TabPFN API ensemble rows available for {disease}")

            for strategy in strategies:
                train_score, validation_score, feature_summary, fit_summary, api_summaries = _build_strategy_scores(
                    api=api,
                    disease=disease,
                    strategy=strategy,
                    train_base=train_base,
                    validation_base=validation_base,
                    prs_cols=prs_cols,
                    covariates=covariates,
                    eligibility_rules=eligibility_rules,
                    n_folds=n_folds,
                    seed=seed,
                    thinking_effort=args.thinking_effort,
                    thinking_timeout_s=args.thinking_timeout_s,
                    thinking_metric=args.thinking_metric,
                    tabpfn_config=tabpfn_config,
                    tabpfn_systems=tabpfn_systems,
                    train_prediction_mode=args.train_prediction_mode,
                    clip_eps=args.clip_eps,
                    score_scale=args.score_scale,
                )
                train_score_by_patient = pd.DataFrame(
                    {
                        patient_col: train_base[patient_col].to_numpy(),
                        f"prs_{disease}": train_score.to_numpy(),
                    }
                )
                validation_score_by_patient = pd.DataFrame(
                    {
                        patient_col: validation_base[patient_col].to_numpy(),
                        f"prs_{disease}": validation_score.to_numpy(),
                    }
                )
                strategy_train_scores[strategy] = strategy_train_scores[strategy].merge(
                    train_score_by_patient,
                    on=patient_col,
                    how="left",
                    validate="one_to_one",
                )
                strategy_validation_scores[strategy] = strategy_validation_scores[strategy].merge(
                    validation_score_by_patient,
                    on=patient_col,
                    how="left",
                    validate="one_to_one",
                )
                all_feature_summaries.append(feature_summary)
                all_fit_summaries.append(fit_summary)
                all_api_summaries.extend(api_summaries)
    finally:
        api.close()

    for strategy in strategies:
        strategy_dir = out_dir / strategy
        strategy_dir.mkdir(parents=True, exist_ok=True)
        write_table(
            _reindex_scores(strategy_train_scores[strategy], train_phenotype[[patient_col]], patient_col),
            strategy_dir / "train_prs_scores.csv",
        )
        write_table(
            _reindex_scores(strategy_validation_scores[strategy], validation_covariates[[patient_col]], patient_col),
            strategy_dir / "validation_prs_scores.csv",
        )

    if all_feature_summaries:
        write_table(pd.concat(all_feature_summaries, ignore_index=True), out_dir / "tabpfn_api_feature_summary.csv")
    if all_fit_summaries:
        write_table(pd.DataFrame(all_fit_summaries), out_dir / "tabpfn_api_fit_summary.csv")
    if all_api_summaries:
        (out_dir / "tabpfn_api_call_summary.json").write_text(
            json.dumps(all_api_summaries, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    if not args.no_validation_commands:
        _write_validation_commands(
            out_dir=out_dir,
            strategies=strategies,
            args=args,
            labels_arg=labels_arg,
            covariates_arg=covariates_arg,
            patient_col=patient_col,
        )

    print(f"Wrote Prior Labs TabPFN API ensemble PRS outputs to {out_dir}")


if __name__ == "__main__":
    main()
