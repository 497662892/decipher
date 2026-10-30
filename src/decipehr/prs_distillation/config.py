"""Strict, deliberately small executable configuration for the v1 specification."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


@dataclass
class Optimization:
    max_epochs: int = 2000
    evaluate_every: int = 10
    patience: int = 200
    min_delta: float = 1e-6
    learning_rate: float = 0.01
    retry_learning_rate: float = 0.003
    gradient_clip: float = 10.0
    seeds: list[int] = field(default_factory=lambda: [17, 29, 43])
    lambdas: list[float] = field(default_factory=lambda: [0, 0.25, 0.5, 0.75, 1])
    gate_ridges: list[float] = field(default_factory=lambda: [1e-4, 1e-3, 1e-2])
    baseline_ridge: float = 1e-4
    scale_ridge: float = 1e-3
    tie_tolerance: float = 1e-4


@dataclass
class Config:
    schema_version: int = 1
    output_root: str = "outputs/prs_distillation_v1"
    run_name: str = "phase_a_20261026"
    traits: list[str] = field(default_factory=lambda: ["T2D", "cad", "bc"])
    # These departures from the supplied age/cohort contract require an explicit choice.
    cohort_policy: str = "original_strict"
    allow_standardized_age: bool = False
    train_path: str = "data/processed/phenotype.csv"
    evaluation_path: str = "data/processed/official_validation_phenotype.csv"
    panel_selection: str = "outputs/external_pgs/holdout_wrapper_20261012/SELECTION.json"
    ancestry_path: str = "outputs/final_ancestry_1000g/combined/target_1000G_hard_labels_pc10_margin0.5.train_val.tsv"
    qc_prepared: str = "outputs/internal_prs/nyu_20261011/prepared"
    checkpoint: str = "outputs/tabpfn-model-cache/tabpfn-v3-classifier-v3_default.ckpt"
    teacher_package: str = "8.0.1"
    teacher_estimators: int = 16
    teacher_device: str = "cuda"
    teacher_preprocessing: str = "student_v1"
    split_seed: int = 20261026
    teacher_seed: int = 20261026
    bootstrap_seed: int = 20261027
    bootstrap_replicates: int = 1000
    optimization: Optimization = field(default_factory=Optimization)

    def to_dict(self):
        return asdict(self)


def load_config(path: str | Path) -> Config:
    raw = yaml.safe_load(Path(path).read_text())
    if not isinstance(raw, dict):
        raise ValueError("Configuration must be a mapping")
    raw = dict(raw)
    try:
        opt = Optimization(**raw.pop("optimization", {}))
        cfg = Config(**raw, optimization=opt)
    except TypeError as exc:
        raise ValueError(f"Unknown or invalid configuration field: {exc}") from exc
    if cfg.schema_version != 1 or cfg.cohort_policy not in {
        "original_strict", "audited_qc", "original_full_sensitivity",
    }:
        raise ValueError("Unsupported schema_version or cohort_policy")
    if cfg.teacher_preprocessing not in {"student_v1", "historical_external"}:
        raise ValueError("Unsupported teacher_preprocessing")
    if type(cfg.allow_standardized_age) is not bool:
        raise ValueError("allow_standardized_age must be a YAML boolean")
    if not cfg.traits or len(set(cfg.traits)) != len(cfg.traits) or set(cfg.traits) - {"T2D", "cad", "bc"}:
        raise ValueError("Expected a unique subset of T2D, cad, bc")
    if cfg.teacher_estimators != 16 or cfg.teacher_package != "8.0.1":
        raise ValueError("The verified local external-PGS teacher is tabpfn 8.0.1 / V3 / 16 estimators")
    if opt.seeds != [17, 29, 43] or opt.lambdas != [0, 0.25, 0.5, 0.75, 1]:
        raise ValueError("The Phase A seeds and lambda grid are fixed by the v1 protocol")
    if opt.gate_ridges != [1e-4, 1e-3, 1e-2]:
        raise ValueError("The Phase A gate ridge grid is fixed by the v1 protocol")
    if min(opt.max_epochs, opt.evaluate_every, opt.patience, cfg.bootstrap_replicates) < 1:
        raise ValueError("Epoch and bootstrap counts must be positive")
    if min(opt.learning_rate, opt.retry_learning_rate, opt.gradient_clip) <= 0:
        raise ValueError("Learning rates and gradient_clip must be positive")
    if min(opt.baseline_ridge, opt.scale_ridge, opt.min_delta, opt.tie_tolerance) < 0:
        raise ValueError("Penalties and tolerances must be nonnegative")
    return cfg
