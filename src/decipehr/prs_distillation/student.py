"""The single specified formula, deletion ablations, and full-batch optimization."""

from __future__ import annotations

import copy
import math
from pathlib import Path

import joblib
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from decipehr.external_pgs import sha256_file, write_json
from decipehr.prs_distillation.config import Optimization


class Student(nn.Module):
    def __init__(self, k, n_ancestry, n_basis, n_baseline, *, variant="full", seed=17,
                 target_mean=0.5):
        super().__init__()
        if k < 1 or variant not in {"full", "metadata", "static_gate", "constant_scale"}:
            raise ValueError("Invalid student dimensions or variant")
        self.dimensions = dict(k=k, n_ancestry=n_ancestry, n_basis=n_basis, n_baseline=n_baseline)
        self.variant = variant
        self.baseline = nn.Parameter(torch.zeros(n_baseline, dtype=torch.float64))
        with torch.no_grad():
            self.baseline[0] = math.log(np.clip(target_mean, 1e-6, 1 - 1e-6) /
                                        (1 - np.clip(target_mean, 1e-6, 1 - 1e-6)))
        if variant == "metadata":
            return
        self.u0 = nn.Parameter(torch.tensor(math.log(0.25), dtype=torch.float64))
        if variant != "constant_scale":
            self.theta_s = nn.Parameter(torch.zeros(n_basis, dtype=torch.float64))
        if k > 1:
            self.c = nn.Parameter(torch.zeros(k, dtype=torch.float64))
            if variant != "static_gate":
                generator = torch.Generator().manual_seed(seed)
                self.w = nn.Parameter(torch.randn(k, n_ancestry, dtype=torch.float64,
                                                 generator=generator) * 0.01)

    def centered_gate(self):
        w = self.w - self.w.mean(dim=0) if hasattr(self, "w") else None
        c = self.c - self.c.mean() if hasattr(self, "c") else None
        return w, c

    def forward(self, data):
        b = data["baseline"] @ self.baseline
        z, a, phi = data["z"], data["a"], data["phi"]
        if self.variant == "metadata":
            return {"logit": b, "probability": b.sigmoid(), "baseline_logit": b}
        w, c = self.centered_gate()
        if c is None:
            pi = torch.ones_like(z)
        else:
            g = c.expand(len(z), -1) if w is None else a @ w.T + c
            pi = torch.softmax(g, dim=1)
        log_scale = self.u0.expand(len(z))
        if hasattr(self, "theta_s"):
            log_scale = log_scale + phi @ self.theta_s
        scale = log_scale.exp()
        beta = scale[:, None] * pi
        contribution = beta * z
        eta = b + contribution.sum(dim=1)
        return {"logit": eta, "probability": eta.sigmoid(), "baseline_logit": b,
                "genetic_score": (pi * z).sum(dim=1), "age_scale": scale,
                "weights": pi, "effective_coefficients": beta, "contributions": contribution}

    def penalty(self, gate_ridge, options):
        penalty = options.baseline_ridge * self.baseline[1:].square().sum()
        w, c = self.centered_gate()
        for parameter in (w, c):
            if parameter is not None:
                penalty = penalty + gate_ridge * parameter.square().sum()
        if hasattr(self, "theta_s"):
            penalty = penalty + options.scale_ridge * self.theta_s.square().sum()
        return penalty

    def canonical_state(self):
        state = {k: v.detach().clone() for k, v in self.state_dict().items()}
        for key in ("w", "c"):
            if key in state:
                state[key] -= state[key].mean(dim=0)
        return state


def tensors(data):
    return {key: torch.as_tensor(data[key], dtype=torch.float64)
            for key in ("z", "a", "phi", "baseline")}


def soft_targets(y, q, lam):
    if not 0 <= lam <= 1:
        raise ValueError("lambda must be in [0,1]")
    def checked(value):
        result = (value.detach().clone().to(torch.float64) if isinstance(value, torch.Tensor)
                  else torch.tensor(np.asarray(value), dtype=torch.float64))
        if result.ndim != 1 or not torch.isfinite(result).all() or ((result < 0) | (result > 1)).any():
            raise ValueError("Targets must be finite one-dimensional probabilities")
        return result
    if lam == 0:
        return checked(y)
    if q is None:
        raise ValueError("Distillation requires audited teacher probabilities")
    if lam == 1:
        return checked(q)
    y, q = checked(y), checked(q)
    if y.shape != q.shape:
        raise ValueError("Label and teacher-target lengths differ")
    return (1 - lam) * y + lam * q


def predict(model, data):
    with torch.no_grad():
        result = {k: v.numpy() for k, v in model(tensors(data)).items()}
    if not all(np.isfinite(v).all() for v in result.values()):
        raise FloatingPointError("Nonfinite student prediction")
    if "weights" in result:
        np.testing.assert_allclose(result["weights"].sum(axis=1), 1, atol=1e-12)
        np.testing.assert_allclose(result["effective_coefficients"].sum(axis=1), result["age_scale"])
        np.testing.assert_allclose(result["baseline_logit"] + result["contributions"].sum(axis=1),
                                   result["logit"], atol=1e-12)
        if (result["age_scale"] <= 0).any():
            raise FloatingPointError("Nonpositive age scale")
    return result


def fit_student(train, y, q, *, lam, gate_ridge, seed, options: Optimization,
                tune=None, tune_y=None, tune_q=None, variant="full", epochs=None):
    """Selection sees H only; a fixed-epoch refit has no query arguments."""
    target = soft_targets(y, q, lam)
    tx = tensors(train)
    vx = tensors(tune) if tune is not None else None
    vt = soft_targets(tune_y, tune_q, 1 if lam == 1 else 0) if vx is not None else None
    if epochs is not None and vx is not None:
        raise ValueError("Refit cannot receive tuning/evaluation data")
    failures = []
    for attempt, lr in enumerate([options.learning_rate, options.retry_learning_rate]):
        logs = []
        model = Student(tx["z"].shape[1], tx["a"].shape[1], tx["phi"].shape[1],
                        tx["baseline"].shape[1], variant=variant, seed=seed,
                        target_mean=float(target.mean()))
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0)
        best, best_epoch, state = float("inf"), 0, None
        try:
            for epoch in range(1, (epochs or options.max_epochs) + 1):
                optimizer.zero_grad()
                out = model(tx)
                loss = F.binary_cross_entropy_with_logits(out["logit"], target)
                total = loss + model.penalty(gate_ridge, options)
                if not torch.isfinite(total):
                    raise FloatingPointError(f"Nonfinite training loss at epoch {epoch}")
                total.backward()
                grad = torch.nn.utils.clip_grad_norm_(model.parameters(), options.gradient_clip,
                                                      error_if_nonfinite=True)
                optimizer.step()
                if epoch % options.evaluate_every == 0 or epoch == (epochs or options.max_epochs):
                    with torch.no_grad():
                        current = model(vx or tx)
                        metric = float(F.binary_cross_entropy_with_logits(
                            current["logit"], vt if vx is not None else target))
                        if not np.isfinite(metric) or any(not torch.isfinite(v).all()
                                                         for v in current.values()):
                            raise FloatingPointError(f"Nonfinite prediction at epoch {epoch}")
                        row = {"epoch": epoch, "loss": float(loss.detach()), "metric": metric,
                               "gradient_norm": float(grad), "learning_rate": lr, "attempt": attempt}
                        if "age_scale" in current:
                            row.update(scale_min=float(current["age_scale"].min()),
                                       scale_max=float(current["age_scale"].max()))
                        logs.append(row)
                        if vx is None or metric < best - options.min_delta:
                            best, best_epoch = metric, epoch
                            state = copy.deepcopy(model.canonical_state())
                    if vx is not None and epoch - best_epoch >= options.patience:
                        break
            if state is None:
                raise FloatingPointError("No valid checkpoint")
            model.load_state_dict(state)
            info = {"best_epoch": best_epoch, "selection_metric": best,
                    "metric_type": "teacher_cross_entropy" if lam == 1 else "true_label_log_loss",
                    "learning_rate": lr, "numerical_failures": failures, "logs": logs}
            if vx is not None:
                with torch.no_grad():
                    info["true_label_log_loss"] = float(F.binary_cross_entropy_with_logits(
                        model(vx)["logit"], soft_targets(tune_y, None, 0)))
            return model, info
        except (FloatingPointError, RuntimeError) as exc:
            if isinstance(exc, RuntimeError) and "non-finite" not in str(exc):
                raise
            failures.append({"attempt": attempt, "learning_rate": lr, "error": str(exc),
                             "last_logs": logs[-3:]})
    raise FloatingPointError(f"Both predeclared numerical attempts failed: {failures}")


class StudentBundle:
    def __init__(self, model, preprocessing, metadata):
        self.model, self.preprocessing, self.metadata = model, preprocessing, metadata

    def predict(self, patient_table, return_components=True):
        result = predict(self.model, self.preprocessing.transform(patient_table))
        return result if return_components else result["probability"]

    def save(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=False)
        torch.save(self.model.canonical_state(), path / "student_state.pt")
        joblib.dump(self.preprocessing, path / "preprocessing.joblib")
        write_json(path / "model_metadata.json", {
            **self.metadata, "dimensions": self.model.dimensions, "variant": self.model.variant,
            "preprocessing": self.preprocessing.metadata(),
        })
        write_json(path / "bundle_hashes.json", {p.name: sha256_file(p) for p in path.iterdir()})


def load_student_bundle(path):
    """Load trusted local artifacts only; inference does not import TabPFN or fit anything."""
    import json

    path = Path(path)
    hashes = json.loads((path / "bundle_hashes.json").read_text())
    for name, expected in hashes.items():
        if sha256_file(path / name) != expected:
            raise ValueError(f"Bundle checksum mismatch: {name}")
    meta = json.loads((path / "model_metadata.json").read_text())
    model = Student(**meta["dimensions"], variant=meta["variant"])
    model.load_state_dict(torch.load(path / "student_state.pt", map_location="cpu", weights_only=True))
    return StudentBundle(model, joblib.load(path / "preprocessing.joblib"), meta)
