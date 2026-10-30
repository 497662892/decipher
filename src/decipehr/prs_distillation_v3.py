"""Fixed-PRS sign-aware students; independent of the sealed v1 implementation."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import torch
from scipy.special import expit, softmax
from torch import nn
from torch.nn import functional as F

from decipehr.external_pgs import sha256_file, write_json
from decipehr.prs_distillation.data import verify_files
from decipehr.prs_distillation.preprocessing import AGE, Preprocessor
from decipehr.prs_distillation.student import soft_targets


@dataclass
class SignAwarePreprocessor:
    fixed: Preprocessor
    reference_age: float
    age_unit: float
    age_boundaries: np.ndarray

    @classmethod
    def fit(cls, frame, prs):
        fixed = Preprocessor.fit(frame, prs)
        ages = frame[AGE].to_numpy(dtype=float)
        return cls(fixed, float(np.median(ages)), 1.0,
                   np.unique(np.quantile(ages, [0, 1/3, 2/3, 1])))

    @property
    def n_segments(self):
        return len(self.age_boundaries) - 1

    def age_transform(self, age):
        age = np.asarray(age, dtype=float)
        used = np.clip(age, self.age_boundaries[0], self.age_boundaries[-1])
        t = (used - self.reference_age) / self.age_unit
        b = (self.age_boundaries - self.reference_age) / self.age_unit
        basis = np.clip(t[:, None] - b[:-1], 0, np.diff(b)) - np.clip(-b[:-1], 0, np.diff(b))
        return t, basis

    def transform(self, frame):
        original = self.fixed.transform(frame)
        t, basis = self.age_transform(frame[AGE])
        # Explicitly discard the old unconstrained age columns.
        c = original["baseline"][:, 1 + original["phi"].shape[1] + original["a"].shape[1]:]
        nonage = np.column_stack([np.ones(len(frame)), original["a"], c])
        return {"z": original["z"], "a": original["a"], "nonage": nonage, "t": t,
                "age_basis": basis, "age_out_of_support": original["age_out_of_support"],
                "sex_out_of_support": original["sex_out_of_support"]}

    def metadata(self):
        return {"fixed_preprocessing": self.fixed.metadata(), "reference_age": self.reference_age,
                "age_unit": self.age_unit, "age_units": "provided age_sd; NOT years or decades",
                "age_boundaries": self.age_boundaries.tolist(), "n_segments": self.n_segments,
                "constant_age": self.n_segments == 0, "age_extrapolation": "constant_in_both_branches",
                "nonage_columns": ["intercept", *[f"PC{i}_sd" for i in range(1, 11)],
                                   *([f"sex__{s}" for s in self.fixed.categories] if len(self.fixed.categories) > 1 else [])]}


def as_tensors(data):
    return {k: torch.as_tensor(data[k], dtype=torch.float64)
            for k in ["z", "a", "nonage", "t", "age_basis"]}


class SignAwareStudent(nn.Module):
    def __init__(self, k, n_ancestry, n_nonage, n_segments, *, variant="M0", seed=17, target_mean=.5):
        super().__init__()
        if k < 1 or variant not in {"M0", "M1", "M2"}:
            raise ValueError("Invalid fixed-PRS student dimensions or variant")
        self.dimensions = dict(k=k, n_ancestry=n_ancestry, n_nonage=n_nonage, n_segments=n_segments)
        self.variant = variant
        self.base = nn.Parameter(torch.zeros(n_nonage, dtype=torch.float64))
        with torch.no_grad():
            p = np.clip(target_mean, 1e-6, 1-1e-6)
            self.base[0] = math.log(p/(1-p))
        self.log_r = nn.Parameter(torch.zeros(k, dtype=torch.float64))
        if n_segments:
            self.age_slopes = nn.Parameter(torch.zeros(n_segments, dtype=torch.float64))
        else:
            self.register_buffer("age_slopes", torch.zeros(0, dtype=torch.float64))
        if variant == "M0" or not n_segments:
            self.register_buffer("d", torch.zeros(k, dtype=torch.float64))
        else:
            self.d = nn.Parameter(torch.zeros(1 if variant == "M1" else k, dtype=torch.float64))
        if k > 1:
            generator = torch.Generator().manual_seed(seed)
            self.w = nn.Parameter(torch.randn(k, n_ancestry, generator=generator, dtype=torch.float64)*.01)
            self.c = nn.Parameter(torch.zeros(k, dtype=torch.float64))

    def centered_gate(self):
        return ((self.w-self.w.mean(dim=0), self.c-self.c.mean()) if hasattr(self, "w") else (None, None))

    def effective_d(self):
        return self.d.expand(self.dimensions["k"])

    def forward(self, x):
        z = x["z"]
        w, c = self.centered_gate()
        pi = torch.ones_like(z) if w is None else torch.softmax(x["a"] @ w.T + c, dim=1)
        r, d = self.log_r.exp(), self.effective_d()
        dt = x["t"][:, None] * d
        multiplier = torch.exp(dt * z.sign())
        transformed = z * multiplier
        beta = pi * r * multiplier
        reference_contribution = pi * r * z
        contribution = beta * z
        base_reference = x["nonage"] @ self.base
        baseline_age = x["age_basis"] @ self.age_slopes
        reference_logit = base_reference + reference_contribution.sum(dim=1)
        age_delta = baseline_age + (contribution-reference_contribution).sum(dim=1)
        eta = base_reference + baseline_age + contribution.sum(dim=1)
        return {"logit": eta, "probability": eta.sigmoid(), "base_reference_logit": base_reference,
                "baseline_age": baseline_age, "reference_age_logit": reference_logit,
                "age_delta": age_delta, "pi": pi, "r": r.expand_as(z), "d": d.expand_as(z),
                "z": z, "transformed_prs": transformed, "age_multiplier": multiplier,
                "effective_coefficients": beta, "contributions": contribution,
                "reference_contributions": reference_contribution, "abs_d_t": dt.abs()}

    def penalty(self, rho, kappa, baseline_ridge):
        value = baseline_ridge*(self.base[1:].square().sum()+self.age_slopes.square().sum())
        w, c = self.centered_gate()
        if w is not None:
            value = value + rho*(w.square().sum()+c.square().sum())
        return value + kappa*(self.log_r.square().mean()+self.effective_d().square().mean())

    def project(self):
        with torch.no_grad():
            self.d.clamp_(min=0)
            self.age_slopes.clamp_(min=0)

    def canonical_state(self):
        state = {k: v.detach().clone() for k, v in self.state_dict().items()}
        if "w" in state:
            state["w"] -= state["w"].mean(dim=0)
            state["c"] -= state["c"].mean()
        return state


def from_m0(model, variant):
    if model.variant != "M0" or variant not in {"M1", "M2"}:
        raise ValueError("Warm starts require an M0 model and an M1/M2 target")
    new = SignAwareStudent(**model.dimensions, variant=variant)
    state = new.state_dict()
    for key, value in model.canonical_state().items():
        if key != "d":
            state[key] = value.clone()
    new.load_state_dict(state)
    return new


def predict(model, x):
    with torch.no_grad():
        result = {k: v.numpy() for k, v in model(as_tensors(x)).items()}
    if not all(np.isfinite(v).all() for v in result.values()):
        raise FloatingPointError("Nonfinite v3 prediction/component")
    return result


def numpy_forward(model, x):
    """Independent reconstruction using the positive/negative decomposition."""
    state = {k: v.numpy() for k, v in model.canonical_state().items()}
    z = x["z"]
    pi = softmax(x["a"] @ state["w"].T + state["c"], axis=1) if "w" in state else np.ones_like(z)
    dt = x["t"][:, None] * np.broadcast_to(state["d"], z.shape)
    phi = np.maximum(z, 0)*np.exp(dt)-np.maximum(-z, 0)*np.exp(-dt)
    base = x["nonage"] @ state["base"]
    age = x["age_basis"] @ state["age_slopes"]
    contributions = pi*np.exp(state["log_r"])*phi
    ref = base + (pi*np.exp(state["log_r"])*z).sum(axis=1)
    eta = base + age + contributions.sum(axis=1)
    return {"logit": eta, "probability": expit(eta), "contributions": contributions,
            "reference_age_logit": ref, "age_delta": eta-ref, "transformed_prs": phi}


def fit_student(train, y, q, *, variant, lam, rho, kappa, seed, options,
                tune=None, tune_y=None, epochs=None, warm=None):
    if (epochs is None) == (tune is None):
        raise ValueError("Choose either train-only internal tuning or locked-step refit")
    tx = as_tensors(train)
    target = soft_targets(y, q, lam)
    vx = as_tensors(tune) if tune is not None else None
    vy = soft_targets(tune_y, None, 0) if tune is not None else None
    if variant != "M0" and warm is None:
        raise ValueError("M1/M2 must start from a same-scope trained M0")
    failures = []
    for attempt, lr in enumerate([options.learning_rate, options.retry_learning_rate]):
        model = (from_m0(warm, variant) if warm is not None else
                 SignAwareStudent(tx["z"].shape[1], tx["a"].shape[1], tx["nonage"].shape[1],
                                  tx["age_basis"].shape[1], variant=variant, seed=seed,
                                  target_mean=float(target.mean())))
        if warm is not None:
            with torch.no_grad():
                torch.testing.assert_close(model(tx)["logit"], warm(tx)["logit"], rtol=0, atol=1e-12)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0)
        best, best_step, state, logs = float("inf"), 0, None, []
        max_abs_dt = 0.0
        try:
            for step in range(1, (epochs or options.max_epochs)+1):
                optimizer.zero_grad()
                output = model(tx)
                loss_data = F.binary_cross_entropy_with_logits(output["logit"], target)
                penalty = model.penalty(rho, kappa, options.baseline_ridge)
                loss = loss_data + penalty
                if not torch.isfinite(loss) or not torch.isfinite(output["effective_coefficients"]).all():
                    raise FloatingPointError("Nonfinite v3 loss or scale")
                max_abs_dt = max(max_abs_dt, float(output["abs_d_t"].detach().max()))
                loss.backward()
                norm = nn.utils.clip_grad_norm_(model.parameters(), options.gradient_clip, error_if_nonfinite=True)
                optimizer.step()
                model.project()
                if step % options.evaluate_every == 0 or step == (epochs or options.max_epochs):
                    with torch.no_grad():
                        current = model(vx if vx is not None else tx)
                        score = float(F.binary_cross_entropy_with_logits(current["logit"], vy if vy is not None else target))
                        if not np.isfinite(score) or not all(torch.isfinite(v).all() for v in current.values()):
                            raise FloatingPointError("Nonfinite v3 tuning/refit prediction")
                        logs.append({"step": step, "soft_target_bce": float(loss_data.detach()),
                                     "penalty": float(penalty.detach()), "score": score,
                                     "gradient_norm": float(norm), "max_abs_d_t": max_abs_dt,
                                     "max_r": float(model.log_r.exp().max()), "max_d": float(model.d.max()),
                                     "max_multiplier": float(current["age_multiplier"].max()),
                                     "max_beta": float(current["effective_coefficients"].max())})
                        if vx is None or score < best-options.min_delta:
                            best, best_step, state = score, step, copy.deepcopy(model.canonical_state())
                    if vx is not None and step-best_step >= options.patience:
                        break
            if state is None:
                raise FloatingPointError("No valid v3 checkpoint")
            model.load_state_dict(state)
            return model, {"best_step": best_step, "selection_metric": best,
                           "metric_type": "H_true_label_log_loss" if vx is not None else "D_soft_target_BCE",
                           "learning_rate": lr, "numerical_failures": failures, "logs": logs,
                           "warm_start_exact": warm is not None, "max_abs_d_t": max_abs_dt}
        except (FloatingPointError, RuntimeError) as exc:
            if isinstance(exc, RuntimeError) and "non-finite" not in str(exc):
                raise
            failures.append({"attempt": attempt, "learning_rate": lr, "error": str(exc)})
    raise FloatingPointError(f"Both predeclared numerical attempts failed: {failures}")


def save_bundle(path, model, preprocessing, metadata):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    torch.save(model.canonical_state(), path / "state.pt")
    joblib.dump(preprocessing, path / "preprocessing.joblib")
    write_json(path / "metadata.json", {**metadata, "dimensions": model.dimensions,
                                        "variant": model.variant, "preprocessing": preprocessing.metadata(),
                                        "prediction_scale": "direct_probability", "phase_b": False})
    write_json(path / "HASHES.json", {p.name: sha256_file(p) for p in path.iterdir()})


def load_bundle(path):
    import json

    path = Path(path)
    verify_files(path, json.loads((path / "HASHES.json").read_text()))
    metadata = json.loads((path / "metadata.json").read_text())
    model = SignAwareStudent(**metadata["dimensions"], variant=metadata["variant"])
    model.load_state_dict(torch.load(path / "state.pt", map_location="cpu", weights_only=True))
    if (model.d < 0).any() or (model.age_slopes < 0).any():
        raise ValueError("Saved v3 model violates projected constraints")
    return model, joblib.load(path / "preprocessing.joblib"), metadata
