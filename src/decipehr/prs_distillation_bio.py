"""Small monotone biological-prior students and explicit warm-start diagnostics.

These are predictive structural priors, not identified biological mechanisms.
Frozen v3 code and artifacts are intentionally left untouched.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import torch
from scipy.special import expit, softmax
from torch import nn
from torch.nn import functional as F

from decipehr.external_pgs import sha256_file, write_json
from decipehr.prs_distillation.preprocessing import Preprocessor
from decipehr.prs_distillation.student import soft_targets
from decipehr.prs_distillation_v3 import SignAwarePreprocessor, SignAwareStudent, as_tensors


@dataclass
class BioPreprocessor:
    age: SignAwarePreprocessor
    prs_knots: np.ndarray

    @classmethod
    def fit(cls, frame, prs, age_segments=3):
        fixed = Preprocessor.fit(frame, prs)
        age = SignAwarePreprocessor(fixed, float(frame.age_sd.median()), 1.0,
                                   np.unique(np.quantile(frame.age_sd, np.linspace(0, 1, age_segments+1))))
        z = fixed.transform(frame)["z"]
        return cls(age, np.quantile(z, [.1, .3, .7, .9], axis=0).T)

    def transform(self, frame):
        x = self.age.transform(frame)
        x["hinges"] = np.maximum(x["z"][:, :, None]-self.prs_knots, 0)-np.maximum(-self.prs_knots, 0)
        return x

    def metadata(self):
        return {**self.age.metadata(), "prs_shape_knots": self.prs_knots.tolist(),
                "prs_shape_reference": "zero on the frozen context-standardized PRS scale",
                "prs_shape_extrapolation": "linear with positive end slopes"}


def tensors(x):
    return {**as_tensors(x), "hinges": torch.as_tensor(x["hinges"], dtype=torch.float64)}


class BioStudent(SignAwareStudent):
    def __init__(self, k, n_ancestry, n_nonage, n_segments, *, variant="M0", shape=False, seed=17, target_mean=.5):
        super().__init__(k, n_ancestry, n_nonage, n_segments, variant=variant, seed=seed, target_mean=target_mean)
        self.has_shape = shape
        if shape:
            self.log_slopes = nn.Parameter(torch.zeros(k, 5, dtype=torch.float64))
        else:
            self.register_buffer("log_slopes", torch.zeros(k, 5, dtype=torch.float64))

    def shape_slopes(self):
        # Per-PRS geometric-mean-one slope removes its common-scale ambiguity with r.
        return (self.log_slopes-self.log_slopes.mean(1, keepdim=True)).exp()

    def forward(self, x):
        slopes = self.shape_slopes()
        shaped = x["z"]*slopes[:, 0] + (x["hinges"]*(slopes[:, 1:]-slopes[:, :-1])).sum(2)
        output = super().forward({**x, "z": shaped})
        output["fixed_prs"] = x["z"]
        output["shaped_prs"] = shaped
        return output

    def penalty(self, rho, kappa_r, kappa_d, kappa_shape, baseline_ridge):
        value = super().penalty(rho, 0, baseline_ridge)
        value = value + kappa_r*self.log_r.square().mean() + kappa_d*self.effective_d().square().mean()
        return value + kappa_shape*(self.log_slopes-self.log_slopes.mean(1, keepdim=True)).square().mean()


def initialize(x, y_soft, *, variant, shape, seed, warm=None):
    model = BioStudent(x["z"].shape[1], x["a"].shape[1], x["nonage"].shape[1], x["age_basis"].shape[1],
                       variant=variant, shape=shape, seed=seed, target_mean=float(y_soft.mean()))
    if warm is not None:
        state = model.state_dict()
        if (warm.effective_d() != 0).any():
            raise ValueError("Warm start must have zero age interaction")
        for key, value in warm.state_dict().items():
            if key != "d":
                if key not in state or state[key].shape != value.shape:
                    raise ValueError("Incompatible warm-start dimensions")
                state[key] = value.detach().clone()
        model.load_state_dict(state)
        with torch.no_grad():
            torch.testing.assert_close(model(x)["logit"], warm(x)["logit"], atol=1e-12, rtol=0)
    return model


def fit(train, y, q, *, variant, shape, lam, rho, kappa_r, kappa_d, kappa_shape, seed,
        learning_rate=.01, max_steps=2000, patience=200, evaluate_every=10, baseline_ridge=1e-4,
        tune=None, tune_y=None, fixed_steps=None, warm=None, d_only=False):
    if (fixed_steps is None) == (tune is None):
        raise ValueError("Use either H selection or a locked refit step count")
    if fixed_steps is not None and fixed_steps < 0:
        raise ValueError("Negative step count")
    tx = tensors(train)
    target = soft_targets(y, q, lam)
    vx, vy = (tensors(tune), soft_targets(tune_y, None, 0)) if tune is not None else (tx, target)
    model = initialize(tx, target, variant=variant, shape=shape, seed=seed, warm=warm)
    if d_only:
        if warm is None or variant == "M0" or not isinstance(model.d, nn.Parameter):
            raise ValueError("d-only diagnostic requires a nonconstant-age warm-start interaction")
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name == "d")
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=learning_rate)
    logs = []

    def measure(step):
        with torch.no_grad():
            output, val = model(tx), model(vx)
            if not all(torch.isfinite(v).all() for v in [*output.values(), *val.values()]):
                raise FloatingPointError("Nonfinite biological-prior model")
            loss_data = F.binary_cross_entropy_with_logits(output["logit"], target)
            penalty = model.penalty(rho, kappa_r, kappa_d, kappa_shape, baseline_ridge)
            score = float(F.binary_cross_entropy_with_logits(val["logit"], vy))
            logs.append({"step": step, "selection_loss": score, "training_soft_bce": float(loss_data),
                         "training_penalty": float(penalty), "max_d": float(model.d.max()),
                         "max_abs_d_t": float(output["abs_d_t"].max()),
                         "max_r": float(model.log_r.exp().max()), "max_shape_slope": float(model.shape_slopes().max())})
            return score

    # Step zero is a valid candidate, including the exact nested parent.
    best, best_step, state = measure(0), 0, copy.deepcopy(model.state_dict())
    limit = max_steps if fixed_steps is None else fixed_steps
    for step in range(1, limit+1):
        optimizer.zero_grad()
        output = model(tx)
        loss = F.binary_cross_entropy_with_logits(output["logit"], target)
        loss = loss + model.penalty(rho, kappa_r, kappa_d, kappa_shape, baseline_ridge)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite objective; no silent clipping or numerical fallback")
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 10, error_if_nonfinite=True)
        optimizer.step()
        model.project()
        if step % evaluate_every == 0 or step == limit:
            score = measure(step)
            if fixed_steps is not None or score < best-1e-6:
                best, best_step, state = score, step, copy.deepcopy(model.state_dict())
            if fixed_steps is None and step-best_step >= patience:
                break
    model.load_state_dict(state)
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    return model, {"best_step": best_step, "selection_loss": best, "logs": logs,
                   "step_zero_eligible": True, "warm_start_exact": warm is not None,
                   "d_only": d_only, "optimizer_reset": True, "learning_rate": learning_rate,
                   "selection_scope": "H_true_label" if tune is not None else "locked_steps"}


def predict(model, x):
    with torch.no_grad():
        return {k: v.numpy() for k, v in model(tensors(x)).items()}


def numpy_predict(model, x):
    state = {k: v.detach().numpy() for k, v in model.state_dict().items()}
    if "w" in state:
        w, c = state["w"]-state["w"].mean(0), state["c"]-state["c"].mean()
        pi = softmax(x["a"] @ w.T+c, axis=1)
    else:
        pi = np.ones_like(x["z"])
    slopes = np.exp(state["log_slopes"]-state["log_slopes"].mean(1, keepdims=True))
    # Independently sum each centered piecewise interval, including linear tails.
    h = x["hinges"]
    pieces = np.concatenate([(x["z"]-h[:, :, 0])[:, :, None], h[:, :, :-1]-h[:, :, 1:], h[:, :, -1:]], axis=2)
    g = (pieces*slopes).sum(2)
    dt = x["t"][:, None]*np.broadcast_to(state["d"], x["z"].shape)
    phi = np.maximum(g, 0)*np.exp(dt)-np.maximum(-g, 0)*np.exp(-dt)
    eta = x["nonage"] @ state["base"] + x["age_basis"] @ state["age_slopes"]
    eta += (pi*np.exp(state["log_r"])*phi).sum(1)
    return {"probability": expit(eta), "logit": eta, "shaped_prs": g}


def save(path, model, prep, metadata):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    torch.save(model.state_dict(), path / "state.pt")
    joblib.dump(prep, path / "preprocessing.joblib")
    write_json(path / "metadata.json", {**metadata, "dimensions": model.dimensions,
        "variant": model.variant, "shape": model.has_shape, "preprocessing": prep.metadata()})
    write_json(path / "HASHES.json", {p.name: sha256_file(p) for p in path.iterdir()})


def load(path):
    import json

    from decipehr.prs_distillation.data import verify_files

    path = Path(path)
    verify_files(path, json.loads((path / "HASHES.json").read_text()))
    meta = json.loads((path / "metadata.json").read_text())
    model = BioStudent(**meta["dimensions"], variant=meta["variant"], shape=meta["shape"])
    model.load_state_dict(torch.load(path / "state.pt", weights_only=True, map_location="cpu"))
    return model, joblib.load(path / "preprocessing.joblib"), meta
