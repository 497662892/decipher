"""Monotone additive / biological-pair students using sklearn's mature booster.

The pair model constrains TOTAL risk to increase with age and each risk-oriented
PRS, while allowing the PRS contrast to strengthen or attenuate across age.
Every tree branch uses at most two explicitly allowed features, enabling exact
anchored main-effect / pair-effect decomposition on the logit scale.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import expit
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import log_loss

from decipehr.prs_distillation_bio import BioPreprocessor


@dataclass
class Features:
    prep: BioPreprocessor
    names: list[str]
    reference: np.ndarray

    @classmethod
    def fit(cls, frame, prs):
        prep = BioPreprocessor.fit(frame, prs)
        names = [*prep.age.fixed.kept_prs, "age_sd_centered", *prep.metadata()["nonage_columns"][1:]]
        temporary = cls(prep, names, np.zeros(len(names)))
        reference = np.median(temporary.transform(frame), axis=0)
        reference[:len(prep.age.fixed.kept_prs)+11] = 0  # PRS, reference age, PC1-PC10
        return cls(prep, names, reference)

    def transform(self, frame):
        x = self.prep.transform(frame)
        return np.column_stack([x["z"], x["t"], x["nonage"][:, 1:]])

    @property
    def k(self):
        return len(self.prep.age.fixed.kept_prs)

    def constraints(self, family):
        monotone = [1]*(self.k+1)+[0]*(len(self.names)-self.k-1)
        groups = [{i} for i in range(len(self.names))]
        if family == "PAIR":
            groups += [{k, j} for k in range(self.k) for j in range(self.k, self.k+11)]
            groups += [{self.k, j} for j in range(self.k+11, len(self.names))]
        elif family != "GAM":
            raise ValueError("Unknown low-order family")
        return monotone, groups


def soft_binary_rows(x, target):
    target = np.asarray(target, dtype=float)
    if target.shape != (len(x),) or not np.isfinite(target).all() or ((target < 0) | (target > 1)).any():
        raise ValueError("Invalid soft binary target")
    # Per-patient weighted binary loss equals soft-label BCE exactly.
    return np.concatenate([x, x]), np.r_[np.zeros(len(x), dtype=int), np.ones(len(x), dtype=int)], np.r_[1-target, target]


def fit_booster(x, target, features, *, family, leaves, l2, iterations=400):
    monotone, groups = features.constraints(family)
    model = HistGradientBoostingClassifier(loss="log_loss", learning_rate=.05, max_iter=max(1, iterations),
        max_leaf_nodes=leaves, max_depth=3, min_samples_leaf=80, l2_regularization=l2,
        max_bins=32, categorical_features=None, monotonic_cst=monotone, interaction_cst=groups,
        early_stopping=False, random_state=17)
    xx, yy, weights = soft_binary_rows(x, target)
    model.fit(xx, yy, sample_weight=weights)
    return model


def select_iteration(model, x, y, target_mean, tolerance=1e-4):
    scores = [{"iterations": 0, "H_log_loss": log_loss(y, np.full(len(y), target_mean))}]
    for step, p in enumerate(model.staged_predict_proba(x), 1):
        if step % 10 == 0 or step == 1 or step == model.n_iter_:
            scores.append({"iterations": step, "H_log_loss": log_loss(y, p[:, 1])})
    best = min(row["H_log_loss"] for row in scores)
    chosen = next(row for row in scores if row["H_log_loss"] <= best+tolerance)
    return chosen, scores


def active_pairs(model, allowed_groups):
    found = set()
    for stage in model._predictors:
        for tree in stage:
            nodes = tree.nodes
            def walk(index, used, nodes=nodes):
                node = nodes[index]
                if node["is_leaf"]:
                    if len(used) > 2 or not any(used <= group for group in allowed_groups):
                        raise ValueError("A fitted branch violates the declared biological interaction graph")
                    if len(used) == 2:
                        found.add(tuple(sorted(used)))
                    return
                used = used | {int(node["feature_idx"])}
                walk(int(node["left"]), used)
                walk(int(node["right"]), used)
            walk(0, set())
    return sorted(found)


def raw_predict(bundle, x):
    if bundle["iterations"] == 0:
        rate = bundle["target_mean"]
        return np.full(len(x), np.log(rate)-np.log1p(-rate))
    return bundle["model"].decision_function(x)


def predict(bundle, x):
    return expit(raw_predict(bundle, x))


def explain(bundle, x):
    reference = bundle["features"].reference
    baseline = float(raw_predict(bundle, reference[None])[0])
    first = []
    for i in range(x.shape[1]):
        z = np.tile(reference, (len(x), 1))
        z[:, i] = x[:, i]
        first.append(raw_predict(bundle, z)-baseline)
    first = np.column_stack(first)
    pairs, second = bundle["active_pairs"], []
    for i, j in pairs:
        z = np.tile(reference, (len(x), 1))
        z[:, [i, j]] = x[:, [i, j]]
        second.append(raw_predict(bundle, z)-baseline-first[:, i]-first[:, j])
    second = np.column_stack(second) if second else np.zeros((len(x), 0))
    np.testing.assert_allclose(baseline+first.sum(1)+second.sum(1), raw_predict(bundle, x), atol=2e-10, rtol=1e-10)
    return baseline, first, second


def bundle_model(model, features, family, iterations, target_mean, metadata):
    _, groups = features.constraints(family)
    pairs = active_pairs(model, groups) if iterations else []
    return {"model": model, "features": features, "family": family, "iterations": iterations,
            "target_mean": float(target_mean), "active_pairs": pairs, "metadata": metadata}


def verify(bundle, x):
    sample = x[np.linspace(0, len(x)-1, min(32, len(x))).astype(int)]
    explain(bundle, sample)
    minimum = 0.0
    for feature in range(bundle["features"].k+1):
        for row in sample[:8]:
            grid = np.linspace(x[:, feature].min()-1, x[:, feature].max()+1, 41)
            changed = np.tile(row, (len(grid), 1))
            changed[:, feature] = grid
            delta = np.diff(raw_predict(bundle, changed))
            minimum = min(minimum, float(delta.min()))
            if delta.min() < -1e-10:
                raise ValueError("Fitted booster violates age/PRS monotonicity")
    return {"exact_anchored_decomposition": True, "max_interaction_order": 2,
            "active_pair_count": len(bundle["active_pairs"]), "minimum_monotone_difference": minimum}
