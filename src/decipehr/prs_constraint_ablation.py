"""Matched low-order models differing only in their monotonicity constraints."""

from dataclasses import dataclass

import numpy as np

from decipehr.prs_monotone_boosting import Features, explain, raw_predict


@dataclass
class AblationFeatures(Features):
    constraint: str = "AGE"

    def constraints(self, family):
        monotone, groups = super().constraints(family)
        if self.constraint not in {"AGE", "PRS", "FREE"}:
            raise ValueError("Unknown monotonicity ablation")
        if self.constraint in {"AGE", "FREE"}:
            monotone[:self.k] = [0]*self.k
        if self.constraint in {"PRS", "FREE"}:
            monotone[self.k] = 0
        return monotone, groups


def verify_constraints(bundle, x):
    sample = x[np.linspace(0, len(x)-1, min(32, len(x))).astype(int)]
    explain(bundle, sample)
    flags, _ = bundle["features"].constraints(bundle["family"])
    minimum = 0.0
    for feature in np.flatnonzero(flags):
        for row in sample[:8]:
            grid = np.linspace(x[:, feature].min()-1, x[:, feature].max()+1, 41)
            changed = np.tile(row, (len(grid), 1))
            changed[:, feature] = grid
            delta = np.diff(raw_predict(bundle, changed))
            minimum = min(minimum, float(delta.min()))
            if delta.min() < -1e-10:
                raise ValueError("A declared monotone coordinate decreased")
    return {"exact_anchored_decomposition": True, "max_interaction_order": 2,
        "constraint": bundle["features"].constraint, "active_pair_count": len(bundle["active_pairs"]),
        "monotone_features": [bundle["features"].names[i] for i in np.flatnonzero(flags)],
        "minimum_declared_monotone_difference": minimum}
