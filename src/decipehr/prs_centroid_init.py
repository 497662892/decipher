"""Train-context centroid warm starts for the existing linear ancestry gate."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import torch

from decipehr.prs_distillation.student import soft_targets
from decipehr.prs_distillation_bio import initialize, tensors

GROUPS = ("AFR", "AMR", "EAS", "EUR", "SAS")


def source_mapping(manifest):
    """Use recorded single-ancestry GWAS provenance, never an inferred display name."""
    rows = []
    for row in manifest.itertuples():
        metadata = json.loads(row.source_ancestry)
        group, reason = "UNMAPPED", "multi_ancestry_or_unspecified"
        if isinstance(metadata, str):
            if metadata in GROUPS:
                group, reason = metadata, "explicit_local_component"
            else:
                reason = "broad_ASN_not_assumed_EAS" if metadata == "ASN" else "unspecified_local_component"
        elif isinstance(metadata, dict) and not metadata.get("multi"):
            dist = {k: float(v) for k, v in metadata.get("dist", {}).items() if float(v) > 0}
            if len(dist) == 1:
                name, fraction = next(iter(dist.items()))
                if name in GROUPS and np.isclose(fraction, 100):
                    group, reason = name, "single_ancestry_GWAS_100_percent"
        rows.append({"trait": row.trait, "prs": row.column, "display_name": row.display_name,
                     "source_ancestry": row.source_ancestry, "mapped_group": group, "mapping_reason": reason})
    result = pd.DataFrame(rows)
    if not result.prs.is_unique:
        raise ValueError("PRS mapping must be unique")
    return result


def centroid_parameters(ancestry, hard_labels, prs, mapping, *, minimum_group_n=10):
    """All inputs refer only to the fit context, in its standardized clinical PC basis."""
    ancestry = np.asarray(ancestry, dtype=float)
    labels = np.asarray(hard_labels)
    if ancestry.ndim != 2 or len(ancestry) != len(labels) or not np.isfinite(ancestry).all():
        raise ValueError("Invalid training ancestry coordinates")
    means, counts = {}, {}
    for group in GROUPS:
        mask = labels == f"{group}-like"
        counts[group] = int(mask.sum())
        if counts[group] >= minimum_group_n:
            means[group] = ancestry[mask].mean(axis=0)
    if len(means) < 2:
        raise ValueError("At least two supported ancestry groups are required")
    centers = np.stack(list(means.values()))
    squared = ((centers[:, None] - centers[None, :]) ** 2).sum(axis=2)
    distances = squared[np.triu_indices(len(centers), 1)]
    distances = distances[distances > 1e-12]
    if not len(distances):
        raise ValueError("Ancestry centers are indistinguishable")
    # A typical pair of centers starts with a distance-logit difference of one nat.
    tau_squared = float(np.median(distances) / 2)
    groups = mapping.set_index("prs").loc[prs, "mapped_group"].tolist()
    w, c = np.zeros((len(prs), ancestry.shape[1])), np.zeros(len(prs))
    assigned = []
    for k, group in enumerate(groups):
        supported = group in means
        if supported:
            mu = means[group]
            w[k], c[k] = mu / tau_squared, -float(mu @ mu) / (2 * tau_squared)
        assigned.append({"prs": prs[k], "mapped_group": group, "center_used": supported})
    return w, c, {"tau_squared": tau_squared, "minimum_group_n": minimum_group_n,
                  "group_counts": counts, "centers": {k: v.tolist() for k, v in means.items()},
                  "mapped_prs": assigned, "per_prs_prior": "equal; no ancestry-count reweighting",
                  "ambiguous_patients_used_for_centers": False,
                  "unmapped_policy": "zero centroid offset; retain original small random gate row"}


def initialized_model(x, y, q, *, lam, seed, w, c):
    """Keep the exact original random perturbation and every nongate initial parameter."""
    target = soft_targets(y, q, lam)
    model = initialize(tensors(x), target, variant="M0", shape=False, seed=seed)
    if tuple(model.w.shape) != w.shape or tuple(model.c.shape) != c.shape:
        raise ValueError("Centroid parameter dimensions do not match the frozen PRS panel")
    with torch.no_grad():
        model.w.add_(torch.as_tensor(w, dtype=torch.float64))
        model.c.add_(torch.as_tensor(c, dtype=torch.float64))
    return model
