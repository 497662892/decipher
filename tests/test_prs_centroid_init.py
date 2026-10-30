"""Independent distance-form checks and strict ancestry provenance tests."""

import json

import numpy as np
import pandas as pd
from scipy.special import softmax

from decipehr.prs_centroid_init import centroid_parameters, initialized_model, source_mapping
from decipehr.prs_distillation.student import soft_targets
from decipehr.prs_distillation_bio import initialize, tensors


def test_single_source_mapping_does_not_guess_target_or_broad_asian_ancestry():
    rows = [
        ("x", "trainedforAFR_EURweights", {"dist": {"EUR": 100}}, "EUR"),
        ("y", "EAS_optimized", {"dist": {"EUR": 30, "EAS": 70}}, "UNMAPPED"),
        ("z", "Asian", "ASN", "UNMAPPED"),
        ("w", "local", "EUR", "EUR"),
        ("u", "EUR_name_only", {"dist": {"NR": 100}}, "UNMAPPED"),
    ]
    manifest = pd.DataFrame([{"trait": "test", "column": p, "display_name": name,
                             "source_ancestry": json.dumps(meta)} for p, name, meta, _ in rows])
    mapped = source_mapping(manifest).set_index("prs").mapped_group.to_dict()
    assert mapped == {p: group for p, _, _, group in rows}


def test_centroid_logits_equal_distance_softmax_and_keep_unmapped_offsets_zero():
    rng = np.random.default_rng(37)
    centers = np.array([[2., 0., 0.], [-2., 1., 0.], [0., -2., 1.]])
    train = np.concatenate([rng.normal(c, .3, size=(20, 3)) for c in centers])
    labels = np.repeat(["AFR-like", "EUR-like", "EAS-like"], 20)
    prs = ["a", "b", "c"]
    mapping = pd.DataFrame({"prs": prs, "mapped_group": ["AFR", "EUR", "EAS"]})
    w, c, info = centroid_parameters(train, labels, prs, mapping)
    query = rng.normal(size=(50, 3))
    mu = np.array([info["centers"][g] for g in mapping.mapped_group])
    distances = ((query[:, None] - mu[None, :]) ** 2).sum(axis=2)
    np.testing.assert_allclose(softmax(query @ w.T + c, axis=1),
                               softmax(-distances / (2 * info["tau_squared"]), axis=1), atol=1e-14)
    unknown = pd.concat([mapping, pd.DataFrame({"prs": ["u"], "mapped_group": ["UNMAPPED"]})])
    w2, c2, _ = centroid_parameters(train, labels, [*prs, "u"], unknown)
    np.testing.assert_array_equal(w2[-1], 0)
    assert c2[-1] == 0
    # Ambiguous patients are retained by training but cannot move the proxy centers.
    w3, c3, _ = centroid_parameters(np.vstack([train, np.ones((4, 3)) * 1000]),
                                    np.r_[labels, np.repeat("AMBIGUOUS", 4)], prs, mapping)
    np.testing.assert_array_equal(w3, w)
    np.testing.assert_array_equal(c3, c)


def test_only_gate_initialization_changes_and_random_perturbations_are_paired():
    rng = np.random.default_rng(42)
    x = {"z": rng.normal(size=(40, 3)), "a": rng.normal(size=(40, 2)),
         "nonage": np.column_stack([np.ones(40), rng.normal(size=(40, 2))]),
         "age_basis": np.zeros((40, 3)), "t": np.zeros(40), "hinges": np.zeros((40, 3, 4))}
    y, q = rng.integers(2, size=40), rng.uniform(.1, .9, size=40)
    w, c = rng.normal(size=(3, 2)), rng.normal(size=3)
    for seed in [17, 29, 43]:
        random = initialize(tensors(x), soft_targets(y, q, .75), variant="M0", shape=False, seed=seed)
        centered = initialized_model(x, y, q, lam=.75, seed=seed, w=w, c=c)
        for key, value in random.state_dict().items():
            delta = centered.state_dict()[key].numpy() - value.numpy()
            np.testing.assert_allclose(delta, w if key == "w" else c if key == "c" else 0, atol=1e-15)
