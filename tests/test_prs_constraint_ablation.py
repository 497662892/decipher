import numpy as np
from scipy.special import expit
from test_prs_monotone_boosting import sample

from decipehr.prs_constraint_ablation import AblationFeatures, verify_constraints
from decipehr.prs_monotone_boosting import bundle_model, fit_booster, raw_predict


def test_ablation_preserves_feature_map_and_changes_only_declared_constraints():
    frame = sample()
    feature = AblationFeatures.fit(frame, ["PRS_A", "PRS_B"])
    x = feature.transform(frame)
    graphs = []
    for mode, expected in [("AGE", [0, 0, 1]), ("PRS", [1, 1, 0]), ("FREE", [0, 0, 0])]:
        feature.constraint = mode
        mask, graph = feature.constraints("PAIR")
        assert mask[:3] == expected and not any(mask[3:])
        np.testing.assert_array_equal(feature.transform(frame), x)
        graphs.append(graph)
    assert graphs[0] == graphs[1] == graphs[2]


def test_age_only_can_fit_negative_conditional_prs_effect_while_preserving_age():
    frame = sample()
    features = AblationFeatures.fit(frame, ["PRS_A", "PRS_B"])
    x = features.transform(frame)
    q = expit(-2*x[:, 0]+x[:, features.k])
    fitted = fit_booster(x, q, features, family="GAM", leaves=4, l2=1, iterations=100)
    bundle = bundle_model(fitted, features, "GAM", 100, q.mean(), {})
    audit = verify_constraints(bundle, x)
    assert audit["monotone_features"] == ["age_sd_centered"]
    z = np.tile(features.reference, (51, 1))
    z[:, 0] = np.linspace(-1, 1, 51)
    eta = raw_predict(bundle, z)
    assert eta[-1] < eta[0]-.5
