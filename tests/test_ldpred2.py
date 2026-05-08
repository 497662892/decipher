from decipehr.ldpred2 import ldpred2_model_tag, ldpred2_parameter_grid


def test_ldpred2_parameter_grid_crosses_h2_p_and_sparse_values():
    grid = ldpred2_parameter_grid(h2_grid=[0.1, 0.2], p_grid=[0.01], sparse_grid=[False, True])

    assert grid.to_dict("records") == [
        {"h2": 0.1, "p": 0.01, "sparse": False},
        {"h2": 0.1, "p": 0.01, "sparse": True},
        {"h2": 0.2, "p": 0.01, "sparse": False},
        {"h2": 0.2, "p": 0.01, "sparse": True},
    ]
    assert ldpred2_model_tag(0.1, 0.01, True) == "h20p1_p0p01_sparse1"
