import pandas as pd

from decipehr.prs import compute_prs_from_dosage, pt_parameter_grid


def test_pt_parameter_grid_size():
    grid = pt_parameter_grid([1.0, 0.5], [0.2, 0.8])
    assert len(grid) == 4


def test_compute_prs_from_dosage():
    genotype = pd.DataFrame(
        {
            "patient_id": ["P1", "P2"],
            "rs1": [0, 2],
            "rs2": [1, 1],
        }
    )
    weights = pd.DataFrame({"SNP": ["rs1", "rs2"], "BETA": [0.5, -1.0]})
    prs = compute_prs_from_dosage(genotype, weights)
    assert prs["prs"].tolist() == [-1.0, 0.0]

