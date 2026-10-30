"""Scientific boundary checks for validation-only kinship exclusions."""

import importlib
from pathlib import Path

import numpy as np
import pandas as pd


def test_group_filter_is_transitive_but_close_filter_requires_direct_train_edge(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    report = importlib.import_module("report_prs_validation_kinship_filters")
    train = pd.DataFrame({"sampleid": ["t1", "t2"], "kinship_group": ["g1", "g2"]})
    val = pd.DataFrame({"sampleid": ["direct", "indirect", "second", "onlyval", "unknown"],
                        "kinship_group": ["g1", "g1", "g2", "g3", "UNRESOLVED_QC::unknown"]})
    edges = pd.DataFrame({"IID1": ["direct", "direct", "t2", "onlyval"],
                          "IID2": ["t1", "indirect", "second", "val_relative"],
                          "KINSHIP": [.177, .1, .176999, .4]})
    original = train.copy(deep=True)
    group, direct, unknown = report.removal_masks(train, val, edges)
    np.testing.assert_array_equal(group, [True, True, True, False, False])
    np.testing.assert_array_equal(direct, [True, False, False, False, False])
    np.testing.assert_array_equal(unknown, [False, False, False, False, True])
    pd.testing.assert_frame_equal(train, original)


def test_context_excludes_ineligible_training_relatives(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    report = importlib.import_module("report_prs_validation_kinship_filters")
    train = pd.DataFrame({"sampleid": ["eligible"], "kinship_group": ["g1"]})
    val = pd.DataFrame({"sampleid": ["v"], "kinship_group": ["g2"]})
    edges = pd.DataFrame({"IID1": ["ineligible"], "IID2": ["v"], "KINSHIP": [.4]})
    group, direct, _ = report.removal_masks(train, val, edges)
    assert not group.any() and not direct.any()
