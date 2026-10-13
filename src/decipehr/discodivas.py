"""Adapters for official DiscoDivas interpolation and train-only PRS harmonization."""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.linear_model import LinearRegression

from decipehr.external_pgs import sha256_file, write_json
from decipehr.external_pgs_benchmark import DISCODIVAS_METHODS, _ids_hash
from decipehr.schema import require_columns, validate_unique

REFERENCE_PCS = [f"refPC{i}" for i in range(1, 11)]
ANCESTRIES = ["AFR", "AMR", "EAS", "EUR", "SAS"]
COMPARISON_METHODS = DISCODIVAS_METHODS


@dataclass(frozen=True)
class ScoreBank:
    disease: str
    source_ids: tuple[str, ...]
    groups: tuple[str, ...]
    combination: np.ndarray
    provenance: str

    @property
    def model_ids(self) -> list[str]:
        return [f"external_{group}" for group in self.groups]


def comparison_bank(disease: str, coefficient_mapping: pd.DataFrame) -> ScoreBank:
    if disease == "T2D":
        expected_ids = [f"PGS{i:06d}" for i in range(5353, 5378)]
        mapping = coefficient_mapping.set_index("pgs_id").loc[expected_ids]
        if not mapping.index.is_unique or len(coefficient_mapping) != 25:
            raise ValueError("Expected the complete 25-row published D-PRISM mapping")
        combination = np.zeros((25, 5))
        for i, accession in enumerate(expected_ids):
            target, origin = ANCESTRIES[i // 5], ANCESTRIES[i % 5]
            row = mapping.loc[accession]
            if row.trained_for != target or row.weight_source != origin:
                raise ValueError("D-PRISM trained-for/source mapping is inconsistent")
            combination[i, i // 5] = float(row.published_coefficient)
        if not np.isfinite(combination).all() or (combination < 0).any():
            raise ValueError("Invalid published D-PRISM coefficients")
        return ScoreBank(disease, tuple(expected_ids), tuple(ANCESTRIES), combination,
                         "Published D-PRISM 5x5 coefficients; train-only component scaling")
    if disease == "bc":
        return ScoreBank(disease, ("PGS005378", "PGS004869", "PGS018538"),
                         ("AFR", "EUR", "EAS"), np.eye(3),
                         "Heterogeneous frozen models; source/development population proxies")
    if disease == "cad":
        return ScoreBank(disease, ("PGS003725", "PGS012558"), ("EUR", "EAS"), np.eye(2),
                         "EUR-tuned GPS_Mult and EAS-targeted MIXPRS+; 1000G proxies")
    raise ValueError(f"Unsupported disease: {disease}")


def official_interpolation(
    *, pcs: pd.DataFrame, scores: pd.DataFrame, centers: pd.DataFrame,
    groups: tuple[str, ...], r_script: str | Path, expected_sha256: str,
    out_dir: str | Path, rscript: str = "Rscript",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run unmodified upstream code with data.frame I/O and sorted surrogate IDs."""
    if sha256_file(r_script) != expected_sha256:
        raise ValueError("Official DiscoDivas source checksum mismatch")
    if len(groups) < 2 or len(set(groups)) != len(groups):
        raise ValueError("At least two distinct model reference centers are required")
    for frame in (pcs, scores):
        require_columns(frame, ["sampleid"], "DiscoDivas input")
        validate_unique(frame, "sampleid", "DiscoDivas input")
    if set(pcs.sampleid) != set(scores.sampleid):
        raise ValueError("DiscoDivas PRS and PC patient sets differ")
    require_columns(pcs, REFERENCE_PCS, "DiscoDivas PCs")
    require_columns(scores, list(groups), "DiscoDivas PRSs")
    require_columns(centers, ["POP", *[f"PC{i}" for i in range(1, 11)]], "PC medians")
    validate_unique(centers, "POP", "reference centers")
    medians = centers.set_index("POP").loc[
        list(groups), [f"PC{i}" for i in range(1, 11)],
    ].reset_index()
    x = pcs[REFERENCE_PCS].to_numpy(dtype=float)
    m = medians.iloc[:, 1:].to_numpy(dtype=float)
    if not np.isfinite(x).all() or not np.isfinite(m).all():
        raise ValueError("Nonfinite PC coordinates")
    if not np.isfinite(scores[list(groups)].to_numpy(dtype=float)).all():
        raise ValueError("Nonfinite PRS inputs")
    distances = cdist(m, m)
    if np.linalg.cond(distances) > 1e12 or np.any(cdist(x, m) <= 1e-12):
        raise ValueError("Degenerate DiscoDivas reference geometry")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=False)
    # Upstream's final IID vector follows PC input order while merges sort IDs.
    # Non-numeric sorted surrogates preserve alignment without modifying its code.
    ordered = pcs.sort_values("sampleid").reset_index(drop=True).copy()
    ordered["IID"] = [f"p{i:08d}" for i in range(len(ordered))]
    ordered[["IID", "sampleid"]].to_csv(out / "patient_mapping.tsv", sep="\t", index=False)
    ordered[["IID", *REFERENCE_PCS]].to_csv(out / "pcs.tsv", sep="\t", index=False)
    medians.to_csv(out / "medians.tsv", sep="\t", index=False)
    values = ordered[["IID", "sampleid"]].merge(
        scores, on="sampleid", how="left", validate="one_to_one",
    )
    paths = []
    for group in groups:
        path = out / f"{group}.tsv"
        values[["IID", group]].rename(columns={group: "PRS"}).to_csv(
            path, sep="\t", index=False,
        )
        paths.append(str(path))
    launcher = Path(__file__).resolve().parents[2] / "scripts/run_official_discodivas.R"
    command = [rscript, str(launcher), "-m", str(out / "medians.tsv"),
               "-p", str(out / "pcs.tsv"), "--prs.list", ",".join(paths),
               "-s", "IID,PRS", "-A", ",".join(["1"] * len(groups)),
               "--regress.PCA", "FALSE", "--print.coef", "TRUE", "-o", str(out / "combined")]
    write_json(out / "command.json", command)
    environment = {**os.environ, "DECIPEHR_DISCODIVAS_SOURCE": str(Path(r_script).resolve())}
    write_json(out / "upstream_provenance.json", {
        "upstream_path": str(r_script), "upstream_sha256": expected_sha256,
        "launcher_path": str(launcher), "launcher_sha256": sha256_file(launcher),
        "compatibility_option": "datatable.fread.datatable=FALSE",
        "regress_PCA": False, "quality_parameters": [1] * len(groups),
    })
    with (out / "upstream.log").open("w") as log:
        subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT, env=environment)
    coefficients = pd.read_csv(out / "combined.coef.tsv.gz", sep="\t", dtype={"IID": str})
    combined = pd.read_csv(out / "combined.tsv.gz", sep="\t", dtype={"IID": str})
    for frame in (coefficients, combined):
        if not frame.IID.is_unique or set(frame.IID) != set(ordered.IID):
            raise ValueError("Official output lost or duplicated patient IDs")
    weights = ordered[["IID", "sampleid"]].merge(
        coefficients, on="IID", validate="one_to_one",
    )
    weight_columns = [f"{group}.a" for group in groups]
    a = weights[weight_columns].to_numpy(dtype=float)
    if not np.isfinite(a).all() or not np.allclose(a.sum(axis=1), 1, atol=1e-10):
        raise ValueError("Invalid official interpolation coefficients")
    predicted = ordered[["IID", "sampleid"]].merge(combined, on="IID", validate="one_to_one")
    # Verify the official score against its exported coefficients, including ID alignment.
    expected = np.sum(values[list(groups)].to_numpy(dtype=float) * a, axis=1)
    np.testing.assert_allclose(predicted.PRS, expected, atol=1e-10, rtol=1e-10)
    weights = weights[["sampleid", *weight_columns]].rename(
        columns={f"{group}.a": f"disco_a_{group}" for group in groups},
    )
    predicted = predicted[["sampleid", "PRS"]].rename(columns={"PRS": "discodivas_prs"})
    return predicted, weights


@dataclass
class TrainScoreHarmonizer:
    """Fit standard scaling/OLS only on context, then apply published combinations."""

    bank: ScoreBank
    allowed_fit_ids: frozenset[str]
    fit_audit: list[dict] = field(default_factory=list)

    def __call__(self, context: pd.DataFrame, query: pd.DataFrame):
        if not set(context.sampleid).issubset(self.allowed_fit_ids):
            raise ValueError("Score harmonization attempted to fit outside official train")
        if set(context.sampleid) & set(query.sampleid):
            raise ValueError("Context and query patients overlap")
        sources = list(self.bank.source_ids)
        weight_columns = [f"disco_a_{group}" for group in self.bank.groups]
        for frame in (context, query):
            require_columns(frame, [*sources, *REFERENCE_PCS, *weight_columns], "Harmonization")
            if not np.isfinite(frame[[*sources, *REFERENCE_PCS, *weight_columns]]).all().all():
                raise ValueError("Harmonization inputs must be complete and finite")
        raw = context[sources].to_numpy(dtype=float)
        means, scales = raw.mean(axis=0), raw.std(axis=0, ddof=1)
        if not np.isfinite(scales).all() or (scales <= 0).any():
            raise ValueError("Constant source PRS in training context")
        metascores = ((raw - means) / scales) @ self.bank.combination
        pc_model = LinearRegression().fit(context[REFERENCE_PCS], metascores)
        residuals = metascores - pc_model.predict(context[REFERENCE_PCS])
        residual_means = residuals.mean(axis=0)
        residual_scales = residuals.std(axis=0, ddof=1)
        if (residual_scales <= 0).any() or not np.isfinite(residual_scales).all():
            raise ValueError("Degenerate PC-residualized PRS")
        transformed = []
        for frame in (context, query):
            scores = ((frame[sources].to_numpy(dtype=float) - means) / scales)
            metas = scores @ self.bank.combination
            z = (metas - pc_model.predict(frame[REFERENCE_PCS]) - residual_means) / residual_scales
            current = frame.copy()
            current[self.bank.model_ids] = z
            # Reuse the coefficients generated by the unmodified official R script.
            current["discodivas_prs"] = np.sum(z * frame[weight_columns].to_numpy(), axis=1)
            transformed.append(current)
        audit = {"context_ids_sha256": _ids_hash(context), "query_ids_sha256": _ids_hash(query),
                 "n_context": len(context), "n_query": len(query)}
        self.fit_audit.append(audit)
        parameters = []
        for i, name in enumerate(sources):
            parameters.append({"parameter_stage": "published_component_scaling", "feature": name,
                               "mean": means[i], "scale": scales[i], "ddof": 1, **audit})
        for i, name in enumerate(self.bank.model_ids):
            parameters.append({
                "parameter_stage": "prs_pc_residualization", "feature": name,
                "mean": residual_means[i], "scale": residual_scales[i], "ddof": 1,
                "intercept": pc_model.intercept_[i],
                "pc_coefficients": json.dumps(pc_model.coef_[i].tolist()), **audit,
            })
        return (*transformed, pd.DataFrame(parameters))
