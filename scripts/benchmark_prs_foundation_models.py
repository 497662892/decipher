"""Local, frozen-split Kumo Tabular and TabPFN 3.5 predictions; no val labels."""

from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from decipehr.external_pgs import sha256_file, write_json
from decipehr.prs_distillation.config import digest
from decipehr.prs_distillation.data import verify_files
from decipehr.prs_distillation.experiment import load_inputs
from decipehr.prs_distillation.preprocessing import AGE, PCS, assert_boundary, frame_hash, id_hash
from decipehr.tabpfn_ensemble import TabPFNFeatureTransformer


def read(path):
    return json.loads(Path(path).read_text())


def predict(backend, x, y, query, checkpoint, *, estimators=16):
    torch.manual_seed(42)
    np.random.seed(42)
    if backend == "tabpfn35":
        from tabpfn import TabPFNClassifier
        from tabpfn.constants import ModelVersion

        model = TabPFNClassifier.create_default_for_version(ModelVersion.V3_5,
            n_estimators=estimators, random_state=42, device="cuda", model_path=str(checkpoint),
            balance_probabilities=False)
        model.fit(x, y)
        all_p = np.asarray(model.predict_proba(query))
        classes = np.asarray(model.classes_)
    else:
        import sdm

        model = sdm.models.KumoTabular(task="classification", size="large", device="cuda")
        ctx = pd.DataFrame(x).rename(columns=lambda c: f"x{c}").assign(target=np.asarray(y, dtype=int))
        qry = pd.DataFrame(query).rename(columns=lambda c: f"x{c}")
        stypes = sdm.infer_stypes(ctx, overrides={"target": "categorical"})
        context = sdm.TableTensor.from_pandas(ctx, stypes=stypes, device="cuda")
        queries = sdm.TableTensor.from_pandas(qry, stypes={c: stypes[c] for c in qry}, device="cuda")
        generator = torch.Generator(device="cuda").manual_seed(42)
        with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.float16):
            output = model(x_context=context.drop_columns("target"), y_context=context[:, "target"],
                           x_query=queries, num_estimators=estimators, estimator_batch_size=1, generator=generator)
        result = output.to_pandas()
        classes = np.asarray([int(c) for c in result.columns])
        all_p = result.to_numpy()
    if set(classes) != {0, 1} or all_p.shape != (len(query), 2):
        raise ValueError("Unexpected positive-class mapping or probability shape")
    if not np.isfinite(all_p).all() or (all_p < 0).any() or (all_p > 1).any():
        raise ValueError("Nonfinite or invalid probabilities")
    np.testing.assert_allclose(all_p.sum(1), 1, atol=2e-5)
    probability = all_p[:, np.flatnonzero(classes == 1)[0]]
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return probability, classes.tolist()


def main():
    os.umask(0o077)
    for name in ["TABPFN_NO_BROWSER", "TABPFN_DISABLE_TELEMETRY", "HF_HUB_DISABLE_TELEMETRY", "HF_HUB_OFFLINE"]:
        os.environ[name] = "1"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["tabpfn35", "kumo"], required=True)
    parser.add_argument("--run-dir", type=Path, default=Path("outputs/foundation_model_probe_20261026/benchmark"))
    parser.add_argument("--source-run", type=Path, default=Path("outputs/prs_distillation_v1/phase_a_fulltrain_20261026_3d4334d2_0161ad36d950"))
    parser.add_argument("--eligibility-config", required=True)
    parser.add_argument("--trait", choices=["T2D", "cad", "bc"])
    args = parser.parse_args()
    from decipehr.eligibility import filter_eligible, load_eligibility_rules

    source = args.source_run
    audit = read(source / "audit_report.json")
    if args.eligibility_config != audit["paths"]["eligibility"]:
        raise ValueError("Eligibility contract mismatch")
    rules = load_eligibility_rules(args.eligibility_config)
    download = read(Path("outputs/foundation_model_probe_20261026") / f'{"tabpfn35" if args.backend == "tabpfn35" else "kumo"}_download.json')
    checkpoint = Path(download["path"]).resolve()
    if sha256_file(checkpoint) != download["sha256"]:
        raise ValueError("Checkpoint receipt mismatch")
    root = args.run_dir / args.backend
    root.mkdir(parents=True, exist_ok=True)
    contract = {"backend": args.backend, "checkpoint": download, "source_complete_sha256": sha256_file(source / "COMPLETE.json"),
        "script_sha256": sha256_file(__file__), "packages": {p: importlib.metadata.version(p) for p in
        ["numpy", "pandas", "scikit-learn", "torch", "tabpfn" if args.backend == "tabpfn35" else "structured-data-models"]},
        "estimators": 16, "seed": 42, "scope": "same features, train-only context, direct probabilities",
        "private_data_sent": False, "evaluation_labels_read": False, "cohort_policy": audit["cohort_policy"]}
    if (root / "CONTRACT.json").exists() and read(root / "CONTRACT.json") != contract:
        raise ValueError("Existing benchmark contract differs")
    write_json(root / "CONTRACT.json", contract)
    if not (root / "SMOKE.json").exists():
        from sklearn.datasets import make_classification

        x, y = make_classification(n_samples=160, n_features=8, random_state=17)
        start = time.monotonic()
        p, classes = predict(args.backend, x[:120], y[:120], x[120:], checkpoint, estimators=2)
        write_json(root / "SMOKE.json", {"status": "passed", "n_predictions": len(p), "classes": classes,
            "elapsed_seconds": time.monotonic()-start, "synthetic_only": True})
        print(f"{args.backend} synthetic smoke passed", flush=True)
    split = pd.read_parquet(source / "split_manifest.parquet").set_index("sampleid")
    for trait in [args.trait] if args.trait else ["T2D", "cad", "bc"]:
        d, e, manifest = load_inputs(source, trait)
        for frame in [d, e]:
            if len(filter_eligible(frame, trait, rules)) != len(frame):
                raise ValueError("Cohort eligibility mismatch")
        prs = manifest.column.tolist()
        roles = split.loc[d.sampleid]
        u, h = d.loc[roles.internal_role.eq("U").to_numpy()], d.loc[roles.internal_role.eq("H").to_numpy()]
        scopes = [("U_to_H", u, h.drop(columns="y")), ("D_to_E", d, e)]
        for fold in range(5):
            mask = roles.final_teacher_fold.to_numpy() == fold
            scopes.append((f"D_oof_{fold}", d.loc[~mask], d.loc[mask].drop(columns="y")))
        for scope, context, query in scopes:
            assert_boundary(context, query, require_group_disjoint=scope != "D_to_E")
            out = root / trait / scope
            fingerprint = digest({"contract": contract, "context": frame_hash(context), "query": frame_hash(query), "prs": prs})
            if (out / "COMPLETE.json").exists():
                saved = read(out / "COMPLETE.json")
                if saved["fingerprint"] != fingerprint:
                    raise ValueError("Existing prediction scope mismatch")
                verify_files(out, saved["output_sha256"])
                continue
            print(f"{args.backend}/{trait}/{scope}: {len(context)} context, {len(query)} query", flush=True)
            start = time.monotonic()
            prep = TabPFNFeatureTransformer.fit(context, prs_cols=prs, covariates=[AGE, "sex", *PCS], include_covariates=True)
            x, xq = prep.transform(context).to_numpy(), prep.transform(query).to_numpy()
            q, classes = predict(args.backend, x, context.y.to_numpy(), xq, checkpoint)
            out.mkdir(parents=True, exist_ok=True)
            query[["sampleid", "kinship_group"]].assign(probability=q).to_parquet(out / "predictions.parquet", index=False)
            write_json(out / "COMPLETE.json", {"fingerprint": fingerprint, "context_ids_hash": id_hash(context),
                "query_ids_hash": id_hash(query), "context_n": len(context), "query_n": len(query),
                "columns": list(prep.output_columns), "classes": classes, "evaluation_labels_read": False,
                "elapsed_seconds": time.monotonic()-start, "max_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
                "output_sha256": {"predictions.parquet": sha256_file(out / "predictions.parquet")}})
        print(f"Finished {args.backend}/{trait}", flush=True)
    write_json(root / "COMPLETE.json", {"status": "predictions_complete", "contract": contract,
        "traits": [args.trait] if args.trait else ["T2D", "cad", "bc"], "evaluation_labels_read": False,
        "output_sha256": {str(p.relative_to(root)): sha256_file(p) for p in root.rglob("*")
                          if p.is_file() and p != root / "COMPLETE.json"}})


if __name__ == "__main__":
    main()
