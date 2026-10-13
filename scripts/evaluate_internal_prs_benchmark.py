"""Run one disease/outer-fold partition from a frozen NYU benchmark manifest."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd
from evaluate_external_pgs_cv import _tabpfn_factory

from decipehr.discodivas import comparison_bank
from decipehr.external_pgs import sha256_file, write_json
from decipehr.internal_prs_benchmark import METHODS, evaluate_partition


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default="outputs/internal_prs/nyu_20261011")
    parser.add_argument("--disease", choices=["T2D", "cad", "bc"], required=True)
    parser.add_argument("--partition", choices=["0", "1", "2", "3", "4", "development"], required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--output-suffix", default="")
    args = parser.parse_args()
    os.umask(0o077)
    root = Path(args.run_dir)
    prepared = root / "prepared"
    manifest = json.loads((prepared / "manifest.json").read_text())
    for name, digest in manifest["code_sha256"].items():
        if sha256_file(name) != digest:
            raise ValueError(f"Code changed after input freeze: {name}")
    for name, digest in manifest["output_sha256"].items():
        if sha256_file(prepared / name) != digest:
            raise ValueError(f"Prepared input changed: {name}")
    context = pd.read_parquet(prepared / f"{args.disease}.training.parquet")
    if args.partition == "development":
        query = pd.read_parquet(prepared / f"{args.disease}.development.parquet")
        context = context.loc[context.development_context].copy()
    else:
        fold = int(args.partition)
        query, context = context.loc[context.fold == fold].copy(), context.loc[context.fold != fold].copy()
    out = root / "partitions" / args.disease / (args.partition + args.output_suffix)
    out.mkdir(parents=True, exist_ok=False)
    bank = comparison_bank(args.disease, pd.read_csv(manifest["source_paths"]["mapping"]))

    def save_method(method, frame, selection, fits, parameters):
        frame.to_csv(out / f"{method}.predictions.csv", index=False, float_format="%.17g")
        pd.DataFrame(selection).to_csv(out / "selection.partial.csv", index=False)
        pd.DataFrame(fits).to_csv(out / "fits.partial.csv", index=False)
        write_json(out / "parameters.partial.json", parameters)

    predictions, selection, fits, parameters = evaluate_partition(
        context, query, disease=args.disease, full_ids=manifest["panels"][args.disease]["full_ids"],
        bank=bank, methods=tuple(args.methods), seed=manifest["seed"],
        tabpfn_factory=_tabpfn_factory(args.device, manifest["source_paths"]["checkpoint"]),
        on_method=save_method,
    )
    predictions["partition"] = args.partition
    predictions.to_csv(out / "predictions.csv", index=False, float_format="%.17g")
    selection.to_csv(out / "selection.csv", index=False)
    fits.to_csv(out / "fits.csv", index=False)
    write_json(out / "parameters.json", parameters)
    write_json(out / "COMPLETE.json", {
        "arguments": vars(args), "status": "complete", "n_context": len(context), "n_query": len(query),
        "prepared_manifest_sha256": sha256_file(prepared / "manifest.json"),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "output_sha256": {name: sha256_file(out / name) for name in (
            "predictions.csv", "selection.csv", "fits.csv", "parameters.json")},
    })
    print(f"Complete: {args.disease}, partition={args.partition}, methods={len(args.methods)}", flush=True)


if __name__ == "__main__":
    main()
