"""Fresh local V3 inference to audit frozen validation teacher-cache reproducibility."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from decipehr.external_pgs import sha256_file, write_json
from decipehr.prs_distillation.config import load_config
from decipehr.prs_distillation.data import verify_files
from decipehr.prs_distillation.experiment import load_inputs
from decipehr.prs_distillation.teacher import make_teacher, teacher_predict


def main():
    os.umask(0o077)
    torch.set_num_threads(1)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True)
    args = p.parse_args()
    protocol = json.loads((args.run_dir / "PROTOCOL.json").read_text())
    source = Path(protocol["identity"]["config"]["source_run"])
    cfg = load_config(source / "resolved_config.yaml")
    out = args.run_dir / "teacher_replay"
    rows = []
    for trait in cfg.traits:
        d, e, manifest = load_inputs(source, trait)
        contract = json.loads((source / "core" / trait / "teacher_contract.json").read_text())
        verify_files(Path.cwd(), contract["implementation_sha256"])
        assert sha256_file(contract["checkpoint"]) == contract["checkpoint_sha256"]
        q = teacher_predict(d, e, manifest.column.tolist(), directory=out / trait, contract=contract,
            trait=trait, manifest_hash=sha256_file(source / trait / "feature_manifest.tsv"),
            fold_id="D_to_E", factory=lambda: make_teacher(cfg))
        cached = np.load(args.run_dir / "inputs" / trait / "targets.npz")["q_E"]
        delta = np.abs(q-cached)
        row = {"trait": trait, "n": len(q), "mean_absolute_difference": float(delta.mean()),
               "max_absolute_difference": float(delta.max()), "within_5e5": bool((delta <= 5e-5).all()),
               "evaluation_labels_used": False}
        rows.append(row)
        print(row, flush=True)
    write_json(out / "REPLAY.json", {"status": "passed" if all(x["within_5e5"] for x in rows) else "mismatch",
        "comparisons": rows, "script_sha256": sha256_file(__file__), "original_caches_modified": False,
        "output_sha256": {str(p.relative_to(out)): sha256_file(p) for p in out.rglob("*") if p.is_file()}})


if __name__ == "__main__":
    main()
