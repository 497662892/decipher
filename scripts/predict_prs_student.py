"""Predict with saved PRS students; requires no training context or TabPFN checkpoint."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from decipehr.external_pgs import sha256_file, write_json
from decipehr.io import read_table, write_table
from decipehr.prs_student_inference import ARMS, predict_student


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--trait", choices=["T2D", "cad", "bc"], required=True)
    parser.add_argument("--arm", choices=ARMS, default="M0_KD")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--eligibility-config", required=True)
    args = parser.parse_args()
    os.umask(0o077)
    receipt = Path(str(args.output) + ".json")
    if args.output.exists() or receipt.exists():
        raise FileExistsError("Use a new output path; existing predictions are preserved")
    read_options = {} if args.input.suffix == ".parquet" else {"dtype": {"sampleid": str}}
    frame = read_table(args.input, **read_options)
    result = predict_student(args.run_dir, frame, trait=args.trait, arm=args.arm,
                            seed=args.seed, eligibility_config=args.eligibility_config)
    write_table(result, args.output)
    write_json(receipt, {
        "status": "prediction_complete", "trait": args.trait,
        "requested_arm": args.arm, "resolved_arm": result.arm.iloc[0], "seed": args.seed,
        "n_input": len(frame), "n_predicted": len(result), "n_ineligible": len(frame)-len(result),
        "input_sha256": sha256_file(args.input), "output_sha256": sha256_file(args.output),
        "eligibility_sha256": sha256_file(args.eligibility_config),
        **result.attrs,
        "labels_used": False, "new_fits": 0, "tabpfn_checkpoint_required": False,
    })
    print(f"Predicted {len(result)} eligible patients; excluded {len(frame)-len(result)}; {args.output}")


if __name__ == "__main__":
    main()
