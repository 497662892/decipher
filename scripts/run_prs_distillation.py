"""Audited Phase A fixed-PRS prototype distillation, using local TabPFN only."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from decipehr.prs_distillation.config import load_config


def main():
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["audit", "smoke", "run-core", "ablate", "report", "evaluate-official"])
    parser.add_argument("--config", default="configs/prs_distillation_v1.yaml")
    parser.add_argument("--eligibility-config", required=True)
    parser.add_argument("--run-dir")
    parser.add_argument("--trait", choices=["T2D", "cad", "bc"])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.command == "evaluate-official":
        parser.error("Phase B is not implemented; direct diagnostics are not official Stage-2 results")
    if args.trait is not None and args.trait not in cfg.traits:
        parser.error("Trait is outside the configured scope")
    if args.dry_run:
        print(json.dumps({"command": args.command, "config": cfg.to_dict(), "run_dir": args.run_dir,
                          "eligibility_config": args.eligibility_config, "trait": args.trait,
                          "dry_run": True}, indent=2))
        return
    if args.command == "audit":
        from decipehr.prs_distillation.data import prepare_run
        root, audit = prepare_run(cfg, args.eligibility_config)
        print(json.dumps({"run_dir": str(root), "status": audit["status"], "blockers": audit["blockers"]}))
        if audit["blockers"]:
            raise SystemExit(2)
        return
    if not args.run_dir or not args.trait:
        parser.error("Running/reporting requires --run-dir and --trait")
    audit = json.loads((Path(args.run_dir) / "audit_report.json").read_text())
    if str(Path(args.eligibility_config)) != audit["paths"]["eligibility"]:
        parser.error("Eligibility configuration differs from the resolved audit")
    if args.command == "report":
        from decipehr.prs_distillation.reporting import report
        metrics, _ = report(args.run_dir, cfg, args.trait)
        print(metrics.to_string(index=False))
    else:
        from decipehr.prs_distillation.experiment import run_ablations, run_core
        if args.command == "ablate":
            out = run_ablations(args.run_dir, cfg, args.trait)
        else:
            smoke = args.command == "smoke"
            out = run_core(args.run_dir, cfg, args.trait, smoke=smoke)
            if smoke:
                from decipehr.prs_distillation.reporting import report
                report(args.run_dir, cfg, args.trait, smoke=True)
        print(f"Completed {args.command}: {out}")


if __name__ == "__main__":
    main()
