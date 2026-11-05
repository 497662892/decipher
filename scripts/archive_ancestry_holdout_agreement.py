"""Audit the completed holdout comparison and bundle its aggregate-only report."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from run_ancestry_holdout_agreement import BASELINE, SCOPES, saved_student
from run_ancestry_transfer import TRAITS, read

from decipehr.external_pgs import sha256_file, write_json
from decipehr.prs_distillation.data import verify_files
from decipehr.prs_distillation.preprocessing import frame_hash


def independent_statistics(x, y, row):
    """Read back point estimates using direct NumPy/Pandas formulas."""
    values = {
        "pearson_probability": np.corrcoef(x, y)[0, 1],
        "spearman": np.corrcoef(pd.Series(x).rank(), pd.Series(y).rank())[0, 1],
        "mae_probability": np.abs(y-x).mean(),
        "mean_student_minus_teacher": (y-x).mean(),
        "rmse_probability": np.sqrt(np.square(y-x).mean()),
        "mean_teacher": x.mean(), "mean_student": y.mean(),
    }
    for key, value in values.items():
        np.testing.assert_allclose(value, row[key], rtol=0, atol=1e-12)


def main():
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    root = parser.parse_args().out_dir
    report = root / "report"
    prepared, measured = read(root / "PREPARED.json"), read(root / "VERIFIED.json")
    verify_files(Path.cwd(), prepared["input_sha256"])
    verify_files(root, measured["output_sha256"])
    verify_files(Path.cwd(), measured["source_sha256"])
    baseline = read(BASELINE / "VERIFIED.json")
    verify_files(BASELINE, baseline["output_sha256"])
    verify_files(root / "projection", read(root / "projection/COMPLETE.json")["output_sha256"])
    projection = read(root / "projection/PROTOCOL.json")
    verify_files(Path.cwd(), projection["input_sha256"])
    for name, previous in projection["genotype_files"].items():
        stat = Path(name).stat()
        assert stat.st_size == previous["bytes"] and stat.st_mtime_ns == previous["mtime_ns"]
    browser = read(report / "assets/qa/BROWSER_VERIFIED.json")
    visual = read(root / "VISUAL_REVIEW.json")
    html_sha = sha256_file(report / "report_zh.html")
    assert browser["status"] == visual["status"] == "passed"
    assert browser["html_sha256"] == visual["html_sha256"] == html_sha
    verify_files(root, visual["image_sha256"])
    assert browser["filter_combinations"] == 4 and not browser["external_requests"] and not browser["page_errors"]
    metrics = pd.read_csv(report / "agreement.csv")
    stability = pd.read_csv(report / "cross_scope_stability.csv")
    available = pd.read_csv(root / "availability.csv").set_index("trait")
    assignments = pd.read_parquet(root / "private/trainval_folds.parquet").set_index("sampleid")
    scopes, replay = [], []
    all_holdout_ids = set()
    for trait in TRAITS:
        folder = root / "private" / trait
        holdout = pd.read_parquet(folder / "holdout.parquet")
        assert not set(holdout.columns) & {"y", *TRAITS}
        assert holdout.sampleid.is_unique and len(holdout) == available.loc[trait, "n_holdout"]
        if trait == "bc":
            assert holdout.sex.eq("Female").all()
        all_holdout_ids.update(holdout.sampleid)
        query_ids = sorted(holdout.sampleid)
        by_scope = {}
        for scope in SCOPES:
            dest = folder / scope
            receipt = read(dest / "COMPLETE.json")
            assert sha256_file(dest / "COMPLETE.json") == measured["prediction_receipts_sha256"][f"{trait}/{scope}"]
            verify_files(dest, receipt["output_sha256"])
            assert not receipt["holdout_outcomes_used"] and not receipt["new_hyperparameter_search"]
            name = "train" if scope == "train_only" else "trainval"
            context = pd.read_parquet(folder / f"{name}.parquet")
            assert len(context) == available.loc[trait, f"n_{name}"]
            assert context.sampleid.is_unique and not set(context.sampleid) & set(query_ids)
            assert receipt["context_hash"] == frame_hash(context) and receipt["query_hash"] == frame_hash(holdout)
            p = pd.read_parquet(dest / "predictions.parquet")
            assert p.trait.eq(trait).all() and p.scope.eq(scope).all()
            assert not p.duplicated(["sampleid", "arm"]).any()
            wide = p.pivot(index="sampleid", columns="arm", values="probability").sort_index()
            assert list(wide.index) == query_ids and not wide.isna().any().any()
            assert set(wide) == {"TabPFN_v3", "M0_KD", "H_SELECTED"}
            assert np.isfinite(wide.to_numpy()).all() and wide.ge(0).all().all() and wide.le(1).all().all()
            if trait == "T2D":
                np.testing.assert_array_equal(wide.M0_KD, wide.H_SELECTED)
            by_scope[scope] = wide
            for arm in ["M0_KD", "H_SELECTED"]:
                row = metrics.loc[metrics.trait.eq(trait) & metrics.scope.eq(scope) & metrics.student.eq(arm)].squeeze()
                x, y = wide.TabPFN_v3.to_numpy(), wide[arm].to_numpy()
                independent_statistics(x, y, row)
                k = int(row.top_decile_n)
                a = set(wide.TabPFN_v3.nlargest(k, keep="first").index)
                b = set(wide[arm].nlargest(k, keep="first").index)
                assert len(a & b) == row.top_decile_intersection
                np.testing.assert_allclose(len(a & b)/k, row.top_decile_overlap, rtol=0, atol=1e-12)
                resolved = "M0_KD" if arm == "M0_KD" else receipt["selected_student"]
                model_root = BASELINE / "private" / trait / "reference_pc/final" if scope == "train_only" else dest
                fresh = saved_student(model_root / "models" / resolved, holdout, resolved)
                expected = wide.loc[holdout.sampleid, arm].to_numpy()
                np.testing.assert_allclose(fresh, expected, rtol=0, atol=1e-12)
                replay.append({"trait": trait, "scope": scope, "arm": arm,
                               "max_absolute_error": float(np.abs(fresh-expected).max())})
            if scope == "trainval_refit":
                folds = assignments.loc[context.sampleid, "fold"].to_numpy()
                for fold in range(5):
                    cached = read(dest / "teacher_oof" / str(fold) / "cache.json")["fingerprint"]
                    assert cached["require_group_disjoint"] and cached["shared_kinship_groups"] == 0
                    used = pd.read_parquet(dest / "teacher_oof" / str(fold) / "scope.parquet")
                    assert set(used.loc[used.role.eq("query"), "sampleid"]) == set(context.loc[folds == fold, "sampleid"])
                    assert set(used.loc[used.role.eq("context"), "sampleid"]) == set(context.loc[folds != fold, "sampleid"])
            scopes.append({key: receipt[key] for key in ["trait", "scope", "n_context", "n_holdout", "gpu", "slurm_job_id"]})
        for arm in ["TabPFN_v3", "M0_KD", "H_SELECTED"]:
            row = stability.loc[stability.trait.eq(trait) & stability.model.eq(arm)].squeeze()
            independent_statistics(by_scope["train_only"][arm].to_numpy(), by_scope["trainval_refit"][arm].to_numpy(), row)
    assert len(all_holdout_ids) == 3922 and len(scopes) == 6 and len(metrics) == 12 and len(stability) == 9
    scheduler = read(root / "SCHEDULER.json")
    job_ids = [scheduler[k] for k in ["projection", "predict_t2d_cad", "predict_bc", "report"]]
    account = subprocess.run(["sacct", "-j", ",".join(job_ids), "-X", "-P", "-n",
        "--format=JobID%64,State,ExitCode,Partition,Elapsed"], check=True, capture_output=True, text=True).stdout
    jobs = [dict(zip(["job_id", "state", "exit_code", "partition", "elapsed"], line.split("|"), strict=True))
            for line in account.splitlines() if line.strip()]
    assert len(jobs) == 5 and all(j["state"] == "COMPLETED" and j["exit_code"] == "0:0" for j in jobs), jobs
    write_json(root / "FINAL_REVIEW.json", {
        "status": "completed_and_verified", "holdout_patients": len(all_holdout_ids),
        "prediction_scopes": scopes, "frozen_source_report_files_unchanged": len(baseline["output_sha256"]),
        "input_hashes_rechecked": len(prepared["input_sha256"]), "holdout_outcomes_used": False,
        "point_estimates_independently_recomputed": True, "combined_oof_scope_audit": "passed",
        "student_prediction_replay": replay, "slurm_accounting": jobs,
        "checks": read(root / "CHECKS.json"), "report_sha256": html_sha,
        "browser_receipt_sha256": sha256_file(report / "assets/qa/BROWSER_VERIFIED.json"),
        "visual_review_sha256": sha256_file(root / "VISUAL_REVIEW.json"),
        "m0_results": json.loads(metrics.loc[metrics.student.eq("M0_KD")].to_json(orient="records", double_precision=15)),
        "limits": ["Prediction agreement does not measure outcome accuracy or calibration",
                   "Patient bootstrap conditions on fitted predictions and does not model unknown holdout kinship",
                   "Whole genotype files verified by size/mtime plus exact sample/marker coverage, not a new full-file hash"],
    })
    snapshots = [Path(__file__), Path("scripts/verify_ancestry_holdout_report.cjs"),
                 Path("tests/test_prediction_agreement.py"), *sorted(Path("slurm").glob("ancestry_holdout_*.sbatch"))]
    for source in snapshots:
        (root / "source_snapshot" / source.name).write_bytes(source.read_bytes())
    (root / "README.txt").write_text(
        "Holdout 预测一致性：新的标准参考 PC，train-only 与 train+validation 两套模型。\n\n"
        "打开 report/report_zh.html 即可离线浏览；训练范围和学生选项会同步切换统计与图形。\n"
        "T2D/CAD 各 3922 人；女性 BC 2187 人。模型间及训练范围间均为同一批患者。\n"
        "M0_KD 是统一的蒸馏学生；既定方案为 T2D=M0_KD、CAD=HGB_PAIR_KD、BC=G_A6_KD。\n"
        "T2D 两个学生选项相同。合并训练使用共享五折 OOF 标签与原定超参数，不重新选模型。\n"
        "Pearson 与 Spearman 分别描述概率线性相关与排序；MAE 单位 pp=概率百分点。\n"
        "最高 10% 重叠=两模型各自最高 ceil(0.1*n) 人的交集人数/该人数。\n"
        "Holdout 无结局标签，相关性不是 AUROC 或疾病校准性能。\n"
        "1000 次患者配对 bootstrap 固定模型，未包含重训练不确定性或未知亲缘依赖。\n\n"
        "report/ 包含汇总 CSV、8 份 SVG 与 8 份 PDF。报告自带图形与下载数据，无外部请求。\n"
        "VERIFIED.json、FINAL_REVIEW.json 与浏览器回执记录计算、来源和检查；source_snapshot/ 保存脚本。\n"
        "本压缩包只含汇总报告、脚本和回执；逐人预测、患者清单、基因型与模型留在原始受限目录。\n"
    )
    members = [root / name for name in ["README.txt", "PROTOCOL.json", "PREPARED.json", "VERIFIED.json",
               "FINAL_REVIEW.json", "CHECKS.json", "SCHEDULER.json", "VISUAL_REVIEW.json", "availability.csv"]]
    members += [root / "projection" / name for name in ["COMPLETE.json", "PROTOCOL.json"]]
    members += [report / "assets/qa/BROWSER_VERIFIED.json", *sorted((root / "source_snapshot").glob("*"))]
    members += [root / name for name in visual["image_sha256"]]
    members += [root / name for name in measured["output_sha256"]]
    members = sorted(set(members))
    assert all(p.is_file() and "private" not in p.relative_to(root).parts and p.suffix != ".parquet" for p in members)
    manifest = {str(p.relative_to(root)): sha256_file(p) for p in members}
    write_json(root / "BUNDLE_MANIFEST.json", manifest)
    members.append(root / "BUNDLE_MANIFEST.json")
    archive = root / "ancestry_holdout_agreement_zh.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in members:
            z.write(p, str(p.relative_to(root)))
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None and set(z.namelist()) == {*manifest, "BUNDLE_MANIFEST.json"}
        for name, expected in manifest.items():
            assert hashlib.sha256(z.read(name)).hexdigest() == expected
    write_json(root / "ARCHIVE_VERIFIED.json", {"status": "passed", "archive": archive.name,
        "sha256": sha256_file(archive), "members": len(members), "contains_individual_predictions": False,
        "contains_models_or_genotypes": False, "aggregate_only": True})
    print(json.dumps({"status": "completed_and_verified", "comparisons": len(metrics), "jobs": jobs,
                      "archive": str(archive)}, indent=2))


if __name__ == "__main__":
    main()
