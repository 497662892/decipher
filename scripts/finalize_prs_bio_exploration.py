"""Verify every experimental scope and assemble the aggregate bilingual report."""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import zipfile
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit
from xml.etree import ElementTree

import pandas as pd
from run_prs_distillation_bio import verify

from decipehr.external_pgs import sha256_file, write_json
from decipehr.prs_distillation.data import verify_files


def read(path):
    return json.loads(Path(path).read_text())


class OfflineReport(HTMLParser):
    def __init__(self):
        super().__init__()
        self.images = 0
        self.links = []

    def handle_starttag(self, tag, attrs):
        values = dict(attrs)
        if tag == "a" and values.get("href"):
            self.links.append(values["href"])
        if tag == "script":
            raise ValueError("Offline reports must not load scripts")
        if tag == "img":
            src = values.get("src", "")
            assert src.startswith("data:image/svg+xml;base64,")
            ElementTree.fromstring(base64.b64decode(src.split(",", 1)[1], validate=True))
            self.images += 1


def main():
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.run_dir
    _, cfg = verify(root)
    receipts, models, fits, budgets, secondary_models, secondary_fits = {}, 0, 0, [], 0, 0
    for source in [Path(cfg["source_run"]), Path(cfg["v3_run"])]:
        complete = read(source / "COMPLETE.json")
        verify_files(source, complete["output_sha256"])
        receipts[str(source / "COMPLETE.json")] = sha256_file(source / "COMPLETE.json")
    for scope_root in [root, *[root / "nested" / f"outer{i}" for i in range(5)]]:
        verify(scope_root)
        if scope_root != root:
            verify_files(scope_root, read(scope_root / "PREPARED.json")["output_sha256"])
        for trait in cfg["traits"]:
            core = scope_root / "core" / trait
            receipt = read(core / "PREDICTIONS_COMPLETE.json")
            assert receipt["models"] == 72 and not receipt["smoke"] and not receipt["evaluation_labels_read"]
            verify_files(core, receipt["output_sha256"])
            lock = read(core / "SELECTION_LOCK.json")
            assert lock["tuning_fit_count"] == 300 and not lock["selection_uses_val"]
            for arm, choice in lock["choices"].items():
                meta = read(core / "models" / arm / "seed17/metadata.json")
                budgets.append({"scope_root": str(scope_root), "trait": trait, "arm": arm,
                    "stage_steps": choice["steps"],
                    "parent_steps": lock["choices"][choice["parent"]]["steps"] if choice["parent"] else 0,
                    "full_batch_patients": meta["preprocessing"]["fixed_preprocessing"]["fit_n"],
                    "seeds": "17/29/43", "lambda": choice["settings"]["lam"],
                    "learning_rate": choice["settings"]["learning_rate"], "H_loss_mean3": choice["H_loss"]})
            audit = read(core / "MODEL_VERIFICATION.json")["models"]
            assert len(audit) == 72 and all(x["constraints_passed"] and x["save_load_exact"] for x in audit)
            low = scope_root / "low_order" / trait
            low_receipt = read(low / "COMPLETE.json")
            assert low_receipt["models"] == 4 and not low_receipt["evaluation_labels_read"]
            verify_files(low, low_receipt["output_sha256"])
            verify_files(Path.cwd(), read(low / "CONTRACT.json")["code_sha256"])
            assert len(pd.read_csv(low / "tuning_results.csv")) == 40
            assert all(x["saved_replay_exact"] and x["exact_anchored_decomposition"] for x in low_receipt["model_verification"])
            models += receipt["models"]+low_receipt["models"]
            fits += lock["tuning_fit_count"]+40
            for p in [core / "PREDICTIONS_COMPLETE.json", low / "COMPLETE.json", scope_root / "PROTOCOL.json"]:
                receipts[str(p)] = sha256_file(p)
            for prior in ["AGE", "PRS", "FREE"]:
                directory = scope_root / "constraint_ablation" / prior / trait
                receipt = read(directory / "COMPLETE.json")
                assert receipt["models"] == 4 and not receipt["evaluation_labels_read"]
                verify_files(directory, receipt["output_sha256"])
                verify_files(Path.cwd(), read(directory / "CONTRACT.json")["code_sha256"])
                assert len(pd.read_csv(directory / "tuning_results.csv")) == 40
                assert all(x["saved_replay_exact"] and x["exact_anchored_decomposition"] for x in receipt["model_verification"])
                secondary_models += 4
                secondary_fits += 40
                receipts[str(directory / "COMPLETE.json")] = sha256_file(directory / "COMPLETE.json")
        print(f"Verified model scope: {scope_root}", flush=True)
    for backend in ["tabpfn35", "kumo"]:
        directory = Path("outputs/foundation_model_probe_20261026/benchmark") / backend
        receipt = read(directory / "COMPLETE.json")
        verify_files(directory, receipt["output_sha256"])
        assert not receipt["evaluation_labels_read"] and not receipt["contract"]["private_data_sent"]
        receipts[str(directory / "COMPLETE.json")] = sha256_file(directory / "COMPLETE.json")
    assert read(root / "NESTED_ISOLATION_AUDIT.json")["status"] == "passed"
    verify_files(Path.cwd(), read(root / "verification/TESTS.json")["code_sha256"])
    for p in [root / "NESTED_ISOLATION_AUDIT.json", root / "verification/TESTS.json"]:
        receipts[str(p)] = sha256_file(p)
    replay_path = root / "teacher_replay/REPLAY.json"
    replay = read(replay_path)
    assert replay["status"] == "passed" and len(replay["comparisons"]) == 3
    assert all(x["max_absolute_difference"] == 0 for x in replay["comparisons"])
    assert replay["script_sha256"] == sha256_file("scripts/replay_prs_teacher.py")
    verify_files(replay_path.parent, replay["output_sha256"])
    receipts[str(replay_path)] = sha256_file(replay_path)
    for backend in ["tabpfn3", "tabpfn35", "kumo"]:
        directory = root / "constraint_probe" / backend
        receipt = read(directory / "COMPLETE.json")
        verify_files(directory, receipt["output_sha256"])
        assert receipt["contract"]["script_sha256"] == sha256_file("scripts/probe_prs_foundation_constraints.py")
        receipts[str(directory / "COMPLETE.json")] = sha256_file(directory / "COMPLETE.json")
    assert models == 1368 and fits == 6120
    assert secondary_models == 216 and secondary_fits == 2160
    out = root / "report"
    pd.DataFrame(budgets).to_csv(out / "all_parametric_training_budgets.csv", index=False)
    report_qa, summaries = [], []
    for scope in ["val", "nested"]:
        directory = out / scope
        verify_files(directory, read(directory / "VERIFIED.json")["output_sha256"])
        assert read(directory / "EVALUATION.json")["report_script_sha256"] == sha256_file("scripts/report_prs_distillation_bio.py")
        for lang in ["zh", "en"]:
            p = directory / f"bio_distillation_{lang}.html"
            text = p.read_text()
            checker = OfflineReport()
            checker.feed(text)
            assert checker.images == (5 if scope == "val" else 2)
            if lang == "en":
                assert not re.search("[\u4e00-\u9fff]", text)
            report_qa.append({"file": str(p), "embedded_svg_images": checker.images, "offline_assets": True,
                              "english_without_cjk": lang == "en", "sha256": sha256_file(p)})
        with zipfile.ZipFile(directory / "bio_distillation_report.zip") as archive:
            assert archive.testzip() is None
            for name in archive.namelist():
                assert archive.read(name) == (directory / name).read_bytes()
        metrics = pd.read_csv(directory / "metrics.csv")
        summaries.append(metrics.loc[metrics.arm.isin(["BIO_POLICY", "M0_KD", "TabPFN_v3", "TabPFN_v3.5", "Kumo_Tabular"])])
        receipts[str(directory / "VERIFIED.json")] = sha256_file(directory / "VERIFIED.json")
    summary = pd.concat(summaries, ignore_index=True)
    summary.to_csv(out / "primary_comparison.csv", index=False)
    directory = out / "constraint_probe"
    verify_files(directory, read(directory / "VERIFIED.json")["output_sha256"])
    receipts[str(directory / "VERIFIED.json")] = sha256_file(directory / "VERIFIED.json")
    for lang in ["zh", "en"]:
        p = directory / f"constraint_probe_{lang}.html"
        checker = OfflineReport()
        checker.feed(p.read_text())
        assert checker.images == 1
        if lang == "en":
            assert not re.search("[\u4e00-\u9fff]", p.read_text())
        report_qa.append({"file": str(p), "embedded_svg_images": 1, "offline_assets": True,
                          "english_without_cjk": lang == "en", "sha256": sha256_file(p)})
    secondary_summaries = []
    for scope in ["val", "nested"]:
        directory = out / f"constraint_ablation_{scope}"
        verify_files(directory, read(directory / "VERIFIED.json")["output_sha256"])
        assert read(directory / "VERIFIED.json")["script_sha256"] == sha256_file("scripts/report_prs_constraint_ablation.py")
        receipts[str(directory / "VERIFIED.json")] = sha256_file(directory / "VERIFIED.json")
        metrics = pd.read_csv(directory / "metrics.csv")
        secondary_summaries.append(metrics.loc[metrics.arm.isin(["STRICT_SELECTED", "AGE_SELECTED", "PRS_SELECTED", "FREE_SELECTED", "TabPFN_v3", "TabPFN_v3.5", "Kumo_Tabular"])])
        for lang in ["zh", "en"]:
            p = directory / f"constraint_ablation_{lang}.html"
            checker = OfflineReport()
            checker.feed(p.read_text())
            assert checker.images == 1
            if lang == "en":
                assert not re.search("[\u4e00-\u9fff]", p.read_text())
            report_qa.append({"file": str(p), "embedded_svg_images": 1, "offline_assets": True,
                              "english_without_cjk": lang == "en", "sha256": sha256_file(p)})
    secondary_summary = pd.concat(secondary_summaries, ignore_index=True)
    secondary_summary.to_csv(out / "constraint_comparison.csv", index=False)
    directory = out / "relatedness_diagnostic"
    receipt = read(directory / "VERIFIED.json")
    verify_files(directory, receipt["output_sha256"])
    assert receipt["script_sha256"] == sha256_file("scripts/report_prs_relatedness_diagnostic.py")
    assert receipt["private_membership_sha256"] == sha256_file(root / "relatedness_diagnostic/private_membership.parquet")
    assert receipt["no_model_refitting_or_reselection"] and receipt["teacher_replay_exact"]
    receipts[str(directory / "VERIFIED.json")] = sha256_file(directory / "VERIFIED.json")
    relatedness_summary = pd.read_csv(directory / "metrics.csv").loc[lambda x:
        x.stratum.eq("no_known_shared_family") & x.arm.isin(["PRIMARY_SELECTED", "TabPFN_v3", "TabPFN_v3.5", "Kumo_Tabular"])]
    for lang in ["zh", "en"]:
        p = directory / f"relatedness_{lang}.html"
        checker = OfflineReport()
        checker.feed(p.read_text())
        assert checker.images == 1
        if lang == "en":
            assert not re.search("[\u4e00-\u9fff]", p.read_text())
        report_qa.append({"file": str(p), "embedded_svg_images": 1, "offline_assets": True,
                          "english_without_cjk": lang == "en", "sha256": sha256_file(p)})
    for lang in ["zh", "en"]:
        def t(zh, en, lang=lang):
            return zh if lang == "zh" else en
        title = t("PRS 可解释蒸馏实验总览", "Interpretable PRS distillation: experiment overview")
        body = f'<!doctype html><html lang="{lang}"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{title}</title>'
        body += '<style>body{font:16px/1.7 system-ui;color:#16314b;max-width:1150px;margin:35px auto;padding:20px;background:#f5f8fa}section{background:white;padding:25px;margin:20px 0}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:8px;border-bottom:1px solid #dde5ea;text-align:left}a{color:#087f70}.note{background:#fff5df;padding:15px}</style>'
        body += f'<h1>{title}</h1><p>'+t("完成 6,120 次主实验候选拟合与 2,160 次二级约束消融拟合，共 1,584 个最终可解释模型；另完成 TabPFN 3.5 与 Kumo Tabular 的本地对照。",
            "Completed 6,120 primary candidate fits and 2,160 secondary constraint-ablation fits, yielding 1,584 final interpretable models, plus local TabPFN 3.5 and Kumo Tabular comparisons.")+'</p>'
        conclusions = out / "CONCLUSIONS.json"
        if conclusions.exists():
            body += '<section>'+read(conclusions)[lang]+'</section>'
        body += '<p class="note">'+t("结构、软标签权重、正则化和训练步数只在训练内部选择。原始 val 是开发评估；嵌套五折验证每折重新构建 Teacher 软标签并重新选择 Student。两者都不能代替新的外部验证。",
            "Structure, soft-label weight, penalties and steps are selected inside training. Original val is developmental. Nested five-fold evaluation rebuilds teacher soft targets and reselects the student within each outer fold. Neither replaces new external validation.")+'</p>'
        body += '<section><h2>'+t("关键诊断：排除已知同组验证患者后的差距", "Key diagnostic: gap in validation patients without a known shared training family")+'</h2>'
        body += f'<p><a href="relatedness_diagnostic/relatedness_{lang}.html">'+t("查看亲缘分层、差距分解与配对置信区间", "Open family strata, gap decomposition and paired intervals")+'</a></p>'
        body += '<p>'+t("这是冻结模型的事后分层；没有重新训练或按该子集选模型。原 Teacher 的新推理与缓存逐项完全一致。无已知同组不代表所有亲缘关系都已排除。",
            "This is post-hoc stratification of frozen models, with no refitting or subset-based reselection. Fresh original-teacher inference exactly matches the cached predictions. No known shared group does not establish absence of all relatedness.")+'</p>'
        body += relatedness_summary[["trait", "arm", "n", "log_loss", "auroc", "nagelkerke_r2"]].to_html(index=False, float_format=lambda v: f"{v:.4f}", border=0)+'</section>'
        for scope in ["val", "nested"]:
            heading = t("原始验证集", "Original validation") if scope == "val" else t("Train-only 嵌套五折 OOF", "Train-only nested five-fold OOF")
            body += f'<section><h2>{heading}</h2><p><a href="{scope}/bio_distillation_{lang}.html">'+t("打开完整报告：公式、训练、曲线、区间与分组", "Open the full report: equations, training, curves, intervals and subgroups")+'</a></p>'
            m = summary.loc[summary.scope.eq(scope), ["trait", "arm", "n", "log_loss", "auroc", "brier", "nagelkerke_r2"]]
            body += m.to_html(index=False, float_format=lambda v: f"{v:.4f}", border=0)+'</section>'
        body += '<p>'+t("BIO_POLICY 指训练内部选出的结构；不同疾病、不同外折可以选择不同模型。原 Teacher 为 TabPFN v3，新基础模型均为独立对照。所有结果使用直接概率。",
            "BIO_POLICY denotes the train-selected structure; diseases and outer folds may select different models. The teacher remains TabPFN v3, and newer foundation models are independent comparators. All results use direct probabilities.")+'</p>'
        body += f'<p><a href="constraint_probe/constraint_probe_{lang}.html">'+t("探索性诊断：Teacher 是否符合年龄与 PRS 单调先验", "Exploratory diagnostic: compatibility of teacher responses with age and PRS monotonicity")+'</a></p>'
        for scope in ["val", "nested"]:
            body += '<section><h2>'+t(f"二级约束消融：{scope}", f"Secondary constraint ablation: {scope}")+'</h2>'
            body += f'<p><a href="constraint_ablation_{scope}/constraint_ablation_{lang}.html">'+t("打开匹配对照、公式、实际预算与区间", "Open matched comparisons, equations, actual budgets and intervals")+'</a></p>'
            m = secondary_summary.loc[secondary_summary.scope.eq(scope), ["trait", "arm", "log_loss", "auroc", "nagelkerke_r2"]]
            body += m.to_html(index=False, float_format=lambda v: f"{v:.4f}", border=0)+'</section>'
        body += '</html>'
        (out / f"index_{lang}.html").write_text(body)
    for p in out.rglob("*.html"):
        checker = OfflineReport()
        checker.feed(p.read_text())
        for link in checker.links:
            url = urlsplit(link)
            if not url.scheme and url.path:
                assert (p.parent / unquote(url.path)).exists(), (str(p), link)
        if p.name.startswith("index_"):
            if p.name == "index_en.html":
                assert not re.search("[\u4e00-\u9fff]", p.read_text())
            report_qa.append({"file": str(p), "embedded_svg_images": checker.images, "offline_assets": True,
                              "english_without_cjk": p.name == "index_en.html", "sha256": sha256_file(p)})
    for p in out.rglob("*.csv"):
        assert not set(pd.read_csv(p, nrows=0).columns) & {"sampleid", "pat_mrn_id", "MRN", "kinship_group"}, str(p)
    write_json(out / "HTML_QA.json", {"checks": report_qa, "browser_render_tested": False,
        "local_links_exist": True, "csv_identifier_columns_absent": True,
        "method": "HTML parse, embedded SVG XML validation, English text and local link checks, exact archive replay"})
    snapshot_files = set(read(root / "PROTOCOL.json")["identity"]["code_sha256"])
    snapshot_files.update(["scripts/nested_prs_distillation_bio.py", "scripts/benchmark_prs_foundation_models.py",
        "scripts/run_prs_monotone_boosting.py", "scripts/report_prs_distillation_bio.py",
        "scripts/finalize_prs_bio_exploration.py", "src/decipehr/prs_monotone_boosting.py",
        "scripts/replay_prs_teacher.py", "scripts/report_prs_relatedness_diagnostic.py",
        "scripts/probe_prs_foundation_constraints.py", "scripts/report_prs_constraint_probe.py",
        "slurm/prs_foundation_constraint_probe.sbatch",
        "src/decipehr/prs_constraint_ablation.py", "tests/test_prs_constraint_ablation.py",
        "scripts/run_prs_constraint_ablation.py", "scripts/report_prs_constraint_ablation.py",
        "slurm/prs_constraint_ablation.sbatch", "slurm/prs_constraint_ablation_report.sbatch",
        "tests/test_prs_monotone_boosting.py", "tests/test_prs_bio_reporting.py",
        "slurm/prs_bio_nested_teacher.sbatch", "slurm/prs_bio_nested_student.sbatch",
        "slurm/prs_foundation_models.sbatch", "slurm/prs_monotone_boosting.sbatch", "slurm/prs_bio_report.sbatch"])
    for name in sorted(snapshot_files):
        destination = root / "source_snapshot" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and sha256_file(destination) != sha256_file(name):
            raise ValueError(f"Source snapshot changed: {name}")
        destination.write_bytes(Path(name).read_bytes())
    archive_path = out / "bio_exploration_bilingual.zip"
    members = [p for p in out.rglob("*") if p.is_file() and p.suffix != ".zip"]
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for p in members:
            archive.write(p, p.relative_to(out))
    with zipfile.ZipFile(archive_path) as archive:
        assert archive.testzip() is None
        assert all(archive.read(str(p.relative_to(out))) == p.read_bytes() for p in members)
    write_json(root / "COMPLETE.json", {"status": "completed_and_verified", "final_models": models+secondary_models,
        "candidate_fits": fits+secondary_fits, "primary_models": models, "primary_candidate_fits": fits,
        "secondary_models": secondary_models, "secondary_candidate_fits": secondary_fits,
        "main_scopes": 3, "nested_scopes": 15,
        "assessment_used_for_selection": False, "phase_b": False, "all_patient_data_remain_local": True,
        "original_val_role": "developmental", "secondary_design_after_primary_evaluation": True,
        "relatedness_diagnostic_posthoc": True, "external_confirmation_completed": False,
        "upstream_receipt_sha256": receipts, "finalizer_sha256": sha256_file(__file__),
        "source_snapshot_sha256": {str(p.relative_to(root)): sha256_file(p)
                                  for p in (root / "source_snapshot").rglob("*") if p.is_file()},
        "verification": read(root / "verification/TESTS.json"),
        "aggregate_output_sha256": {str(p.relative_to(root)): sha256_file(p) for p in out.rglob("*") if p.is_file()}})
    print(f"Verified complete: {root}; {fits+secondary_fits} candidate fits, {models+secondary_models} final models", flush=True)


if __name__ == "__main__":
    main()
