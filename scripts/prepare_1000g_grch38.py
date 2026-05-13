#!/usr/bin/env python3
"""Download and convert the 1000G GRCh38 reference for ancestry PCA."""

from __future__ import annotations

import argparse
import shlex
import shutil
import subprocess
import urllib.request
from pathlib import Path
from typing import Any

from decipehr.onekg import (
    DEFAULT_1000G_GRCH38_RELEASE_URL,
    DEFAULT_1000G_MANIFEST_NAME,
    DEFAULT_1000G_PANEL_URL,
    DEFAULT_1000G_README_NAME,
    build_plink_convert_command,
    build_plink_merge_command,
    normalize_1000g_panel,
    one_kg_vcf_name,
    parse_chromosomes,
    pfile_paths,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/resources/1000g/grch38"),
        help="Directory for raw downloads, normalized panel, and PLINK2 output.",
    )
    parser.add_argument("--chromosomes", default="1-22", help="Autosome list, e.g. 1-22 or 1,2,5-8.")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--plink2", default="plink2")
    parser.add_argument("--release-url", default=DEFAULT_1000G_GRCH38_RELEASE_URL)
    parser.add_argument("--panel-url", default=DEFAULT_1000G_PANEL_URL)
    parser.add_argument(
        "--download-tool",
        choices=["auto", "wget", "curl", "python"],
        default="auto",
        help="Downloader for remote files. auto prefers wget, then curl, then Python urllib.",
    )
    parser.add_argument("--skip-download", action="store_true", help="Use files already in downloads/.")
    parser.add_argument("--download-only", action="store_true", help="Download files but do not run PLINK2.")
    parser.add_argument("--panel-only", action="store_true", help="Only download and normalize the sample panel.")
    parser.add_argument("--force-download", action="store_true", help="Re-download files even if present.")
    parser.add_argument("--force-convert", action="store_true", help="Re-run PLINK2 conversion/merge.")
    parser.add_argument(
        "--keep-all-samples",
        action="store_true",
        help=(
            "Do not restrict the VCF import to samples in the 1000G Phase 3 panel. "
            "Default is to keep only samples with super_pop labels."
        ),
    )
    parser.add_argument("--skip-merge", action="store_true", help="Do not merge chromosome PGENs.")
    parser.add_argument("--merged-prefix-name", default="1000g_GRCh38")
    return parser.parse_args()


def file_ready(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def pfile_ready(prefix: Path) -> bool:
    return all(file_ready(path) for path in pfile_paths(prefix))


def choose_download_tool(requested: str) -> str:
    if requested != "auto":
        return requested
    if shutil.which("wget") is not None:
        return "wget"
    if shutil.which("curl") is not None:
        return "curl"
    return "python"


def run_command(command: list[str]) -> None:
    print("+ " + shlex.join(command), flush=True)
    subprocess.run(command, check=True)


def download_file(url: str, out: Path, *, tool: str, force: bool) -> None:
    if file_ready(out) and not force:
        print(f"Using existing {out}", flush=True)
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url} -> {out}", flush=True)

    if tool == "wget":
        run_command(["wget", "--continue", "--tries", "3", "--timeout", "60", "-O", str(out), url])
    elif tool == "curl":
        run_command(
            ["curl", "-L", "--retry", "3", "--connect-timeout", "60", "-C", "-", "-o", str(out), url]
        )
    elif tool == "python":
        tmp = out.with_name(out.name + ".tmp")
        urllib.request.urlretrieve(url, tmp)
        tmp.replace(out)
    else:
        raise ValueError(f"Unknown download tool: {tool}")


def write_sources(
    *,
    out: Path,
    release_url: str,
    panel_url: str,
    chromosomes: list[int],
) -> None:
    rows = [
        ("grch38_release_dir", release_url),
        ("sample_panel", panel_url),
        ("release_manifest", f"{release_url.rstrip('/')}/{DEFAULT_1000G_MANIFEST_NAME}"),
        ("release_readme", f"{release_url.rstrip('/')}/{DEFAULT_1000G_README_NAME}"),
    ]
    for chrom in chromosomes:
        vcf_name = one_kg_vcf_name(chrom)
        rows.append((f"chr{chrom}_vcf", f"{release_url.rstrip('/')}/{vcf_name}"))
        rows.append((f"chr{chrom}_vcf_index", f"{release_url.rstrip('/')}/{vcf_name}.tbi"))

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        handle.write("resource\turl\n")
        for resource, url in rows:
            handle.write(f"{resource}\t{url}\n")


def main() -> None:
    args = parse_args()
    chromosomes = parse_chromosomes(args.chromosomes)
    download_tool = choose_download_tool(args.download_tool)

    out_dir = args.out_dir
    downloads_dir = out_dir / "downloads"
    pgen_dir = out_dir / "pgen"
    panel_raw = downloads_dir / Path(args.panel_url).name
    panel_out = out_dir / "1000g_panel.tsv"
    keep_out = out_dir / "1000g_panel.keep"
    sources_out = out_dir / "1000g_GRCh38.sources.tsv"
    summary_out = out_dir / "1000g_GRCh38.resources.summary.json"
    merged_prefix = out_dir / args.merged_prefix_name
    merge_list = pgen_dir / "1000g_GRCh38.pmerge_list.txt"

    out_dir.mkdir(parents=True, exist_ok=True)
    downloads_dir.mkdir(parents=True, exist_ok=True)
    pgen_dir.mkdir(parents=True, exist_ok=True)

    write_sources(
        out=sources_out,
        release_url=args.release_url,
        panel_url=args.panel_url,
        chromosomes=chromosomes,
    )

    if not args.skip_download:
        download_file(args.panel_url, panel_raw, tool=download_tool, force=args.force_download)
        if not args.panel_only:
            for name in [DEFAULT_1000G_MANIFEST_NAME, DEFAULT_1000G_README_NAME]:
                download_file(
                    f"{args.release_url.rstrip('/')}/{name}",
                    downloads_dir / name,
                    tool=download_tool,
                    force=args.force_download,
                )

    if not file_ready(panel_raw):
        raise SystemExit(f"Missing raw 1000G panel file: {panel_raw}")
    panel_summary = normalize_1000g_panel(panel_raw, panel_out, keep_out)

    chromosome_prefixes: list[Path] = []
    downloaded_vcfs: list[Path] = []
    if not args.panel_only:
        for chrom in chromosomes:
            vcf_name = one_kg_vcf_name(chrom)
            vcf_path = downloads_dir / vcf_name
            tbi_path = downloads_dir / f"{vcf_name}.tbi"
            if not args.skip_download:
                download_file(
                    f"{args.release_url.rstrip('/')}/{vcf_name}",
                    vcf_path,
                    tool=download_tool,
                    force=args.force_download,
                )
                download_file(
                    f"{args.release_url.rstrip('/')}/{vcf_name}.tbi",
                    tbi_path,
                    tool=download_tool,
                    force=args.force_download,
                )
            if not file_ready(vcf_path):
                raise SystemExit(f"Missing 1000G VCF: {vcf_path}")
            downloaded_vcfs.append(vcf_path)

            out_prefix = pgen_dir / f"chr{chrom}.1000g_GRCh38"
            chromosome_prefixes.append(out_prefix)
            if not args.download_only:
                if pfile_ready(out_prefix) and not args.force_convert:
                    print(f"Using existing {out_prefix}.pgen/.pvar/.psam", flush=True)
                else:
                    keep_file = None if args.keep_all_samples else keep_out
                    command = build_plink_convert_command(
                        plink2=args.plink2,
                        vcf=vcf_path,
                        out_prefix=out_prefix,
                        threads=args.threads,
                        keep_file=keep_file,
                    )
                    run_command(command)

    if chromosome_prefixes:
        with merge_list.open("w", encoding="utf-8") as handle:
            for prefix in chromosome_prefixes:
                handle.write(str(prefix) + "\n")

    if chromosome_prefixes and not args.download_only and not args.panel_only and not args.skip_merge:
        missing = [str(prefix) for prefix in chromosome_prefixes if not pfile_ready(prefix)]
        if missing:
            raise SystemExit(f"Cannot merge because chromosome PGENs are missing: {missing[:5]}")
        if pfile_ready(merged_prefix) and not args.force_convert:
            print(f"Using existing {merged_prefix}.pgen/.pvar/.psam", flush=True)
        else:
            run_command(
                build_plink_merge_command(
                    plink2=args.plink2,
                    merge_list=merge_list,
                    out_prefix=merged_prefix,
                    threads=args.threads,
                )
            )

    summary: dict[str, Any] = {
        "release_url": args.release_url,
        "panel_url": args.panel_url,
        "chromosomes": chromosomes,
        "download_tool": download_tool,
        "restrict_to_panel_samples": not args.keep_all_samples,
        "out_dir": str(out_dir),
        "downloads_dir": str(downloads_dir),
        "sources": str(sources_out),
        "raw_panel": str(panel_raw),
        "panel": panel_summary,
        "vcfs": [str(path) for path in downloaded_vcfs],
        "chromosome_pgen_prefixes": [str(prefix) for prefix in chromosome_prefixes],
        "merge_list": str(merge_list) if chromosome_prefixes else None,
        "merged_prefix": str(merged_prefix) if chromosome_prefixes and not args.skip_merge else None,
        "mode": {
            "panel_only": bool(args.panel_only),
            "download_only": bool(args.download_only),
            "skip_download": bool(args.skip_download),
            "skip_merge": bool(args.skip_merge),
        },
    }
    write_json(summary_out, summary)

    print(f"1000G resource summary: {summary_out}", flush=True)
    print(f"Normalized 1000G panel: {panel_out}", flush=True)
    if chromosome_prefixes and not args.skip_merge:
        print(f"Merged PLINK2 prefix: {merged_prefix}", flush=True)


if __name__ == "__main__":
    main()
