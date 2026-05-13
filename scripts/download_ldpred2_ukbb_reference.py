#!/usr/bin/env python3
"""Download the public UKB European LD reference used by LDpred2."""

from __future__ import annotations

import argparse
import http.cookiejar
import re
import shutil
import tarfile
import tempfile
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

HM3_PLUS_MAP_URL = "https://figshare.com/ndownloader/files/37802721"
HM3_PLUS_GDRIVE_ID = "17dyKGA2PZjMsivlYb_AjDmuZjM1RIGvs"


def html_hidden_value(body: bytes, name: str) -> str | None:
    match = re.search(
        rb'name="' + re.escape(name.encode("ascii")) + rb'"\s+value="([^"]+)"',
        body,
    )
    return match.group(1).decode("utf-8") if match else None


def html_form_action(body: bytes) -> str:
    match = re.search(rb'<form[^>]+id="download-form"[^>]+action="([^"]+)"', body)
    if match:
        return match.group(1).decode("utf-8")
    return "https://drive.usercontent.google.com/download"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map-out", type=Path, default=Path("data/resources/ldpred2/map_hm3_plus.rds"))
    parser.add_argument("--corr-dir", type=Path, default=Path("data/resources/ldpred2/corr_hm3_plus"))
    parser.add_argument(
        "--archive-out",
        type=Path,
        default=Path("data/resources/ldpred2/corr_hm3_plus_archive"),
        help="Archive path prefix. The detected suffix is added if missing.",
    )
    parser.add_argument("--force", action="store_true", help="Re-download files even if outputs already exist.")
    return parser.parse_args()


def retrieve_url(url: str, out: Path, force: bool) -> None:
    if out.exists() and out.stat().st_size > 0 and not force:
        print(f"Using existing {out}")
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    print(f"Downloading {url} -> {out}")
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(out)


def google_drive_download(file_id: str, out_prefix: Path, force: bool) -> Path:
    existing = sorted(out_prefix.parent.glob(out_prefix.name + ".*"))
    existing = [path for path in existing if path.is_file() and path.stat().st_size > 0]
    if existing and not force:
        print(f"Using existing {existing[0]}")
        return existing[0]

    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    cookie_jar = http.cookiejar.CookieJar()
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cookie_jar))
    base_url = "https://drive.google.com/uc"
    params = {"export": "download", "id": file_id}

    response = opener.open(f"{base_url}?{urllib.parse.urlencode(params)}")
    content_type = response.headers.get("Content-Type", "").lower()
    token = None
    action_url = base_url
    if "text/html" in content_type:
        body = response.read()
        token = next(
            (
                cookie.value
                for cookie in cookie_jar
                if cookie.name.startswith("download_warning")
            ),
            None,
        )
        if token is None:
            match = re.search(rb"confirm=([0-9A-Za-z_]+)", body)
            token = match.group(1).decode("ascii") if match else None
        if token is None:
            token = html_hidden_value(body, "confirm")
        action_url = html_form_action(body)
        uuid = html_hidden_value(body, "uuid")
        if uuid is not None:
            params["uuid"] = uuid

    if token is not None:
        params["confirm"] = token
        response = opener.open(f"{action_url}?{urllib.parse.urlencode(params)}")
    elif "text/html" in content_type:
        raise SystemExit("Google Drive did not return a downloadable archive or confirmation token.")

    body_iter = iter(lambda: response.read(1024 * 1024), b"")
    content_type = response.headers.get("Content-Type", "")
    suffix = ".zip"
    if "gzip" in content_type or "tar" in content_type:
        suffix = ".tar.gz"
    out = out_prefix if out_prefix.suffix else out_prefix.with_suffix(suffix)
    tmp = out.with_suffix(out.suffix + ".tmp")
    print(f"Downloading Google Drive file {file_id} -> {out}")
    with tmp.open("wb") as handle:
        for chunk in body_iter:
            if chunk:
                handle.write(chunk)
    tmp.replace(out)
    return out


def extract_archive(archive: Path, corr_dir: Path, force: bool) -> None:
    existing = sorted(corr_dir.glob("LD_with_blocks_chr*.rds"))
    if len(existing) >= 22 and not force:
        print(f"Using existing LD matrices in {corr_dir}")
        return

    corr_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=str(corr_dir.parent)) as tmp_name:
        tmp_dir = Path(tmp_name)
        if zipfile.is_zipfile(archive):
            print(f"Extracting {archive}")
            with zipfile.ZipFile(archive) as zf:
                zf.extractall(tmp_dir)
        elif tarfile.is_tarfile(archive):
            print(f"Extracting {archive}")
            with tarfile.open(archive) as tf:
                tf.extractall(tmp_dir)
        else:
            raise SystemExit(f"Downloaded file is not a zip/tar archive: {archive}")

        files = sorted(tmp_dir.rglob("LD_with_blocks_chr*.rds"))
        if len(files) < 22:
            raise SystemExit(f"Expected 22 LD matrix files, found {len(files)} in {archive}")
        for src in files:
            dst = corr_dir / src.name
            if dst.exists() and not force:
                continue
            shutil.move(str(src), dst)
            print(f"Wrote {dst}")


def main() -> None:
    args = parse_args()
    retrieve_url(HM3_PLUS_MAP_URL, args.map_out, args.force)
    archive = google_drive_download(HM3_PLUS_GDRIVE_ID, args.archive_out, args.force)
    extract_archive(archive, args.corr_dir, args.force)
    print("LDpred2 UKB HM3+ reference is ready.")


if __name__ == "__main__":
    main()
