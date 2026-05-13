#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

TOKEN_PATHS = (
    Path.home() / ".cache" / "tabpfn" / "auth_token",
    Path.home() / ".tabpfn" / "token",
)


def _package_version(package: str) -> str | None:
    try:
        return version(package)
    except PackageNotFoundError:
        return None


def _print_status(name: str, ok: bool, detail: str) -> None:
    prefix = "ok" if ok else "missing"
    print(f"{prefix}: {name}: {detail}")


def _has_cached_token() -> bool:
    return any(path.is_file() and path.stat().st_size > 0 for path in TOKEN_PATHS)


def _configure_environment(args: argparse.Namespace) -> None:
    if args.model_cache_dir:
        os.environ["TABPFN_MODEL_CACHE_DIR"] = str(Path(args.model_cache_dir).expanduser())
    if not args.allow_browser:
        os.environ.setdefault("TABPFN_NO_BROWSER", "1")
    os.environ.setdefault("TABPFN_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    if "PYTORCH_ALLOC_CONF" not in os.environ and "PYTORCH_CUDA_ALLOC_CONF" in os.environ:
        os.environ["PYTORCH_ALLOC_CONF"] = os.environ["PYTORCH_CUDA_ALLOC_CONF"]
        os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)


def _check_packages() -> bool:
    ok = True
    for package in ("tabpfn", "torch"):
        found = _package_version(package)
        _print_status(package, found is not None, found or "not installed")
        ok = ok and found is not None
    return ok


def _check_torch() -> bool:
    try:
        import torch
    except Exception as exc:  # noqa: BLE001
        _print_status("torch import", False, f"{type(exc).__name__}: {exc}")
        return False

    detail = (
        f"version={torch.__version__}, cuda_build={torch.version.cuda}, "
        f"cuda_available={torch.cuda.is_available()}"
    )
    _print_status("torch import", True, detail)
    return True


def _check_tabpfn_cache() -> bool:
    try:
        from tabpfn.model_loading import get_cache_dir
    except Exception as exc:  # noqa: BLE001
        _print_status("TabPFN cache", False, f"{type(exc).__name__}: {exc}")
        return False

    cache_dir = get_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)
    _print_status("TabPFN cache", True, str(cache_dir))
    return True


def _check_auth(require_auth: bool) -> bool:
    token_env = bool(os.environ.get("TABPFN_TOKEN", "").strip())
    token_file = _has_cached_token()
    if token_env:
        _print_status("TabPFN auth", True, "TABPFN_TOKEN is set")
        return True
    if token_file:
        _print_status("TabPFN auth", True, "cached token file exists")
        return True
    _print_status("TabPFN auth", not require_auth, "no token found")
    return not require_auth


def _verify_license(hf_repo_id: str) -> bool:
    try:
        from tabpfn.browser_auth import ensure_license_accepted

        ensure_license_accepted(hf_repo_id=hf_repo_id)
    except Exception as exc:  # noqa: BLE001
        _print_status("TabPFN license", False, f"{type(exc).__name__}: {exc}")
        return False
    _print_status("TabPFN license", True, f"accepted for {hf_repo_id}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Check local TabPFN installation and auth wiring.")
    parser.add_argument("--model-cache-dir", default=None)
    parser.add_argument("--require-auth", action="store_true")
    parser.add_argument("--verify-license", action="store_true")
    parser.add_argument("--hf-repo-id", default="tabpfn_2_6")
    parser.add_argument("--allow-browser", action="store_true")
    args = parser.parse_args()

    _configure_environment(args)

    checks = [
        _check_packages(),
        _check_torch(),
        _check_tabpfn_cache(),
        _check_auth(args.require_auth or args.verify_license),
    ]
    if args.verify_license:
        checks.append(_verify_license(args.hf_repo_id))

    return 0 if all(checks) else 1


if __name__ == "__main__":
    exit_code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)
