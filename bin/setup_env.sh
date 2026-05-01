#!/usr/bin/env bash
set -euo pipefail

uv sync --extra dev
uv run python -c "import decipehr; print(decipehr.__version__)"

