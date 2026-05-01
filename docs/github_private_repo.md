# GitHub Private Repository Plan

Use GitHub for code and review only. Do not upload real competition data or generated patient-level artifacts.

## Commit

Commit these classes of files:

- source code under `src/`;
- pipeline entry points under `scripts/`;
- shell wrappers under `bin/`;
- tests using synthetic fixtures only;
- configs with placeholder paths;
- README and documentation.

Do not commit:

- real genotype, phenotype, summary statistics, or submission files;
- model weights or run outputs;
- notebooks with bulky output;
- secrets or tokens.

## Suggested First Commit

```bash
git add .gitignore pyproject.toml README.md configs bin scripts src tests docs jupyter-notebook/README.md
git commit -m "Initialize reproducible PRS baseline scaffold"
```

Then create a private GitHub repository and push the branch.

## CI

CI should run only synthetic-data checks:

```bash
uv sync --extra dev
uv run --extra dev pytest
uv run --extra dev ruff check .
```

Do not configure CI jobs that require access to private DREAM data.

