#!/usr/bin/env bash
# Reproduz o projeto do zero: dependências, testes, notebooks em ordem e README.
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ ! -f data/raw/cs-training.csv ]]; then
    echo "Faltam os dados: coloque cs-training.csv e cs-test.csv em data/raw/ (veja o README)." >&2
    exit 1
fi

uv sync
uv run pytest -q

for notebook in notebooks/0*.ipynb; do
    echo "executando $notebook"
    uv run jupyter nbconvert --to notebook --execute --inplace \
        --ExecutePreprocessor.timeout=1800 "$notebook"
done

uv run python scripts/build_readme.py
