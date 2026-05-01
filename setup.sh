#!/usr/bin/env bash
# =============================================================================
# setup.sh — One-shot project bootstrap for MRI-CT Synthesis
# Requirements: uv installed (curl -LsSf https://astral.sh/uv/install.sh | sh)
# =============================================================================

set -euo pipefail

# ── Colour helpers ────────────────────────────────────────────────────────────
GREEN='\033[0;32m'
CYAN='\033[0;36m'
NC='\033[0m'  # No Colour

echo -e "${CYAN}[INFO]${NC} Creating virtual environment (.venv) with Python 3.11 …"
uv venv --python 3.11 .venv

echo -e "${CYAN}[INFO]${NC} Installing dependencies from pyproject.toml …"
uv sync

echo -e "${CYAN}[INFO]${NC} Creating project directories structure …"
mkdir -p \
    data/brain \
    checkpoints \
    outputs/figures \
    outputs/logs \
    src \
    tests

touch src/__init__.py

echo -e "${GREEN}[OK]${NC} Setup complete!"
echo "Run 'uv run python -m src.main --help' after building the pipeline."
