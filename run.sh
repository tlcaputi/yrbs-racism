#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────
# End-to-end pipeline: download → parse → analyze → manuscript
# ──────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")/scripts" && pwd)"

echo "=== Step 0: Download YRBS 2023 data ==="
bash "$SCRIPT_DIR/00_download_data.sh"

echo ""
echo "=== Step 1: Parse raw data ==="
python3 "$SCRIPT_DIR/01_parse_yrbs.py"

echo ""
echo "=== Step 2: Run analysis (R + zelig2) ==="
Rscript "$SCRIPT_DIR/02_analysis.R"

echo ""
echo "=== Step 3: Build manuscript (table, figure, PDF) ==="
python3 "$SCRIPT_DIR/03_build_manuscript.py"

echo ""
echo "=== Pipeline complete ==="
echo "Outputs:"
echo "  draft-v2.pdf  — LaTeX manuscript"
echo "  figures/figure1.pdf — Dose-response figure"
