#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────
# Download YRBS 2023 National Public-Use Data from CDC
# ──────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BASE_DIR="$(dirname "$SCRIPT_DIR")"
DATA_DIR="$BASE_DIR/data"

mkdir -p "$DATA_DIR"

CDC_BASE="https://www.cdc.gov/yrbs/files/2023"

echo "Downloading YRBS 2023 National data..."

# ASCII data file
if [ ! -f "$DATA_DIR/XXH2023_YRBS_Data.dat" ]; then
    echo "  Downloading ASCII data file..."
    curl -L -o "$DATA_DIR/XXH2023_YRBS_Data.dat" "$CDC_BASE/XXH2023_YRBS_Data.dat"
else
    echo "  ASCII data file already exists, skipping."
fi

# SPSS syntax (column specifications)
if [ ! -f "$DATA_DIR/2023XXH-SPSS.sps" ]; then
    echo "  Downloading SPSS syntax file..."
    curl -L -o "$DATA_DIR/2023XXH-SPSS.sps" "$CDC_BASE/2023XXH-SPSS.sps"
else
    echo "  SPSS syntax file already exists, skipping."
fi

# SAS input program
if [ ! -f "$DATA_DIR/2023XXH-SAS-Input-Program.sas" ]; then
    echo "  Downloading SAS input program..."
    curl -L -o "$DATA_DIR/2023XXH-SAS-Input-Program.sas" "$CDC_BASE/2023XXH-SAS-Input-Program.sas"
else
    echo "  SAS input program already exists, skipping."
fi

echo "Done. Data files in: $DATA_DIR"
