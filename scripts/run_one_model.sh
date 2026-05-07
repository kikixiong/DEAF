#!/bin/bash
# Run a single API model on the 6-CSV "ASS+ARS" subset, OE format only.
# Resume-safe (skips eval_ids already in the output CSV).
# Concurrent: 8 workers (override with WORKERS=N).
#
# Pre-requisites:
#   export OPENROUTER_API_KEY=sk-or-v1-...
#   (and DEEPSEEK_API_KEY=... if you also want to judge)
#
# Usage:
#   bash scripts/run_one_model.sh gpt-4o-audio
#   bash scripts/run_one_model.sh gemini-2.5-flash
#   WORKERS=12 bash scripts/run_one_model.sh gemini-3-flash

set -e
cd "$(dirname "$0")/.."

MODEL=${1:?usage: run_one_model.sh <model-name>}
PY=${PY:-python}
DATA_DIR=${DEAF_DATA_DIR:-./data/raw}
WORKERS=${WORKERS:-8}

mkdir -p data/prompts/oe_only results/raw logs

EVAL_CSVS=(
    "esc_eval.csv"
    "bsc_eval.csv"
    "sic_eval.csv"
    "esc_match_eval.csv"
    "bsc_match_eval.csv"
    "sic_match_eval.csv"
)

# 1. Build OE-only versions of each CSV (idempotent).
for f in "${EVAL_CSVS[@]}"; do
    src="data/prompts/$f"
    dst="data/prompts/oe_only/$f"
    if [ -f "$src" ] && { [ ! -f "$dst" ] || [ "$src" -nt "$dst" ]; }; then
        $PY - <<PYEOF
import csv
with open("$src", newline="", encoding="utf-8") as fi:
    rows = [r for r in csv.DictReader(fi) if r["format"] == "oe"]
with open("$dst", "w", newline="", encoding="utf-8") as fo:
    w = csv.DictWriter(fo, fieldnames=rows[0].keys())
    w.writeheader()
    w.writerows(rows)
print(f"  built $dst  rows={len(rows)}")
PYEOF
    fi
done

# 2. Run inference per CSV (sequential, each file uses its own thread pool).
for f in "${EVAL_CSVS[@]}"; do
    eval_path="data/prompts/oe_only/$f"
    [ ! -f "$eval_path" ] && { echo "SKIP missing: $eval_path"; continue; }
    log="logs/${MODEL}_${f%.csv}.log"
    echo
    echo "===== $MODEL @ $f (workers=$WORKERS)  log=$log ====="
    $PY src/run_inference.py \
        --model "$MODEL" \
        --eval-csv "$eval_path" \
        --models-yaml configs/models.yaml \
        --output-dir results/raw \
        --data-dir "$DATA_DIR" \
        --workers "$WORKERS" 2>&1 | tee "$log"
done

echo
echo "$MODEL done.  raw: results/raw/${MODEL}_*.csv"
