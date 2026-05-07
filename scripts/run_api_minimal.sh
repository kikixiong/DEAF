#!/bin/bash
# Minimal-cost run of the 3 API models on the 6-CSV subset required for ASS + ARS
# (no matched-baseline CSVs). OE format only by default.
#
# Pre-requisites:
#   export OPENROUTER_API_KEY=sk-or-v1-...      (https://openrouter.ai/keys)
#   export DEEPSEEK_API_KEY=sk-...               (used by the judge step)
#
# Output:
#   results/raw/<model>_{esc,bsc,sic,esc_match,bsc_match,sic_match}.csv
#   results/judged/<model>_*.csv
#
# Coverage: 6 CSVs / model (3 conflict + 3 matched-control), OE only by default.
# Total per model ~ 8,272 OE rows; estimated cost across 3 models ~ $25-50.

set -e
cd "$(dirname "$0")/.."

PY=${PY:-python}
DATA_DIR=${DEAF_DATA_DIR:-./data/raw}
WORKERS=${WORKERS:-8}
ONLY_OE=${ONLY_OE:-1}

if [ -z "$OPENROUTER_API_KEY" ]; then
    echo "ERROR: OPENROUTER_API_KEY is not set." >&2
    exit 1
fi

if [ $# -eq 0 ]; then
    MODELS=(gpt-4o-audio gemini-2.5-flash gemini-3-flash)
else
    MODELS=("$@")
fi

EVAL_CSVS=(
    "esc_eval.csv"
    "bsc_eval.csv"
    "sic_eval.csv"
    "esc_match_eval.csv"
    "bsc_match_eval.csv"
    "sic_match_eval.csv"
)

mkdir -p results/raw results/judged results/metrics logs

# Optionally derive _oe.csv variants once.
if [ "$ONLY_OE" = "1" ]; then
    OE_DIR="data/prompts/oe_only"
    mkdir -p "$OE_DIR"
    for csv_file in "${EVAL_CSVS[@]}"; do
        src="data/prompts/$csv_file"
        dst="$OE_DIR/$csv_file"
        [ ! -f "$src" ] && continue
        if [ ! -f "$dst" ] || [ "$src" -nt "$dst" ]; then
            $PY - <<PYEOF
import csv
with open("$src", newline="", encoding="utf-8") as fi:
    rows = [r for r in csv.DictReader(fi) if r["format"] == "oe"]
if rows:
    with open("$dst", "w", newline="", encoding="utf-8") as fo:
        w = csv.DictWriter(fo, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)
print(f"  {len(rows)} OE rows -> $dst")
PYEOF
        fi
    done
    PROMPT_DIR="$OE_DIR"
else
    PROMPT_DIR="data/prompts"
fi

# 1. Inference.
echo
echo "=== inference ($([ "$ONLY_OE" = "1" ] && echo "OE only" || echo "all formats")) ==="
for model in "${MODELS[@]}"; do
    for csv_file in "${EVAL_CSVS[@]}"; do
        eval_path="$PROMPT_DIR/$csv_file"
        [ ! -f "$eval_path" ] && { echo "  SKIP missing: $eval_path"; continue; }
        log="logs/${model}_${csv_file%.csv}.log"
        echo
        echo "--- $model on $csv_file ---"
        $PY src/run_inference.py \
            --model "$model" \
            --eval-csv "$eval_path" \
            --models-yaml configs/models.yaml \
            --output-dir results/raw \
            --data-dir "$DATA_DIR" \
            --workers "$WORKERS" 2>&1 | tee "$log"
    done
done

# 2. Judge.
if [ -z "$DEEPSEEK_API_KEY" ]; then
    echo; echo "WARN: DEEPSEEK_API_KEY not set; skipping judge."; exit 0
fi
echo
echo "=== LLM judge (DeepSeek, OE) ==="
$PY src/judge.py \
    --results-dir results/raw \
    --output-dir results/judged \
    --oe-only 2>&1 | tee logs/judge.log

# 3. Metrics.
echo
echo "=== compute metrics ==="
$PY src/compute_metrics.py \
    --judged-dir results/judged \
    --output-dir results/metrics

echo
echo "All done."
