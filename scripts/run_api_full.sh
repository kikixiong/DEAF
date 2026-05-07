#!/bin/bash
# Run GPT-4o-Audio + Gemini 2.5 Flash + Gemini 3 Flash on the FULL DEAF eval set.
# 8 CSVs per model x 3 models = 24 inference jobs.
# Resume-safe: rerunning skips eval_ids already in the output CSV.
#
# Pre-requisites:
#   export OPENROUTER_API_KEY=sk-or-v1-...      (https://openrouter.ai/keys)
#   export DEEPSEEK_API_KEY=sk-...               (for the judge step)
#
# Usage:
#   bash scripts/run_api_full.sh                # all 3 models
#   bash scripts/run_api_full.sh gpt-4o-audio   # one model
#   bash scripts/run_api_full.sh gemini-2.5-flash gemini-3-flash
#
# Output:
#   results/raw/<model>_{esc,bsc,sic,esc_match,bsc_match,sic_match,esc_matched_baseline,bsc_matched_baseline}.csv
#   results/judged/<model>_*.csv  (after the judge step)
#   results/metrics/main_results.csv

set -e
cd "$(dirname "$0")/.."

PY=${PY:-python}
DATA_DIR=${DEAF_DATA_DIR:-./data/raw}
WORKERS=${WORKERS:-8}

if [ -z "$OPENROUTER_API_KEY" ]; then
    echo "ERROR: OPENROUTER_API_KEY is not set." >&2
    echo "       Get a key at https://openrouter.ai/keys, then:" >&2
    echo "       export OPENROUTER_API_KEY=sk-or-v1-..." >&2
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
    "esc_matched_baseline_eval.csv"
    "bsc_matched_baseline_eval.csv"
)

mkdir -p results/raw results/judged results/metrics logs

# 0. Verify OpenRouter has credit.
echo "=== checking OpenRouter credits ==="
$PY - <<'EOF'
import json, os, sys, urllib.request
key = os.environ["OPENROUTER_API_KEY"]
req = urllib.request.Request(
    "https://openrouter.ai/api/v1/credits",
    headers={"Authorization": f"Bearer {key}"},
)
try:
    d = json.loads(urllib.request.urlopen(req, timeout=10).read())["data"]
except Exception as e:
    print(f"  WARN: could not check credits ({e}); proceeding.")
    sys.exit(0)
avail = d["total_credits"] - d["total_usage"]
print(f"  total_credits = ${d['total_credits']:.2f}")
print(f"  total_usage   = ${d['total_usage']:.2f}")
print(f"  available     = ${avail:.2f}")
if avail < 5:
    print("  >>> ERROR: OpenRouter balance is too low. Top up at:")
    print("            https://openrouter.ai/settings/credits")
    sys.exit(1)
EOF

# 1. Inference for each (model, eval_csv).
echo
echo "=== inference plan ==="
echo "  models : ${MODELS[*]}"
echo "  eval   : ${#EVAL_CSVS[@]} CSVs (full set)"
echo "  workers: $WORKERS"
echo "  data   : $DATA_DIR"

for model in "${MODELS[@]}"; do
    for csv_file in "${EVAL_CSVS[@]}"; do
        eval_path="data/prompts/$csv_file"
        if [ ! -f "$eval_path" ]; then
            echo "  SKIP (no such file): $eval_path"
            continue
        fi

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

# 2. LLM judge.
if [ -z "$DEEPSEEK_API_KEY" ]; then
    echo
    echo "WARN: DEEPSEEK_API_KEY not set; skipping judge step."
    echo "      Export it then re-run only the judge:"
    echo "        python src/judge.py --results-dir results/raw --output-dir results/judged"
    exit 0
fi

echo
echo "=== LLM judge (DeepSeek, OE only) ==="
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
echo "  raw      -> results/raw/<model>_*.csv"
echo "  judged   -> results/judged/<model>_*.csv"
echo "  metrics  -> results/metrics/main_results.csv"
