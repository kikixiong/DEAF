#!/bin/bash
# Run a local-GPU model on the full DEAF eval set.
#
# Each local model lives in its own conda env (see docs/models/). Activate
# the env first, then run this script.
#
# Pre-requisites for each model:
#   qwen2-audio       -> conda activate qwen2_audio  + QWEN2_AUDIO_MODEL set
#   salmonn           -> conda activate salmon_env   + SALMONN_DIR set
#   desta2            -> conda activate desta2       + DESTA_DIR + DESTA2_MODEL set
#   audio-flamingo-3  -> conda activate af3          + AF3_DIR + AF3_MODEL set
#
# Local adapters use ~100% of one GPU, so workers stays at 1.
#
# Usage:
#   conda activate qwen2_audio
#   bash scripts/run_local_model.sh qwen2-audio

set -e
cd "$(dirname "$0")/.."

MODEL=${1:?usage: run_local_model.sh <model-name>}
PY=${PY:-python}
DATA_DIR=${DEAF_DATA_DIR:-./data/raw}

mkdir -p results/raw logs

EVAL_CSVS=(
    "esc_eval.csv"
    "bsc_eval.csv"
    "sic_eval.csv"
    "esc_match_eval.csv"
    "bsc_match_eval.csv"
    "sic_match_eval.csv"
)

for csv_file in "${EVAL_CSVS[@]}"; do
    eval_path="data/prompts/$csv_file"
    [ ! -f "$eval_path" ] && { echo "SKIP missing: $eval_path"; continue; }
    log="logs/${MODEL}_${csv_file%.csv}.log"
    echo
    echo "===== $MODEL @ $csv_file  log=$log ====="
    $PY src/run_inference.py \
        --model "$MODEL" \
        --eval-csv "$eval_path" \
        --models-yaml configs/models.yaml \
        --output-dir results/raw \
        --data-dir "$DATA_DIR" \
        --workers 1 2>&1 | tee "$log"
done

echo
echo "$MODEL done.  raw: results/raw/${MODEL}_*.csv"
