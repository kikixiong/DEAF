"""Step 3: Run model inference on eval CSVs.

For local-GPU models, activate the right conda env first (see docs/models/):
    conda activate qwen2_audio
    python src/run_inference.py --model qwen2-audio --eval-csv data/prompts/esc_eval.csv

For API models, just export the relevant key and run with workers > 1:
    export OPENROUTER_API_KEY=sk-or-v1-...
    python src/run_inference.py --model gpt-4o-audio --eval-csv data/prompts/esc_eval.csv --workers 8

Features:
    - Checkpoint/resume: skips eval_ids already present in the output CSV
    - Progress bar via tqdm
    - Auto-retry with exponential backoff on transient errors (max 3)
    - ${ENV_VAR} placeholders in models.yaml are expanded from os.environ
"""

import argparse
import csv
import importlib
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml
from tqdm import tqdm

# Make `adapters` importable when running this script from project root.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def build_prompt(row: dict) -> str:
    """Build the full text prompt sent to the model from one eval row.

    MCQ: [context prompt] + question + options + "Reply with exactly one letter: ..."
    OE : [context prompt] + question
    """
    parts = []

    if row["prompt"]:
        parts.append(f"Context: {row['prompt']}")

    if row["format"] == "mcq":
        parts.append(f"{row['question']}")
        parts.append(row["options"])
        opt_lines = [l for l in row["options"].split("\n") if l.strip()]
        letters = [l.strip()[0] for l in opt_lines if l.strip()]
        if len(letters) == 2:
            parts.append(f"Reply with exactly one letter: {letters[0]} or {letters[1]}.")
        else:
            parts.append(f"Reply with exactly one letter: {', '.join(letters[:-1])}, or {letters[-1]}.")
    else:
        parts.append(row["question"])

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Adapter loader
# ---------------------------------------------------------------------------

ADAPTER_MAP = {
    "qwen2-audio":      ("adapters.qwen2_audio",     "Qwen2AudioAdapter"),
    "audio-flamingo-3": ("adapters.audio_flamingo3", "AudioFlamingo3Adapter"),
    "desta2":           ("adapters.desta2",          "DeSTA2Adapter"),
    "salmonn":          ("adapters.salmonn",         "SALMONNAdapter"),
    "qwen3-omni":       ("adapters.qwen3_omni",      "Qwen3OmniAdapter"),
    "gpt-4o":           ("adapters.gpt4o",           "GPT4oAdapter"),
    "gpt-4o-audio":     ("adapters.gpt4o",           "GPT4oAdapter"),
    "gemini-2.5-flash": ("adapters.gemini",          "GeminiAdapter"),
    "gemini-3-flash":   ("adapters.gemini",          "GeminiAdapter"),
}


def load_adapter(model_name: str, models_cfg: dict):
    """Dynamically load and initialize the correct model adapter."""
    model_cfg = None
    for m in models_cfg["models"]:
        if m["name"] == model_name:
            model_cfg = m
            break
    if model_cfg is None:
        raise ValueError(f"Model '{model_name}' not found in models.yaml")

    module_path, class_name = ADAPTER_MAP[model_name]
    mod = importlib.import_module(module_path)
    adapter_cls = getattr(mod, class_name)

    adapter = adapter_cls()
    adapter.load(model_cfg)
    return adapter


# ---------------------------------------------------------------------------
# ${ENV_VAR} substitution in YAML values
# ---------------------------------------------------------------------------

_ENV_PAT = re.compile(r"\$\{([A-Z0-9_]+)\}")


def _expand_env_in_obj(obj):
    """Recursively replace ${VAR} occurrences in strings with os.environ[VAR]."""
    if isinstance(obj, str):
        def repl(match):
            name = match.group(1)
            return os.environ.get(name, match.group(0))
        return _ENV_PAT.sub(repl, obj)
    if isinstance(obj, list):
        return [_expand_env_in_obj(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _expand_env_in_obj(v) for k, v in obj.items()}
    return obj


# ---------------------------------------------------------------------------
# Main inference loop
# ---------------------------------------------------------------------------

def _process_one(adapter, row: dict, data_dir: str, max_retry: int) -> dict:
    """Run a single inference with retry. Thread-safe (no shared mutable state)."""
    audio_path = os.path.join(data_dir, row["audio_path"])
    prompt = build_prompt(row)

    response = ""
    elapsed = 0.0
    for attempt in range(1, max_retry + 1):
        try:
            t0 = time.time()
            response = adapter.infer(audio_path, prompt, fmt=row.get("format", "oe"))
            elapsed = time.time() - t0
            break
        except Exception as e:
            if attempt == max_retry:
                response = f"ERROR: {e}"
            else:
                time.sleep(min(2 ** attempt, 30))
    return {
        "eval_id": row["eval_id"],
        "response": response,
        "response_time": f"{elapsed:.2f}",
    }


def run_inference(
    eval_csv: str,
    model_name: str,
    models_yaml: str,
    output_dir: str,
    data_dir: str,
    max_retry: int = 3,
    workers: int = 1,
):
    with open(models_yaml) as f:
        models_cfg = yaml.safe_load(f)
    models_cfg = _expand_env_in_obj(models_cfg)

    with open(eval_csv, newline="", encoding="utf-8") as f:
        eval_rows = list(csv.DictReader(f))
    print(f"Loaded {len(eval_rows)} eval rows from {eval_csv}")

    os.makedirs(output_dir, exist_ok=True)
    conflict = Path(eval_csv).stem.replace("_eval", "")
    out_csv = os.path.join(output_dir, f"{model_name}_{conflict}.csv")

    done_ids = set()
    existing_rows = []
    if os.path.exists(out_csv):
        with open(out_csv, newline="", encoding="utf-8") as f:
            existing_rows = list(csv.DictReader(f))
            done_ids = {r["eval_id"] for r in existing_rows}
        print(f"Resuming: {len(done_ids)} already completed")

    remaining = [r for r in eval_rows if r["eval_id"] not in done_ids]
    print(f"Remaining: {len(remaining)} rows to process (workers={workers})")

    if not remaining:
        print("Nothing to do.")
        return

    print(f"Loading model: {model_name} ...")
    adapter = load_adapter(model_name, models_cfg)
    print("Model loaded.\n")

    out_fields = ["eval_id", "model", "response", "response_time"]

    write_lock = threading.Lock()
    write_header = not os.path.exists(out_csv) or len(existing_rows) == 0

    with open(out_csv, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=out_fields)
        if write_header:
            writer.writeheader()
            f.flush()

        if workers <= 1:
            # Serial path - keep this for local GPU adapters.
            for row in tqdm(remaining, desc=f"Inference [{model_name}]"):
                result = _process_one(adapter, row, data_dir, max_retry)
                with write_lock:
                    writer.writerow({"model": model_name, **result})
                    f.flush()
        else:
            # Parallel path for HTTP API adapters (GIL released during requests).
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {
                    pool.submit(_process_one, adapter, row, data_dir, max_retry): row
                    for row in remaining
                }
                with tqdm(total=len(futures), desc=f"Inference [{model_name}] x{workers}") as pbar:
                    for fut in as_completed(futures):
                        try:
                            result = fut.result()
                        except Exception as e:
                            row = futures[fut]
                            result = {
                                "eval_id": row["eval_id"],
                                "response": f"ERROR: {e}",
                                "response_time": "0.00",
                            }
                        with write_lock:
                            writer.writerow({"model": model_name, **result})
                            f.flush()
                        pbar.update(1)

    print(f"\nDone. Results saved to {out_csv}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run DEAF model inference")
    parser.add_argument("--model", type=str, required=True,
                        choices=list(ADAPTER_MAP.keys()),
                        help="Model name to run")
    parser.add_argument("--eval-csv", type=str, required=True,
                        help="Path to eval CSV (e.g. data/prompts/esc_eval.csv)")
    parser.add_argument("--models-yaml", type=str, default="configs/models.yaml")
    parser.add_argument("--output-dir", type=str, default="results/raw")
    parser.add_argument("--data-dir", type=str,
                        default=os.environ.get("DEAF_DATA_DIR", "data/raw"),
                        help="Base dir; audio_path in eval CSV is relative to it. "
                             "Override with --task to nest under data-dir/Task<N>/.")
    parser.add_argument("--task", type=str, default="",
                        help="Optional task subdir name (e.g. Task1, Task2, Task3).")
    parser.add_argument("--max-retry", type=int, default=3)
    parser.add_argument("--workers", type=int, default=1,
                        help="Concurrent workers. Use 8-16 for HTTP API adapters; "
                             "keep 1 for local GPU adapters.")
    args = parser.parse_args()

    data_dir = args.data_dir
    if args.task:
        data_dir = os.path.join(args.data_dir, args.task)

    run_inference(
        eval_csv=args.eval_csv,
        model_name=args.model,
        models_yaml=args.models_yaml,
        output_dir=args.output_dir,
        data_dir=data_dir,
        max_retry=args.max_retry,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
