# Pipeline guide

End-to-end walkthrough of the seven scripts under `src/`.

```
       raw audio ─┐
                  │
                  ▼
   1. parse_metadata.py  ──>  data/metadata/{esc,bsc,sic}_clips.csv
                  │
                  ▼
   2. generate_eval.py   ──>  data/prompts/{esc,bsc,sic}_eval.csv
                  │
                  ▼
   3. run_inference.py   ──>  results/raw/<model>_<conflict>.csv
                  │
                  ▼
   4. judge.py           ──>  results/judged/<model>_<conflict>.csv
                  │
                  ▼
   5. compute_metrics.py ──>  results/metrics/main_results.csv
                  │
                  ▼
   6. visualize.py       ──>  results/figures/*.png
                  │
                  ▼
   7. validate.py        ──>  console report
```

Scripts at any step are idempotent and resume-safe; rerun is cheap.

---

## Step 1 — parse_metadata.py

Walks `Task1/`, `Task2/`, `Task3/` under `--data-dir`, extracts the structured
fields out of each filename (see [data.md](data.md)) and writes one CSV per
conflict type into `data/metadata/`.

```bash
python src/parse_metadata.py \
    --data-dir "$DEAF_DATA_DIR" \
    --output-dir data/metadata \
    --tasks 1,2,3
```

Skipping a task is fine — pass `--tasks 1` for ESC only, etc.

## Step 2 — generate_eval.py

Expands every mismatched clip into 6 eval rows (3 levels × 2 formats). The
random L2 prompt and the MCQ option order both use `--seed 42` so the output
is reproducible across machines.

```bash
python src/generate_eval.py \
    --metadata-dir data/metadata \
    --output-dir data/prompts \
    --seed 42
```

You should see `esc_eval.csv`, `bsc_eval.csv`, `sic_eval.csv` plus the
matched-control variants. Row counts in the public release:

| File          | Rows     |
|---------------|----------|
| esc_eval.csv  | ~14,000  |
| bsc_eval.csv  | ~13,500  |
| sic_eval.csv  | ~2,000   |

## Step 3 — run_inference.py

Picks one model from `configs/models.yaml`, loads the matching adapter, and
streams responses into `results/raw/<model>_<eval>.csv`. Resume-safe: skips
any `eval_id` already present in the output file.

```bash
# API model — go wide on workers
python src/run_inference.py \
    --model gpt-4o-audio \
    --eval-csv data/prompts/esc_eval.csv \
    --workers 8

# Local-GPU model — keep workers=1
python src/run_inference.py \
    --model qwen2-audio \
    --eval-csv data/prompts/esc_eval.csv \
    --workers 1
```

`--data-dir` defaults to `$DEAF_DATA_DIR` (or `./data/raw`); use `--task
Task1` if your audio sits one level deeper under `Task1/`, `Task2/`, `Task3/`.

## Step 4 — judge.py

MCQ rows are scored deterministically by extracting the chosen letter; OE
rows go through a DeepSeek call that returns `C` / `T` / `O`.

```bash
export DEEPSEEK_API_KEY=sk-...
python src/judge.py \
    --results-dir results/raw \
    --eval-dir data/prompts \
    --output-dir results/judged \
    --oe-only            # skip MCQ (already scored locally) and save tokens
```

## Step 5 — compute_metrics.py

Aggregates judgments into:

- `main_results.csv` — per (model, conflict, level, format)
- `summary.csv` — L1/L2/L3 columns side-by-side per (model, conflict, format)
- `sic_by_dimension.csv`, `sic_summary_by_dim.csv` — SIC AGE/GDR/CMB × EX/IM
- `bsc_by_mismatch.csv`, `bsc_by_snr.csv`, `bsc_edi.csv` — BSC breakdowns
- `latex_tables.txt` — paste-ready LaTeX

```bash
python src/compute_metrics.py \
    --judged-dir results/judged \
    --output-dir results/metrics
```

## Step 6 — visualize.py

Generates four PNG plots into `results/figures/`. Set the matplotlib backend
to `Agg` if you're running headlessly on a server (the script does this
automatically).

```bash
python src/visualize.py \
    --metrics-dir results/metrics \
    --output-dir results/figures
```

## Step 7 — validate.py

Top-to-bottom sanity check: counts rows, looks for missing audio files,
checks that `Acc + TDS + OR ≈ 100`, and reports duplicates.

```bash
python src/validate.py
```

---

## Adding a new model

1. Create `src/adapters/<name>.py` subclassing `BaseAdapter` (`load`, `infer`).
2. Add an entry to `ADAPTER_MAP` in `src/run_inference.py`.
3. Add an entry to `configs/models.yaml`. Use `${ENV_VAR}` for any secret.
4. Optional: add a `docs/models/<name>.md` covering install + gotchas.

`load(model_cfg)` is called once before any `infer()`; use it to put the model
on the GPU and stash references on `self`. `infer(audio_path, prompt, fmt)`
must return a string. The framework handles retries, resume, and parallelism.
