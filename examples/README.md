# examples/

Two files to help you verify your install:

## `sample_eval.csv`

Twelve rows lifted from the real DEAF eval set, covering every combination
of conflict (ESC / BSC / SIC) × level (L1 / L2 / L3) plus one MCQ row per
task so you can see both formats side-by-side. The `audio_path` column
points at the original filenames as they appear under
`audios_EMIS/` / `noisy_speech/` / `SIC_clips/`; replace with your own
audio paths if you want to run inference against this CSV.

You do not need the real DEAF audio to *read* this file — it is a
specification of the eval schema (eval_id, clip_id, audio_path,
conflict_type, level, format, question, options, prompt, ground_truth,
ground_truth_letter, trap_label, trap_letter, sub_dimension, mention_type,
mismatch_type, snr).

## `smoke_test.py`

A one-shot script that:
1. Imports every adapter module under `src/adapters/` and reports
   whether the dependencies are installed.
2. If you've set `OPENROUTER_API_KEY` (and optionally
   `SILICONFLOW_API_KEY`), runs one real inference call against the API
   adapters on a single audio file and prints the response.

```bash
# import-only check (no audio, no keys needed)
python examples/smoke_test.py

# import + one real call per API adapter
export OPENROUTER_API_KEY=sk-or-v1-...
python examples/smoke_test.py path/to/some.wav
```

The script intentionally does **not** load any local-GPU model — those
need their own conda env and downloaded weights. See
[`../docs/models/`](../docs/models/) for the per-model setup.
