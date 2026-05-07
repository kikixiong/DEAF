# API models — GPT-4o-Audio, Gemini 2.5/3 Flash

All three are routed through OpenRouter using OpenAI-compatible chat
completions. One key works for all three.

## 1. Get an OpenRouter key

1. Sign up at https://openrouter.ai
2. Generate a key under https://openrouter.ai/keys
3. Top up credits at https://openrouter.ai/settings/credits
   - The full DEAF sweep (3 models × ~30k OE rows) costs ~$25–$50 at the time
     of the rebuttal cycle.

## 2. Set the env var

```bash
export OPENROUTER_API_KEY=sk-or-v1-...
```

Or add it to `.env` and `set -a; source .env; set +a`.

## 3. Verify connectivity

`run_api_full.sh` checks remaining credit before any inference call. To
sanity-check by hand on a single audio file:

```bash
python - <<'EOF'
import os, sys
sys.path.insert(0, "src")
from adapters.gpt4o import GPT4oAdapter
a = GPT4oAdapter()
a.load({
    "model_id": "openai/gpt-4o-audio-preview",
    "base_url": "https://openrouter.ai/api/v1",
    "api_key":  os.environ["OPENROUTER_API_KEY"],
    "max_tokens": 64,
})
print(a.infer("path/to/some.wav", "What emotion is the speaker expressing?"))
EOF
```

## 4. Run the full sweep

```bash
bash scripts/run_api_full.sh                         # all 3 models
bash scripts/run_api_full.sh gpt-4o-audio            # one model
WORKERS=12 bash scripts/run_api_full.sh gemini-2.5-flash
```

## Notes

- `temperature=0` is hard-coded in the adapters for reproducibility; do not
  change unless you also change the rest of the eval methodology.
- All three models accept `input_audio` blocks with `wav`, `mp3`, `flac`,
  `ogg`, `m4a`. Anything else falls back to `wav`.
- Rate limits are per-key on OpenRouter; if you hit 429s, drop `WORKERS`.
- `run_inference.py` has retry-with-backoff (3 attempts, exponential); a
  failed row writes `ERROR: ...` so you can grep and rerun later.
