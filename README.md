# DEAF — Diagnostic Evaluation of Acoustic Faithfulness

A benchmark and evaluation pipeline for testing whether Audio Large Language
Models (Audio LLMs) actually listen to the audio, or fall back on textual
priors when audio and text disagree.

DEAF crafts audio clips where an acoustic ground-truth signal (vocal emotion,
background environment, or speaker identity) deliberately conflicts with the
semantic content of the speech, then probes the model with prompts of
escalating misleading strength to measure how much the model trusts the audio
versus the text.

---

## Three conflict tasks

| Code | Conflict          | Acoustic ground truth          | Trap (text-driven cue)                 |
|------|-------------------|--------------------------------|----------------------------------------|
| ESC  | Emotion–Semantic  | Speaker's vocal emotion        | Emotion described in the words spoken  |
| BSC  | Background–Sound  | Real environmental background  | Environment described in the words     |
| SIC  | Speaker Identity  | Voice age / gender             | Age / gender implied by the words      |

The audio was synthesised so the acoustic property is unambiguous; only the
semantic content lies. A faithful Audio LLM should ignore the words and
report the acoustic truth.

## Four prompt levels

For every mismatched clip we ask the same question under four conditions:

| Level   | Audio                  | Prompt                                  | What it tests                          |
|---------|------------------------|-----------------------------------------|----------------------------------------|
| MATCH   | Matched / no conflict  | none                                    | Baseline: can the model classify at all? |
| L1      | Mismatched audio       | none                                    | Does the audio alone fool the model?   |
| L2      | Matched audio          | Misleading text-only prompt             | Does a wrong prompt overturn correct audio? |
| L3      | Mismatched audio       | Prompt aligned with the textual trap    | Hardest: audio and prompt both push toward the trap |

L2's misleading prompt is sampled with a fixed seed (42); MCQ option order is
shuffled per row to remove position bias.

## Two response formats

Every (clip, level) pair is evaluated twice:

- **MCQ** — 2-to-4 letter multiple choice. Scored deterministically.
- **OE**  — open-ended free-form. Scored by an LLM judge (DeepSeek by default)
  into one of three classes: **C**orrect / **T**rap / **O**ther.

## Five metrics

| Metric | Meaning                                                                  |
|--------|--------------------------------------------------------------------------|
| ACC    | % of responses scored Correct (acoustic ground truth)                    |
| TDS    | % of responses scored Trap (text-driven misleading cue) — *Text Dominance Score* |
| OR     | % scored Other (refusal / off-topic / ambiguous) — *Other Rate*          |
| ASS    | Per-clip *Audio Sensitivity Score*: drop in ACC from MATCH to L3         |
| ARS    | *Acoustic Robustness Score*: harmonic mean of ACC and ASS                |

---

## Repository layout

```
DEAF/
├── src/
│   ├── parse_metadata.py       # Step 1: raw audio -> data/metadata/*_clips.csv
│   ├── generate_eval.py        # Step 2: clips -> data/prompts/*_eval.csv (MATCH/L1/L2/L3 x MCQ/OE)
│   ├── run_inference.py        # Step 3: drive any adapter on an eval CSV; resume-safe
│   ├── judge.py                # Step 4: MCQ deterministic + OE via LLM judge
│   ├── compute_metrics.py      # Step 5: ACC/TDS/OR + breakdowns + LaTeX tables
│   ├── visualize.py            # Step 6: radar / bar / scatter / stacked plots
│   ├── validate.py             # Step 7: end-to-end sanity checks
│   └── adapters/               # One file per model
│       ├── base.py
│       ├── gpt4o.py            # GPT-4o-Audio   (OpenRouter)
│       ├── gemini.py           # Gemini 2.5/3 Flash (OpenRouter)
│       ├── qwen3_omni.py       # Qwen3-Omni     (SiliconFlow)
│       ├── qwen2_audio.py      # Qwen2-Audio    (HuggingFace, GPU)
│       ├── salmonn.py          # SALMONN-7B     (upstream repo, GPU)
│       ├── desta2.py           # DeSTA2-8B-beta (upstream repo, GPU)
│       └── audio_flamingo3.py  # NVIDIA AF3     (upstream repo, GPU)
├── configs/models.yaml         # Single registry; uses ${ENV_VAR} placeholders
├── scripts/                    # Top-level convenience runners
│   ├── run_one_model.sh        # one API model on the 6-CSV ASS+ARS subset
│   ├── run_api_full.sh         # 3 API models on the full 8-CSV set + judge + metrics
│   ├── run_api_minimal.sh      # 3 API models on the 6-CSV subset + judge + metrics
│   └── run_local_model.sh      # one local-GPU model on the 6-CSV subset
├── docs/
│   ├── pipeline.md             # End-to-end how-to
│   ├── data.md                 # Data sources and on-disk layout
│   ├── benchmark.md            # Prompt design + metric definitions
│   └── models/                 # Per-model setup
│       ├── api_models.md
│       ├── qwen2_audio.md
│       ├── salmonn.md
│       ├── desta2.md
│       ├── audio_flamingo3.md
│       └── qwen3_omni.md
├── requirements.txt            # Common deps
├── requirements-api.txt        # Adds openai for the judge
├── requirements-local.txt      # Adds torch + transformers + librosa
├── .env.example                # Copy to .env and fill in
└── LICENSE                     # MIT
```

---

## Quick start (API-only, ~10 minutes)

```bash
# 1. Install
git clone https://github.com/kikixiong/DEAF.git
cd DEAF
pip install -r requirements-api.txt

# 2. Configure
cp .env.example .env
# Edit .env to set OPENROUTER_API_KEY and DEEPSEEK_API_KEY.
# Then load it into your shell:
set -a; source .env; set +a

# 3. Lay out your audio data under $DEAF_DATA_DIR (see docs/data.md)
#    Expected layout:
#      $DEAF_DATA_DIR/Task1/audios_EMIS/*.wav      # ESC
#      $DEAF_DATA_DIR/Task2/noisy_speech/*.wav     # BSC
#      $DEAF_DATA_DIR/Task3/SIC_clips/*.wav        # SIC

# 4. Build metadata + eval CSVs
python src/parse_metadata.py --data-dir "$DEAF_DATA_DIR"
python src/generate_eval.py

# 5. Run one API model on a tiny subset to sanity-check wiring
python src/run_inference.py \
    --model gpt-4o-audio \
    --eval-csv data/prompts/esc_eval.csv \
    --workers 8

# 6. Or kick off the full 3-model x 8-CSV sweep
bash scripts/run_api_full.sh
```

## Quick start (local-GPU model)

Each local model lives in its own conda env. See the per-model docs in
`docs/models/` for clone commands, model downloads and conda recipes; then:

```bash
conda activate qwen2_audio
export QWEN2_AUDIO_MODEL=/path/to/Qwen2-Audio-7B-Instruct
bash scripts/run_local_model.sh qwen2-audio
```

---

## Models supported out of the box

| Name              | Type  | Provider     | Notes                                              |
|-------------------|-------|--------------|----------------------------------------------------|
| gpt-4o-audio      | API   | OpenRouter   | `openai/gpt-4o-audio-preview`                      |
| gemini-2.5-flash  | API   | OpenRouter   | `google/gemini-2.5-flash`                          |
| gemini-3-flash    | API   | OpenRouter   | `google/gemini-3-flash-preview`                    |
| qwen3-omni        | API   | SiliconFlow  | `Qwen/Qwen3-Omni-30B-A3B-Instruct`                 |
| qwen2-audio       | local | HuggingFace  | `Qwen/Qwen2-Audio-7B-Instruct`, ~17 GB GPU         |
| salmonn           | local | upstream     | SALMONN-7B; needs upstream repo + Vicuna + Whisper |
| desta2            | local | upstream     | DeSTA2-8B-beta; ships its own `desta` package      |
| audio-flamingo-3  | local | upstream     | NVIDIA AF3; uses NVIDIA's `llava` package          |

Adding a new model = one Python file under `src/adapters/`; see `docs/pipeline.md`.

---

## Citation

If you use DEAF, please cite the paper:

```bibtex
@inproceedings{deaf2026,
  title  = {DEAF: A Benchmark for Diagnostic Evaluation of Acoustic Faithfulness in Audio Language Models},
  author = {TBD},
  year   = {2026},
  note   = {Under review at ACL ARR 2026}
}
```

## License

MIT — see [LICENSE](LICENSE).
