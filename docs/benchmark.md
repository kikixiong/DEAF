# Benchmark design

## Why DEAF

Audio LLMs report acoustic information through a language head. When the
language pretraining strongly favours one answer (the words say "happy") but
the audio carries a different signal (the voice sounds angry), the model
must choose. DEAF measures *which* it chooses, across three orthogonal
acoustic dimensions and four prompt levels.

## Three conflict tasks

| Code | Conflict          | Acoustic ground truth       | Trap (text-driven cue)              |
|------|-------------------|-----------------------------|-------------------------------------|
| ESC  | Emotion–Semantic  | vocal emotion               | emotion described in the words      |
| BSC  | Background–Sound  | real environmental noise    | environment described in the words  |
| SIC  | Speaker Identity  | voice age / gender          | age / gender implied by the words   |

A clip is *matched* when the acoustic ground truth and the text agree, and
*mismatched* otherwise. Only mismatched clips are used to compute the conflict
metrics; matched clips serve as the L2 audio source and as the MATCH baseline.

## Four prompt levels

| Level   | Audio                  | Prompt                                  | Reads as                          |
|---------|------------------------|-----------------------------------------|-----------------------------------|
| MATCH   | Matched                | none                                    | "Can you classify at all?"        |
| L1      | Mismatched             | none                                    | "Audio alone — does it fool you?" |
| L2      | Matched                | Misleading text-only prompt             | "Will a wrong prompt overturn correct audio?" |
| L3      | Mismatched             | Prompt aligned with the textual trap    | "Worst case: audio + prompt both lie." |

L2 prompts are sampled at random (seed = 42) from the set of *wrong* answers
that are valid MCQ options for the same clip. L3 prompts are pinned to the
textual trap, which makes them reproducible per clip.

## Two response formats

| Format | What the model is asked          | How it is scored                          |
|--------|----------------------------------|-------------------------------------------|
| MCQ    | Choose A / B / C / D             | Deterministic letter extraction (regex)   |
| OE     | Free-form open-ended description | LLM judge classifies into C / T / O       |

The judge prompt (in `src/judge.py`) is task-aware: ESC asks about emotion,
BSC about environment, SIC about gender or age. The judge accepts semantic
matches (`"joyful" ≈ "happy"`, `"sounds elderly" ≈ "old person"`) and
classifies hedging or refusals as O.

## Five metrics

Let `n_C`, `n_T`, `n_O` be the number of C / T / O judgments in a group of
size `N = n_C + n_T + n_O`.

| Symbol | Definition                                  | What it measures                       |
|--------|---------------------------------------------|----------------------------------------|
| ACC    | n_C / N × 100                               | How often the model reports the audio  |
| TDS    | n_T / N × 100                               | How often the model follows the textual trap |
| OR     | n_O / N × 100                               | How often the model dodges the question (Other Rate) |
| ASS    | ACC<sub>MATCH</sub> − ACC<sub>L3</sub>      | How much pressure shifts the model     |
| ARS    | 2 · ACC · ASS / (ACC + ASS)                 | Robustness: high ACC *and* low fragility |

Always: ACC + TDS + OR = 100 (within rounding). `validate.py` checks this
holds row-by-row.

## Reproducibility

Both prompt sampling and MCQ option shuffling use a fixed seed (42).
Inference is deterministic for the API models (`temperature=0`) and for the
deterministic-decoding local models (`do_sample=False`). The DeepSeek judge
also uses `temperature=0`. Per-row resume in `run_inference.py` means
checkpoint files are forward-compatible: rerunning a partially-completed
file is safe.

## Limitations

- The L2 misleading prompt uses the matched audio of the same acoustic
  category. This prevents L2 from being trivially solvable by ignoring the
  audio entirely, but it does mean L2 ≢ "text-only baseline".
- The OE judge is itself an LLM; the judge prompt is fixed and the judge
  model is held constant within an experiment, but absolute numbers are
  judge-dependent. Reporting *deltas* across models / levels is more robust
  than reporting absolute ACC.
- The benchmark is currently English-only. Cross-lingual extension is left
  to future work.
