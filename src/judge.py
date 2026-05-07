"""Step 4: Judge model responses.

MCQ responses are scored deterministically by extracting the chosen letter and
comparing against ground_truth_letter / trap_letter.

OE (open-ended) responses are scored with an LLM judge (DeepSeek by default),
which classifies each response into one of three categories:
    C - Correct: aligns with the acoustic ground truth
    T - Trap:    aligns with the misleading textual cue
    O - Other:   refusal / off-topic / ambiguous

Output: results/judged/<model_name>.csv
"""

import argparse
import csv
import os
import re
import time
from pathlib import Path

from tqdm import tqdm


# ---------------------------------------------------------------------------
# MCQ judge (deterministic)
# ---------------------------------------------------------------------------

def extract_letter(response: str, valid_letters: str = "ABCD") -> str:
    """Extract the chosen option letter from a model response.

    Args:
        response: Raw model output.
        valid_letters: Letters considered valid (e.g. "AB" for 2-choice).
    """
    response = response.strip()
    valid = set(valid_letters.upper())

    if len(response) == 1 and response.upper() in valid:
        return response.upper()

    # Pattern 1: "A." / "A)" / "A:" / "A " (option-selection style).
    m = re.match(r'^([A-Z])\s*[.):\s]', response)
    if m and m.group(1).upper() in valid:
        return m.group(1).upper()

    # Pattern 2: "answer is A" / "choose B" / "select C".
    m = re.search(r'(?:answer|choose|select|option)\s*(?:is\s*)?([A-Z])\b', response, re.IGNORECASE)
    if m and m.group(1).upper() in valid:
        return m.group(1).upper()

    # Pattern 3: isolated valid letter with word boundary.
    pat = '|'.join(re.escape(l) for l in sorted(valid))
    m = re.search(rf'(?<![a-zA-Z])({pat})(?:\.|,|\)|\b)', response)
    if m:
        # Reject if it looks like the English article "A" followed by a noun.
        pos = m.end()
        rest = response[pos:pos + 15].lstrip()
        if m.group(1) == "A" and rest and rest[0].islower():
            pass
        else:
            return m.group(1).upper()

    return ""


def _valid_letters_from_eval(eval_info: dict) -> str:
    """Determine the valid option letters from the eval options string."""
    opts = eval_info.get("options", "")
    if not opts:
        return "ABCD"
    letters = []
    for line in opts.split("\n"):
        line = line.strip()
        if line and len(line) >= 2 and line[0].isalpha() and line[1] in ".):":
            letters.append(line[0].upper())
    return "".join(letters) if letters else "ABCD"


def judge_mcq(response: str, gt_letter: str, trap_letter: str,
              valid_letters: str = "ABCD") -> str:
    """Classify an MCQ response into C / T / O."""
    chosen = extract_letter(response, valid_letters)
    if not chosen:
        return "O"
    if chosen == gt_letter:
        return "C"
    if chosen == trap_letter:
        return "T"
    return "O"


# ---------------------------------------------------------------------------
# OE judge (LLM-based)
# ---------------------------------------------------------------------------

TASK_DESCRIPTIONS = {
    "ESC": (
        "Emotion Speech Classification (ESC). "
        "The model was given an audio clip in which the speaker's vocal emotion "
        "(e.g., angry, happy, sad) may conflict with the emotional context of the "
        "spoken text. The model was asked to identify the speaker's emotion purely "
        "from their voice (tone, pitch, speaking style), NOT from the semantic "
        "content of what is being said."
    ),
    "BSC": (
        "Background Sound Classification (BSC). "
        "The model was given an audio clip in which the background environmental "
        "sounds (e.g., kitchen noises, traffic, birdsong) may conflict with the "
        "environment described in the speech content. The model was asked to "
        "identify the real background environment based on what it hears in the "
        "audio, NOT from what the speaker talks about."
    ),
    "SIC": (
        "Speaker Identity Classification (SIC). "
        "The model was given an audio clip in which the speaker's voice "
        "characteristics (age and/or gender) may conflict with demographic traits "
        "implied by the speech content (e.g., a young voice reading text about "
        "retirement). The model was asked to identify the speaker's age or gender "
        "based on their voice, NOT from the semantic content of the speech."
    ),
}

JUDGE_SYSTEM_PROMPT = """\
You are an expert evaluator for the DEAF (Diagnostic Evaluation of Acoustic Faithfulness) \
benchmark. The benchmark tests whether Audio Large Language Models (Audio LLMs) can \
correctly perceive and report acoustic information from audio, even when textual or \
semantic cues in the audio conflict with the true acoustic signal.

Your job is to classify a model's free-form (open-ended) response into exactly one of \
three categories:
  C - Correct: the response semantically aligns with the ground-truth answer.
  T - Trap:    the response aligns with the trap answer (the misleading textual cue).
  O - Other:   the response does not clearly match either, or is vague / refused / erroneous."""

JUDGE_USER_TEMPLATE = """\
## Task Description
{task_description}

## Evaluation Inputs
- **Correct answer (acoustic ground truth):** {ground_truth}
- **Trap answer (text-biased misleading cue):** {trap_label}

## Model's Response
\"{response}\"

## Classification Rules
1. **Semantic matching**: The response does NOT need to use the exact same words as the \
correct or trap answer. Judge by semantic equivalence. For example:
   - "happy" ~ "joyful" ~ "cheerful" (all align with a "happy" ground truth)
   - "a young man" aligns with both "young" (age) and "male" (gender)
   - "sounds elderly" ~ "old person" ~ "senior" (all align with "elderly")
   - "restaurant" ~ "dining area" ~ "people eating" (all align with a restaurant environment)
2. **Ambiguous or hedging responses**: If the model mentions BOTH the correct and trap \
answers (e.g., "could be happy or sad"), classify as O.
3. **Refusals or errors**: If the response is a refusal, error message, or completely \
irrelevant to the question, classify as O.
4. **Partial match**: If the response partially matches (e.g., for a combined \
age+gender question, only one attribute is correct), classify based on the specific \
attribute being asked about.

Output ONLY a single letter: C, T, or O."""


def _call_deepseek(system_prompt: str, user_prompt: str,
                   api_key: str, max_retry: int = 3) -> str:
    """Call DeepSeek API for OE judging."""
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("pip install openai  (needed for DeepSeek API)")

    client = OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com",
    )

    for attempt in range(1, max_retry + 1):
        try:
            resp = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=4,
                temperature=0,
            )
            answer = resp.choices[0].message.content.strip().upper()
            if answer in ("C", "T", "O"):
                return answer
            for ch in answer:
                if ch in ("C", "T", "O"):
                    return ch
            return "O"
        except Exception as e:
            print(f"  [DeepSeek] Attempt {attempt}/{max_retry} failed: {e}")
            if attempt < max_retry:
                time.sleep(2 ** attempt)
    return "O"


def judge_oe(response: str, ground_truth: str, trap_label: str,
             conflict_type: str, api_key: str) -> str:
    """Judge an OE response using DeepSeek."""
    base_task = conflict_type.split("_")[0]  # ESC / BSC / SIC
    task_desc = TASK_DESCRIPTIONS.get(base_task, f"Audio classification task: {conflict_type}")

    user_prompt = JUDGE_USER_TEMPLATE.format(
        task_description=task_desc,
        ground_truth=ground_truth,
        trap_label=trap_label,
        response=response,
    )
    return _call_deepseek(JUDGE_SYSTEM_PROMPT, user_prompt, api_key)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Judge DEAF model responses (MCQ + OE)")
    parser.add_argument("--results-dir", type=str, default="results/raw",
                        help="Directory with raw model output CSVs")
    parser.add_argument("--eval-dir", type=str, default="data/prompts",
                        help="Directory with eval CSVs (for ground truth lookup)")
    parser.add_argument("--output-dir", type=str, default="results/judged")
    parser.add_argument("--judge", type=str, default="deepseek", choices=["deepseek"])
    parser.add_argument("--api-key", type=str, default=None,
                        help="API key (or set DEEPSEEK_API_KEY env var)")
    parser.add_argument("--oe-only", action="store_true",
                        help="Only judge OE responses (skip MCQ rows entirely)")
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("DEEPSEEK_API_KEY", "")

    eval_lookup = {}
    for eval_csv in Path(args.eval_dir).glob("*_eval.csv"):
        with open(eval_csv, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                eval_lookup[row["eval_id"]] = row

    print(f"Loaded {len(eval_lookup)} eval definitions")

    os.makedirs(args.output_dir, exist_ok=True)
    out_fields = [
        "eval_id", "model", "conflict_type", "level", "format",
        "response", "judgment", "ground_truth", "trap_label",
        "sub_dimension", "mention_type", "mismatch_type", "snr",
    ]

    for raw_csv in sorted(Path(args.results_dir).glob("*.csv")):
        print(f"\nJudging: {raw_csv.name}")
        with open(raw_csv, newline="", encoding="utf-8") as f:
            raw_rows = list(csv.DictReader(f))

        out_csv = Path(args.output_dir) / raw_csv.name
        judged = []

        for row in tqdm(raw_rows, desc=f"  {raw_csv.stem}"):
            eval_id = row["eval_id"]
            eval_info = eval_lookup.get(eval_id)
            if eval_info is None:
                print(f"  [WARN] eval_id not found: {eval_id}")
                continue

            fmt = eval_info["format"]
            gt = eval_info["ground_truth"]
            trap = eval_info["trap_label"]
            response = row["response"]

            if args.oe_only and fmt == "mcq":
                continue

            if fmt == "mcq":
                valid = _valid_letters_from_eval(eval_info)
                judgment = judge_mcq(
                    response,
                    eval_info["ground_truth_letter"],
                    eval_info["trap_letter"],
                    valid_letters=valid,
                )
            else:
                if not api_key:
                    judgment = "O"
                else:
                    judgment = judge_oe(
                        response, gt, trap,
                        eval_info["conflict_type"], api_key,
                    )

            judged.append({
                "eval_id": eval_id,
                "model": row["model"],
                "conflict_type": eval_info["conflict_type"],
                "level": eval_info["level"],
                "format": fmt,
                "response": response,
                "judgment": judgment,
                "ground_truth": gt,
                "trap_label": trap,
                "sub_dimension": eval_info.get("sub_dimension", ""),
                "mention_type": eval_info.get("mention_type", ""),
                "mismatch_type": eval_info.get("mismatch_type", ""),
                "snr": eval_info.get("snr", ""),
            })

        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=out_fields)
            writer.writeheader()
            writer.writerows(judged)

        c = sum(1 for r in judged if r["judgment"] == "C")
        t = sum(1 for r in judged if r["judgment"] == "T")
        o = sum(1 for r in judged if r["judgment"] == "O")
        total = len(judged) or 1
        print(f"  Results: {len(judged)} rows  C={c} ({100*c/total:.1f}%)  "
              f"T={t} ({100*t/total:.1f}%)  O={o} ({100*o/total:.1f}%)")
        print(f"  Saved -> {out_csv}")


if __name__ == "__main__":
    main()
