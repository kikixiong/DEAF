"""Step 2: Generate evaluation questions + prompts for ESC, BSC and SIC.

Reads data/metadata/{esc,bsc,sic}_clips.csv -> data/prompts/{esc,bsc,sic}_eval.csv.
Each mismatched clip expands to 6 rows: 3 Levels x 2 Formats (MCQ + OE).
Matched clips are used as L2 audio sources (no audio-text conflict).

Key design:
  - L1: audio-only, no prompt -> tests acoustic vs semantic conflict
  - L2: matched/neutral audio + misleading prompt -> tests prompt robustness
  - L3: mismatched audio + semantic-aligned prompt -> hardest (double trap)
  - L2 prompt trap is randomly chosen (not fixed opposite), seed=42
  - MCQ options are shuffled per-row to prevent position bias
"""

import argparse
import csv
import random
from pathlib import Path


# ---------------------------------------------------------------------------
# ESC prompt templates
# ---------------------------------------------------------------------------

EMOTION_DESC = {
    "happy": "happy / joyful",
    "sad": "sad / sorrowful",
    "angry": "angry / frustrated",
    "neutral": "neutral / calm",
}

ESC_QUESTION = "Based on the speaker's voice, what emotion is being expressed?"
ESC_QUESTION_OE = "Listen carefully to the speaker's voice. Describe the emotion conveyed by the speaker."

ESC_OPTS = {
    "happy": "Happy / Joyful",
    "sad": "Sad / Sorrowful",
    "angry": "Angry / Frustrated",
    "neutral": "Neutral / Calm",
}


# ---------------------------------------------------------------------------
# BSC prompt templates
# ---------------------------------------------------------------------------

ENV_DESC = {
    "DWASHING": "a laundry room with a washing machine running",
    "DKITCHEN": "a kitchen during meal preparation",
    "DLIVING": "a living room with household activities",
    "NFIELD": "an open field with natural sounds",
    "NRIVER": "near a river or stream with flowing water",
    "NPARK": "a park with birds and people",
    "OOFFICE": "a quiet office with computers and typing",
    "OHALLWAY": "a hallway in an office building",
    "OMEETING": "a meeting room with people talking",
    "PSTATION": "a busy train or metro station",
    "PCAFETER": "a cafeteria with people eating and chatting",
    "PRESTO": "a restaurant during a busy meal time",
    "STRAFFIC": "a busy street with traffic noise",
    "SPSQUARE": "a public square with crowds",
    "SCAFE": "an outdoor cafe on a street",
    "TMETRO": "inside a metro train",
    "TBUS": "inside a bus",
    "TCAR": "inside a car while driving",
}

BSC_QUESTION = "Based on the background sounds you hear, where was this audio most likely recorded?"
BSC_QUESTION_OE = "Listen carefully to the background sounds in this audio. Describe the environment where this recording was most likely made."


# ---------------------------------------------------------------------------
# SIC prompt templates
# ---------------------------------------------------------------------------

SIC_Q_GENDER = "Based on the voice you hear, what is the speaker's gender?"
SIC_Q_GENDER_OE = "Listen to the speaker's voice carefully. What gender does the speaker sound like?"

SIC_Q_AGE = "Based on the voice you hear, how old does the speaker sound?"
SIC_Q_AGE_OE = "Listen to the speaker's voice carefully. How old does the speaker sound?"

SIC_Q_CMB_GENDER = SIC_Q_GENDER  # for CMB, ask gender
SIC_Q_CMB_AGE = SIC_Q_AGE  # for CMB, ask age

SIC_OPTS_GENDER = {"male": "Male", "female": "Female"}
SIC_OPTS_AGE = {
    "young": "A young person (under 30)",
    "elderly": "An elderly person (over 60)",
}

GENDER_DESC = {"male": "a male speaker", "female": "a female speaker"}
AGE_DESC = {"young": "a young person", "elderly": "an elderly person"}


# ---------------------------------------------------------------------------
# MCQ option shuffler
# ---------------------------------------------------------------------------

def _make_mcq_options(options_dict: dict, correct_key: str,
                      trap_key: str | None = None) -> tuple[str, str, str]:
    """
    Shuffle MCQ options. Return (options_str, correct_letter, trap_letter).
    """
    items = list(options_dict.items())
    random.shuffle(items)

    letters = "ABCDEFGHIJ"
    lines = []
    correct_letter = ""
    trap_letter = ""
    for i, (key, text) in enumerate(items):
        letter = letters[i]
        lines.append(f"{letter}. {text}")
        if key == correct_key:
            correct_letter = letter
        if trap_key and key == trap_key:
            trap_letter = letter

    return "\n".join(lines), correct_letter, trap_letter


def _random_other(gt: str, pool: list[str]) -> str:
    """Pick a random item from pool that is not gt."""
    candidates = [x for x in pool if x != gt]
    if not candidates:
        return gt
    return random.choice(candidates)


# ---------------------------------------------------------------------------
# ESC eval generator
# ---------------------------------------------------------------------------

def generate_esc_eval(clips: list[dict]) -> list[dict]:
    """Generate eval rows for ESC clips."""

    # Build matched clip lookup: speech_emotion -> list of matched clips
    matched_lookup = {}
    for c in clips:
        if c["is_matched"] == "True":
            matched_lookup.setdefault(c["speech_emotion"], []).append(c)

    rows = []
    for c in clips:
        if c["is_matched"] == "True":
            continue  # only test mismatched clips

        speech_emo = c["speech_emotion"]   # acoustic ground truth
        text_emo = c["text_emotion"]       # semantic trap

        # Find matched clip for L2 (same speech emotion, matched)
        matched_candidates = matched_lookup.get(speech_emo, [])
        l2_clip = random.choice(matched_candidates) if matched_candidates else c

        for level in (1, 2, 3):
            if level == 1:
                prompt = ""
                trap = text_emo
                audio_path = c["audio_path"]
            elif level == 2:
                # Misleading prompt (random wrong emotion)
                l2_trap = _random_other(speech_emo, list(EMOTION_DESC.keys()))
                prompt = f"The speaker sounds {EMOTION_DESC.get(l2_trap, l2_trap)}."
                trap = l2_trap
                audio_path = l2_clip["audio_path"]
            else:  # level 3
                # Prompt reinforces text emotion (semantic trap)
                prompt = f"The speaker sounds {EMOTION_DESC.get(text_emo, text_emo)}."
                trap = text_emo
                audio_path = c["audio_path"]

            for fmt in ("mcq", "oe"):
                question = ESC_QUESTION if fmt == "mcq" else ESC_QUESTION_OE
                if fmt == "mcq":
                    opts_str, gt_letter, trap_letter = _make_mcq_options(
                        ESC_OPTS, speech_emo, trap
                    )
                else:
                    opts_str, gt_letter, trap_letter = "", "", ""

                mention = c.get("mention_type", "")
                # Map mention to EX/IM style for metrics
                if mention == "explicit":
                    mention_code = "EX"
                elif mention == "implicit":
                    mention_code = "IM"
                else:
                    mention_code = ""

                rows.append({
                    "eval_id": f"{c['clip_id']}_L{level}_{fmt.upper()}",
                    "clip_id": c["clip_id"],
                    "audio_path": audio_path,
                    "conflict_type": "ESC",
                    "level": level,
                    "format": fmt,
                    "question": question,
                    "options": opts_str,
                    "prompt": prompt,
                    "ground_truth": EMOTION_DESC.get(speech_emo, speech_emo),
                    "ground_truth_letter": gt_letter,
                    "trap_label": EMOTION_DESC.get(trap, trap),
                    "trap_letter": trap_letter,
                    "sub_dimension": "",
                    "mention_type": mention_code,
                    "mismatch_type": "",
                    "snr": "",
                })

    return rows


# ---------------------------------------------------------------------------
# BSC eval generator
# ---------------------------------------------------------------------------

def generate_bsc_eval(clips: list[dict]) -> list[dict]:
    """Generate eval rows for BSC clips."""
    all_bg_envs = sorted(set(c["background_env"] for c in clips))

    # Build matched clip lookup: bg_env -> list of matched clips
    matched_lookup = {}
    for c in clips:
        if c["is_matched"] == "True":
            matched_lookup.setdefault(c["background_env"], []).append(c)

    rows = []
    for c in clips:
        if c["is_matched"] == "True":
            continue  # only test on mismatched clips

        bg_env = c["background_env"]   # acoustic ground truth
        text_env = c["text_env"]       # semantic trap

        # Build MCQ options: bg_env (correct), text_env (trap), + 2 random distractors
        opt_keys = list(dict.fromkeys([bg_env, text_env]))  # dedupe, preserve order
        others = [e for e in all_bg_envs if e not in opt_keys]
        random.shuffle(others)
        while len(opt_keys) < 4 and others:
            opt_keys.append(others.pop(0))
        options_dict = {k: ENV_DESC.get(k, k) for k in opt_keys}

        # Find matched clip for L2 (same bg_env, matched)
        matched_candidates = matched_lookup.get(bg_env, [])
        l2_clip = random.choice(matched_candidates) if matched_candidates else c

        for level in (1, 2, 3):
            if level == 1:
                prompt = ""
                trap = text_env
                audio_path = c["audio_path"]
            elif level == 2:
                # Random misleading prompt — must be from MCQ options so trap_letter is valid
                l2_trap = _random_other(bg_env, list(opt_keys))
                prompt = f"This audio was recorded in {ENV_DESC.get(l2_trap, l2_trap.lower())}."
                trap = l2_trap
                audio_path = l2_clip["audio_path"]
            else:  # level 3
                # Prompt aligned with semantic (text_env) — hardest
                prompt = f"This audio was recorded in {ENV_DESC.get(text_env, text_env.lower())}."
                trap = text_env
                audio_path = c["audio_path"]

            for fmt in ("mcq", "oe"):
                question = BSC_QUESTION if fmt == "mcq" else BSC_QUESTION_OE
                if fmt == "mcq":
                    opts_str, gt_letter, trap_letter = _make_mcq_options(
                        options_dict, bg_env, trap
                    )
                else:
                    opts_str, gt_letter, trap_letter = "", "", ""

                rows.append({
                    "eval_id": f"{c['clip_id']}_L{level}_{fmt.upper()}",
                    "clip_id": c["clip_id"],
                    "audio_path": audio_path,
                    "conflict_type": "BSC",
                    "level": level,
                    "format": fmt,
                    "question": question,
                    "options": opts_str,
                    "prompt": prompt,
                    "ground_truth": ENV_DESC.get(bg_env, bg_env),
                    "ground_truth_letter": gt_letter,
                    "trap_label": ENV_DESC.get(trap, trap),
                    "trap_letter": trap_letter,
                    "sub_dimension": "",
                    "mention_type": "",
                    "mismatch_type": c["mismatch_type"],
                    "snr": c["snr"],
                })

    return rows


# ---------------------------------------------------------------------------
# SIC eval generator
# ---------------------------------------------------------------------------

def generate_sic_eval(clips: list[dict]) -> list[dict]:
    """Generate eval rows for SIC clips."""
    # Build matched clip lookup for L2 audio: (voice_gender, voice_age) -> matched clips
    matched_lookup = {}
    for c in clips:
        if c["is_matched"] == "True" and c["sub_dimension"] != "NEU":
            key = (c["voice_gender"], c["voice_age"])
            matched_lookup.setdefault(key, []).append(c)

    rows = []
    for c in clips:
        sub_dim = c["sub_dimension"]

        # Skip neutral clips (used as L2 audio, not as test targets)
        if sub_dim == "NEU":
            continue

        # Skip matched clips (no conflict to test)
        if c["is_matched"] == "True":
            continue

        voice_gender = c["voice_gender"]
        voice_age = c["voice_age"]
        semantic_gender = c["semantic_gender"]
        semantic_age = c["semantic_age"]

        # Determine which question to ask based on sub-dimension
        if sub_dim == "GDR":
            question_mcq = SIC_Q_GENDER
            question_oe = SIC_Q_GENDER_OE
            options_dict = SIC_OPTS_GENDER
            gt_key = voice_gender        # acoustic truth
            trap_key = semantic_gender   # text trap
        elif sub_dim == "AGE":
            question_mcq = SIC_Q_AGE
            question_oe = SIC_Q_AGE_OE
            options_dict = SIC_OPTS_AGE
            gt_key = voice_age
            trap_key = semantic_age
        elif sub_dim == "CMB":
            # For CMB, generate TWO sets of questions (gender + age)
            for q_type in ("gender", "age"):
                if q_type == "gender":
                    question_mcq = SIC_Q_CMB_GENDER
                    question_oe = SIC_Q_GENDER_OE
                    options_dict = SIC_OPTS_GENDER
                    gt_key = voice_gender
                    trap_key = semantic_gender
                else:
                    question_mcq = SIC_Q_CMB_AGE
                    question_oe = SIC_Q_AGE_OE
                    options_dict = SIC_OPTS_AGE
                    gt_key = voice_age
                    trap_key = semantic_age

                _generate_sic_levels(
                    rows, c, sub_dim, q_type,
                    question_mcq, question_oe,
                    options_dict, gt_key, trap_key,
                    matched_lookup,
                )
            continue  # already handled
        else:
            continue

        _generate_sic_levels(
            rows, c, sub_dim, sub_dim.lower(),
            question_mcq, question_oe,
            options_dict, gt_key, trap_key,
            matched_lookup,
        )

    return rows


def _generate_sic_levels(
    rows, c, sub_dim, q_suffix,
    question_mcq, question_oe,
    options_dict, gt_key, trap_key,
    matched_lookup,
):
    """Generate L1/L2/L3 × MCQ/OE for one SIC clip + question type."""
    # Find matched clip for L2 with same voice attributes (voice = text identity)
    voice_key = (c["voice_gender"], c["voice_age"])
    l2_candidates = matched_lookup.get(voice_key, [])
    l2_clip = random.choice(l2_candidates) if l2_candidates else c

    for level in (1, 2, 3):
        if level == 1:
            prompt = ""
            trap = trap_key
            audio_path = c["audio_path"]
        elif level == 2:
            # Misleading prompt — random wrong identity
            if "gender" in q_suffix:
                pool = list(SIC_OPTS_GENDER.keys())
                l2_trap = _random_other(gt_key, pool)
                prompt = f"The speaker is {GENDER_DESC.get(l2_trap, l2_trap)}."
            else:
                pool = list(SIC_OPTS_AGE.keys())
                l2_trap = _random_other(gt_key, pool)
                prompt = f"The speaker sounds like {AGE_DESC.get(l2_trap, l2_trap)}."
            trap = l2_trap
            audio_path = l2_clip["audio_path"]
        else:  # level 3
            # Prompt aligned with semantic content
            if "gender" in q_suffix:
                prompt = f"The speaker is {GENDER_DESC.get(trap_key, trap_key)}."
            else:
                prompt = f"The speaker sounds like {AGE_DESC.get(trap_key, trap_key)}."
            trap = trap_key
            audio_path = c["audio_path"]

        # Use q_suffix to disambiguate CMB eval_ids
        suffix = f"_{q_suffix}" if sub_dim == "CMB" else ""

        for fmt in ("mcq", "oe"):
            question = question_mcq if fmt == "mcq" else question_oe
            if fmt == "mcq":
                opts_str, gt_letter, trap_letter = _make_mcq_options(
                    options_dict, gt_key, trap
                )
            else:
                opts_str, gt_letter, trap_letter = "", "", ""

            rows.append({
                "eval_id": f"{c['clip_id']}{suffix}_L{level}_{fmt.upper()}",
                "clip_id": c["clip_id"],
                "audio_path": audio_path,
                "conflict_type": "SIC",
                "level": level,
                "format": fmt,
                "question": question,
                "options": opts_str,
                "prompt": prompt,
                "ground_truth": gt_key,
                "ground_truth_letter": gt_letter,
                "trap_label": trap,
                "trap_letter": trap_letter,
                "sub_dimension": sub_dim,
                "mention_type": c["mention_type"],
                "mismatch_type": "",
                "snr": "",
            })


# ---------------------------------------------------------------------------
# CSV I/O
# ---------------------------------------------------------------------------

EVAL_FIELDS = [
    "eval_id", "clip_id", "audio_path", "conflict_type", "level", "format",
    "question", "options", "prompt", "ground_truth", "ground_truth_letter",
    "trap_label", "trap_letter",
    # Extra metadata for fine-grained metrics
    "sub_dimension",   # SIC: AGE/GDR/CMB; BSC: within/cross
    "mention_type",    # SIC: EX/IM; BSC: N/A
    "mismatch_type",   # BSC: within/cross; SIC: N/A
    "snr",             # BSC: -10/-5/0/5/10; SIC: N/A
]


def _read_clips_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_eval_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=EVAL_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate eval questions + prompts")
    parser.add_argument("--metadata-dir", type=str, default="data/metadata")
    parser.add_argument("--output-dir", type=str, default="data/prompts")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    meta_dir = Path(args.metadata_dir)
    prompt_dir = Path(args.output_dir)

    # ESC
    esc_clips = _read_clips_csv(meta_dir / "esc_clips.csv")
    if esc_clips:
        esc_eval = generate_esc_eval(esc_clips)
        _write_eval_csv(prompt_dir / "esc_eval.csv", esc_eval)
        print(f"[ESC] Generated {len(esc_eval)} eval rows")
    else:
        print("[ESC] No clips found, skipping")

    # BSC
    bsc_clips = _read_clips_csv(meta_dir / "bsc_clips.csv")
    if bsc_clips:
        bsc_eval = generate_bsc_eval(bsc_clips)
        _write_eval_csv(prompt_dir / "bsc_eval.csv", bsc_eval)
        print(f"[BSC] Generated {len(bsc_eval)} eval rows")
    else:
        print("[BSC] No clips found, skipping")

    # SIC
    sic_clips = _read_clips_csv(meta_dir / "sic_clips.csv")
    if sic_clips:
        sic_eval = generate_sic_eval(sic_clips)
        _write_eval_csv(prompt_dir / "sic_eval.csv", sic_eval)
        print(f"[SIC] Generated {len(sic_eval)} eval rows")
    else:
        print("[SIC] No clips found, skipping")

    print("\nDone. Check output directory for eval CSVs.")


if __name__ == "__main__":
    main()
