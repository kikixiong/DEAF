"""Step 7: Validation checks for the DEAF pipeline.

Checks:
  1. Data integrity   - metadata CSVs, audio files exist
  2. Eval integrity   - eval CSVs well-formed, no duplicate eval_ids
  3. Inference results - all eval_ids covered, no empty responses
  4. Judge results    - all judgments valid (C/T/O)
  5. Metrics sanity   - Acc+TDS+OR ~ 100, values in range
"""

import argparse
import csv
import os
from collections import Counter
from pathlib import Path


def _load_csv(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _warn(msg: str):
    print(f"  [WARN] {msg}")


def _ok(msg: str):
    print(f"  [OK]   {msg}")


# ---------------------------------------------------------------------------
# 1. Metadata checks
# ---------------------------------------------------------------------------

def check_metadata(metadata_dir: str):
    print("\n=== 1. Metadata Checks ===")
    for name in ("esc_clips.csv", "bsc_clips.csv", "sic_clips.csv"):
        path = os.path.join(metadata_dir, name)
        if not os.path.exists(path):
            _warn(f"Missing: {path}")
            continue
        rows = _load_csv(path)
        _ok(f"{name}: {len(rows)} rows")
        if not rows:
            _warn(f"{name} is empty")
            continue
        # Check required columns
        required = {"clip_id"}
        missing = required - set(rows[0].keys())
        if missing:
            _warn(f"{name} missing columns: {missing}")
        # Check for duplicates
        ids = [r["clip_id"] for r in rows]
        dupes = [k for k, v in Counter(ids).items() if v > 1]
        if dupes:
            _warn(f"{name} has {len(dupes)} duplicate clip_ids: {dupes[:5]}")


# ---------------------------------------------------------------------------
# 2. Audio file checks
# ---------------------------------------------------------------------------

def check_audio(data_dir: str, metadata_dir: str):
    print("\n=== 2. Audio File Checks ===")
    raw_dir = os.path.join(data_dir, "raw")
    if not os.path.isdir(raw_dir):
        _warn(f"Raw audio dir not found: {raw_dir}")
        return

    # Check audio files referenced in metadata
    for name in ("esc_clips.csv", "bsc_clips.csv", "sic_clips.csv"):
        meta_path = os.path.join(metadata_dir, name)
        if not os.path.exists(meta_path):
            continue
        rows = _load_csv(meta_path)
        missing = 0
        for r in rows:
            audio_field = r.get("audio_path", r.get("file_path", ""))
            if audio_field:
                full = os.path.join(raw_dir, audio_field)
                if not os.path.exists(full):
                    missing += 1
        if missing:
            _warn(f"{name}: {missing}/{len(rows)} audio files missing in {raw_dir}")
        else:
            _ok(f"{name}: all {len(rows)} audio files present")


# ---------------------------------------------------------------------------
# 3. Eval CSV checks
# ---------------------------------------------------------------------------

def check_eval(eval_dir: str):
    print("\n=== 3. Eval CSV Checks ===")
    if not os.path.isdir(eval_dir):
        _warn(f"Eval dir not found: {eval_dir}")
        return

    all_eval_ids = []
    for ef in sorted(Path(eval_dir).glob("*_eval.csv")):
        rows = _load_csv(str(ef))
        _ok(f"{ef.name}: {len(rows)} rows")

        if not rows:
            _warn(f"{ef.name} is empty")
            continue

        # Check required columns
        required = {"eval_id", "clip_id", "audio_path", "conflict_type",
                     "level", "format", "question", "ground_truth", "trap_label"}
        missing = required - set(rows[0].keys())
        if missing:
            _warn(f"{ef.name} missing columns: {missing}")

        # Check eval_id uniqueness within file
        ids = [r["eval_id"] for r in rows]
        dupes = [k for k, v in Counter(ids).items() if v > 1]
        if dupes:
            _warn(f"{ef.name} has {len(dupes)} duplicate eval_ids")

        all_eval_ids.extend(ids)

        # Check formats
        formats = set(r["format"] for r in rows)
        levels = set(r["level"] for r in rows)
        _ok(f"  formats={formats}, levels={levels}")

        # MCQ rows should have options and letters
        mcq_rows = [r for r in rows if r["format"] == "mcq"]
        if mcq_rows:
            empty_opts = sum(1 for r in mcq_rows if not r.get("options"))
            empty_gt = sum(1 for r in mcq_rows
                          if not r.get("ground_truth_letter"))
            if empty_opts:
                _warn(f"  {empty_opts} MCQ rows missing options")
            if empty_gt:
                _warn(f"  {empty_gt} MCQ rows missing ground_truth_letter")

    # Global uniqueness
    dupes = [k for k, v in Counter(all_eval_ids).items() if v > 1]
    if dupes:
        _warn(f"Global duplicate eval_ids across files: {len(dupes)}")
    else:
        _ok(f"All {len(all_eval_ids)} eval_ids globally unique")


# ---------------------------------------------------------------------------
# 4. Inference result checks
# ---------------------------------------------------------------------------

def check_inference(results_dir: str, eval_dir: str):
    print("\n=== 4. Inference Result Checks ===")
    if not os.path.isdir(results_dir):
        _warn(f"Results dir not found: {results_dir}")
        return

    # Gather expected eval_ids
    expected_ids = set()
    if os.path.isdir(eval_dir):
        for ef in Path(eval_dir).glob("*_eval.csv"):
            for r in _load_csv(str(ef)):
                expected_ids.add(r["eval_id"])

    for rf in sorted(Path(results_dir).glob("*.csv")):
        rows = _load_csv(str(rf))
        _ok(f"{rf.name}: {len(rows)} rows")

        if not rows:
            continue

        # Check for empty responses
        empty = sum(1 for r in rows if not r.get("response", "").strip())
        errors = sum(1 for r in rows
                     if r.get("response", "").startswith("ERROR:"))
        if empty:
            _warn(f"  {empty} empty responses")
        if errors:
            _warn(f"  {errors} error responses")

        # Coverage
        result_ids = {r["eval_id"] for r in rows}
        missing = expected_ids - result_ids
        if missing:
            _warn(f"  {len(missing)} eval_ids not covered")


# ---------------------------------------------------------------------------
# 5. Judge result checks
# ---------------------------------------------------------------------------

def check_judged(judged_dir: str):
    print("\n=== 5. Judge Result Checks ===")
    if not os.path.isdir(judged_dir):
        _warn(f"Judged dir not found: {judged_dir}")
        return

    for jf in sorted(Path(judged_dir).glob("*.csv")):
        rows = _load_csv(str(jf))
        _ok(f"{jf.name}: {len(rows)} rows")

        if not rows:
            continue

        # Check judgments are valid
        judgments = Counter(r.get("judgment", "") for r in rows)
        invalid = {k: v for k, v in judgments.items() if k not in ("C", "T", "O")}
        if invalid:
            _warn(f"  Invalid judgments: {invalid}")

        c = judgments.get("C", 0)
        t = judgments.get("T", 0)
        o = judgments.get("O", 0)
        total = len(rows)
        print(f"    C={c} ({100*c/total:.1f}%)  "
              f"T={t} ({100*t/total:.1f}%)  "
              f"O={o} ({100*o/total:.1f}%)")


# ---------------------------------------------------------------------------
# 6. Metrics sanity checks
# ---------------------------------------------------------------------------

def check_metrics(metrics_dir: str):
    print("\n=== 6. Metrics Sanity Checks ===")
    main_csv = os.path.join(metrics_dir, "main_results.csv")
    if not os.path.exists(main_csv):
        _warn(f"Not found: {main_csv}")
        return

    rows = _load_csv(main_csv)
    _ok(f"main_results.csv: {len(rows)} rows")

    for r in rows:
        acc = float(r["acc"])
        tds = float(r["tds"])
        or_ = float(r["or"])
        total = acc + tds + or_

        # Check sum ≈ 100
        if abs(total - 100.0) > 1.0:
            _warn(f"  {r['model']}/{r['conflict']}/L{r['level']}/{r['format']}: "
                  f"Acc+TDS+OR = {total:.1f} (expected ~100)")

        # Range checks
        for name, val in [("Acc", acc), ("TDS", tds), ("OR", or_)]:
            if val < 0 or val > 100:
                _warn(f"  {r['model']}: {name}={val} out of range [0,100]")

    summary_csv = os.path.join(metrics_dir, "summary.csv")
    if os.path.exists(summary_csv):
        srows = _load_csv(summary_csv)
        for r in srows:
            ass = float(r.get("ASS", -1))
            ars = float(r.get("ARS", -1))
            if ass > 1.0:
                _warn(f"  {r['model']}: ASS={ass} > 1.0")
            if ars > 1.0:
                _warn(f"  {r['model']}: ARS={ars} > 1.0")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Validate DEAF pipeline")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--metadata-dir", type=str, default="data/metadata")
    parser.add_argument("--eval-dir", type=str, default="data/prompts")
    parser.add_argument("--results-dir", type=str, default="results/raw")
    parser.add_argument("--judged-dir", type=str, default="results/judged")
    parser.add_argument("--metrics-dir", type=str, default="results/metrics")
    args = parser.parse_args()

    print("=" * 60)
    print("DEAF Pipeline Validation")
    print("=" * 60)

    check_metadata(args.metadata_dir)
    check_audio(args.data_dir, args.metadata_dir)
    check_eval(args.eval_dir)
    check_inference(args.results_dir, args.eval_dir)
    check_judged(args.judged_dir)
    check_metrics(args.metrics_dir)

    print("\n" + "=" * 60)
    print("Validation complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
