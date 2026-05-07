"""Step 5: Compute DEAF metrics from judged results.

Outputs:
  results/metrics/main_results.csv       - per (model, conflict, level, format)
  results/metrics/summary.csv            - per (model, format) aggregated across levels
  results/metrics/sic_by_dimension.csv   - SIC: per (model, sub_dimension, mention_type, level, format)
  results/metrics/bsc_by_mismatch.csv    - BSC: per (model, mismatch_type, level, format)
  results/metrics/bsc_by_snr.csv         - BSC: per (model, snr, level, format)
  results/metrics/bsc_edi.csv            - BSC EDI = Acc(cross) - Acc(within)
  results/metrics/latex_tables.txt       - LaTeX-ready tables
"""

import argparse
import csv
import os
from collections import defaultdict
from pathlib import Path


def compute_group_metrics(rows: list[dict]) -> dict:
    """Compute Acc, TDS, OR for a group of judged rows."""
    total = len(rows)
    if total == 0:
        return {"total": 0, "correct": 0, "trap": 0, "other": 0,
                "acc": 0.0, "tds": 0.0, "or": 0.0}

    c = sum(1 for r in rows if r["judgment"] == "C")
    t = sum(1 for r in rows if r["judgment"] == "T")
    o = total - c - t

    return {
        "total": total,
        "correct": c,
        "trap": t,
        "other": o,
        "acc": round(100 * c / total, 1),
        "tds": round(100 * t / total, 1),
        "or": round(100 * o / total, 1),
    }


def main():
    parser = argparse.ArgumentParser(description="Compute DEAF metrics")
    parser.add_argument("--judged-dir", type=str, default="results/judged")
    parser.add_argument("--metadata-dir", type=str, default="data/metadata")
    parser.add_argument("--output-dir", type=str, default="results/metrics")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    all_rows = []
    for jf in sorted(Path(args.judged_dir).glob("*.csv")):
        with open(jf, newline="", encoding="utf-8") as f:
            all_rows.extend(list(csv.DictReader(f)))

    if not all_rows:
        print("No judged results found.")
        return

    print(f"Loaded {len(all_rows)} judged rows")

    # 1. Main results: per (model, conflict, level, format).
    groups = defaultdict(list)
    for r in all_rows:
        key = (r["model"], r["conflict_type"], r["level"], r["format"])
        groups[key].append(r)

    main_fields = ["model", "conflict", "level", "format",
                   "total", "correct", "trap", "other", "acc", "tds", "or"]
    main_rows = []
    for (model, conflict, level, fmt), rows in sorted(groups.items()):
        metrics = compute_group_metrics(rows)
        main_rows.append({
            "model": model, "conflict": conflict,
            "level": level, "format": fmt,
            **metrics,
        })

    main_csv = os.path.join(args.output_dir, "main_results.csv")
    _write_csv(main_csv, main_rows, main_fields)
    print(f"Wrote {main_csv}")

    # 2. Summary: per (model, conflict, format) with L1/L2/L3 columns.
    summary_groups = defaultdict(lambda: defaultdict(list))
    for r in all_rows:
        summary_groups[(r["model"], r["conflict_type"], r["format"])][r["level"]].append(r)

    summary_fields = ["model", "conflict", "format",
                      "L1_acc", "L2_acc", "L3_acc",
                      "L1_tds", "L2_tds", "L3_tds"]
    summary_rows = []
    for (model, conflict, fmt), level_dict in sorted(summary_groups.items()):
        row = {"model": model, "conflict": conflict, "format": fmt}
        for lvl in ("1", "2", "3"):
            m = compute_group_metrics(level_dict.get(lvl, []))
            row[f"L{lvl}_acc"] = m["acc"]
            row[f"L{lvl}_tds"] = m["tds"]
        summary_rows.append(row)

    summary_csv = os.path.join(args.output_dir, "summary.csv")
    _write_csv(summary_csv, summary_rows, summary_fields)
    print(f"Wrote {summary_csv}")

    # 3. SIC by dimension.
    sic_rows = [r for r in all_rows if r["conflict_type"] == "SIC"]
    if sic_rows:
        _compute_sic_breakdown(sic_rows, args.output_dir)

    # 4. BSC by mismatch_type / snr / EDI.
    bsc_rows = [r for r in all_rows if r["conflict_type"] == "BSC"]
    if bsc_rows:
        _compute_bsc_breakdown(bsc_rows, args.output_dir)

    # 5. LaTeX tables.
    _print_latex(summary_rows, main_rows, args.output_dir)

    print("\nDone.")


def _compute_sic_breakdown(sic_rows: list[dict], output_dir: str):
    """SIC metrics broken down by sub_dimension x mention_type x level x format."""
    groups = defaultdict(list)
    for r in sic_rows:
        key = (r["model"], r.get("sub_dimension", ""), r.get("mention_type", ""),
               r["level"], r["format"])
        groups[key].append(r)

    fields = ["model", "sub_dimension", "mention_type", "level", "format",
              "total", "correct", "trap", "other", "acc", "tds", "or"]
    rows = []
    for (model, sub_dim, mention, level, fmt), group in sorted(groups.items()):
        metrics = compute_group_metrics(group)
        rows.append({
            "model": model, "sub_dimension": sub_dim,
            "mention_type": mention, "level": level, "format": fmt,
            **metrics,
        })

    csv_path = os.path.join(output_dir, "sic_by_dimension.csv")
    _write_csv(csv_path, rows, fields)
    print(f"Wrote {csv_path}")

    # Aggregated across levels.
    agg_groups = defaultdict(list)
    for r in sic_rows:
        key = (r["model"], r.get("sub_dimension", ""), r.get("mention_type", ""), r["format"])
        agg_groups[key].append(r)

    agg_fields = ["model", "sub_dimension", "mention_type", "format",
                  "total", "acc", "tds", "or"]
    agg_rows = []
    for (model, sub_dim, mention, fmt), group in sorted(agg_groups.items()):
        metrics = compute_group_metrics(group)
        agg_rows.append({
            "model": model, "sub_dimension": sub_dim,
            "mention_type": mention, "format": fmt,
            "total": metrics["total"], "acc": metrics["acc"],
            "tds": metrics["tds"], "or": metrics["or"],
        })

    agg_csv = os.path.join(output_dir, "sic_summary_by_dim.csv")
    _write_csv(agg_csv, agg_rows, agg_fields)
    print(f"Wrote {agg_csv}")


def _compute_bsc_breakdown(bsc_rows: list[dict], output_dir: str):
    """BSC metrics broken down by mismatch_type, snr, plus EDI."""
    mm_groups = defaultdict(list)
    for r in bsc_rows:
        key = (r["model"], r.get("mismatch_type", ""), r["level"], r["format"])
        mm_groups[key].append(r)

    mm_fields = ["model", "mismatch_type", "level", "format",
                 "total", "correct", "trap", "other", "acc", "tds", "or"]
    mm_rows = []
    for (model, mm_type, level, fmt), group in sorted(mm_groups.items()):
        metrics = compute_group_metrics(group)
        mm_rows.append({
            "model": model, "mismatch_type": mm_type,
            "level": level, "format": fmt,
            **metrics,
        })

    mm_csv = os.path.join(output_dir, "bsc_by_mismatch.csv")
    _write_csv(mm_csv, mm_rows, mm_fields)
    print(f"Wrote {mm_csv}")

    snr_groups = defaultdict(list)
    for r in bsc_rows:
        key = (r["model"], r.get("snr", ""), r["level"], r["format"])
        snr_groups[key].append(r)

    snr_fields = ["model", "snr", "level", "format",
                  "total", "correct", "trap", "other", "acc", "tds", "or"]
    snr_rows = []
    for (model, snr, level, fmt), group in sorted(snr_groups.items()):
        metrics = compute_group_metrics(group)
        snr_rows.append({
            "model": model, "snr": snr,
            "level": level, "format": fmt,
            **metrics,
        })

    snr_csv = os.path.join(output_dir, "bsc_by_snr.csv")
    _write_csv(snr_csv, snr_rows, snr_fields)
    print(f"Wrote {snr_csv}")

    # EDI: Acc(cross) - Acc(within) per model x level x format.
    edi_fields = ["model", "level", "format",
                  "acc_within", "acc_cross", "edi",
                  "tds_within", "tds_cross"]
    edi_rows = []

    edi_groups = defaultdict(lambda: defaultdict(list))
    for r in bsc_rows:
        mm = r.get("mismatch_type", "")
        if mm in ("within", "cross"):
            key = (r["model"], r["level"], r["format"])
            edi_groups[key][mm].append(r)

    for (model, level, fmt), mm_dict in sorted(edi_groups.items()):
        w = compute_group_metrics(mm_dict.get("within", []))
        x = compute_group_metrics(mm_dict.get("cross", []))
        edi = round(x["acc"] - w["acc"], 1) if w["total"] > 0 and x["total"] > 0 else 0
        edi_rows.append({
            "model": model, "level": level, "format": fmt,
            "acc_within": w["acc"], "acc_cross": x["acc"], "edi": edi,
            "tds_within": w["tds"], "tds_cross": x["tds"],
        })

    edi_csv = os.path.join(output_dir, "bsc_edi.csv")
    _write_csv(edi_csv, edi_rows, edi_fields)
    print(f"Wrote {edi_csv}")


def _print_latex(summary_rows, main_rows, output_dir):
    """Generate LaTeX table strings."""
    lines = []

    lines.append("% === Summary Table (Model x Conflict x Level) ===")
    lines.append(r"\begin{tabular}{l l l | c c c | c c c}")
    lines.append(r"\toprule")
    lines.append(r"Model & Conflict & Format & L1 Acc & L2 Acc & L3 Acc & L1 TDS & L2 TDS & L3 TDS \\")
    lines.append(r"\midrule")
    for r in summary_rows:
        lines.append(
            f"{r['model']} & {r['conflict']} & {r['format']} & "
            f"{r['L1_acc']} & {r['L2_acc']} & {r['L3_acc']} & "
            f"{r['L1_tds']} & {r['L2_tds']} & {r['L3_tds']} \\\\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append("")

    sic_dim_csv = os.path.join(output_dir, "sic_summary_by_dim.csv")
    if os.path.exists(sic_dim_csv):
        with open(sic_dim_csv, newline="", encoding="utf-8") as f:
            sic_dim_rows = list(csv.DictReader(f))

        lines.append("% === SIC by Sub-dimension x Mention Type ===")
        lines.append(r"\begin{tabular}{l l l l | c c c}")
        lines.append(r"\toprule")
        lines.append(r"Model & Dim & Mention & Format & Acc & TDS & OR \\")
        lines.append(r"\midrule")
        for r in sic_dim_rows:
            lines.append(
                f"{r['model']} & {r['sub_dimension']} & {r['mention_type']} & "
                f"{r['format']} & {r['acc']} & {r['tds']} & {r['or']} \\\\"
            )
        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        lines.append("")

    edi_csv = os.path.join(output_dir, "bsc_edi.csv")
    if os.path.exists(edi_csv):
        with open(edi_csv, newline="", encoding="utf-8") as f:
            edi_rows = list(csv.DictReader(f))

        lines.append("% === BSC EDI (Acc_cross - Acc_within) ===")
        lines.append(r"\begin{tabular}{l l l | c c c}")
        lines.append(r"\toprule")
        lines.append(r"Model & Level & Format & Acc(within) & Acc(cross) & EDI \\")
        lines.append(r"\midrule")
        for r in edi_rows:
            lines.append(
                f"{r['model']} & L{r['level']} & {r['format']} & "
                f"{r['acc_within']} & {r['acc_cross']} & {r['edi']} \\\\"
            )
        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")

    latex_path = os.path.join(output_dir, "latex_tables.txt")
    with open(latex_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Wrote {latex_path}")


def _write_csv(path, rows, fields):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
