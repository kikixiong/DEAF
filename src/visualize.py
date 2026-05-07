"""Step 6: Visualize DEAF evaluation results.

Outputs (in results/figures/):
  radar_by_model.png       - Radar chart: L1/L2/L3 Acc per model
  tds_bar.png              - TDS bar chart by conflict x level
  mcq_vs_oe_scatter.png    - MCQ Acc vs OE Acc scatter
  error_distribution.png   - Stacked bar of C/T/O proportions
"""

import argparse
import csv
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_csv(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52",
           "#8172B3", "#937860", "#DA8BC3", "#8C8C8C"]


# ---------------------------------------------------------------------------
# 1. Radar chart: L1/L2/L3 Acc per model (MCQ)
# ---------------------------------------------------------------------------

def plot_radar(summary_rows, out_dir):
    mcq_rows = [r for r in summary_rows if r["format"] == "mcq"]
    if not mcq_rows:
        print("[Radar] No MCQ rows, skipping")
        return

    categories = ["L1 Acc", "L2 Acc", "L3 Acc"]
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))

    for i, r in enumerate(mcq_rows):
        vals = [float(r["L1_acc"]), float(r["L2_acc"]), float(r["L3_acc"])]
        vals += vals[:1]
        color = COLORS[i % len(COLORS)]
        ax.plot(angles, vals, "o-", linewidth=2, label=r["model"], color=color)
        ax.fill(angles, vals, alpha=0.1, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=12)
    ax.set_ylim(0, 100)
    ax.set_title("DEAF Accuracy by Level (MCQ)", fontsize=14, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))

    path = os.path.join(out_dir, "radar_by_model.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


# ---------------------------------------------------------------------------
# 2. TDS bar chart by conflict × level
# ---------------------------------------------------------------------------

def plot_tds_bar(main_rows, out_dir):
    mcq = [r for r in main_rows if r["format"] == "mcq"]
    if not mcq:
        print("[TDS bar] No MCQ rows, skipping")
        return

    models = sorted(set(r["model"] for r in mcq))
    conflicts = sorted(set(r["conflict"] for r in mcq))
    levels = sorted(set(r["level"] for r in mcq))

    # Group labels: "ESC-L1", "ESC-L2", ...
    group_labels = [f"{c}-L{l}" for c in conflicts for l in levels]
    x = np.arange(len(group_labels))
    width = 0.8 / max(len(models), 1)

    fig, ax = plt.subplots(figsize=(max(10, len(group_labels) * 1.2), 6))

    for i, model in enumerate(models):
        tds_vals = []
        for c in conflicts:
            for l in levels:
                match = [r for r in mcq
                         if r["model"] == model and r["conflict"] == c and r["level"] == l]
                tds_vals.append(float(match[0]["tds"]) if match else 0)
        offset = (i - len(models) / 2 + 0.5) * width
        ax.bar(x + offset, tds_vals, width, label=model,
               color=COLORS[i % len(COLORS)])

    ax.set_xlabel("Conflict Type - Level")
    ax.set_ylabel("TDS (%)")
    ax.set_title("Text Dominance Score (TDS) by Conflict and Level")
    ax.set_xticks(x)
    ax.set_xticklabels(group_labels, rotation=45, ha="right")
    ax.legend()
    ax.set_ylim(0, 100)

    path = os.path.join(out_dir, "tds_bar.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


# ---------------------------------------------------------------------------
# 3. MCQ vs OE scatter
# ---------------------------------------------------------------------------

def plot_mcq_vs_oe(summary_rows, out_dir):
    # Match model pairs: same model, mcq vs oe
    models = set(r["model"] for r in summary_rows)
    pairs = []
    for m in models:
        mcq = [r for r in summary_rows if r["model"] == m and r["format"] == "mcq"]
        oe = [r for r in summary_rows if r["model"] == m and r["format"] == "oe"]
        if mcq and oe:
            pairs.append((m, float(mcq[0]["L1_acc"]), float(oe[0]["L1_acc"])))

    if len(pairs) < 2:
        print("[MCQ vs OE] Not enough pairs, skipping")
        return

    fig, ax = plt.subplots(figsize=(7, 7))
    for i, (model, mcq_acc, oe_acc) in enumerate(pairs):
        ax.scatter(mcq_acc, oe_acc, s=100, color=COLORS[i % len(COLORS)],
                   zorder=3)
        ax.annotate(model, (mcq_acc, oe_acc), fontsize=9,
                    xytext=(5, 5), textcoords="offset points")

    lim = max(100, max(max(p[1], p[2]) for p in pairs) + 5)
    ax.plot([0, lim], [0, lim], "k--", alpha=0.3, label="y=x")
    ax.set_xlabel("MCQ L1 Accuracy (%)")
    ax.set_ylabel("OE L1 Accuracy (%)")
    ax.set_title("MCQ vs Open-Ended Accuracy (L1)")
    ax.legend()
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)

    path = os.path.join(out_dir, "mcq_vs_oe_scatter.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


# ---------------------------------------------------------------------------
# 4. Error distribution (stacked C/T/O)
# ---------------------------------------------------------------------------

def plot_error_distribution(main_rows, out_dir):
    mcq = [r for r in main_rows if r["format"] == "mcq"]
    if not mcq:
        print("[Error dist] No MCQ rows, skipping")
        return

    models = sorted(set(r["model"] for r in mcq))
    levels = sorted(set(r["level"] for r in mcq))

    labels = [f"{m}\nL{l}" for m in models for l in levels]
    acc_vals, tds_vals, or_vals = [], [], []

    for m in models:
        for l in levels:
            match = [r for r in mcq if r["model"] == m and r["level"] == l]
            if match:
                # Average across conflicts for this model-level
                accs = [float(r["acc"]) for r in match]
                tdss = [float(r["tds"]) for r in match]
                ors = [float(r["or"]) for r in match]
                acc_vals.append(sum(accs) / len(accs))
                tds_vals.append(sum(tdss) / len(tdss))
                or_vals.append(sum(ors) / len(ors))
            else:
                acc_vals.append(0)
                tds_vals.append(0)
                or_vals.append(0)

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 0.8), 6))

    ax.bar(x, acc_vals, label="Correct (C)", color="#55A868")
    ax.bar(x, tds_vals, bottom=acc_vals, label="Trap (T)", color="#C44E52")
    bottoms = [a + t for a, t in zip(acc_vals, tds_vals)]
    ax.bar(x, or_vals, bottom=bottoms, label="Other (O)", color="#8C8C8C")

    ax.set_ylabel("Proportion (%)")
    ax.set_title("Response Distribution: Correct / Trap / Other")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.legend()
    ax.set_ylim(0, 105)

    path = os.path.join(out_dir, "error_distribution.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Visualize DEAF results")
    parser.add_argument("--metrics-dir", type=str, default="results/metrics")
    parser.add_argument("--output-dir", type=str, default="results/figures")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    summary_csv = os.path.join(args.metrics_dir, "summary.csv")
    main_csv = os.path.join(args.metrics_dir, "main_results.csv")

    if not os.path.exists(summary_csv) or not os.path.exists(main_csv):
        print("Metrics CSVs not found. Run compute_metrics.py first.")
        return

    summary_rows = _load_csv(summary_csv)
    main_rows = _load_csv(main_csv)

    plot_radar(summary_rows, args.output_dir)
    plot_tds_bar(main_rows, args.output_dir)
    plot_mcq_vs_oe(summary_rows, args.output_dir)
    plot_error_distribution(main_rows, args.output_dir)

    print("\nAll figures generated.")


if __name__ == "__main__":
    main()
