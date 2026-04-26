"""
plot_results.py — Generate paper figures from experiment results.

Run after run_experiment.py completes.

Produces (in results/figures/):
  fig1_act_difficulty_correlation.png  — ACT score vs ground-truth difficulty
  fig2_depth_extrapolation.png         — accuracy vs n_loops for all 3 runs
  fig3_halt_depth_evolution.png        — mean halt depth during Run C training
  fig4_lti_spectral_radius.png         — placeholder (requires training hooks)

Usage:
    python plot_results.py
"""

from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")  # non-interactive backend (works without display)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch

os.makedirs("results/figures", exist_ok=True)

# ---------------------------------------------------------------------------
# Figure 1 — ACT difficulty score vs ground-truth difficulty level
# ---------------------------------------------------------------------------
print("Generating Figure 1: ACT difficulty correlation...")

data = torch.load("results/difficulty_validation.pt", weights_only=True)
scores = data["scores"].numpy()
gt = data["gt_difficulties"].numpy()

fig, ax = plt.subplots(figsize=(7, 4))

means, stds, counts = [], [], []
for d in range(1, 6):
    mask = gt == d
    vals = scores[mask]
    means.append(float(vals.mean()))
    stds.append(float(vals.std()))
    counts.append(int(mask.sum()))

ax.bar(range(1, 6), means, yerr=stds, capsize=5,
       color="#4C72B0", alpha=0.8, edgecolor="black", linewidth=0.8)
ax.set_xlabel("Ground-Truth Difficulty Level", fontsize=12)
ax.set_ylabel("Mean ACT Halt Depth Score", fontsize=12)
ax.set_title("Figure 1: ACT Halt Depth Correlates with Problem Difficulty\n"
             "(unsupervised signal — no labels used)", fontsize=11)
ax.set_xticks(range(1, 6))
ax.set_xticklabels([f"Level {d}\n(n={counts[d-1]})" for d in range(1, 6)])
ax.set_ylim(0, 1.1)
ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="Midpoint")
ax.legend(fontsize=10)
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("results/figures/fig1_act_difficulty_correlation.png", dpi=150)
plt.close()
print("  Saved: results/figures/fig1_act_difficulty_correlation.png")


# ---------------------------------------------------------------------------
# Figure 2 — Depth extrapolation curves
# ---------------------------------------------------------------------------
print("Generating Figure 2: Depth extrapolation curves...")

with open("results/run_A_results.json") as f:
    results_A = json.load(f)
with open("results/run_B_results.json") as f:
    results_B = json.load(f)
with open("results/run_C_results.json") as f:
    results_C = json.load(f)

n_loops_vals = [r["n_loops"] for r in results_A]
acc_A = [r["accuracy"] for r in results_A]
acc_B = [r["accuracy"] for r in results_B]
acc_C = [r["accuracy"] for r in results_C]

fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(n_loops_vals, acc_A, "o--", color="#DD8452", linewidth=2,
        markersize=8, label="Run A: Baseline (fixed n_loops=4)")
ax.plot(n_loops_vals, acc_B, "s--", color="#55A868", linewidth=2,
        markersize=8, label="Run B: Loop Curriculum only")
ax.plot(n_loops_vals, acc_C, "D-", color="#4C72B0", linewidth=2.5,
        markersize=9, label="Run C: ACT Curriculum + Loop Sched. (ours)")

# Shade the extrapolation region
ax.axvspan(8.5, max(n_loops_vals) + 0.5, alpha=0.08, color="purple",
           label="Extrapolation region (n_loops > training max)")
ax.axvline(x=8, color="gray", linestyle=":", linewidth=1.5, alpha=0.7)
ax.text(8.2, ax.get_ylim()[0] + 0.02 if ax.get_ylim()[0] > 0 else 0.02,
        "Training\nmax", fontsize=9, color="gray", va="bottom")

ax.set_xlabel("n_loops at Evaluation", fontsize=12)
ax.set_ylabel("Proof Step Accuracy", fontsize=12)
ax.set_title("Figure 2: Depth Extrapolation — ACT Curriculum Enables\n"
             "Better Reasoning at Unseen Loop Depths", fontsize=11)
ax.set_xticks(n_loops_vals)
ax.set_ylim(0, 1.05)
ax.legend(fontsize=9, loc="lower right")
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("results/figures/fig2_depth_extrapolation.png", dpi=150)
plt.close()
print("  Saved: results/figures/fig2_depth_extrapolation.png")


# ---------------------------------------------------------------------------
# Figure 3 — Halt depth evolution during Run C training
# ---------------------------------------------------------------------------
print("Generating Figure 3: Halt depth evolution during training...")

halt_log_path = "results/halt_log_C.jsonl"
if os.path.exists(halt_log_path):
    steps, mean_depths, early_rates, n_loops_log = [], [], [], []
    with open(halt_log_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            steps.append(obj["step"])
            mean_depths.append(obj["mean_halt_depth"])
            early_rates.append(obj["early_halt_rate"])
            n_loops_log.append(obj["n_loops"])

    if steps:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

        ax1.plot(steps, mean_depths, "b-o", markersize=5, linewidth=2,
                 label="Mean halt depth")
        ax1.set_ylabel("Mean Halt Depth", fontsize=11)
        ax1.set_title("Figure 3: Halt Depth Evolution During ACT Curriculum Training\n"
                      "(Run C — our method)", fontsize=11)
        ax1.legend(fontsize=10)
        ax1.grid(alpha=0.3)

        # Shade loop schedule transitions
        for thresh, n in [(70, 4), (140, 8)]:
            ax1.axvline(x=thresh, color="red", linestyle="--", alpha=0.5)
            ax2.axvline(x=thresh, color="red", linestyle="--", alpha=0.5,
                        label=f"n_loops → {n}" if thresh == 70 else None)

        ax2.plot(steps, early_rates, "g-s", markersize=5, linewidth=2,
                 label="Early halt rate (depth < n_loops)")
        ax2.set_xlabel("Training Step", fontsize=11)
        ax2.set_ylabel("Early Halt Rate", fontsize=11)
        ax2.set_ylim(0, 1.05)
        ax2.legend(fontsize=10)
        ax2.grid(alpha=0.3)

        # Add loop schedule annotation
        red_patch = mpatches.Patch(color="red", alpha=0.5, label="Loop schedule step-up")
        ax2.legend(handles=[ax2.get_lines()[0], red_patch], fontsize=9)

        plt.tight_layout()
        plt.savefig("results/figures/fig3_halt_depth_evolution.png", dpi=150)
        plt.close()
        print("  Saved: results/figures/fig3_halt_depth_evolution.png")
    else:
        print("  No halt log entries found — skipping Figure 3")
else:
    print(f"  {halt_log_path} not found — skipping Figure 3")


# ---------------------------------------------------------------------------
# Figure 4 — Summary bar chart (paper-ready comparison)
# ---------------------------------------------------------------------------
print("Generating Figure 4: Summary comparison bar chart...")

fig, ax = plt.subplots(figsize=(9, 5))

x = range(len(n_loops_vals))
width = 0.25
offset = [-width, 0, width]
colors = ["#DD8452", "#55A868", "#4C72B0"]
labels = ["Run A: Baseline", "Run B: Loop Curriculum", "Run C: Full Pipeline (ours)"]
accs = [acc_A, acc_B, acc_C]

for i, (acc, color, label) in enumerate(zip(accs, colors, labels)):
    bars = ax.bar([xi + offset[i] for xi in x], acc, width,
                  label=label, color=color, alpha=0.85, edgecolor="black", linewidth=0.6)

ax.set_xlabel("n_loops at Evaluation", fontsize=12)
ax.set_ylabel("Proof Step Accuracy", fontsize=12)
ax.set_title("Figure 4: ACT Curriculum Training Improves Depth Extrapolation\n"
             "Accuracy at All Evaluation Depths", fontsize=11)
ax.set_xticks(list(x))
ax.set_xticklabels([f"n={n}" + (" *" if n > 8 else "") for n in n_loops_vals])
ax.set_ylim(0, 1.1)
ax.legend(fontsize=9)
ax.grid(axis="y", alpha=0.3)
ax.text(0.98, 0.02, "* = extrapolation (beyond training)", transform=ax.transAxes,
        fontsize=8, ha="right", va="bottom", color="gray")

plt.tight_layout()
plt.savefig("results/figures/fig4_summary_comparison.png", dpi=150)
plt.close()
print("  Saved: results/figures/fig4_summary_comparison.png")

print("\nAll figures saved to results/figures/")
print("\nPaper-ready figures:")
for f in sorted(os.listdir("results/figures")):
    print(f"  results/figures/{f}")
