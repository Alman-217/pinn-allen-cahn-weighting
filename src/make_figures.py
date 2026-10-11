import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
RESDIR = ROOT / "results"
FIGDIR = ROOT / "figures"
FIGDIR.mkdir(exist_ok=True)

with open(f"{RESDIR}/all_results.json") as f:
    res = json.load(f)
with open(f"{RESDIR}/summary.json") as f:
    summary = json.load(f)

pred = np.load(f"{RESDIR}/predictions.npz")
x, t, U_true = pred["x"], pred["t"], pred["U_true"]
methods = ["baseline", "gradnorm", "self_adaptive", "sa_rad"]
labels = {"baseline": "Baseline (equal weights)",
          "gradnorm": "Gradient-norm balancing",
          "self_adaptive": "Self-adaptive (SA-PINN)",
          "sa_rad": "Self-adaptive + RAD"}
colors = {"baseline": "#888888", "gradnorm": "#1f77b4",
          "self_adaptive": "#d62728", "sa_rad": "#2ca02c"}

fig, ax = plt.subplots(figsize=(7, 5))
for m in methods:
    h = res[m][0]["history"]
    ax.plot(h["epoch"], h["L_total"], label=labels[m], color=colors[m])
ax.set_yscale("log")
ax.set_xlabel("Epoch")
ax.set_ylabel("Total training loss (log scale)")
ax.set_title("Training loss vs. epoch (seed 0)")
ax.legend()
fig.tight_layout()
fig.savefig(f"{FIGDIR}/fig1_loss_curves.png", dpi=150)
plt.close(fig)

fig, ax = plt.subplots(figsize=(6, 4.5))
im = ax.pcolormesh(t, x, U_true.T, shading="auto", cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xlabel("t")
ax.set_ylabel("x")
ax.set_title("Ground-truth solution u(x,t) (finite-difference reference)")
fig.colorbar(im, ax=ax, label="u")
fig.tight_layout()
fig.savefig(f"{FIGDIR}/fig2_ground_truth.png", dpi=150)
plt.close(fig)

fig, axes = plt.subplots(4, 2, figsize=(9, 14))
for i, m in enumerate(methods):
    u_pred = pred[f"pred_{m}"]
    err = np.abs(u_pred - U_true)
    ax0, ax1 = axes[i, 0], axes[i, 1]
    im0 = ax0.pcolormesh(t, x, u_pred.T, shading="auto", cmap="RdBu_r", vmin=-1, vmax=1)
    ax0.set_title(f"{labels[m]}\nprediction")
    ax0.set_ylabel("x")
    fig.colorbar(im0, ax=ax0)
    im1 = ax1.pcolormesh(t, x, err.T, shading="auto", cmap="inferno")
    ax1.set_title("|error|")
    fig.colorbar(im1, ax=ax1)
    if i == 3:
        ax0.set_xlabel("t")
        ax1.set_xlabel("t")
fig.tight_layout()
fig.savefig(f"{FIGDIR}/fig3_predictions_errors.png", dpi=150)
plt.close(fig)

fig, ax = plt.subplots(figsize=(6.5, 4.5))
means = [summary[m]["mean_rel_l2"] for m in methods]
stds = [summary[m]["std_rel_l2"] for m in methods]
xpos = np.arange(len(methods))
bars = ax.bar(xpos, means, yerr=stds, capsize=6,
              color=[colors[m] for m in methods])
ax.set_xticks(xpos)
ax.set_xticklabels([labels[m] for m in methods], rotation=20, ha="right")
ax.set_ylabel("Relative L2 error (mean +/- std over 3 seeds)")
ax.set_title("Solution accuracy by weighting strategy")
fig.tight_layout()
fig.savefig(f"{FIGDIR}/fig4_rel_l2_comparison.png", dpi=150)
plt.close(fig)

fig, ax = plt.subplots(figsize=(6.5, 4.5))
for i, m in enumerate(methods):
    vals = [r["rel_l2"] for r in res[m]]
    ax.scatter([i] * len(vals), vals, color=colors[m], s=60, zorder=3)
    ax.scatter([i], [np.mean(vals)], color="black", marker="_", s=400, zorder=4)
ax.set_xticks(range(len(methods)))
ax.set_xticklabels([labels[m] for m in methods], rotation=20, ha="right")
ax.set_ylabel("Relative L2 error (individual seeds)")
ax.set_title("Run-to-run variance across random seeds")
fig.tight_layout()
fig.savefig(f"{FIGDIR}/fig5_seed_variance.png", dpi=150)
plt.close(fig)

print("Figures saved to", FIGDIR)
