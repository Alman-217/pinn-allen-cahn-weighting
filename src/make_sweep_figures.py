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

with open(f"{RESDIR}/stiffness_sweep.json") as f:
    sweep = json.load(f)
with open(f"{RESDIR}/summary.json") as f:
    orig_summary = json.load(f)

methods = ["baseline", "gradnorm", "self_adaptive"]
labels = {"baseline": "Baseline", "gradnorm": "Gradient-norm balancing", "self_adaptive": "Self-adaptive (SA-PINN)"}
colors = {"baseline": "#888888", "gradnorm": "#1f77b4", "self_adaptive": "#d62728"}

eps_values = [0.1, 0.03, 0.01, 0.003, 0.001, 0.0001]  # include original eps=0.1 run

fig, ax1 = plt.subplots(figsize=(7.5, 5))
for m in methods:
    means, stds = [], []
    for eps in eps_values:
        if eps == 0.1:
            vals = [orig_summary[m]["mean_rel_l2"]]
            means.append(orig_summary[m]["mean_rel_l2"])
            stds.append(orig_summary[m]["std_rel_l2"])
        else:
            runs = sweep[str(eps)][m]
            vals = [r["rel_l2"] for r in runs]
            means.append(np.mean(vals))
            stds.append(np.std(vals))
    ax1.errorbar(eps_values, means, yerr=stds, marker="o", label=labels[m],
                 color=colors[m], capsize=4)
ax1.set_xscale("log")
ax1.invert_xaxis()
ax1.set_xlabel("eps (stiffness parameter, log scale, decreasing -> stiffer)")
ax1.set_ylabel("Relative L2 error (mean +/- std, 3 seeds)")
ax1.set_title("PINN accuracy is flat across 3 decades of stiffness")
ax1.legend()
fig.tight_layout()
fig.savefig(f"{FIGDIR}/fig6_stiffness_sweep.png", dpi=150)
plt.close(fig)

u_at_point = [0.08184, 0.11634, 0.11905, 0.11936, 0.11939, 0.11939]
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(eps_values, u_at_point, marker="o", color="#2ca02c")
ax.set_xscale("log")
ax.invert_xaxis()
ax.set_xlabel("eps (log scale, decreasing -> stiffer)")
ax.set_ylabel("True solution u(x=0.25, t=1)")
ax.set_title("Ground-truth solution saturates below eps~0.01\n(unstable-equilibrium escape dynamics, not diffusion, sets the t=1 state)")
fig.tight_layout()
fig.savefig(f"{FIGDIR}/fig7_saturation_mechanism.png", dpi=150)
plt.close(fig)

print("Sweep figures saved.")
