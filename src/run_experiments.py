import json
import time
from pathlib import Path
import numpy as np
import jax.numpy as jnp

import train
from pinn_core import u_fn

ROOT = Path(__file__).resolve().parent.parent

train.EPOCHS = 6000
train.LOG_EVERY = 200

SEEDS = [0, 1, 2]
METHODS = {
    "baseline": train.train_baseline,
    "gradnorm": train.train_gradnorm,
    "self_adaptive": train.train_self_adaptive,
    "sa_rad": train.train_sa_rad,
}

gt = np.load(ROOT / "results" / "ground_truth.npz")
x_grid, t_grid, U_true = gt["x"], gt["t"], gt["U"]
XX, TT = np.meshgrid(x_grid, t_grid)


def evaluate(params):
    x_flat = jnp.array(XX.flatten())
    t_flat = jnp.array(TT.flatten())
    u_pred = np.array(u_fn(params, x_flat, t_flat)).reshape(XX.shape)
    l2_err = np.sqrt(np.mean((u_pred - U_true) ** 2))
    rel_l2 = np.linalg.norm(u_pred - U_true) / np.linalg.norm(U_true)
    return u_pred, l2_err, rel_l2


all_results = {}
predictions = {}

for method_name, train_fn in METHODS.items():
    all_results[method_name] = []
    for seed in SEEDS:
        t0 = time.time()
        params, history, elapsed = train_fn(seed)
        u_pred, l2_err, rel_l2 = evaluate(params)
        run_time = time.time() - t0
        print(f"{method_name} seed={seed}: rel_L2={rel_l2:.4f} l2={l2_err:.4f} "
              f"time={run_time:.1f}s final_loss={history['L_total'][-1]:.3e}", flush=True)
        all_results[method_name].append({
            "seed": seed, "history": history,
            "rel_l2": float(rel_l2), "l2_err": float(l2_err), "time": run_time,
        })
        if seed == SEEDS[0]:
            predictions[method_name] = u_pred

with open(ROOT / "results" / "all_results.json", "w") as f:
    json.dump(all_results, f)

np.savez(ROOT / "results" / "predictions.npz",
         x=x_grid, t=t_grid, U_true=U_true,
         **{f"pred_{k}": v for k, v in predictions.items()})

summary = {}
for method, runs in all_results.items():
    r = [x["rel_l2"] for x in runs]
    summary[method] = {
        "mean_rel_l2": float(np.mean(r)), "std_rel_l2": float(np.std(r)),
        "min_rel_l2": float(np.min(r)), "max_rel_l2": float(np.max(r)),
        "mean_time": float(np.mean([x["time"] for x in runs])),
    }
with open(ROOT / "results" / "summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("DONE")