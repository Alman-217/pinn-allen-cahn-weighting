import json, time
from pathlib import Path
import numpy as np
import jax.numpy as jnp
import train_v2 as tv2
from pinn_core import u_fn

ROOT = Path(__file__).resolve().parent.parent

EPS = 0.0001
LAYERS = (2, 128, 128, 128, 128, 1)
EPOCHS = 20000
SEEDS = [0, 1]
LAM_LRS = [1e-3, 1e-2, 1e-1]

gt = np.load(ROOT / "results" / "ground_truths" / f"gt_eps{EPS}.npz")
x_grid, t_grid, U_true = gt["x"], gt["t"], gt["U"]
XX, TT = np.meshgrid(x_grid, t_grid)
x_flat, t_flat = jnp.array(XX.flatten()), jnp.array(TT.flatten())

results = {"baseline": [], "self_adaptive": {}}

for seed in SEEDS:
    t0 = time.time()
    params, history, elapsed = tv2.train(seed, "baseline", EPS, epochs=EPOCHS, layers=LAYERS)
    u_pred = np.array(u_fn(params, x_flat, t_flat)).reshape(XX.shape)
    rel_l2 = float(np.linalg.norm(u_pred - U_true) / np.linalg.norm(U_true))
    print(f"[matched] baseline seed={seed}: rel_L2={rel_l2:.4f} time={time.time()-t0:.1f}s", flush=True)
    results["baseline"].append({"seed": seed, "rel_l2": rel_l2, "time": time.time() - t0})

for lam_lr in LAM_LRS:
    results["self_adaptive"][str(lam_lr)] = []
    for seed in SEEDS:
        t0 = time.time()
        params, history, elapsed = tv2.train(seed, "self_adaptive", EPS, epochs=EPOCHS,
                                              layers=LAYERS, lam_lr=lam_lr)
        u_pred = np.array(u_fn(params, x_flat, t_flat)).reshape(XX.shape)
        rel_l2 = float(np.linalg.norm(u_pred - U_true) / np.linalg.norm(U_true))
        print(f"[matched] self_adaptive lam_lr={lam_lr} seed={seed}: rel_L2={rel_l2:.4f} "
              f"time={time.time()-t0:.1f}s", flush=True)
        results["self_adaptive"][str(lam_lr)].append({"seed": seed, "rel_l2": rel_l2,
                                                        "time": time.time() - t0})

with open(ROOT / "results" / "replication_check.json", "w") as f:
    json.dump(results, f)
print("REPLICATION CHECK DONE")