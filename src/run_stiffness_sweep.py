import json, time
from pathlib import Path
import numpy as np
import jax.numpy as jnp
import train_v2 as tv2
from pinn_core import u_fn

ROOT = Path(__file__).resolve().parent.parent

EPS_VALUES = [0.03, 0.01, 0.003, 0.001, 0.0001]
METHODS = ["baseline", "gradnorm", "self_adaptive"]
SEEDS = [0, 1, 2]
EPOCHS = 6000
LAYERS = (2, 64, 64, 64, 64, 1)

results = {}

for eps in EPS_VALUES:
    gt = np.load(ROOT / "results" / "ground_truths" / f"gt_eps{eps}.npz")
    x_grid, t_grid, U_true = gt["x"], gt["t"], gt["U"]
    XX, TT = np.meshgrid(x_grid, t_grid)
    x_flat, t_flat = jnp.array(XX.flatten()), jnp.array(TT.flatten())

    results[str(eps)] = {}
    for method in METHODS:
        results[str(eps)][method] = []
        for seed in SEEDS:
            t0 = time.time()
            params, history, elapsed = tv2.train(seed, method, eps, epochs=EPOCHS, layers=LAYERS)
            u_pred = np.array(u_fn(params, x_flat, t_flat)).reshape(XX.shape)
            rel_l2 = float(np.linalg.norm(u_pred - U_true) / np.linalg.norm(U_true))
            print(f"eps={eps} method={method} seed={seed}: rel_L2={rel_l2:.4f} "
                  f"time={time.time()-t0:.1f}s final_loss={history['L_total'][-1]:.3e}", flush=True)
            results[str(eps)][method].append({"seed": seed, "rel_l2": rel_l2,
                                               "time": time.time() - t0,
                                               "final_loss": history["L_total"][-1]})

with open(ROOT / "results" / "stiffness_sweep.json", "w") as f:
    json.dump(results, f)
print("STIFFNESS SWEEP DONE")