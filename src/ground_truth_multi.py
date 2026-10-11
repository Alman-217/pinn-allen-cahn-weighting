from pathlib import Path
import time
import numpy as np
from scipy.integrate import solve_ivp

ROOT = Path(__file__).resolve().parent.parent


def solve_ground_truth(eps, nx, nt=201):
    x = np.linspace(-1, 1, nx, endpoint=False)
    dx = x[1] - x[0]
    t_eval = np.linspace(0, 1, nt)
    u0 = x**2 * np.cos(np.pi * x)

    def rhs(t, u):
        u_xx = (np.roll(u, -1) - 2 * u + np.roll(u, 1)) / dx**2
        return eps**2 * u_xx + u - u**3

    sol = solve_ivp(rhs, [0, 1], u0, t_eval=t_eval, method="Radau",
                    rtol=1e-9, atol=1e-10)
    return x, t_eval, sol.y.T


CONFIGS = [(0.03, 256), (0.01, 384), (0.003, 512), (0.001, 768), (0.0001, 1536)]

if __name__ == "__main__":
    out_dir = ROOT / "results" / "ground_truths"
    out_dir.mkdir(parents=True, exist_ok=True)
    for eps, nx in CONFIGS:
        t0 = time.time()
        x, t, U = solve_ground_truth(eps, nx)
        np.savez(out_dir / f"gt_eps{eps}.npz", x=x, t=t, U=U, eps=eps)
        print(f"eps={eps} nx={nx} done in {time.time()-t0:.1f}s, range=[{U.min():.3f},{U.max():.3f}]")