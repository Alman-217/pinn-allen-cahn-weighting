from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp

ROOT = Path(__file__).resolve().parent.parent
EPS = 0.1


def solve_ground_truth(nx=256, nt=201, eps=EPS, save_path=None):
    x = np.linspace(-1, 1, nx, endpoint=False)  # periodic grid
    dx = x[1] - x[0]
    t_eval = np.linspace(0, 1, nt)
    u0 = x**2 * np.cos(np.pi * x)

    def rhs(t, u):
        u_xx = (np.roll(u, -1) - 2 * u + np.roll(u, 1)) / dx**2
        return eps**2 * u_xx + u - u**3

    sol = solve_ivp(rhs, [0, 1], u0, t_eval=t_eval, method="Radau",
                    rtol=1e-9, atol=1e-10)
    U = sol.y.T  # shape (nt, nx)
    if save_path:
        np.savez(save_path, x=x, t=t_eval, U=U, eps=eps)
    return x, t_eval, U


if __name__ == "__main__":
    (ROOT / "results").mkdir(exist_ok=True)
    x, t, U = solve_ground_truth(save_path=ROOT / "results" / "ground_truth.npz")
    print("Ground truth solved. Shape:", U.shape, "range:", U.min(), U.max())