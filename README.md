# Adaptive Loss-Weighting for PINNs on the Allen-Cahn Equation

A small, reproducible comparison of four loss-weighting strategies for physics-informed neural networks (PINNs) on the 1D Allen-Cahn equation, plus a follow-up stiffness sweep that turned out to say more about the benchmark than about the methods.

## The problem

```
u_t = eps^2 * u_xx + u - u^3,    x in [-1, 1],  t in [0, 1]
u(x, 0) = x^2 * cos(pi * x)
periodic boundary conditions
```

PINNs are trained by minimizing a weighted sum of a PDE-residual loss, an initial-condition loss and a boundary-condition loss. When these terms have very different gradient scales, training can satisfy the dominant term and neglect the others. This project compares four ways of handling that:

| Method | Idea |
|---|---|
| `baseline` | Fixed, equal loss weights (vanilla PINN) |
| `gradnorm` | Heuristic gradient-norm balancing, updated every 200 steps with a moving average |
| `self_adaptive` | A trainable weight per collocation, IC and BC point, updated by gradient ascent while the network descends (SA-PINN, McClenny and Braga-Neto, 2020) |
| `sa_rad` | Self-adaptive weights plus residual-based adaptive resampling of collocation points |

All methods share the same network (MLP `[2, 64, 64, 64, 64, 1]`, tanh), optimizer (Adam, learning rate 1e-3), training budget (6,000 steps) and point counts (400 residual, 100 IC, 100 BC). Error is the relative L2 error against a finite-difference reference solution (method of lines, Radau integrator).

## Results

### Comparison at eps = 0.1 (3 seeds)

| Method | Mean rel. L2 | Std | Min | Max |
|---|---|---|---|---|
| Baseline | 0.342 | 0.174 | 0.164 | 0.578 |
| Gradient-norm balancing | 0.310 | 0.188 | 0.129 | 0.570 |
| Self-adaptive (SA-PINN) | 0.448 | 0.172 | 0.277 | 0.684 |
| Self-adaptive + RAD | 0.450 | 0.203 | 0.255 | 0.730 |

Gradient-norm balancing has the lowest mean error and the best single run, and both self-adaptive variants have the highest mean error. **These gaps are small relative to the seed-to-seed spread (std about 0.18), so with three seeds none of the differences is statistically established.** Every method shows large run-to-run variance, and seed 0 is the hardest seed for all of them.

### Stiffness sweep (eps = 0.03 down to 0.0001; 3 methods x 3 seeds each)

- Mean error is essentially flat across the sweep. Each method's mean stays within about 0.03 of itself (roughly 0.40 to 0.45) for every eps.
- The flat result comes from the benchmark, not the methods. The initial condition places a broad central region near the unstable equilibrium u = 0, and the diffusion-driven escape from it saturates below eps of about 0.01 within t in [0, 1]. The true solution at a fixed point (x = 0.25, t = 1) stops changing there, so the regression target stops getting harder. Doubling the reference grid resolution changed the result by less than 1%, so this is not a grid artifact.
- Self-adaptive weighting has a much larger spread across seeds (std about 0.28 versus about 0.16 for the baseline). In this data it is worse than the baseline on seed 0 and better on seeds 1 and 2 at every eps. With three seeds this is a hypothesis, not an established effect.

**Lesson for benchmark design:** a stiffness sweep is only informative if the reference solution actually changes as the parameter shrinks. Check that before reading anything into a flat method comparison.

## Limitations

- Reduced scale: 6,000 Adam steps, 3 seeds, one PDE, one network size.
- No hyperparameter search. The self-adaptive weight learning rate (1e-2) and the gradient-norm moving average were not tuned.
- The original SA-PINN paper reports self-adaptive weighting winning on Allen-Cahn. This project does not reproduce that, and it does not use the same equation: the paper's form has a 5x-rescaled reaction term and an unsquared diffusion coefficient, while this project uses `u_t = eps^2 u_xx + u - u^3`. A faithful replication would need to match their exact equation, network size and training length. `src/run_replication_check.py` is an unfinished start on this and uses this project's equation, so its results would not be a true replication.
- The saturation finding is specific to this initial condition and time horizon.

## Setup

Requires Python 3.10 or newer.

```bash
python -m venv venv
# Windows:      venv\Scripts\activate
# Mac / Linux:  source venv/bin/activate
pip install "jax[cpu]" numpy scipy matplotlib
```

## Reproducing everything

From the project root:

```bash
python src/run_all.py                # everything, about 40 minutes on CPU
python src/run_all.py --skip-sweep   # skip the 45-run stiffness sweep, about 10 minutes
```

`run_all.py` runs these steps in order and stops at the first failure:

1. `ground_truth.py` (reference solution, eps = 0.1)
2. `ground_truth_multi.py` (reference solutions for the sweep)
3. `run_experiments.py` (4 methods x 3 seeds at eps = 0.1)
4. `make_figures.py` (figures 1 to 5)
5. `run_stiffness_sweep.py` (3 methods x 3 seeds x 5 eps values)
6. `make_sweep_figures.py` (figures 6 and 7)

Generated `.npz` files are not committed. They are rebuilt by the scripts above (see `.gitignore`).

**Expect small numerical differences.** Training a PINN is sensitive to floating-point details, so a rerun on another machine or JAX version will give slightly different numbers. The qualitative findings (flat sweep, high seed variance, no clear separation between methods) should hold; exact values will not match to the last digit.

## Project layout

```
src/
  pinn_core.py              MLP, Allen-Cahn residual, IC/BC residuals (JAX autodiff)
  train.py                  original training loops for the four methods (eps = 0.1)
  train_v2.py               generalized trainer: eps and network size as parameters
  ground_truth.py           finite-difference reference, eps = 0.1
  ground_truth_multi.py     references for the stiffness sweep
  run_experiments.py        main comparison
  run_stiffness_sweep.py    stiffness sweep
  run_replication_check.py  unfinished, see Limitations
  make_figures.py           figures 1 to 5
  make_sweep_figures.py     figures 6 and 7
  run_all.py                runs the whole pipeline
results/                    JSON results (summary.json, all_results.json, stiffness_sweep.json)
figures/                    generated figures
paper/                      write-up
```

## References

- Raissi, Perdikaris, Karniadakis (2019). Physics-informed neural networks. *Journal of Computational Physics* 378.
- McClenny, Braga-Neto (2020). Self-adaptive physics-informed neural networks using a soft attention mechanism. arXiv:2009.04544.
- Wang, Teng, Perdikaris (2021). Understanding and mitigating gradient pathologies in physics-informed neural networks. *SIAM J. Sci. Comput.* 43(5).
- Krishnapriyan et al. (2021). Characterizing possible failure modes in physics-informed neural networks. *NeurIPS* 34.
- Lu, Meng, Mao, Karniadakis (2021). DeepXDE: A deep learning library for solving differential equations. *SIAM Review* 63(1).
