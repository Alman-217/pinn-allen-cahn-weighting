import jax
import jax.numpy as jnp
from jax import random
from functools import partial
import numpy as np
import time

from pinn_core import init_mlp, residual_fn, bc_residual, ic_residual, sample_points

LR = 1e-3
LAM_LR_DEFAULT = 1e-2
LOG_EVERY = 400
GRADNORM_EVERY = 200
RAD_EVERY = 1000


def adam_init(p):
    return (jax.tree_util.tree_map(jnp.zeros_like, p),
            jax.tree_util.tree_map(jnp.zeros_like, p))


def adam_update(params, grads, m, v, t, lr, b1=0.9, b2=0.999, eps_adam=1e-8):
    m = jax.tree_util.tree_map(lambda m_, g: b1 * m_ + (1 - b1) * g, m, grads)
    v = jax.tree_util.tree_map(lambda v_, g: b2 * v_ + (1 - b2) * g**2, v, grads)
    mhat = jax.tree_util.tree_map(lambda m_: m_ / (1 - b1**t), m)
    vhat = jax.tree_util.tree_map(lambda v_: v_ / (1 - b2**t), v)
    new_params = jax.tree_util.tree_map(
        lambda p, mh, vh: p - lr * mh / (jnp.sqrt(vh) + eps_adam), params, mhat, vhat)
    return new_params, m, v


def term_losses(p, x_f, t_f, x_ic, x_bc_t, eps):
    r = residual_fn(p, x_f, t_f, eps)
    ic = ic_residual(p, x_ic)
    bc_u, bc_ux = bc_residual(p, x_bc_t)
    return r, ic, bc_u, bc_ux


def weighted_total(mode, p, x_f, t_f, x_ic, x_bc_t, lam_r, lam_ic, lam_bc, w_ic, w_bc, eps):
    r, ic, bc_u, bc_ux = term_losses(p, x_f, t_f, x_ic, x_bc_t, eps)
    if mode == "baseline":
        L_r = jnp.mean(r**2); L_ic = jnp.mean(ic**2)
        L_bc = jnp.mean(bc_u**2) + jnp.mean(bc_ux**2)
    elif mode == "gradnorm":
        L_r = jnp.mean(r**2)
        L_ic = w_ic * jnp.mean(ic**2)
        L_bc = w_bc * (jnp.mean(bc_u**2) + jnp.mean(bc_ux**2))
    else:
        L_r = jnp.mean((lam_r**2) * r**2)
        L_ic = jnp.mean((lam_ic**2) * ic**2)
        L_bc = jnp.mean((lam_bc**2) * bc_u**2) + jnp.mean((lam_bc**2) * bc_ux**2)
    return L_r + L_ic + L_bc, (L_r, L_ic, L_bc)


@partial(jax.jit, static_argnums=(0,))
def param_step(mode, params, m_p, v_p, epoch, x_f, t_f, x_ic, x_bc_t,
               lam_r, lam_ic, lam_bc, w_ic, w_bc, eps):
    grad_fn = jax.grad(lambda p: weighted_total(
        mode, p, x_f, t_f, x_ic, x_bc_t, lam_r, lam_ic, lam_bc, w_ic, w_bc, eps)[0])
    grads = grad_fn(params)
    params, m_p, v_p = adam_update(params, grads, m_p, v_p, epoch, LR)
    return params, m_p, v_p


@jax.jit
def lambda_step(params, x_f, t_f, x_ic, x_bc_t, lam_r, lam_ic, lam_bc,
                 m_lr, v_lr, m_lic, v_lic, m_lbc, v_lbc, epoch, eps, lam_lr):
    def neg_total(lr_, lic_, lbc_):
        total, _ = weighted_total("self_adaptive", params, x_f, t_f, x_ic, x_bc_t,
                                   lr_, lic_, lbc_, 1.0, 1.0, eps)
        return -total
    g_lr, g_lic, g_lbc = jax.grad(neg_total, argnums=(0, 1, 2))(lam_r, lam_ic, lam_bc)
    lam_r, m_lr, v_lr = adam_update(lam_r, g_lr, m_lr, v_lr, epoch, lam_lr)
    lam_ic, m_lic, v_lic = adam_update(lam_ic, g_lic, m_lic, v_lic, epoch, lam_lr)
    lam_bc, m_lbc, v_lbc = adam_update(lam_bc, g_lbc, m_lbc, v_lbc, epoch, lam_lr)
    return lam_r, lam_ic, lam_bc, m_lr, v_lr, m_lic, v_lic, m_lbc, v_lbc


def grad_norm_of(g):
    flat = jax.tree_util.tree_leaves(g)
    return sum(jnp.sum(jnp.abs(x)) for x in flat)


def train(seed, mode, eps, epochs=6000, layers=(2, 64, 64, 64, 64, 1),
          n_f=400, n_ic=100, n_bc=100, lam_lr=LAM_LR_DEFAULT, log_every=None):
    log_every = log_every or LOG_EVERY
    key = random.PRNGKey(seed)
    key, pkey, dkey = random.split(key, 3)
    params = init_mlp(list(layers), pkey)
    m_p, v_p = adam_init(params)
    x_f, t_f, x_ic, x_bc_t = sample_points(dkey, n_f, n_ic, n_bc)

    lam_r = jnp.ones(n_f); lam_ic = jnp.ones(n_ic); lam_bc = jnp.ones(n_bc)
    w_ic, w_bc = 1.0, 1.0

    if mode in ("self_adaptive", "sa_rad"):
        m_lr, v_lr = adam_init(lam_r)
        m_lic, v_lic = adam_init(lam_ic)
        m_lbc, v_lbc = adam_init(lam_bc)

    history = {"epoch": [], "L_total": []}
    t_start = time.time()

    for epoch in range(1, epochs + 1):
        params, m_p, v_p = param_step(mode, params, m_p, v_p, epoch,
                                       x_f, t_f, x_ic, x_bc_t,
                                       lam_r, lam_ic, lam_bc, w_ic, w_bc, eps)

        if mode in ("self_adaptive", "sa_rad"):
            (lam_r, lam_ic, lam_bc, m_lr, v_lr, m_lic, v_lic, m_lbc, v_lbc) = lambda_step(
                params, x_f, t_f, x_ic, x_bc_t, lam_r, lam_ic, lam_bc,
                m_lr, v_lr, m_lic, v_lic, m_lbc, v_lbc, epoch, eps, lam_lr)

        if mode == "gradnorm" and epoch % GRADNORM_EVERY == 0:
            gr = jax.grad(lambda p: jnp.mean(residual_fn(p, x_f, t_f, eps)**2))(params)
            gic = jax.grad(lambda p: jnp.mean(ic_residual(p, x_ic)**2))(params)
            def bc_loss(p):
                bc_u, bc_ux = bc_residual(p, x_bc_t)
                return jnp.mean(bc_u**2) + jnp.mean(bc_ux**2)
            gbc = jax.grad(bc_loss)(params)
            gr_n, gic_n, gbc_n = (float(grad_norm_of(gr)), float(grad_norm_of(gic)), float(grad_norm_of(gbc)))
            w_ic = 0.9 * w_ic + 0.1 * (gr_n / (gic_n + 1e-8))
            w_bc = 0.9 * w_bc + 0.1 * (gr_n / (gbc_n + 1e-8))

        if mode == "sa_rad" and epoch % RAD_EVERY == 0 and epoch < epochs:
            key, rk = random.split(key)
            cand_x = random.uniform(rk, (2000,), minval=-1.0, maxval=1.0)
            cand_t = random.uniform(rk, (2000,), minval=0.0, maxval=1.0)
            cand_r = jnp.abs(residual_fn(params, cand_x, cand_t, eps))
            probs = np.array(cand_r / (jnp.sum(cand_r) + 1e-8))
            idx = np.random.choice(len(cand_x), size=n_f, replace=True, p=probs)
            x_f, t_f = cand_x[idx], cand_t[idx]
            lam_r = jnp.ones(n_f)
            m_lr, v_lr = adam_init(lam_r)

        if epoch % log_every == 0 or epoch == 1:
            _, (L_r, L_ic, L_bc) = weighted_total(
                mode, params, x_f, t_f, x_ic, x_bc_t, lam_r, lam_ic, lam_bc, w_ic, w_bc, eps)
            history["epoch"].append(epoch)
            history["L_total"].append(float(L_r + L_ic + L_bc))

    elapsed = time.time() - t_start
    return params, history, elapsed
