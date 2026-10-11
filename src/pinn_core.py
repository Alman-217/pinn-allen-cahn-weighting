import jax
import jax.numpy as jnp
from jax import random
import numpy as np

EPS = 0.1
HARD_D = 0.0001
HARD_A = 5.0    

def init_mlp(layer_sizes, key):
    params = []
    keys = random.split(key, len(layer_sizes) - 1)
    for i, k in enumerate(keys):
        fan_in, fan_out = layer_sizes[i], layer_sizes[i + 1]
        std = jnp.sqrt(2.0 / (fan_in + fan_out))  # Xavier
        w = random.normal(k, (fan_in, fan_out)) * std
        b = jnp.zeros(fan_out)
        params.append((w, b))
    return params


def mlp_forward(params, xt):
    h = xt
    for (w, b) in params[:-1]:
        h = jnp.tanh(h @ w + b)
    w, b = params[-1]
    return (h @ w + b)[..., 0]


def u_fn(params, x, t):
    xt = jnp.stack([x, t], axis=-1)
    return mlp_forward(params, xt)

def residual_fn(params, x, t, eps=EPS, mode="easy"):
    def u_scalar(x_, t_):
        return u_fn(params, x_[None], t_[None])[0]

    u_t = jax.vmap(jax.grad(u_scalar, argnums=1))(x, t)
    u_x = jax.vmap(jax.grad(u_scalar, argnums=0))(x, t)

    def u_x_scalar(x_, t_):
        return jax.grad(u_scalar, argnums=0)(x_, t_)

    u_xx = jax.vmap(jax.grad(u_x_scalar, argnums=0))(x, t)
    u = u_fn(params, x, t)
    if mode == "easy":
        return u_t - eps**2 * u_xx - u + u**3
    else: 
        return u_t - HARD_D * u_xx + HARD_A * u**3 - HARD_A * u


def bc_residual(params, t, eps=EPS):
    xl = -jnp.ones_like(t)
    xr = jnp.ones_like(t)
    u_l = u_fn(params, xl, t)
    u_r = u_fn(params, xr, t)

    def u_scalar(x_, t_):
        return u_fn(params, x_[None], t_[None])[0]

    ux_l = jax.vmap(jax.grad(u_scalar, argnums=0))(xl, t)
    ux_r = jax.vmap(jax.grad(u_scalar, argnums=0))(xr, t)
    return (u_l - u_r), (ux_l - ux_r)


def ic_residual(params, x):
    t0 = jnp.zeros_like(x)
    u0_true = x**2 * jnp.cos(jnp.pi * x)
    u0_pred = u_fn(params, x, t0)
    return u0_pred - u0_true

def sample_points(key, n_f, n_ic, n_bc):
    k1, k2, k3 = random.split(key, 3)
    x_f = random.uniform(k1, (n_f,), minval=-1.0, maxval=1.0)
    t_f = random.uniform(k1, (n_f,), minval=0.0, maxval=1.0)
    x_ic = random.uniform(k2, (n_ic,), minval=-1.0, maxval=1.0)
    t_bc = random.uniform(k3, (n_bc,), minval=0.0, maxval=1.0)
    return x_f, t_f, x_ic, t_bc
