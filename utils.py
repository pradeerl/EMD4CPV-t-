import matplotlib.pyplot as plt
import matplotlib
from tqdm import tqdm
import numpy as np
from scipy.integrate import quad
import random
import pandas as pd
from sympy import Symbol,integrate,exp,cos,sin,I,simplify
import seaborn as sns
from multiprocess import Pool
import ot
import ot.plot
from scipy.spatial import distance
from sklearn.preprocessing import normalize
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

from config import *

from config import (x_max, n_iter, N_evt, M, mpi, mpi0, Grho, s, r, A_plus_fit,
    A_zero_fit, A_minus_fit, Abar_plus_fit, Abar_zero_fit, Abar_minus_fit)

# --- Sampling utilities ---

def _trunc_exp_rvs(n, x_max, rng):
    """
    Draw n samples from Exp(Gamma) truncated to [0, x_max].
    Inverse-CDF sampling for stability.
    """
    u = rng.random(n)
    Z = 1.0 - np.exp(-s*x_max)  # truncation norm
    return -np.log(1.0 - Z * u)/s 

def _modulation(x, C, S, tag):
    """
    Modulating factor 1 + tag*(S*sin(Δm t) - C*cos(Δm t)).
    (tag = +1 for \bar{B}^0, tag = -1 for B^0 in the convention used here)
    """
    return 1.0 + tag * (S * np.sin(r * x) - C * np.cos(r * x))


def _pmax_bound(C, S):
    """
    A tight, t-independent upper bound on the modulation.
    For any t, max over tag in {±1} of [1 + tag*(S*sin - C*cos)] <= 1 + sqrt(C^2 + S^2).
    """
    return 1.0 + np.hypot(C, S)

def sample_tagged_times_ij(n_iter, N_evt, C, S, tag, x_max, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    # Safety: if modulation dips below zero anywhere, the model is unphysical.
    # We still sample safely by clipping the modulation at zero inside rejection,
    # but warn the user.
    if 1.0 - np.hypot(C, S) < 0:
        # This can happen if |C|^2 + |S|^2 > 1 or large coefficients are provided.
        # We won't raise, but it's worth flagging upstream if needed.
        pass

    M = _pmax_bound(C, S)  # envelope on the modulation
    total = n_iter * N_evt
    x_samples = np.empty(total, dtype=float)

    got = 0
    trials = 0
    batch = max(256, int(1.5 * total))  # vectorized batches for speed

    while got < total:
        x_prop = _trunc_exp_rvs(batch, x_max, rng)  # proposal times from truncated exp
        w = _modulation(x_prop, C, S, tag)
        w = np.clip(w, 0.0, None)  # robust if user picks unphysical (C,S)
        u = rng.random(batch)
        keep = u < (w / M)  # rejection step relative to envelope
        n_keep = min(keep.sum(), total - got)
        if n_keep > 0:
            x_samples[got:got+n_keep] = x_prop[keep][:n_keep]
            got += n_keep
        trials += batch

    acc = total / trials
    #return x_samples.reshape(n_iter, N_evt), acc
    return x_samples, acc

# -------Amplitude and Dalitz limits -------------

def BW(s, m, gamma=147.4):
    return 1 / (s - m**2 + 1j*m*gamma)

def amp(s_plus, s_minus, A_plus, A_minus, A_zero, m):
    s_0 = 1 - s_plus - s_minus
    return BW(s_plus*M**2, m)*A_plus + BW(s_minus*M**2, m)*A_minus + BW(s_0*M**2, m)*A_zero

def E2_star(S_plus):
    return np.emath.sqrt(S_plus)/2
def E3_star(S_plus):
    return (M**2 - S_plus - mpi**2)/(2*np.emath.sqrt(S_plus))

def s_minus_min(s_plus):
    return (E2_star(s_plus)+E3_star(s_plus))**2 - (np.emath.sqrt((E2_star(s_plus))**2-mpi**2) + (np.emath.sqrt((E3_star(s_plus))**2-mpi**2)))**2

def s_minus_max(s_plus):
    return (E2_star(s_plus)+E3_star(s_plus))**2 - (np.emath.sqrt((E2_star(s_plus))**2-mpi**2) - (np.emath.sqrt((E3_star(s_plus))**2-mpi**2)))**2

# ----Dalitz variables -------

x_plus  = np.arange((2*mpi)**2/M**2, (M-mpi)**2/M**2, Grho/(2*M))
s2_min = min(s_minus_min(x_plus*M**2))/M**2
s2_max = max(s_minus_max(x_plus*M**2))/M**2

x_minus = np.arange(s2_min, s2_max, Grho/(2*M))

X_plus, X_minus = np.meshgrid(x_plus, x_minus)

s_plus = X_plus * M**2
s_minus = X_minus * M**2

orig_dim = len(x_plus)

# ----Masking ---------

def mask(x_plus, x_minus, s_plus, s_minus):
    mask1 = np.zeros_like(X_plus, dtype=bool)
    for i in tqdm(range(len(x_minus)), desc = "Masking"):
        for j in range(len(x_plus)):
            if not inside_dalitz(s_plus[i,j], s_minus[i,j], M, m1, m2, m3):
                mask1[i,j] = True
    return mask1

def inside_dalitz(s_plus, s_minus, M, m1, m2, m3):
    s0 = M**2 - s_plus - s_minus
    if (s_plus < (4*mpi**2)):
        return False    
    if (s_minus < s_minus_min(s_plus)):
        return False
    if (s_plus > ((M-mpi)**2)):
        return False
    if (s_minus > s_minus_max(s_plus)):
        return False
    else:
        return True

# -------Calculate C and S from model ------------

def plotted_c(A_plus, A_minus, A_zero,
        Abar_plus, Abar_minus, Abar_zero):
    A = amp(X_plus, X_minus, A_plus, A_minus, A_zero, m=m)
    Abar = amp(X_plus, X_minus, Abar_plus, Abar_minus, Abar_zero, m=m)
    lam = Abar/A
    C = (1 - (np.abs(lam))**2)/(1 + (np.abs(lam))**2)
    return C

def plotted_s(A_plus, A_minus, A_zero,
        Abar_plus, Abar_minus, Abar_zero):
    A = amp(X_plus, X_minus, A_plus, A_minus, A_zero, m=m)
    Abar = amp(X_plus, X_minus, Abar_plus, Abar_minus, Abar_zero, m=m)
    lam = Abar/A
    C = (1 - (np.abs(lam))**2)/(1 + (np.abs(lam))**2)
    S = -2*np.imag(lam)/(1 + (np.abs(lam))**2)
    return S

cfunc = plotted_c(A_plus_fit, A_minus_fit, A_zero_fit,
        Abar_plus_fit, Abar_minus_fit, Abar_zero_fit)

sfunc = plotted_s(A_plus_fit, A_minus_fit, A_zero_fit,
        Abar_plus_fit, Abar_minus_fit, Abar_zero_fit)

# ----------Calculate C and S from Lapalace transform ------------

def estimate_CS_for_block(x, x_max, s, r):
    n_bins = 2 * x_max

    bins = x.astype(int)
    valid = (bins >= 0) & (bins < n_bins)
    bins = np.where(valid, bins, -1)

    denom     = np.exp(s * x)
    num_cos = denom * np.cos(r * x)
    num_sin = denom * np.sin(r * x)

    denom     = np.where(valid, denom, 0.0)
    num_cos = np.where(valid, num_cos, 0.0)
    num_sin = np.where(valid, num_sin, 0.0)

    bins_flat = bins.ravel()
    valid_flat = bins_flat >= 0
    bins_flat = bins_flat[valid_flat]

    denom_flat    = denom.ravel()[valid_flat]
    num_cos_flat = num_cos.ravel()[valid_flat]
    num_sin_flat = num_sin.ravel()[valid_flat]

    total_idx = n_bins
    n_k      = np.bincount(bins_flat, minlength=total_idx)
    sum_denom    = np.bincount(bins_flat, weights=denom_flat,    minlength=total_idx)
    sum_num_cos = np.bincount(bins_flat, weights=num_cos_flat, minlength=total_idx)
    sum_num_sin = np.bincount(bins_flat, weights=num_sin_flat, minlength=total_idx)

    mask = sum_denom > 0
    C_k = np.zeros_like(sum_denom)
    S_k = np.zeros_like(sum_denom)

    C_k[mask] = 2.0 * (sum_num_cos[mask] / sum_denom[mask])
    S_k[mask] = -2.0 * (sum_num_sin[mask] / sum_denom[mask])

    N_tot = n_k.sum()
    C_est = (C_k * n_k).sum() / N_tot
    S_est = (S_k * n_k).sum() / N_tot
    return C_est, S_est

def compute_CS_grid(cfunc, sfunc, x_max, n_iter, N_evt, s, r, tag=-1):
    N_i, N_j = cfunc.shape
    C_grid = np.empty((N_i, N_j))
    S_grid = np.empty((N_i, N_j))
    p = np.empty(N_i* N_j)

    for i in tqdm(range(N_i), desc="Grid"):
        for j in range(N_j):
            x_ij, _ = sample_tagged_times_ij(
                n_iter=n_iter,
                N_evt=N_evt,
                C=cfunc[i,j],
                S=sfunc[i,j],
                tag=tag,
                x_max=x_max,
                rng=None
            )
            C_ij, S_ij = estimate_CS_for_block(x_ij, x_max, s, r)
            C_grid[i, j] = C_ij
            S_grid[i, j] = S_ij
    return C_grid, S_grid