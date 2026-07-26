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
from utils import *
from config import *
from sklearn.utils import shuffle
from joblib import Parallel, delayed
from scipy.optimize import minimize
from scipy.stats import chi2

start_time = time.perf_counter()

n_draws = 100

def delta(s_plus):
        return np.arctan(m*Gamma/(s_plus - m**2))

def lambda_eps(eps):
    delta_plus  = delta(s_plus)
    delta_minus = delta(s_minus)
    delta_third = delta(M**2 - s_plus - s_minus)
    delta_arr   = np.array([delta_plus, delta_minus, delta_third])  # (3, 68, 68)

    k = np.array([eps*np.pi/100, 0, 0]).reshape(3, 1, 1)
    phi_weak = k
    
    BW_abs = np.array([
        np.abs(BW(s_plus  / M**2, m/M)),
        np.abs(BW(s_minus / M**2, m/M)),
        np.abs(BW((M**2 - s_plus - s_minus) / M**2, m/M))
    ])

    g = np.array([3e-2, 3e-2, 3e-2]).reshape(3, 1, 1)

    A    = np.sum(g * BW_abs * np.exp(1j * (phi_weak + delta_arr)), axis=0)
    Abar = np.sum(g * BW_abs * np.exp(1j * (-phi_weak + delta_arr)), axis=0)

    return Abar / A

def A_eps(eps):
    delta_plus  = delta(s_plus)
    delta_minus = delta(s_minus)
    delta_third = delta(M**2 - s_plus - s_minus)
    delta_arr   = np.array([delta_plus, delta_minus, delta_third])  # (3, 68, 68)

    k = np.array([eps*np.pi/100, 0, 0]).reshape(3, 1, 1)
    phi_weak = k
    
    BW_abs = np.array([
        np.abs(BW(s_plus  / M**2, m/M)),
        np.abs(BW(s_minus / M**2, m/M)),
        np.abs(BW((M**2 - s_plus - s_minus) / M**2, m/M))
    ])  

    g = np.array([3e-2, 3e-2, 3e-2]).reshape(3, 1, 1)

    A    = np.sum(g * BW_abs * np.exp(1j * (phi_weak + delta_arr)), axis=0)
    Abar = np.sum(g * BW_abs * np.exp(1j * (-phi_weak + delta_arr)), axis=0)

    return A


def CS_field(eps):
    lam = lambda_eps(eps)
    abs2 = np.abs(lam)**2
    C = (1.0 - abs2) / (1.0 + abs2)
    S = -2.0 * np.imag(lam) / (1.0 + abs2)
    return C, S  # each (68, 68)

C0, S0 = CS_field(0.0)  # should be ~0 everywhere

def sample_time_for_event(C_evt, S_evt, tag_evt, x_max, rng):

    # --- 1. Envelope for modulation term -------------------------------
    # The modulation term is bounded by sqrt(C^2 + S^2)
    amp = np.hypot(C_evt, S_evt)
    M = 1.0 + amp   # safe envelope for 1 + tag*(S sin - C cos)

    # --- 2. Rejection sampling loop -----------------------------------
    while True:
        # Proposal from truncated exponential
        u = rng.random()
        x_prop = -np.log(1 - u*(1 - np.exp(-s*x_max))) / s

        # Modulation factor
        mod = 1.0 + tag_evt*(S_evt*np.sin(r*x_prop) - C_evt*np.cos(r*x_prop))

        # If modulation dips below zero, reject immediately
        if mod <= 0:
            continue

        # Accept with probability mod / M
        if rng.random() < mod / M:
            return x_prop

def sample_dalitz_point(eps, rng):
    lam = lambda_eps(eps)
    
    delta_plus  = delta(s_plus)
    delta_minus = delta(s_minus)
    delta_third = delta(M**2 - s_plus - s_minus)
    delta_arr   = np.array([delta_plus, delta_minus, delta_third])

    k = np.array([eps*np.pi/100, 0, 0]).reshape(3,1,1)
    phi_weak = k
    
    BW_abs = np.array([
        np.abs(BW(s_plus  / M**2, m/M)),
        np.abs(BW(s_minus / M**2, m/M)),
        np.abs(BW((M**2 - s_plus - s_minus) / M**2, m/M))
    ])

    g = np.array([3e-2, 3e-2, 3e-2]).reshape(3,1,1)

    A = np.sum(g * BW_abs * np.exp(1j*(phi_weak + delta_arr)), axis=0)
    pdf = np.abs(A)**2
    pdf = pdf / pdf.sum()

    flat_idx = rng.choice(pdf.size, p=pdf.ravel())
    i_sp, i_sm = np.unravel_index(flat_idx, pdf.shape)
    return i_sp, i_sm

def log_likelihood_eps(events, eps):
    # Unpack event arrays
    x      = events[:, 0]
    i_sp   = events[:, 1].astype(int)
    i_sm   = events[:, 2].astype(int)
    tag    = events[:, 3]

    C_field, S_field = CS_field(eps)
    C_evt = C_field[i_sp, i_sm]
    S_evt = S_field[i_sp, i_sm]

    mod = 1.0 + tag * (S_evt * np.sin(r * x) - C_evt * np.cos(r * x))

    if np.any(mod <= 0):
        return -np.inf

    denom = s**2 + r**2
    Z_evt = 1/s + tag * (S_evt * (r/denom) - C_evt * (s/denom))
    
    logL = np.sum(np.log(mod) - s * x - np.log(Z_evt))
    return logL

def generate_events_vectorized(N, eps, x_max, tag, rng=None):
    
    if rng is None:
        rng = np.random.default_rng()

    C_field, S_field = CS_field(eps)

    delta_plus  = delta(s_plus)
    delta_minus = delta(s_minus)
    delta_third = delta(M**2 - s_plus - s_minus)
    delta_arr   = np.array([delta_plus, delta_minus, delta_third])

    k = np.array([eps*np.pi/100, 0, 0]).reshape(3, 1, 1)
    BW_abs = np.array([
        np.abs(BW(s_plus  / M**2, m/M)),
        np.abs(BW(s_minus / M**2, m/M)),
        np.abs(BW((M**2 - s_plus - s_minus) / M**2, m/M))
    ])
    g = np.array([3e-2, 3e-2, 3e-2]).reshape(3, 1, 1)

    A   = np.sum(g * BW_abs * np.exp(1j * (k + delta_arr)), axis=0)
    pdf = np.abs(A)**2
    pdf = pdf / pdf.sum()

    flat_idx = rng.choice(pdf.size, size=N, p=pdf.ravel())
    i_sp, i_sm = np.unravel_index(flat_idx, pdf.shape)   # each shape (N,)

    tag_evt = np.full(N, tag, dtype=float)

    C_evt = C_field[i_sp, i_sm]   
    S_evt = S_field[i_sp, i_sm]   

    x_evt = sample_time_vectorized(C_evt, S_evt, tag_evt, x_max, rng)

    events = np.stack([x_evt, i_sp.astype(float), i_sm.astype(float), tag_evt],
                       axis=1)
    return events


def sample_time_vectorized(C_evt, S_evt, tag_evt, x_max, rng, max_rounds=1000):
    
    N = len(C_evt)
    amp = np.hypot(C_evt, S_evt)
    M_env = 1.0 + amp                      

    x_out    = np.empty(N, dtype=float)
    pending  = np.arange(N)                

    Z_trunc = 1.0 - np.exp(-s * x_max)

    for _ in range(max_rounds):
        if len(pending) == 0:
            break

        n_pend = len(pending)
        u      = rng.random(n_pend)
        x_prop = -np.log(1 - Z_trunc * u) / s

        C_p   = C_evt[pending]
        S_p   = S_evt[pending]
        tag_p = tag_evt[pending]
        M_p   = M_env[pending]

        mod = 1.0 + tag_p * (S_p * np.sin(r * x_prop) - C_p * np.cos(r * x_prop))

        valid = mod > 0

        u2 = rng.random(n_pend)
        accept = valid & (u2 < (mod / M_p))

        accepted_idx = pending[accept]
        x_out[accepted_idx] = x_prop[accept]

        pending = pending[~accept]

    if len(pending) > 0:
        for idx in pending:
            x_out[idx] = sample_time_for_event(
                C_evt[idx], S_evt[idx], tag_evt[idx], x_max, rng)

    return x_out

from joblib import Parallel, delayed

def generate_one_set(eps, N, x_max, tag, seed):
    rng = np.random.default_rng(seed)
    return generate_events_vectorized(N, eps, x_max, tag, rng)

from scipy.optimize import differential_evolution, minimize

def fit_eps_DE(events, bounds=(0.0, 200.0)):
    
    def obj(eps_array):
        eps = eps_array[0]
        return -log_likelihood_eps(events, eps)

    # --- 1. Global search with DE -------------------------------------
    res_de = differential_evolution(
        obj,
        bounds=[bounds],
        strategy='best1bin',
        maxiter=2000,
        popsize=25,
        tol=1e-8,
        mutation=(0.5, 1.0),
        recombination=0.9,
        polish=False,     
        seed=42
    )

    eps_global = float(res_de.x[0])

    res_local = minimize(
        obj,
        x0=[eps_global],
        method='BFGS',
        options=dict(gtol=1e-8)
    )

    eps_hat = float(res_local.x[0])
    return eps_hat

def test_stat_DE(events):
    l0 = log_likelihood_eps(events, 0.0)
    eps_hat = fit_eps_DE(events)
    l1 = log_likelihood_eps(events, eps_hat)
    T = -2 * (l0 - l1)
    return T, eps_hat

def compute_cpv(eps, N_evt, x_max, rng=None):
    if rng is None:
        rng = np.random.default_rng()
        T_vals = np.zeros(n_draws)
        T_nvals = np.zeros(n_draws)
        eps_hat = np.zeros(n_draws)
    for i in range(n_draws):
        events_n = generate_one_set(0, N_evt, x_max, tag = 1, seed = int(5*eps))
        T_n , eps_hat_n = test_stat_DE(events_n)
        T_nvals[i] = T_n
        events = generate_one_set(eps, N_evt, x_max, tag = 1, seed = int(5*eps))
        T_k, eps_hat_k = test_stat_DE(events)
        T_vals[i] = T_k
        

    return T_vals, T_nvals

if __name__ == "__main__":

    eps_scan = np.linspace(0,200,10)

    results = np.array(Parallel(n_jobs=-1, backend="loky")(
            delayed(compute_cpv)(eps, N_evt = 1156, x_max = np.inf)
            for eps in tqdm(eps_scan, desc="Computing CPV")
        ), dtype = object)
    T_cpv = results[:,0]
    T_null = results[:,1]
    print(T_null.shape, T_cpv.shape)
    #np.save("null_dist_2d20.npy", T_null)
    #np.save("cpv_dist_2d_phasevary_alt20.npy", T_cpv)
    end_time = time.perf_counter()

    print("Time taken: ", end_time - start_time, "s.")
