import matplotlib.pyplot as plt
import matplotlib
from tqdm import tqdm
import numpy as np
import random
from multiprocess import Pool
import ot
import ot.plot
import time
import cmath
from utils import *
from config import *
from joblib import Parallel, delayed
from scipy.stats import chi2
from scipy.optimize import differential_evolution, minimize

start_time = time.perf_counter()

n_draws = 1000

#Define coordinates and mask the non physical phase space
mask2 = mask(x_plus, x_minus, s_plus, s_minus, X_plus).ravel()
coords = np.stack([X_plus.ravel(), X_minus.ravel()], axis=1)[~mask2]
n_dal = len(coords)
x_max = 10

#Set up the model using the Toy Lagrangian
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

    return A, Abar

def sigma_flat(i):                      
    a, abar = A_eps(i)
    return (np.abs(a)**2 + np.abs(abar)**2).ravel()[~mask2]

#Define C and S
def CS_field(eps):
    lam = lambda_eps(eps)
    abs2 = np.abs(lam)**2
    C = (1.0 - abs2) / (1.0 + abs2)
    S = 2.0 * np.imag(lam) / (1.0 + abs2)
    return C.ravel()[~mask2], S.ravel()[~mask2]  

def CS_mix(eps):
    lam = lambda_eps(0)*cmath.exp(-I*eps)
    abs2 = np.abs(lam)**2
    C = (1.0 - abs2) / (1.0 + abs2)
    S = 2.0 * np.imag(lam) / (1.0 + abs2)
    return C[~mask2], S[~mask2]  

#Set reference values
sig_ref = sigma_flat(0.0)
C0, S0 = CS_field(0)


#Define probability of sampling Dalitz variables
def dalitz_probs(C_flat, S_flat, sigma, tag):
    e, d = np.exp(-s*x_max), s*s + r*r
    I0 = (1 - e) / s
    Ic = (s - e*(s*np.cos(r*x_max) - r*np.sin(r*x_max))) / d
    Is = (r - e*(s*np.sin(r*x_max) + r*np.cos(r*x_max))) / d
    p = sigma * (I0 + tag*(C_flat*Ic - S_flat*Is))
    return p / p.sum()

_e, _d = np.exp(-s * x_max), s * s + r * r
I0 = (1 - _e) / s
Ic = (s - _e * (s * np.cos(r * x_max) - r * np.sin(r * x_max))) / _d
Is = (r - _e * (s * np.sin(r * x_max) + r * np.cos(r * x_max))) / _d
 
#Sample times and Dalitz variables
def sample_events(N, C_flat, S_flat, sigma, tag, rng):
    n_dal   = len(C_flat)

    dal_idx = rng.choice(len(C_flat), size=N, p=dalitz_probs(C_flat, S_flat, sigma, tag))

    C_evt = C_flat[dal_idx]
    S_evt = S_flat[dal_idx]

    amp   = np.hypot(C_evt, S_evt)
    M_env = 1.0 + amp

    Z_trunc = 1.0 - np.exp(-s * x_max)

    t_out   = np.empty(N)
    pending = np.arange(N)

    while len(pending) > 0:

        n_p = len(pending)

        u = rng.random(n_p)

        t_prop = (
            -np.log(1.0 - Z_trunc * u)
            / s
        )

        C_p = C_evt[pending]
        S_p = S_evt[pending]
        M_p = M_env[pending]

        mod = (
            1.0
            + tag * (
                C_p * np.cos(r * t_prop)
                - S_p * np.sin(r * t_prop)
            )
        )

        mod = np.clip(mod, 0.0, None)

        accept = (
            rng.random(n_p)
            < mod / M_p
        )

        t_out[pending[accept]] = t_prop[accept]

        pending = pending[~accept]

    # Flatten exponential exactly
    u_out = 1.0 - np.exp(-s * t_out)

    x1_evt = coords[dal_idx, 0]
    x2_evt = coords[dal_idx, 1]

    return np.column_stack([
        u_out,
        x1_evt,
        x2_evt,
        dal_idx
    ])

#Define time integrals and normalisation factors
_e  = np.exp(-s * x_max)
_d  = s**2 + r**2
I0  = (1 - _e) / s
Ic  = (s - _e*(s*np.cos(r*x_max) - r*np.sin(r*x_max))) / _d
Is  = (r - _e*(s*np.sin(r*x_max) + r*np.cos(r*x_max))) / _d

#Calculate log likelihood for given eps
def log_likelihood_eps(ev_B, ev_Bbar, eps):
    C, S  = CS_field(eps)
    sigma = sig_ref
    
    ll = 0.0
    for ev, tag in ((ev_B, +1), (ev_Bbar, -1)):
        u = ev[:, 0]
        #Recover t
        t = -np.log1p(-u) / s
        b = ev[:, 3].astype(int)

        #Time-integrated weight at each bin
        w_int     = sigma * (I0 + tag * (C * Ic - S * Is))
        w_int     = np.maximum(w_int, 1e-300)
        log_p_dal = np.log(w_int[b]) - np.log(w_int.sum())

        #Time conditional: p(t|bin)
        mod = 1.0 + tag * (C[b] * np.cos(r * t) - S[b] * np.sin(r * t))
        mod = np.maximum(mod, 1e-300)

        #Per-bin normalisation Z_b = I0 + tag*(C_b*Ic - S_b*Is)
        Z_b = I0 + tag * (C[b] * Ic - S[b] * Is)
        Z_b = np.maximum(Z_b, 1e-300)

        log_p_t   = np.log(mod) - s * t - np.log(Z_b)

        ll += np.sum(log_p_dal + log_p_t)

    return ll

#MLE fit: find eps that maximises the combined log-likelihood    
def fit_eps_MLE(ev_B, ev_Bbar, eps, seed=42):
    def obj(p):
        return -log_likelihood_eps(ev_B, ev_Bbar, float(p[0]))

    res_de = differential_evolution(
        obj, bounds=[(0.0, 200)],
        strategy='best1bin', maxiter=500, popsize=15,
        tol=1e-6, polish=False, seed=seed, workers=1
    )
    res_loc = minimize(obj, x0=[res_de.x[0]], method='BFGS',
                       options=dict(gtol=1e-8))
    return float(res_loc.x[0])

#Calculate test statistic -2*(l(eps=0) - l(eps_hat))
def wilks_stat(ev_B, ev_Bbar, eps, seed=42):
    l0      = log_likelihood_eps(ev_B, ev_Bbar, 0.0)
    eps_hat = fit_eps_MLE(ev_B, ev_Bbar, eps, seed=seed)
    l1      = log_likelihood_eps(ev_B, ev_Bbar, eps_hat)
    T       = max(-2.0 * (l0 - l1), 0.0)
    return T, eps_hat

C_0,    S_0     = CS_field(0.0)

#Build null distribution once for all eps
def build_null(eps, N_evt, x_max):
    T_nvals = np.zeros(n_draws)
    for i in tqdm(range(n_draws)):
        seed_base = abs(int(eps * 1000)) * 10000 + i
        rng = np.random.default_rng(seed_base)

        # ── Null: generate at eps=0, fit eps, compute LR ──────────────────
        ev_n_B    = sample_events(N_evt, C_0, S_0, sig_ref, +1, rng)
        ev_n_Bbar = sample_events(N_evt, C_0, S_0, sig_ref, -1, rng)
        T_nvals[i], _ = wilks_stat(ev_n_B, ev_n_Bbar, 0.0, seed=seed_base)
    return T_nvals

#Compute test statistic for CPV events
def compute_cpv(eps, N_evt, x_max):
    T_vals  = np.zeros(n_draws)
    T_nvals = np.zeros(n_draws)
    sig = sig_ref
    C_field, S_field = CS_field(eps)
    
    for i in tqdm(range(n_draws)):
        seed_base = abs(int(eps * 1000)) * 10000 + i
        rng = np.random.default_rng(seed_base)
        ev_B    = sample_events(N_evt, C_field, S_field, sig, +1, rng)
        ev_Bbar = sample_events(N_evt, C_field, S_field, sig, -1, rng)
        
        T_vals[i], _ = wilks_stat(ev_B, ev_Bbar, eps, seed=seed_base + 1)

    return T_vals

if __name__ == "__main__":

    #Scan through all weak phases and save results
    eps_scan = np.linspace(0,200,20)

    results = np.array(Parallel(n_jobs=-1, backend="loky")(
            delayed(compute_cpv)(eps, N_evt = 1000, x_max = 10)
            for eps in tqdm(eps_scan, desc="Computing CPV")
        ), dtype = object)
    results_null = build_null(0.0, N_evt = 1000, x_max = 10)
    T_cpv = results
    T_null = results_null

    
    np.save("NP_null_g1.npy", T_null)    
    np.save("NP_cpv_g1.npy", T_cpv)
    
    end_time = time.perf_counter()

    print("Time taken: ", end_time - start_time, "s.")
