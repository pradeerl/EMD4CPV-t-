import matplotlib.pyplot as plt
import matplotlib
from tqdm import tqdm
import numpy as np
from scipy.integrate import quad
import random
import pandas as pd
from multiprocess import Pool
import ot
import ot.plot
import time
import cmath
from utils import *
from config import *
from joblib import Parallel, delayed

x_max  = 10.0             
N_evt  = 1000            
n_null = 1000             
n_weak = 20
n_pseudo = 100
numItermax  = 1000000

start_time = time.perf_counter()

rng = np.random.default_rng(42)

#Define coordinates and mask the non physical phase space
mask1 = mask(x_plus, x_minus, s_plus, s_minus, X_plus)
mask1 = mask1.ravel()

mask2 = mask(x_plus, x_minus, s_plus, s_minus, X_plus)

coords = np.stack([X_plus.ravel(), X_minus.ravel()], axis = 1)[~mask1]
dalitz_scale = float(coords.max() - coords.min())

#Set up the model using the Toy Lagrangian
def weak_num(i):
    weak_og = np.array([0, 0, i*np.pi/100])
    weak = weak_og[:, np.newaxis, np.newaxis]
    weak_final = np.broadcast_to(weak, (3, x_plus.shape[0], x_plus.shape[0]))
    return weak_final

def delta(s_plus):
    return np.arctan(m*Gamma/(s_plus - m**2))

def phase(i, tag):
    gam = (tag*weak_num(i))+np.array([delta(s_plus),delta(s_minus),delta(M**2-s_plus-s_minus)],dtype =float)
    return gam

def lam(i):
    g = np.array([3e-2, 3e-2, 3e-2]).reshape(3,1,1)
    h = np.array([3e-2, 3e-2, 3e-2]).reshape(3,1,1)
    lden = lnum = np.array([np.abs(BW(s_plus  / M**2, m/M)), np.abs(BW(s_minus / M**2, m/M)), np.abs(BW((M**2-s_plus-s_minus) / M**2, m/M))])

    gam = phase(i, 1)
    gambar = phase(i, -1)
    a = np.sum(g*h*lnum*np.exp(1j*np.array(gam, dtype = float)), axis = 0)
        #+1j*np.array([delta(s_plus),delta(s_minus),delta(1-s_plus-s_minus)],dtype =float)), axis = 0)
    abar = np.sum(g*h*lden*np.exp(1j*np.array(gambar, dtype = float)), axis = 0)
        #+1j*np.array([delta(s_plus),delta(s_minus),delta(1-s_plus-s_minus)],dtype =float)), axis = 0)
    lam = abar/a
    return lam

def amps(i):
    g = np.array([3e-2, 3e-2, 3e-2]).reshape(3,1,1); h = g.copy()
    l = np.array([np.abs(BW(s_plus  / M**2, m/M)),
                  np.abs(BW(s_minus / M**2, m/M)),
                  np.abs(BW((M**2 - s_plus - s_minus) / M**2, m/M))])
    a    = np.sum(g*h*l*np.exp(1j*np.array(phase(i,  1), dtype=float)), axis=0)
    abar = np.sum(g*h*l*np.exp(1j*np.array(phase(i, -1), dtype=float)), axis=0)
    return a, abar

#Define probabilities for Dalitz variable sampling
def sigma_flat(i):
    a, abar = amps(i)
    return (np.abs(a)**2 + np.abs(abar)**2).ravel()[~mask2]

def sigma_mix(i):
    a, abar = amps(0)
    return (np.abs(a)**2 + np.abs(abar)**2).ravel()[~mask2]


def dalitz_probs(C_flat, S_flat, sigma, tag):
    e, d = np.exp(-s*x_max), s*s + r*r
    I0 = (1 - e) / s
    Ic = (s - e*(s*np.cos(r*x_max) - r*np.sin(r*x_max))) / d
    Is = (r - e*(s*np.sin(r*x_max) + r*np.cos(r*x_max))) / d
    p = sigma * (I0 + tag*(C_flat*Ic - S_flat*Is))
    return p / p.sum()

def lam_mix(phase):
    return lam(0)*cmath.exp(-1j*phase)

def prob(events, tag, C, S):
    return np.exp(-s*events[:,0])*(1 + tag*(C*np.cos(r*events[:,0])-S*np.sin(r*events[:,0])))

#Definition of C and S
def CS(i):
    og = lam(i).shape
    temp = lam(i).reshape(og[0]*og[1])
    graph_s = np.zeros(len(temp))
    graph_c = (1 - np.abs(lam(i))**2)/(1+np.abs(lam(i))**2)
    for j in range(len(temp)):
        graph_s[j] = (2*np.imag(temp[j]))/(1+np.abs(temp[j])**2)
    graph_s = graph_s.reshape(og)
    return graph_c.ravel(), graph_s.ravel()

def CS_mix(i):
    og = lam_mix(i).shape
    temp = lam_mix(i).reshape(og[0]*og[1])
    graph_s = np.zeros(len(temp))
    graph_c = (1 - np.abs(lam_mix(i))**2)/(1+np.abs(lam_mix(i))**2)
    for j in range(len(temp)):
        graph_s[j] = (2*np.imag(temp[j]))/(1+np.abs(temp[j])**2)
    graph_s = graph_s.reshape(og)
    return graph_c.ravel(), graph_s.ravel()


mask2      = mask(x_plus, x_minus, s_plus, s_minus, X_plus).ravel()
coords     = np.stack([X_plus.ravel(), X_minus.ravel()], axis=1)[~mask2]
sig_ref = sigma_flat(0.0) 

            
#Sample times and Dalitz variables
def sample_events(N, C_flat, S_flat, tag, rng, sigma_flat):

    n_dal   = len(C_flat)

    dal_idx = rng.choice(len(C_flat), size=N, p=dalitz_probs(C_flat, S_flat, sigma_flat, tag))

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
        x2_evt
    ])



def events_to_weights(events):
    w = np.ones(len(events))/len(events)
    return w / w.sum()


def emd_between(events_a, events_b, weights_a, weights_b, w_dalitz=1.0):
    
    #Extract variables from events 
    du  = np.subtract.outer(events_a[:, 0], events_b[:, 0])
    dx1 = np.subtract.outer(events_a[:, 1], events_b[:, 1])
    dx2 = np.subtract.outer(events_a[:, 2], events_b[:, 2])

    #Define cost function with euclidean metric
    M = du**2 + w_dalitz**2 * (dx1**2 + dx2**2)
    M /= M.max()

    return float(ot.emd2(weights_a, weights_b, M, numItermax=10_000_000))

def run_one_phase(weak_phase, seed_master):
    rng = np.random.default_rng(seed_master)
    sig = sigma_flat(weak_phase)
    
    C_field, S_field = CS(weak_phase)
    C_flat = C_field[~mask2]
    S_flat = S_field[~mask2]

    T_obs  = np.zeros(n_pseudo)
    p_vals = np.zeros(n_pseudo)

    #Compute Test Statistic for pseudo experiments
    for j in tqdm(range(n_pseudo), desc = "EMD Calculation"):
        rng_j = np.random.default_rng(rng.integers(0, 2**31))
        #Sample events
        ev_B    = sample_events(N_evt, C_flat, S_flat, +1, rng_j, sigma_flat=sig)
        ev_Bbar = sample_events(N_evt, C_flat, S_flat, -1, rng_j, sigma_flat=sig)
        #Use uniform weights
        w_B    = events_to_weights(ev_B)
        w_Bbar = events_to_weights(ev_Bbar)

        T_obs[j] = emd_between(ev_B, ev_Bbar, w_B, w_Bbar)
    

        #Permutation null using the same events
        # Convert u back to t for time-flattening before pooling
        
        if j == n_pseudo - 1:
        
            t_B    = -np.log1p(-ev_B[:,0])    / s
            t_Bbar = -np.log1p(-ev_Bbar[:,0]) / s

            # Laplace weights to flatten time envelope
            wL_B    = np.exp(s * t_B);    wL_B    /= wL_B.sum()
            wL_Bbar = np.exp(s * t_Bbar); wL_Bbar /= wL_Bbar.sum()

            # Resample to flat time before pooling
            idx_B_f    = rng_j.choice(N_evt, N_evt, replace=True, p=wL_B)
            idx_Bbar_f = rng_j.choice(N_evt, N_evt, replace=True, p=wL_Bbar)
            pool = np.concatenate([ev_B[idx_B_f], ev_Bbar[idx_Bbar_f]], axis=0)

            # After flattening, use uniform weights
            
            w_unif = np.ones(N_evt) / N_evt
            #Compute null
            T_null_j = np.zeros(n_null)
            for k in range(n_null):
                perm = rng_j.permutation(2 * N_evt)
                h1   = pool[perm[:N_evt]]
                h2   = pool[perm[N_evt:]]
                T_null_j[k] = emd_between(h1, h2, w_unif, w_unif)

    return T_obs, T_null_j


weak_phases = np.linspace(0, 200, n_weak)

master_seed = np.random.SeedSequence(42)
seeds       = master_seed.spawn(n_weak)
seeds_int   = [int(s.generate_state(1)[0]) for s in seeds]


results = Parallel(n_jobs=-1, backend="loky")(
    delayed(run_one_phase)(phase_idx, seed_int)
    for phase_idx, seed_int in tqdm(zip(weak_phases, seeds_int), total=n_weak, desc="Weak phase scan"
    )
)


T_obs_all     = np.array([r[0] for r in results])
T_null_all    = np.array([r[1] for r in results])

np.save("T_obs_all_g3.npy",  T_obs_all)
np.save("T_null_all_g3.npy",  T_null_all)

print("Saved.")

end_time = time.perf_counter()
run_time = end_time - start_time

print(f"Time taken : {run_time:.6f} s")
