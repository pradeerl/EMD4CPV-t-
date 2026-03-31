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
from utils import sample_tagged_times_ij, inside_dalitz
#from OptimizeTest import compute_CS_grid, sample_tagged_times_ij, inside_dalitz

start_time = time.perf_counter()

A_plus_fit = 0.7791729178283396+0j
A_minus_fit = 0.26905935233643435-0.5022575110480937j
A_zero_fit = 0.32222083425921166+0.3379435952217144j
Abar_plus_fit = 0.626809033217352+0j
Abar_minus_fit = 0.585846194443791-0.7817779784075829j
Abar_zero_fit = 0.2092787058892684+0.18609189622307143j

x_max = 10
n_iter = 1
N_evt = 1000

Gd=1.61e-6 #(MeV)
Grho=147.4 #(MeV)
md=1864.84
mrho=775.26
mpi0=135
mpi=140

gamma = 147.4         # MeV from rho
dm = 2.11e-9          # 0.51 ps^-1
m = 770
m_b = 5280
m_pi = 135

M   = m_b      # your B mass
m1  = mpi      # pi+
m2  = mpi      # pi-
m3  = mpi0     # pi0

x_grid = np.linspace(0, 40, 30000)  # ps (for plotting/reference)
Gamma = 2.746e-9 #MeV
Delta_m = 2.11e-9          # MeV
T=2*np.pi/Delta_m
r= Delta_m*T
s= T*Gamma
C=0.3
S=0.4

def BW(s, m, gamma=147.4):
    return 1 / (s - m**2 + 1j*m*gamma)

def amp(s_plus, s_minus, A_plus, A_minus, A_zero, m):
    s_0 = 1 - s_plus - s_minus
    return BW(s_plus*m_b**2, m)*A_plus + BW(s_minus*m_b**2, m)*A_minus + BW(s_0*m_b**2, m)*A_zero

N=400

def E2_star(S_plus):
    return np.emath.sqrt(S_plus)/2
def E3_star(S_plus):
    return (M**2 - S_plus - mpi**2)/(2*np.emath.sqrt(S_plus))

def s_minus_min(s_plus):
    return (E2_star(s_plus)+E3_star(s_plus))**2 - (np.emath.sqrt((E2_star(s_plus))**2-mpi**2) + (np.emath.sqrt((E3_star(s_plus))**2-mpi**2)))**2

def s_minus_max(s_plus):
    return (E2_star(s_plus)+E3_star(s_plus))**2 - (np.emath.sqrt((E2_star(s_plus))**2-mpi**2) - (np.emath.sqrt((E3_star(s_plus))**2-mpi**2)))**2

x_plus  = np.arange((2*mpi)**2/M**2, (m_b-mpi)**2/M**2, Grho/(2*M))
s2_min = min(s_minus_min(x_plus*M**2))/M**2
s2_max = max(s_minus_max(x_plus*M**2))/M**2

x_minus = np.arange(s2_min, s2_max, Grho/(2*M))

X_plus, X_minus = np.meshgrid(x_plus, x_minus)

s_plus = X_plus * M**2
s_minus = X_minus * M**2

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

list_c = np.zeros(100, dtype = object)
list_s = np.zeros(100, dtype = object)
list_prob = np.zeros(100, dtype = object)


orig_dim = len(x_plus)


mask1 = np.zeros_like(X_plus, dtype=bool)
for i in tqdm(range(len(x_minus)), desc = "Masking"):
    for j in range(len(x_plus)):
        if not inside_dalitz(s_plus[i,j], s_minus[i,j], M, m1, m2, m3):
            mask1[i,j] = True
mask2 = mask1
mask2 = mask2.reshape(1,len(x_plus)**2)
mask1 = mask1.ravel()

coords_s = np.stack([X_plus.ravel(), X_minus.ravel()], axis = 1)[~mask1]
M= ot.dist(coords_s, coords_s)
M /= M.max()
numThreads = 8
numItermax = 1000000
emds_c=np.zeros(1000)
emds_s=np.zeros(1000)
emds_cs=np.zeros(1000)

import numpy as np
import ot
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

n_datasets  = 1000
max_workers = 8   # set to os.cpu_count() to use all cores


def compute_emd_single(i, orig_dim, x_max, s, r, cfunc, sfunc, X_plus, X_minus, mask1, M, numItermax):
    """Single iteration — runs in a worker process."""
    x_s, _ = sample_tagged_times_ij(n_iter=1, N_evt=orig_dim**2, C=0.0, S=0.4, tag=1, x_max=x_max, rng=None)
    x_e, _ = sample_tagged_times_ij(n_iter=1, N_evt=orig_dim**2, C=0.0, S=0.0, tag=1, x_max=x_max, rng=None)
    x_cs,_ = sample_tagged_times_ij(n_iter=1, N_evt=orig_dim**2, C=0.7, S=0.4, tag=1, x_max=x_max, rng=None)
    x_c, _ = sample_tagged_times_ij(n_iter=1, N_evt=orig_dim**2, C=0.7, S=0.0, tag=1, x_max=x_max, rng=None)

    p_exp_v = np.exp(-s*x_e)
    p_full_v = (1+cfunc.ravel()*np.cos(r*x_cs)+sfunc.ravel()*np.sin(r*x_cs))*np.exp(-s*x_cs)
    p_cos_v = (1+cfunc.ravel()*np.cos(r*x_s))*np.exp(-s*x_s)
    p_sin_v = (1+sfunc.ravel()*np.sin(r*x_c))*np.exp(-s*x_c)


    p_exp_mask_v = p_exp_v.ravel()[~mask1]
    p_full_mask_v = p_full_v.ravel()[~mask1]
    p_cos_mask_v = p_cos_v.ravel()[~mask1]
    p_sin_mask_v = p_sin_v.ravel()[~mask1]

    print("Successfully masked.")

    p_exp_mask_v = p_exp_mask_v/np.sum(p_exp_mask_v)
    p_full_mask_v = p_full_mask_v/np.sum(p_full_mask_v)
    p_cos_mask_v = p_cos_mask_v/np.sum(p_cos_mask_v)
    p_sin_mask_v = p_sin_mask_v/np.sum(p_sin_mask_v)

    print("Successfully normalized.")

    emd_c=ot.emd2(p_exp_mask_v, p_cos_mask_v, M, numItermax = numItermax)
    emd_s=ot.emd2(p_exp_mask_v, p_sin_mask_v, M, numItermax = numItermax)
    emd_cs=ot.emd2(p_exp_mask_v, p_full_mask_v, M, numItermax = numItermax)
    return i, emd_c, emd_s, emd_cs

from joblib import Parallel, delayed
from tqdm import tqdm

results = Parallel(n_jobs=-1, backend="loky")(   # n_jobs=-1 = all cores
    delayed(compute_emd_single)(
        i, orig_dim, x_max, s, r,
        cfunc, sfunc,
        X_plus, X_minus, mask1,
        M, numItermax
    )
    for i in tqdm(range(n_datasets), desc="Dataset No.")
)

for i, emd_c, emd_s, emd_cs in results:
    emds_c[i] = emd_c
    emds_s[i] = emd_s
    emds_cs[i] = emd_cs

np.save("list of emds_cnz.npy", emds_c)
np.save("list of emds_snz.npy", emds_s)
np.save("list of emds_csnz.npy", emds_cs)
print("Done saving.")

end_time = time.perf_counter()
run_time = end_time - start_time

print(f"Time taken : {run_time:.6f} s")
