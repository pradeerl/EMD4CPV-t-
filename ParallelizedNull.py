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
#from utils import sample_tagged_times_ij, inside_dalitz, mask
from utils import *
from config import *
#from OptimizeTest import compute_CS_grid, sample_tagged_times_ij, inside_dalitz

print(s_plus)
start_time = time.perf_counter()

x_plus  = np.arange((2*mpi)**2/M**2, (m_b-mpi)**2/M**2, Grho/(2*M))
s2_min = min(s_minus_min(x_plus*M**2))/M**2
s2_max = max(s_minus_max(x_plus*M**2))/M**2

x_minus = np.arange(s2_min, s2_max, Grho/(2*M))

X_plus, X_minus = np.meshgrid(x_plus, x_minus)

s_plus = X_plus * M**2
s_minus = X_minus * M**2

orig_dim = len(x_plus)

mask1 = mask(x_plus, x_minus, s_plus, s_minus)
mask1 = mask1.ravel()

coords_s = np.stack([X_plus.ravel(), X_minus.ravel()], axis = 1)[~mask1]
M= ot.dist(coords_s, coords_s)
M /= M.max()
numThreads = 8
numItermax = 1000000
emds_c=np.zeros(100)
emds_s=np.zeros(100)
emds_cs=np.zeros(100)

import numpy as np
import ot
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

n_datasets  = 100
max_workers = 8   # set to os.cpu_count() to use all cores


def compute_emd_single(i, orig_dim, x_max, s, r, cfunc, sfunc, X_plus, X_minus, mask1, M, numItermax):
    #Single iteration — runs in a worker process.
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

    p_exp_mask_v = p_exp_mask_v/np.sum(p_exp_mask_v)
    p_full_mask_v = p_full_mask_v/np.sum(p_full_mask_v)
    p_cos_mask_v = p_cos_mask_v/np.sum(p_cos_mask_v)
    p_sin_mask_v = p_sin_mask_v/np.sum(p_sin_mask_v)

    emd_c=ot.emd2(p_exp_mask_v, p_cos_mask_v, M, numItermax = numItermax)
    emd_s=ot.emd2(p_exp_mask_v, p_sin_mask_v, M, numItermax = numItermax)
    emd_cs=ot.emd2(p_exp_mask_v, p_full_mask_v, M, numItermax = numItermax)
    return i, emd_c, emd_s, emd_cs

from joblib import Parallel, delayed

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


