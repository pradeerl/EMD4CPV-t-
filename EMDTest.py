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
import cmath
from utils import *
from config import *
from sklearn.utils import shuffle
from joblib import Parallel, delayed

start_time = time.perf_counter()

rng = np.random.default_rng(42)

x_plusalt  = np.arange((2*mpi)**2/M**2, (M-mpi)**2/M**2, Grho/(M))
s2_minalt = min(s_minus_min(x_plusalt*M**2))/M**2
s2_maxalt = max(s_minus_max(x_plusalt*M**2))/M**2

x_minusalt = np.arange(s2_minalt, s2_maxalt, Grho/(M))

X_plusalt, X_minusalt = np.meshgrid(x_plusalt, x_minusalt)
s_plusalt = X_plusalt * M**2
s_minusalt = X_minusalt * M**2

mask1 = mask(x_plus, x_minus, s_plus, s_minus, X_plus)
mask1 = mask1.ravel()

mask2 = mask(x_plus, x_minus, s_plus, s_minus, X_plus)
mask2 = mask2.ravel()
mask3 = mask(x_plusalt, x_minusalt, s_plusalt, s_minusalt, X_plusalt)
mask3 = mask3.ravel()

coords = np.stack([X_plus.ravel(), X_minus.ravel()], axis = 1)[~mask1]

loss_mat = ot.dist(coords, coords)
loss_mat /= loss_mat.max()
sample_size = loss_mat.shape[0]
numThreads = 8
n_datasets = 100
numItermax = 1000000
emds_shuffled = np.zeros(n_datasets)
#emds_var = np.zeros(n_datasets)
var_c = np.ones(orig_dim**2)[None,:]*np.linspace(-0.9,0.9,300)[:,None]
emds_var = np.zeros(n_datasets)

x_plus /=  np.max(x_plus)
x_minus /= np.max(x_minus)

def weak_num(i):
    weak_og = np.array([0, 0, i*np.pi/100])
    weak = weak_og[:, np.newaxis, np.newaxis]
    weak_final = np.broadcast_to(weak, (3, x_plus.shape[0], x_plus.shape[0]))
    return weak_final

def delta(s_plus):
		return np.arctan(m*Gamma/(s_plus - m**2))

def phase(i, tag):
	#gam = np.array([-np.pi/1000, -np.pi/, i*2*np.pi/1000], dtype= object).reshape(3,1,1)
	gam = tag*weak_num(i)+np.array([delta(s_plus),delta(s_minus),delta(M**2-s_plus-s_minus)],dtype =float)
	#.reshape(3,1,1)
	return gam

def lam(i):
	g = np.array([3e-2, 3e-2, 3e-2]).reshape(3,1,1)
	h = np.array([3e-2, 3e-2, 3e-2]).reshape(3,1,1)
	#*np.exp(1j*np.array(gam, dtype = float)+1j*np.array([delta(s_plus),delta(s_minus),delta(1-s_plus-s_minus)],dtype =float))
	lden = lnum = np.array([np.abs(BW(s_plus  / M**2, m/M)), np.abs(BW(s_minus / M**2, m/M)), np.abs(BW((M**2-s_plus-s_minus) / M**2, m/M))])

	#lden = np.array([np.abs(BW(s_plus  / M**2, m/M))*Abar_plus_fit, np.abs(BW(s_minus / M**2, m/M))*Abar_minus_fit, np.abs(BW((1-s_plus-s_minus) * M**2, m/M))*Abar_zero_fit])

	gam = phase(i, 1)
	gambar = phase(i, -1)
	a = np.sum(g*h*lnum*np.exp(1j*np.array(gam, dtype = float)), axis = 0)
		#+1j*np.array([delta(s_plus),delta(s_minus),delta(1-s_plus-s_minus)],dtype =float)), axis = 0)
	abar = np.sum(g*h*lden*np.exp(1j*np.array(gambar, dtype = float)), axis = 0)
		#+1j*np.array([delta(s_plus),delta(s_minus),delta(1-s_plus-s_minus)],dtype =float)), axis = 0)
	lam = abar/a
	return lam

def lam_mix(phase):
	return lam(0)*cmath.exp(-I*phase)
	
def c_toy(x_plus, x_minus):
	return x_plus * x_minus * (1-x_plus-x_minus)*np.cos(3*np.pi*(x_plus-x_minus))*np.sin(5*np.pi*(x_plus+x_minus))
def s_toy(x_plus, x_minus):
	return x_plus * x_minus * (1-x_plus-x_minus)*np.sin(3*np.pi*(x_plus-x_minus))*np.sin(5*np.pi*(x_plus+x_minus))

def CS(i):
	og = lam(i).shape
	temp = lam(i).reshape(og[0]*og[1])
	graph_s = np.zeros(len(temp))
	graph_c = (1 - np.abs(lam(i))**2)/(1+np.abs(lam(i))**2)
	for j in range(len(temp)):
		graph_s[j] = (-2*np.imag(temp[j]))/(1+np.abs(temp[j])**2)
	graph_s = graph_s.reshape(og)
	return graph_c.ravel(), graph_s.ravel()

def CS_mix(i):
	og = lam_mix(i).shape
	temp = lam_mix(i).reshape(og[0]*og[1])
	graph_s = np.zeros(len(temp))
	graph_c = (1 - np.abs(lam_mix(i))**2)/(1+np.abs(lam_mix(i))**2)
	for j in range(len(temp)):
		graph_s[j] = (-2*np.imag(temp[j]))/(1+np.abs(temp[j])**2)
	graph_s = graph_s.reshape(og)
	return graph_c.ravel(), graph_s.ravel()

mask2      = mask(x_plus, x_minus, s_plus, s_minus, X_plus).ravel()
coords     = np.stack([X_plus.ravel(), X_minus.ravel()], axis=1)[~mask2]
loss_mat   = ot.dist(coords, coords)
loss_mat  /= loss_mat.max()
sample_size = loss_mat.shape[0]
numItermax  = 1000000
#n_draws     = 25
#numItermax  = 1000000
n_draws     = 100

def weighted_sample_seeded(x, p, sample_size, s, seed):
    rng    = np.random.default_rng(seed)
    log_w  = s * x.ravel()
    #log_w -= log_w.max()
    w      = np.exp(log_w)
    w     /= w.sum()
    idx    = rng.choice(len(p), size=sample_size, replace=False, p=w)
    p_out  = p[idx]

    return p_out / p_out.sum()      

graph_c0, graph_s0 = CS(0)
x1, _ = sample_tagged_times_ij(
		n_iter=1, N_evt=orig_dim**2,
		C= np.mean(graph_c0), S=np.mean(graph_s0),
		tag=1, x_max=x_max, rng=None)

x2, _ = sample_tagged_times_ij(
		n_iter=1, N_evt=orig_dim**2,
		C= np.mean(graph_c0), S=np.mean(graph_s0),
		tag=1, x_max=x_max, rng=None)

p1 = np.exp(-s * x1).ravel()[~mask2]

p2 = np.exp(-s * x2).ravel()[~mask2]


def compute_emd_for_coupling(i):
    graph_c_i, graph_s_i = CS(i)
    
    x3, _ = sample_tagged_times_ij(
		n_iter=1, N_evt=orig_dim**2,
		C=float(np.mean(graph_c_i)), 
		S=float(np.mean(graph_s_i)), 
		tag=1, x_max=x_max, rng=None)

  
    x1_m = x1.ravel()[~mask2]
    x2_m = x2.ravel()[~mask2]
    x3_m = x3.ravel()[~mask2]

    p3 = ((1 + graph_c_i * np.cos(r * x3)
             + graph_s_i * np.sin(r * x3))
          * np.exp(-s * x3)).ravel()[~mask2]

    master      = np.random.SeedSequence(int(5*i))
    p1_seed, p2_seed, p3_seed = master.spawn(3)
    p1_seeds    = p1_seed.spawn(n_draws)
    p2_seeds    = p2_seed.spawn(n_draws)
    p3_seeds    = p3_seed.spawn(n_draws)

    p1_batch = np.array([
        weighted_sample_seeded(x1_m, p1, sample_size, s, seed)
        for seed in p1_seeds
    ])

    p2_batch = np.array([
        weighted_sample_seeded(x2_m, p2, sample_size, s, seed)
        for seed in p2_seeds
    ]) 

    p3_batch = np.array([
        weighted_sample_seeded(x3_m, p3, sample_size, s, seed)
        for seed in p3_seeds
    ])   
    
    emds_c = np.array([
        ot.emd2(p1_batch[k], p2_batch[k], loss_mat, numItermax=numItermax)
        for k in range(n_draws)
    ])
	
    emds_v = np.array([
        ot.emd2(p1_batch[k], p3_batch[k], loss_mat, numItermax=numItermax)
        for k in range(n_draws)
    ])
    
    return emds_c, emds_v
    #float(np.mean(emds_c)), float(np.mean(emds_v))


if __name__ == "__main__":
    
    phase_scan = np.linspace(0,200,100)
    results = Parallel(n_jobs=-1, backend="loky")(
        delayed(compute_emd_for_coupling)(i)
        for i in tqdm(phase_scan, desc="Coupling value")
    )
    results = np.array(results)
    emds_null = np.array(results[:,0,:])
    emds_phase = np.array(results[:,1,:])

    np.save("list of emds_shuffled_alt_g3.npy", emds_null)
    np.save("list of emds_phase_alt_g3.npy", emds_phase)
    print(f"Done saving EMD values.")
    end_time = time.perf_counter()
    run_time = end_time - start_time

    print(f"Time taken : {run_time:.6f} s")
