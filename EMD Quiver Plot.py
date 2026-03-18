import matplotlib.pyplot as plt
import matplotlib
from tqdm import tqdm
import numpy as np
from scipy.integrate import quad, simpson
import random
import matplotlib.colors as colors
import pandas as pd
from sympy import Symbol,integrate,exp,cos,sin,I,simplify
import seaborn as sns
from multiprocess import Pool
import matplotlib.pylab as pl
import ot
import ot.plot
from scipy.spatial import distance
from sklearn.preprocessing import normalize
import time

from OptimizeTest import compute_CS_grid, sample_tagged_times_ij, inside_dalitz

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

x_plus  = np.arange((2*mpi)**2/M**2, (m_b-mpi)**2/M**2, Grho/(4*M))
s2_min = min(s_minus_min(x_plus*M**2))/M**2
s2_max = max(s_minus_max(x_plus*M**2))/M**2

x_minus = np.arange(s2_min, s2_max, Grho/(4*M))

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

mask1 = np.zeros_like(X_plus, dtype=bool)

for i in tqdm(range(len(s_minus))):
    for j in range(len(s_plus)):
        if not inside_dalitz(s_plus[i,j], s_minus[i,j], M, m1, m2, m3):
            mask1[i,j] = True
mask2 = mask1
mask2 = mask2.reshape(1,len(x_plus)**2)
mask1 = mask1.ravel()
orig_dim = len(x_plus)

g1 = np.linspace(1e-3, 1, orig_dim**2)
g2 =np.linspace(1e-3, 1, orig_dim**2)
gamma1 = 3*np.pi/5
gamma2 = 2*np.pi/3

x_v, acc = sample_tagged_times_ij(
				n_iter=1,
                N_evt=orig_dim**2,
                C=0.3,
                S=0.4,
                tag=1,
                x_max=x_max,
                rng=None
            )

x_c, acc = sample_tagged_times_ij(
                n_iter=1,
                N_evt=orig_dim**2,
                C=0.0,
                S=0.4,
                tag=1,
                x_max=x_max,
                rng=None
            )

x_e, acc = sample_tagged_times_ij(
                n_iter=1,
                N_evt=orig_dim**2,
                C=0.0,
                S=0.0,
                tag=1,
                x_max=x_max,
                rng=None
            )

x1 = x_v[~mask1]
coords_s = np.stack([X_plus.ravel(), X_minus.ravel()], axis = 1)[~mask1]

p_exp_v = np.exp(-s*x_e)
p_full_v = (1+cfunc.ravel()*np.cos(r*x_v)+sfunc.ravel()*np.sin(r*x_v))*np.exp(-s*x_v)
p_cos_v = (1+cfunc.ravel()*np.cos(r*x_v))*np.exp(-s*x_v)
p_sin_v = (1+sfunc.ravel()*np.sin(r*x_c))*np.exp(-s*x_c)

coords_s = np.stack([X_plus.ravel(), X_minus.ravel()], axis = 1)[~mask1]


p_exp_mask_v = p_exp_v.ravel()[~mask1]
p_full_mask_v = p_full_v.ravel()[~mask1]
p_cos_mask_v = p_cos_v.ravel()[~mask1]
p_sin_mask_v = p_sin_v.ravel()[~mask1]

print("Successful masking.")

p_exp_mask_v = p_exp_mask_v/np.sum(p_exp_mask_v)
p_full_mask_v = p_full_mask_v/np.sum(p_full_mask_v)
p_cos_mask_v = p_cos_mask_v/np.sum(p_cos_mask_v)
p_sin_mask_v = p_sin_mask_v/np.sum(p_sin_mask_v)

M= ot.dist(coords_s, coords_s)
M /= M.max()

numThreads = 8
numItermax = 1000000
"""
G_exp_cos=ot.emd(p_exp_mask_v, p_cos_mask_v, M, numItermax = numItermax, numThreads = numThreads)
G2_exp_cos=ot.emd2(p_exp_mask_v, p_cos_mask_v, M, numItermax = numItermax, numThreads = numThreads)

print("Done 1")
"""
G_exp_sin=ot.emd(p_exp_mask_v, p_sin_mask_v, M, numItermax = numItermax, numThreads = numThreads)
G2_exp_sin=ot.emd2(p_exp_mask_v, p_sin_mask_v, M, numItermax = numItermax, numThreads = numThreads)

print("Done 2")

G_exp_full=ot.emd(p_exp_mask_v, p_full_mask_v, M, numItermax = numItermax, numThreads = numThreads)
G2_exp_full=ot.emd2(p_exp_mask_v, p_full_mask_v, M, numItermax = numItermax, numThreads = numThreads)

print("Done 3")


print(G2_exp_full)

"""
Nec_sample = 50
idx_ec = np.random.choice(len(coords_s), size=Nec_sample, replace=False)

mass_out = G_exp_cos.sum(axis=1)[:, None]      # (N, 1)
x_src = coords_s                         # (N, 2)
x_bar_ec = (np.dot(G_exp_cos, coords_s)) / mass_out        # (N, 2)

dx_ec = x_bar_ec - x_src                     # (N, 2)

x = x_src[idx_ec, 0]
y = x_src[idx_ec, 1]
u_ec = dx_ec[idx_ec, 0]
v_ec = dx_ec[idx_ec, 1]

A2 = amp(X_plus, X_minus, A_plus_fit, A_minus_fit, A_zero_fit, m)

A_masked = np.ma.masked_array(A2, mask1)

X_plus_mask = X_plus[~mask2.reshape(orig_dim, orig_dim)]
X_minus_mask = X_minus[~mask2.reshape(orig_dim, orig_dim)]

plt.figure(figsize=(10,7))
#plt.contourf(X_plus, X_minus, np.abs(A_masked)**2, levels=600, norm=colors.LogNorm(vmin=np.min(np.abs(A_masked)**2), vmax=np.max(np.abs(A_masked)**2)))

plt.quiver(
    x, y, u_ec, v_ec, color = 'blue',
    angles='xy',
    scale_units='xy',
    scale=1,      # adjust arrow length
    width=0.002,
    alpha=0.8
)

plt.xlabel(r"$x_{1}$")
plt.ylabel(r"$x_{2}$")
plt.title("Optimal Transport Vector Field (Only cosine nonzero)")
plt.axis('equal')
plt.show()
"""
Nes_sample = 500
idx_es = np.random.choice(len(coords_s), size=Nes_sample, replace=False)


mass_out = G_exp_sin.sum(axis=1)[:, None]      # (N, 1)
x_src = coords_s                         # (N, 2)
x_bar_es = (np.dot(G_exp_sin, coords_s)) / mass_out        # (N, 2)

dx_es = x_bar_es - x_src                     # (N, 2)

x = x_src[idx_es, 0]
y = x_src[idx_es, 1]
u_es = dx_es[idx_es, 0]
v_es = dx_es[idx_es, 1]

A2 = amp(X_plus, X_minus, A_plus_fit, A_minus_fit, A_zero_fit, m)

A_masked = np.ma.masked_array(A2, mask1)

X_plus_mask = X_plus[~mask2.reshape(orig_dim, orig_dim)]
X_minus_mask = X_minus[~mask2.reshape(orig_dim, orig_dim)]


plt.figure(figsize=(10,7))
#plt.contourf(X_plus, X_minus, np.abs(A_masked)**2, levels=600, norm=colors.LogNorm(vmin=np.min(np.abs(A_masked)**2), vmax=np.max(np.abs(A_masked)**2)))

plt.quiver(
    x, y, u_es, v_es, color = 'red',
    angles='xy',
    scale_units='xy',
    scale=1,      # adjust arrow length
    width=0.002,
    alpha=0.8
)


plt.xlabel(r"$x_{1}$")
plt.ylabel(r"$x_{2}$")
plt.title("Optimal Transport Vector Field (Only sine nonzero)")
plt.axis('equal')
plt.show()

Nef_sample = 50
idx_ef = np.random.choice(len(coords_s), size=Nef_sample, replace=False)

mass_out = G_exp_full.sum(axis=1)[:, None]      # (N, 1)
x_src = coords_s                         # (N, 2)
x_bar_ef = (np.dot(G_exp_full, coords_s)) / mass_out        # (N, 2)

dx_ef = x_bar_ef - x_src                     # (N, 2)

x = x_src[idx_ef, 0]
y = x_src[idx_ef, 1]
u_ef = dx_ef[idx_ef, 0]
v_ef = dx_ef[idx_ef, 1]

A2 = amp(X_plus, X_minus, A_plus_fit, A_minus_fit, A_zero_fit, m)

A_masked = np.ma.masked_array(A2, mask1)

X_plus_mask = X_plus[~mask2.reshape(orig_dim, orig_dim)]
X_minus_mask = X_minus[~mask2.reshape(orig_dim, orig_dim)]

plt.figure(figsize=(10,7))
#plt.contourf(X_plus, X_minus, np.abs(A_masked)**2, levels=600, norm=colors.LogNorm(vmin=np.min(np.abs(A_masked)**2), vmax=np.max(np.abs(A_masked)**2)))

plt.quiver(
    x, y, u_ef, v_ef, color = 'blue',
    angles='xy',
    scale_units='xy',
    scale=1,      # adjust arrow length
    width=0.002,
    alpha=0.8
)

plt.xlabel(r"$x_{1}$")
plt.ylabel(r"$x_{2}$")
plt.title("Optimal Transport Vector Field (Both sine and cosine nonzero)")
plt.axis('equal')
plt.show()