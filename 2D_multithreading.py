#!/usr/bin/env python
# coding: utf-8

# In[22]:


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


# In[23]:

Gd=1.61e-6 #(MeV)
Grho=147.4 #(MeV)
md=1864.84
mrho=775.26
mpi0=135
mpi=140

g1=g2=g3=np.float64(3*3.14/5)
print(type(g1))

s1=np.linspace(0,1,20)
s2=np.linspace(0,1,20)
mesh1,mesh2=np.meshgrid(s1,s2)

def delta(s,m,G):
    delta_ans=np.arctan(m*G/(s-m**2))
    return delta_ans

delta_D1=delta(mesh1,md,Gd)
print(delta_D1.dtype)
delta_D2=delta(mesh2,md,Gd)
delta_D3=delta(md**2+2*mpi**2+mpi0**2-mesh1-mesh2,md,Gd)
#delta_rho1=delta(s1,mrho,Grho)
#delta_rho2=delta(s2,mrho,Grho)

def amp(G,M,m1,m2):
    p1=np.sqrt(M**4+m1**4+m2**4-2*M**2*m1**2-2*M**2*m2**2-2*m1**2-m2**2)/(2*M)
    ans=np.sqrt(G*8*np.pi*M**2/(p1))
    return ans



A_D_rp_pm = amp(Gd*1.01e-7,md,mrho,mpi)
A_D_rm_pp = amp(Gd*5.15e-5,md,mrho,mpi)
A_D_r0_p0 = amp(Gd*3.86e-5,md,mrho,mpi0)

A_rhop_p0 = amp(Grho,mrho,mpi,mpi0)
A_rhom_m0 = amp(Grho,mrho,mpi,mpi0)
A_rho_pm = amp(Grho,mrho,mpi,mpi)

def cexp(g,d):
    return np.cos(g+d)+1j*np.sin(g+d)

lam=-1j*(A_D_rp_pm*A_rhop_p0*cexp(-g1,delta_D1)+A_D_rm_pp*A_rhom_m0*cexp(-g2,delta_D2)+A_D_r0_p0*A_rho_pm*cexp(-g3,delta_D3))/(A_D_rp_pm*A_rhop_p0*cexp(g1,delta_D1)+A_D_rm_pp*A_rhom_m0*cexp(g2,delta_D2)+A_D_r0_p0*A_rho_pm*cexp(g3,delta_D3))
C12=np.real((1-lam*np.conjugate(lam))/(1+lam*np.conjugate(lam)))
S12=np.real(-2*np.imag(lam)/(1+lam*np.conjugate(lam)))


x_grid = np.linspace(0, 40, 30000)  # ps (for plotting/reference)
Gamma = 0.658 # ps^{-1} (1/tau_Bd with tau~1.52 ps)
Delta_m = 0.506 # ps^{-1}
T=2*np.pi/Delta_m
r= Delta_m*T
s= T*Gamma
C=0.3
S=0.4
#x=t/T


# In[24]:


def rates(C, S, x):
    """
    Return (B0, B0bar) rates vs time for given (C,S) on grid t.
    Unnormalized; proportional to the differential decay rate.
    """
    base = np.exp(-s*x)
    osc = np.cos(r * x)
    sin = np.sin(r * x)
    B0    = base * (1 - C * osc + S * sin)
    B0bar = base * (1 + C * osc - S * sin)
    return B0, B0bar


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


def sample_tagged_times(n, C, S, tag, x_max, rng=None):
    """
    Rejection sample n proper times for a given flavor tag from the
    time-dependent rate  ~  e^{-Gamma t} * [1 + tag*(S*sin(Δm t) - C*cos(Δm t))]
    restricted to t in [0, t_max].

    Parameters
    ----------
    n : int
        Number of samples to return.
    C, S : float
        Direct/interference CPV coefficients.
    tag : int
        +1 for \bar{B}^0, -1 for B^0 (matches the 'rates' convention above).
    t_max : float
        Maximum proper time to sample (ps).
    rng : np.random.Generator or None
        Random generator; if None, uses default.

    Returns
    -------
    t : np.ndarray, shape (n,)
        Sampled proper times (ps).
    accept_rate : float
        Empirical acceptance rate of the rejection sampler.
    """
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
    x_samples = np.empty(n, dtype=float)
    got = 0
    trials = 0
    batch = max(256, int(1.5 * n))  # vectorized batches for speed

    # Accept-reject loop
    while got < n:
        x_prop = _trunc_exp_rvs(batch, x_max, rng)  # proposal times from truncated exp
        w = _modulation(x_prop, C, S, tag)
        w = np.clip(w, 0.0, None)  # robust if user picks unphysical (C,S)
        u = rng.random(batch)
        keep = u < (w / M)  # rejection step relative to envelope
        n_keep = min(keep.sum(), n - got)
        if n_keep > 0:
            x_samples[got:got+n_keep] = x_prop[keep][:n_keep]
            got += n_keep
        trials += batch

    accept_rate = float(n) / float(trials)
    return x_samples, accept_rate


def sample_pair(n_B0, n_B0bar, C, S, x_max, seed=None):
    """
    Convenience wrapper: sample both B^0 and \bar{B}^0 time sets.

    Returns
    -------
    t_B0, t_B0bar, info
    """
    rng = np.random.default_rng(seed)
    # Our 'rates' convention used tag = -1 for B^0 and +1 for \bar{B}^0
    x_B0, acc0 = sample_tagged_times(n_B0, C, S, tag=-1, x_max=x_max, rng=rng)
    x_B0bar, acc1 = sample_tagged_times(n_B0bar, C, S, tag=+1, x_max=x_max, rng=rng)
    info = {"accept_rate_B0": acc0, "accept_rate_B0bar": acc1, "x_max": x_max}
    return x_B0, x_B0bar, info


def demo_sampling(cases, x_max, n_each=20000, seed=1234):
    """
    Plot histograms of sampled times overlaid with the target rate curves.
    'cases' is a list of (C, S, title).
    """
    rng = np.random.default_rng(seed)

    for (C, S, title) in cases:
        x_B0, acc0 = sample_tagged_times(n_each, C, S, tag=-1, x_max=x_max, rng=rng)
        x_B0bar, acc1 = sample_tagged_times(n_each, C, S, tag=+1, x_max=x_max, rng=rng)

        # Rates on plotting grid, normalized to unit area on [0,t_max]
        xx = x_grid[(x_grid >= 0) & (x_grid <= x_max)]
        B0, B0bar = rates(C, S, xx)
        Z0 = np.trapezoid(B0, xx); Z1 = np.trapezoid(B0bar, xx)
        B0n, B0barn = B0/Z0, B0bar/Z1
        #Cplot=(t_B0bar-t_B0)*np.exp(Gamma*tt)*np.cos(Delta_m*tt)

        fig, ax = plt.subplots(1, 1, figsize=(6,4))
        ax.hist(x_B0, bins=80, range=(0, x_max), density=True, histtype='step', label=r"$B^0$ samples", color = "tab:blue")
        ax.hist(x_B0bar, bins=80, range=(0, x_max), density=True, histtype='step', label=r"$\bar B^0$ samples", color = "tab:orange")
        ax.plot(xx, B0n, label=r"$B^0$ target", linewidth=2, alpha=0.8, color = "tab:orange")
        ax.plot(xx, B0barn, label=r"$\bar B^0$ target", linewidth=2, alpha=0.8, color = "tab:blue")
        #ax.plot(tt, Cplot, label=r"C Plot", linewidth=2, alpha=0.8, color = "tab:red")
#weights=(np.exp(-s*x_B0))
        ax.set_xlabel(r"$\mathrm{Dimensionless\ time\ x=\frac{t}{\Gamma}}$")
        ax.set_ylabel(r"$\mathrm{Density}$")
        ax.set_title(title + rf"  $(C={C:.2f},\ S={S:.2f})$" + 
                     f"\naccept: B0={acc0:.2f}, B0bar={acc1:.2f}")
        ax.legend(frameon=False)
        fig.tight_layout()

# Example cases mirroring earlier plots
cases = [
    (0.0, 0.70, "Interference CPV only"),
    (0.70, 0.0, "Direct CPV only"),
    (0.30, 0.50, "Mixed CPV"),
]

demo_sampling(cases, n_each=100000, x_max=1.0, seed=42)


# In[ ]:


#s1=np.linspace(0,1,20)
#s2=np.linspace(0,1,20)
periods=np.arange(1,100,np.pi*2/r)
x_max = 100
n_iter = 300

mesh1,mesh2=np.meshgrid(s1,s2)
trialc=C12
trials=S12


def worker(i, j):
    n = np.zeros(2 * x_max)
    C_box = np.zeros(2 * x_max)
    C_ave = np.float64(0)

    for iter_ in tqdm(range(n_iter)):
        x, acc=sample_tagged_times(n=1000000,C=trialc[i][j],S=0,tag=-1,x_max=x_max,rng=None)

        # n_len counts the number of valid values from this iteration that we should consider for the average below
        n_len = 0
        """buckets = np.zeros(int(max(x) + 1)
        values = np.zeros(int(max(x) + 1, int(max(x) + 1)))
        for e in x:
            k = int(e)
            values[k][buckets[k]] = e
            buckets[k] += 1
        for k in range(0, int(max(x)) + 1):
            C_box[k] = 2*np.mean(np.exp(s*values[k])*np.cos(r*values[k]))/np.mean(np.exp(s*values[k]))"""
        for k in range(0,int(max(x))+1):
            x1=x[(k<x) & (x<k+1)]
            if (len(x1)!=0):
                n[n_len] = len(x1)
                C_box[n_len]= 2*np.mean(np.exp(s*x1)*np.cos(r*x1))/np.mean(np.exp(s*x1))
                n_len += 1
            else:
                continue

        C_ave += np.average(C_box[:n_len],None,n[:n_len])
    return C_ave / n_iter

with Pool(10) as p:
    indices = []
    for i in range(len(s1)):  
        for j in range(len(s2)):
           indices.append((i, j))
    C_calc = np.array(p.starmap(worker, indices)).reshape((len(s1), len(s2)))

np.save("cvalues_phys.npy",C_calc)


import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
"""
# 1. Define your data levels and corresponding colors
levels = [0, 2, 4, 6, 8, 10,1e4]  # Your data levels
colors = ['white','black','red','blue','yellow','orange','green'] # Colors for each level

# 2. Create a ListedColormap
cmap = mcolors.ListedColormap(colors)

# 3. Create a Normalization object to map data values to colormap indices
# This maps the levels directly to the colors in the colormap
norm = mcolors.BoundaryNorm(levels + [levels[-1] + 1], cmap.N) # Add an extra boundary for the last color
"""
levels = [-10,0, 2, 4, 6, 8, 10,1e4]
#levels = np.arange(-10,50,2)
cmap_base = plt.cm.viridis  # Choose a base colormap
custom_colors = cmap_base(np.linspace(0, 1, len(levels) - 1)) # Get colors for the intervals
#custom_cmap = mcolors.ListedColormap(colors)
# Option 2: Creating a colormap from a list of specific colors
#custom_colors = ['blue', 'cyan', 'green', 'yellow', 'orange', 'red','pink','black','white']
custom_cmap = mcolors.ListedColormap(custom_colors)
norm = mcolors.BoundaryNorm(levels, custom_cmap.N)

w=np.log(np.abs(((np.array(C_calc)-np.array(trialc+1e-30))/np.array(trialc+1e-30))*100))
#w=np.array(C_calc)-np.array(trialc)
fig, ax2 = plt.subplots(1, 1, figsize=(10,8))
plt.title(r'Log Percentage Error')
ax2.set_xlabel(r"s1")
ax2.set_ylabel(r"s2")
plt.contourf(mesh1,mesh2,w,levels=levels,cmap=custom_cmap, norm=norm)
plt.colorbar()
plt.savefig("C_calc-C_actual contour (2D)", bbox_inches='tight')
plt.show()


# In[17]:

"""
s1=np.linspace(0,1,50)
s2=np.linspace(0,1,50)
periods=np.arange(1,100,np.pi*2/r)
x_max = 100
n_iter = 100

mesh1,mesh2=np.meshgrid(s1,s2)
trialc=C12(mesh1,mesh2)
trials=S12(mesh1,mesh2)

from multiprocessing import Pool

def worker(i, j):
    n = np.zeros(2 * x_max)
    C_box = np.zeros(2 * x_max)
    C_ave = np.float64(0)

    for iter_ in tqdm(range(n_iter)):
        x, acc=sample_tagged_times(n=1000000,C=trialc[i][j],S=0,tag=-1,x_max=x_max,rng=None)

        # n_len counts the number of valid values from this iteration that we should consider for the average below
        n_len = 0
        values, bins = np.histogram(x, bins=range(int(max(x) + 1)))

        weighted_avg = np.float64(0)
        for k in range(0, int(max(x)) + 1):
            weighted_avg += buckets[k] * 2*np.mean(np.exp(s*values[k])*np.cos(r*values[k]))/np.mean(np.exp(s*values[k]))
        C_ave += weighted_avg / np.sum(buckets)

    return C_ave / n_iter

with Pool(1) as p:
    indices = []
    for i in range(len(s1)):  
        for j in range(len(s2)):
           indices.append((i, j))
    C_calc = np.array(p.starmap(worker, indices)).reshape((len(s1), len(s2)))


# In[ ]:


import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
"""
# 1. Define your data levels and corresponding colors
#levels = [0, 2, 4, 6, 8, 10,1e4]  # Your data levels
#colors = ['white','black','red','blue','yellow','orange','green'] # Colors for each level

# 2. Create a ListedColormap
#cmap = mcolors.ListedColormap(colors)

# 3. Create a Normalization object to map data values to colormap indices
# This maps the levels directly to the colors in the colormap
#norm = mcolors.BoundaryNorm(levels + [levels[-1] + 1], cmap.N) # Add an extra boundary for the last color
"""
levels = [-10,0, 2, 4, 6, 8, 10,1e4]
#levels = np.arange(-10,50,2)
cmap_base = plt.cm.viridis  # Choose a base colormap
custom_colors = cmap_base(np.linspace(0, 1, len(levels) - 1)) # Get colors for the intervals
#custom_cmap = mcolors.ListedColormap(colors)
# Option 2: Creating a colormap from a list of specific colors
#custom_colors = ['blue', 'cyan', 'green', 'yellow', 'orange', 'red','pink','black','white']
custom_cmap = mcolors.ListedColormap(custom_colors)
norm = mcolors.BoundaryNorm(levels, custom_cmap.N)

w=np.log(np.abs(((np.array(C_calc)-np.array(trialc+1e-30))/np.array(trialc+1e-30))*100))
#w=np.array(C_calc)-np.array(trialc)
fig, ax2 = plt.subplots(1, 1, figsize=(10,8))
plt.title('log(((C_calc-C_actual)/C_actual)*100)')
ax2.set_xlabel(r"s1")
ax2.set_ylabel(r"s2")
plt.contourf(mesh1,mesh2,w,levels=levels,cmap=custom_cmap, norm=norm)
plt.colorbar()
plt.savefig("C_calc-C_actual contour (2D)", bbox_inches='tight')
plt.show()


# In[241]:


s1=np.linspace(0,1,20)
s2=np.linspace(0,1,20)
mesh1,mesh2=np.meshgrid(s1,s2)
trialc=C12(mesh1,mesh2)
trials=S12(mesh1,mesh2)

q=np.array(trialc)

levels = np.arange(-0.2,0.3,0.05)
cmap_base = plt.cm.viridis  # Choose a base colormap
colors = cmap_base(np.linspace(0, 1, len(levels) - 1)) # Get colors for the intervals
custom_cmap = mcolors.ListedColormap(colors)

# Option 2: Creating a colormap from a list of specific colors
#custom_colors = ['blue', 'cyan', 'green', 'yellow', 'orange', 'red','pink','black','white']
#custom_cmap = mcolors.ListedColormap(custom_colors)
norm = mcolors.BoundaryNorm(levels, custom_cmap.N)


fig, ax2 = plt.subplots(1, 1, figsize=(10,8))
plt.title('C_actual')
ax2.set_xlabel(r"s1")
ax2.set_ylabel(r"s2")
plt.contourf(mesh1,mesh2,q,cmap=custom_cmap, norm=norm)
plt.colorbar()
plt.savefig("C_actual contour (2D)", bbox_inches='tight')
plt.show()


# In[104]:


periods=np.arange(1,100,np.pi*2/r)
x_max = 100
C_calc=[]
#x, acc=sample_tagged_times(n=20000,C=0.2,S=0,tag=-1,x_max=x_max,rng=None)
for i in periods:
    x, acc=sample_tagged_times(n=20000,C=0.8,S=0.3,tag=-1,x_max=i,rng=None)
    C_calc=np.append(C_calc,(2*np.mean(np.exp(s*x)*np.cos(r*x))/np.mean(np.exp(s*x))))
    S_calc=np.append(S_calc,(2*np.mean(np.exp(s*x)*np.sin(r*x))/np.mean(np.exp(s*x))))
plt.plot(C_calc)
print(np.mean(C_calc))


# In[38]:


xunif = np.linspace(0, 1, 30000)
#t = np.linspace(0, 8, 100)  # ps
periods=np.arange(1,100,np.pi*2/r)
C=0.3
S=0.5


def monte(C,S):
    C_calc=np.array([])
    C_err=np.array([])
    S_calc=np.array([])
    C_calc_bar=np.array([])
    C_err_bar=np.array([])
    numx=np.array([])
    x_max=10

    for i in periods:
        n=20000
        x, acc = sample_tagged_times(n, C, S, -1, i, rng=None)
        C_val=2*np.mean(np.exp(s*x)*np.cos(r*x))/np.mean(np.exp(s*x))
        C_calc=np.append(C_calc,C_val)
        numx=np.append(numx,i)
    return numx, C_calc
x, acc = sample_tagged_times(20000, C, S, -1, 1000, rng=None)

numx, C_calc = monte(C, S)
fig, ax2 = plt.subplots(1, 1, figsize=(5,4))
#plt.axhline(C)
plt.axhline(np.mean(C_calc))
plt.plot(numx, C_calc, color='red')
plt.title('C_calc vs Number of Periods')
ax2.set_xlabel(r"Number of Periods")
ax2.set_ylabel(r"C_calc")
plt.show()
print(C,np.mean(C_calc))
print(s)


# In[88]:


C_final=[]
s_var=[]
q=np.linspace(1e-2,300,100)

for a in range(0,25):
    for k in q:
        s=k
        #for j in range(0,25):
        x_max = 100
        C_calc=[]
        S_calc=[]
        n=[]
        x, acc=sample_tagged_times(n=20000,C=0.3,S=0.7,tag=-1,x_max=x_max,rng=None)

        for i in range(0,int(max(x))+1):
                #print("starting range: ",i, " to ",i+1, "for s= ",s)
            x1=x[(i<x) & (x<i+1)]
                #print(x1)
            n=np.append(n,len(x1))
            C_calc=np.append(C_calc,(2*np.mean(np.exp(k*x1)*np.cos(r*x1))/np.mean(np.exp(k*x1))))
            S_calc=np.append(S_calc,-(2*np.mean(np.exp(k*x1)*np.sin(r*x1))/np.mean(np.exp(k*x1))))
#plt.plot(C_calc)
#print(np.mean(C_calc))
#print(np.mean(S_calc))
            #print("stop")
        C_final=np.append(C_final,np.average(C_calc,None,n))
        s_var=np.append(s_var,k)
            #print("n: ",n)


# In[89]:


fig, ax2 = plt.subplots(1, 1, figsize=(5,4))
data=pd.DataFrame({"s":s_var,"C":C_final})
plt.title("C_final (weighted average) vs s")
sns.lineplot(data=data,x="s",y="C",errorbar='ci')
plt.axhline(np.mean(C_final),color='red')
ax2.set_xlabel(r"s")
ax2.set_ylabel(r"C_final")
plt.savefig("1D test in 2D (large s)")
plt.show()


# In[39]:


C_final=[]
s_var=[]
q=np.linspace(1e-2,20,100)

for a in range(0,25):
    for k in q:
        s=k
        #for j in range(0,25):
        x_max = 100
        C_calc=[]
        S_calc=[]
        n=[]
        x, acc=sample_tagged_times(n=20000,C=0.3,S=0.7,tag=-1,x_max=x_max,rng=None)

        for i in range(0,int(max(x))+1):
                #print("starting range: ",i, " to ",i+1, "for s= ",s)
            x1=x[(i<x) & (x<i+1)]
                #print(x1)
            n=np.append(n,len(x1))
            C_calc=np.append(C_calc,(2*np.mean(np.exp(k*x1)*np.cos(r*x1))/np.mean(np.exp(k*x1))))
            S_calc=np.append(S_calc,-(2*np.mean(np.exp(k*x1)*np.sin(r*x1))/np.mean(np.exp(k*x1))))
#plt.plot(C_calc)
#print(np.mean(C_calc))
#print(np.mean(S_calc))
            #print("stop")
            C_final=np.append(C_final,np.average(C_calc,None,n))
            s_var=np.append(s_var,k)
            #print("n: ",n)


# In[40]:


fig, ax2 = plt.subplots(1, 1, figsize=(5,4))
data=pd.DataFrame({"s":s_var,"C":C_final})
plt.title("C_final (weighted average) vs s")
sns.lineplot(data=data,x="s",y="C",errorbar='ci')
plt.axhline(np.mean(C_final),color='red')
print(np.mean(C_final))
ax2.set_xlabel(r"s")
ax2.set_ylabel(r"C_final")
plt.savefig("1D test in 2D")
plt.show()


# In[80]:


C_final=[]
for j in range(0,50):
        x_max = 100
        C_calc=[]
        S_calc=[]
        n=[]
        x, acc=sample_tagged_times(n=20000,C=0.3,S=0.7,tag=-1,x_max=x_max,rng=None)

        for i in range(0,2):
            x1=x[(i<x) & (x<i+1)]
            n=np.append(n,len(x1))
            #print(n)
            C_calc=np.append(C_calc,(2*np.mean(np.exp(s*x1)*np.cos(r*x1))/np.mean(np.exp(s*x1))))
            S_calc=np.append(S_calc,-(2*np.mean(np.exp(s*x1)*np.sin(r*x1))/np.mean(np.exp(s*x1))))
#plt.plot(C_calc)
#print(np.mean(C_calc))
#print(np.mean(S_calc))
        C_final=np.append(C_final,np.average(C_calc,None,n))
plt.title('C_calc vs Number of Periods')
plt.axhline(np.mean(C_final))
print(np.mean(C_final))
#plt.axhline(np.mean(S_calc))
plt.plot(C_final)
ax2.set_xlabel(r"s")
ax2.set_ylabel(r"C_calc")
#plt.savefig("1D test in 2D")
plt.show()


# In[112]:


s1=np.linspace(0,1,50)
s2=np.linspace(0,1,50)
periods=np.arange(1,100,np.pi*2/r)

mesh1,mesh2=np.meshgrid(s1,s2)
trialc=C12(mesh1,mesh2)
trials=S12(mesh1,mesh2)
n=[]
C_box=[]
C_calc=np.zeros((len(s1),len(s2)))
print(C_calc.shape)
                 #,len(periods))) 
for i in range(len(s1)):
    for j in range(len(s2)):
        for k in range(0,100):
            x, acc=sample_tagged_times(n=1000000,C=trialc[i][j],S=0,tag=-1,x_max=100,rng=None)
            for k in range(0,int(max(x))+1):
                x1=x[(k<x) & (x<k+1)]
                n=np.append(n,len(x1))
                C_box=np.append(C_box, 2*np.mean(np.exp(s*x1)*np.cos(r*x1))/np.mean(np.exp(s*x1)))
            C_ave=np.average(C_box,None,n)
        C_calc[i,j]=C_ave
        n=[]
        C_box=[]


# In[111]:


import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
"""
# 1. Define your data levels and corresponding colors
#levels = [0, 2, 4, 6, 8, 10,1e4]  # Your data levels
#colors = ['white','black','red','blue','yellow','orange','green'] # Colors for each level

# 2. Create a ListedColormap
#cmap = mcolors.ListedColormap(colors)

# 3. Create a Normalization object to map data values to colormap indices
# This maps the levels directly to the colors in the colormap
#norm = mcolors.BoundaryNorm(levels + [levels[-1] + 1], cmap.N) # Add an extra boundary for the last color
"""
levels = [-10,0, 2, 4, 6, 8, 10,1e4]
#levels = np.arange(-10,50,2)
cmap_base = plt.cm.viridis  # Choose a base colormap
custom_colors = cmap_base(np.linspace(0, 1, len(levels) - 1)) # Get colors for the intervals
#custom_cmap = mcolors.ListedColormap(colors)
# Option 2: Creating a colormap from a list of specific colors
#custom_colors = ['blue', 'cyan', 'green', 'yellow', 'orange', 'red','pink','black','white']
custom_cmap = mcolors.ListedColormap(custom_colors)
norm = mcolors.BoundaryNorm(levels, custom_cmap.N)

w=np.log(np.abs(((np.array(C_calc)-np.array(trialc+1e-25))/np.array(trialc+1e-25))*100))
#w=np.array(C_calc)-np.array(trialc)
fig, ax2 = plt.subplots(1, 1, figsize=(10,8))
plt.title('log(((C_calc-C_actual)/C_actual)*100)')
ax2.set_xlabel(r"s1")
ax2.set_ylabel(r"s2")
plt.contourf(mesh1,mesh2,w,levels=levels,cmap=custom_cmap, norm=norm)
plt.colorbar()
plt.savefig("C_calc-C_actual contour (2D)", bbox_inches='tight')
plt.show()


# In[132]:


C=C12(0.5,0.3)
print(C)
C_box=[]
C_calc=[]
n=[]
for i in range(0,100):
    x,acc=sample_tagged_times(1000000,tag=-1,C=C12(0.5,0.3),S=0,x_max=100,rng=None)
    for k in range(0,int(max(x))+1):
            x1=x[(k<x) & (x<k+1)]
            n=np.append(n,len(x1))
            C_box=np.append(C_box, 2*np.mean(np.exp(s*x1)*np.cos(r*x1))/np.mean(np.exp(s*x1)))
    C_calc=np.append(C_calc,np.average(C_box,None,n))
plt.plot(C_calc)
plt.axhline(C)

"""
# In[ ]:




