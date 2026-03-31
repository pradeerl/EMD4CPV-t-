import numpy as np

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

x_grid = np.linspace(0, 40, 30000) # (for plotting/reference)
Gamma = 2.746e-9 #MeV
Delta_m = 2.11e-9          # MeV
T=2*np.pi/Delta_m
r= Delta_m*T
s= T*Gamma
C=0.3
S=0.4