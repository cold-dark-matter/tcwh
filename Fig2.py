# -*- coding: utf-8 -*-
"""
Created on Wed Jul 23 18:24:10 2025

@author: Richita
"""

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import networkx as nx
from numba import njit
import scipy.sparse as sp
import time
import pickle
start_time = time.time()

# ---------------------
# Parameters
# ---------------------
a = 4
b = 1
N = 100
dim = 1
x0 = 0.5
d1 =0.5
r = np.zeros(N)
d = np.ones(N) * 0.05
r[0] = 0.2  # trigger node

# ---------------------
# Generate Network
# ---------------------

average_degree=6
p = average_degree/(N-1)
G = nx.gnp_random_graph(N, p,directed=False)

#------scalefree----
# m=2
# G = nx.barabasi_albert_graph(N, m)
#-----------WS--------------------------
# k = 6      #nearest neighbors
# p = 0.2     # rewiring probability

# G = nx.watts_strogatz_graph(N, k, p)

#--------------------------------------------
# Build Sparse A
# ---------------------
A = nx.to_numpy_array(G).T
np.fill_diagonal(A, 0)
A_sparse = sp.csr_matrix(A)

# ---------------------
# Build A1 
# ---------------------
i_idx, j_idx, k_idx, val_idx = [], [], [], []

for i in range(N):
    for j in range(N):
        for k in range(N):
            val = A[i, j] * A[j, k] * A[k, i]
            if val != 0:
                i_idx.append(i)
                j_idx.append(j)
                k_idx.append(k)
                val_idx.append(val)

i_idx = np.array(i_idx, dtype=np.int32)
j_idx = np.array(j_idx, dtype=np.int32)
k_idx = np.array(k_idx, dtype=np.int32)
val_idx = np.array(val_idx, dtype=np.float64)

# ---------------------
# HOI
# ---------------------
@njit
def compute_hoi(x, i_idx, j_idx, k_idx, vals, N):
    hoi = np.zeros(N)
    for n in range(len(vals)):
        i = i_idx[n]
        j = j_idx[n]
        k = k_idx[n]
        hoi[i] += vals[n] * x[j] * x[k]
    return hoi

# ---------------------
# Model 
# ---------------------
def model(states, t, r):
    x = states[0:dim * N:dim]
    dx = np.zeros_like(states)

    linear_term = A_sparse.dot(x)
    hoi_term = compute_hoi(x, i_idx, j_idx, k_idx, val_idx, N)

    dx[0:dim * N:dim] = -a * (x - x0)**3 + b * (x - x0) + r + d * linear_term + d1 * hoi_term
    return dx

# ---------------------
# Integration
# ---------------------
x_current = np.zeros(N * dim)

t_current = 1
t_end = 100
t_step = 1

times = [t_current]
states = [x_current.copy()]
tipped_counts = [np.count_nonzero(x_current > 0.5)]

while t_current < t_end:
    print(f"t = {t_current}")
    t_span = [t_current, t_current + t_step]
    sol = odeint(model, x_current, t_span, args=(r,))
    x_current = sol[1]
    t_current += t_step

    times.append(t_current)
    states.append(x_current.copy())
    tipped_counts.append(np.count_nonzero(x_current > 0.5))
tipped_counts=np.array(tipped_counts)



plt.plot(times, np.array(tipped_counts), '-', color='crimson', linewidth=2)
plt.xlabel("Time")
plt.ylabel("Fraction of tipped nodes")
plt.show()

end_time = time.time()

print('Time taken:', end_time - start_time, 'seconds')


