# -*- coding: utf-8 -*-
"""
Created on Mon Aug 18 18:59:25 2025

@author: Richita
"""

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import scipy.sparse
import pickle
import time
from numba import njit

start_time = time.time()


a = 4.0
b = 1.0
A = np.load("A_sparse_414.npy")
N = A.shape[0]
dim = 1
x0 = 0.5
d1 = -0.05
r = np.zeros(N)
d = 0.08
r[0] = 0.2  


# A_sparse = scipy.sparse.load_npz("A.npz")


with open("A1_414.pkl", "rb") as f:
    A1_data = pickle.load(f)

i_idx = A1_data["i"]
j_idx = A1_data["j"]
k_idx = A1_data["k"]
A1_vals = A1_data["values"]


@njit
def compute_hoi(x, i_idx, j_idx, k_idx, vals, N):
    hoi_term = np.zeros(N)
    for n in range(len(vals)):
        i = i_idx[n]
        j = j_idx[n]
        k = k_idx[n]
        hoi_term[i] += vals[n] * x[j] * x[k]
    return hoi_term


def model(states, t, r):
    x = states[0:dim * N:dim]
    dx = np.zeros_like(states)
    linear_term = np.dot(A,x)
    hoi_term = compute_hoi(x, i_idx, j_idx, k_idx, A1_vals, N)
    dx[0:dim * N:dim] = -a * (x - x0)**3 + b * (x - x0) + r + d * linear_term + (d1 * hoi_term)/2
    return dx

x_current = np.zeros(N * dim)
t_current = 0
t_end = 100
t_step = 1

times = [t_current]
states = [x_current.copy()]
tipped_counts = [np.count_nonzero(x_current > 0.5)]

while t_current < t_end:
    print(t_current)

    t_span = [t_current, t_current + t_step]
    sol = odeint(model, x_current, t_span, args=(r,))
    x_current = sol[-1]
    t_current += t_step
    times.append(t_current)
    states.append(x_current.copy())
    tipped_counts.append(np.count_nonzero(x_current > 0.5))

# ---------------------
# Plot
# ---------------------
plt.plot(times, np.array(tipped_counts), '-', color='crimson', linewidth=2)
plt.xlabel("Time")
plt.ylabel("No. of tipped nodes")
plt.tight_layout()
plt.show()

print("Time taken:", time.time() - start_time, "seconds")
# np.savetxt("tipped_counts_vs_time_random_d_0.08_d1_-0.05_undirected_facebook_414_repulsive_hoi.txt",np.column_stack((times,np.array(tipped_counts))))

