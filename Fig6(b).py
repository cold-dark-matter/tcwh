# -*- coding: utf-8 -*-
"""
Created on Fri Aug 22 12:13:58 2025

@author: Richita
"""

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import pickle
import time
from numba import njit
import multiprocessing as mp

# ---------------------
# Parameter
# ---------------------
a = 4.0
b = 1.0
A = np.load("A_sparse_3980.npy")          # adjacency matrix (NxN)
N = A.shape[0]
dim = 1
x0 = 0.5
r = np.zeros(N); r[0] = 0.2          # trigger node

with open("A1_3980.pkl", "rb") as f:
    A1_data = pickle.load(f)
i_idx = np.asarray(A1_data["i"], dtype=np.int64)
j_idx = np.asarray(A1_data["j"], dtype=np.int64)
k_idx = np.asarray(A1_data["k"], dtype=np.int64)
A1_vals = np.asarray(A1_data["values"], dtype=np.float64)

# ---------------------
# HOI
# ---------------------
@njit
def compute_hoi(x, i_idx, j_idx, k_idx, vals, N):
    hoi_term = np.zeros(N)
    for n in range(len(vals)):
        i = i_idx[n]
        j = j_idx[n]
        k = k_idx[n]
        hoi_term[i] += vals[n] * x[j] * x[k]
    return hoi_term

# ---------------------
# Model 
# ---------------------
def make_model(d, d1):
    def model(states, t, r):
        x = states[0:dim * N:dim]
        dx = np.zeros_like(states)
        linear_term = A.dot(x)
        hoi_term = compute_hoi(x, i_idx, j_idx, k_idx, A1_vals, N)
        dx[0:dim * N:dim] = -a * (x - x0) ** 3 + b * (x - x0) + r + d * linear_term + d1 * hoi_term 
        return dx
    return model


def simulate_cell(args):
    ii, jj, d1, d, t_end, t_step = args
    x_current = np.zeros(N * dim)
    t_current = 0.0
    model = make_model(d, d1)

    while t_current < t_end:
        t_span = [t_current, t_current + t_step]
        sol = odeint(model, x_current, t_span, args=(r,), atol=1e-10, rtol=1e-8)
        x_current = sol[-1]
        t_current += t_step

    final_tipped = np.count_nonzero(x_current > 0.5)
    tipped = 1 if final_tipped > 1 else 0
    return ii, jj, tipped

def main():
    start_time = time.time()


    d_vals  = np.linspace(0.0, 0.1, 50)   # linear coupling
    d1_vals = np.linspace(0.0, 0.3, 50)   # HOI coupling
    t_end   = 100.0
    t_step  = 1.0


    tasks = [(ii, jj, float(d1), float(d), t_end, t_step)
             for ii, d1 in enumerate(d1_vals)
             for jj, d  in enumerate(d_vals)]

    with mp.Pool(processes=mp.cpu_count()-4) as pool:
        results = pool.map(simulate_cell, tasks)


    tipping_matrix = np.zeros((len(d1_vals), len(d_vals)))
    for ii, jj, tipped in results:
        tipping_matrix[ii, jj] = tipped


    plt.figure(figsize=(7, 5))
    plt.imshow(
        tipping_matrix,
        origin="lower",
        aspect="auto",
        extent=[d_vals[0], d_vals[-1], d1_vals[0], d1_vals[-1]],
        cmap="coolwarm",
        vmin=0, vmax=1,
    )
    plt.colorbar(label="Tipped (1 = yes, 0 = no)")
    plt.xlabel("Linear coupling d")
    plt.ylabel("HOI coupling d1")
    # plt.title("Tipping matrix (parallel)")
    plt.tight_layout()
    plt.show()

    import pandas as pd
    rows = []
    for ii, d1 in enumerate(d1_vals):
        for jj, d in enumerate(d_vals):
            rows.append({"d": d, "d1": d1, "value": tipping_matrix[ii, jj]})
    pd.DataFrame(rows).to_csv("tipping_matrix_parallel_3980.csv", index=False)

    print("Saved tipping_matrix_parallel_3980.csv")
    print(f"Time taken: {time.time() - start_time:.2f} s")

if __name__ == "__main__":

    main()

