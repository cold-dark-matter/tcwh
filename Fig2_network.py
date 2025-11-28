

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import networkx as nx
from numba import njit
import scipy.sparse as sp
import time

start_time = time.time()

a = 4
b = 1
N = 100
dim = 1
x0 = 0.5
d1 = 5
r_base = np.zeros(N)
d = np.ones(N) * 0.05
r_base[0] = 0.2  # trigger node 0

average_degree = 6
p = average_degree / (N - 1)


deriv_tol = 1e-5 

@njit
def compute_hoi(x, i_idx, j_idx, k_idx, vals, N):
    hoi = np.zeros(N)
    for n in range(len(vals)):
        i = i_idx[n]
        j = j_idx[n]
        k = k_idx[n]
        hoi[i] += vals[n] * x[j] * x[k]
    return hoi


def model(states, t, r):
    x = states[0:dim * N:dim]
    dx = np.zeros_like(states)

    linear_term = A_sparse.dot(x)
    hoi_term = compute_hoi(x, i_idx, j_idx, k_idx, val_idx, N)

    dx[0:dim * N:dim] = (-a * (x - x0) ** 3  + b * (x - x0)+ r + (d * linear_term) + (d1 * hoi_term / 2.0))
    return dx


def run_one_realization(seed=None):
    global A_sparse, i_idx, j_idx, k_idx, val_idx

    # Network
    G = nx.gnp_random_graph(N, p, seed=seed, directed=False)

    # ---------------------BA------------------------------------------------
    # m = 2
    # G = nx.barabasi_albert_graph(N, m)

    # ---------------------WS------------------------------------------------
    # k = 6      # nearest neighbors
    # p_ws = 0.2 # rewiring probability
    # G = nx.watts_strogatz_graph(N, k, p_ws)

    A = nx.to_numpy_array(G).T
    A_sparse = sp.csr_matrix(A)


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

    # Integrate
    x_current = np.zeros(N * dim)
    r = r_base.copy()

    t_current = 1.0
    t_end = 100.0
    t_step = 0.01

    times = [t_current]
    tipped_counts = [np.count_nonzero(x_current > 0.5)]

    while t_current < t_end:
        t_span = [t_current, t_current + t_step]
        sol = odeint(model, x_current, t_span, args=(r,))
        x_current = sol[1]
        t_current += t_step


        dx_current = model(x_current, t_current, r)
        max_abs_dx = np.max(np.abs(dx_current[0:dim * N:dim]))
        if max_abs_dx < deriv_tol:
            break

        times.append(t_current)
        tipped_counts.append(np.count_nonzero(x_current > 0.5))

    tipped_counts = np.array(tipped_counts)
    return times, tipped_counts



num_runs = 100  
cascade_threshold_count = 2

cascade_runs = 0
any_tipping_runs = 0
max_counts = []

for run in range(num_runs):
    print(f"=== Realization {run+1}/{num_runs} ===")
    times, tipped_counts = run_one_realization(seed=2 + run)
    max_tip = int(tipped_counts.max())
    max_counts.append(max_tip)

    if max_tip > 0:
        any_tipping_runs += 1
    if max_tip > cascade_threshold_count:
        cascade_runs += 1

end_time = time.time()

print(f"\nSummary over {num_runs} realizations:")
print(f" - Any tipping: {any_tipping_runs}/{num_runs} ({any_tipping_runs/num_runs:.1%})")
print(f" - Cascades (> {cascade_threshold_count} nodes): {cascade_runs}/{num_runs} ({cascade_runs/num_runs:.1%})")
print(f" - Mean of max tipped count: {np.mean(max_counts):.2f} / N={N}")
print('Time taken:', end_time - start_time, 'seconds')


p_cascade = cascade_runs / num_runs  # cascade frequency

# Normal approximation 95% CI for a binomial proportion
se = np.sqrt(p_cascade * (1 - p_cascade) / num_runs)
z = 1.96  # for 95% CI
ci_low = p_cascade - z * se
ci_high = p_cascade + z * se



yerr = [[p_cascade - ci_low], [ci_high - p_cascade]] 


# ----- Plot bar with error bar -----
plt.figure(figsize=(4, 5))

# Single bar at x=0
plt.bar([0], [p_cascade], yerr=yerr, capsize=8)
plt.xticks([0], ['ER'])
plt.ylabel('Cascade probability')
plt.ylim(0, 1)


plt.tight_layout()
plt.show()