#!/usr/bin/env python3
"""Generate pre-computed TurboQuant codebooks for d=256.

Standalone — no torch dependency, only numpy + scipy.
"""
import json
import math
import os

import numpy as np
from scipy import integrate, special


def beta_pdf(x, d):
    if d <= 2:
        raise ValueError(f"d={d} too small")
    log_const = special.gammaln(d / 2.0) - 0.5 * np.log(np.pi) - special.gammaln((d - 1) / 2.0)
    exponent = (d - 3) / 2.0
    x = np.clip(x, -1 + 1e-15, 1 - 1e-15)
    return np.exp(log_const + exponent * np.log(1 - x**2))


def _conditional_mean(lo, hi, d):
    num, _ = integrate.quad(lambda x: x * beta_pdf(np.array([x]), d)[0], lo, hi)
    den, _ = integrate.quad(lambda x: beta_pdf(np.array([x]), d)[0], lo, hi)
    return num / den if den > 1e-30 else (lo + hi) / 2.0


def compute_lloyd_max_codebook(d, bits, max_iter=200, tol=1e-12):
    n_clusters = 2**bits
    x_grid = np.linspace(-1 + 1e-10, 1 - 1e-10, 10000)
    pdf_vals = beta_pdf(x_grid, d)
    cdf_vals = np.cumsum(pdf_vals) * (x_grid[1] - x_grid[0])
    cdf_vals /= cdf_vals[-1]

    quantile_edges = np.linspace(0, 1, n_clusters + 1)
    centroids = np.zeros(n_clusters)
    for i in range(n_clusters):
        q_mid = (quantile_edges[i] + quantile_edges[i + 1]) / 2.0
        idx = min(np.searchsorted(cdf_vals, q_mid), len(x_grid) - 1)
        centroids[i] = x_grid[idx]

    prev_cost = float("inf")
    for _ in range(max_iter):
        boundaries = np.zeros(n_clusters + 1)
        boundaries[0] = -1.0
        boundaries[-1] = 1.0
        for i in range(n_clusters - 1):
            boundaries[i + 1] = (centroids[i] + centroids[i + 1]) / 2.0
        new_centroids = np.array([_conditional_mean(boundaries[i], boundaries[i + 1], d) for i in range(n_clusters)])
        cost = 0.0
        for i in range(n_clusters):
            c = new_centroids[i]
            val, _ = integrate.quad(lambda x, c=c: (x - c) ** 2 * beta_pdf(np.array([x]), d)[0], boundaries[i], boundaries[i + 1])
            cost += val
        centroids = new_centroids
        if abs(prev_cost - cost) < tol:
            break
        prev_cost = cost

    boundaries = np.zeros(n_clusters + 1)
    boundaries[0] = -1.0
    boundaries[-1] = 1.0
    for i in range(n_clusters - 1):
        boundaries[i + 1] = (centroids[i] + centroids[i + 1]) / 2.0

    return {
        "centroids": centroids.tolist(),
        "boundaries": boundaries.tolist(),
        "mse_per_coord": float(cost),
        "mse_total": float(cost * d),
        "d": d,
        "bits": bits,
    }


out_dir = os.environ.get('CODEBOOK_OUT_DIR',
    os.path.join(os.path.dirname(__file__), '..', 'external', 'turboquant', 'turboquant', 'codebooks'))
os.makedirs(out_dir, exist_ok=True)

for d in [256]:
    for bits in [1, 2, 3, 4]:
        path = os.path.join(out_dir, f"codebook_d{d}_b{bits}.json")
        if os.path.exists(path):
            print(f"Already exists: {path}")
            continue
        print(f"Computing codebook d={d}, bits={bits}...")
        cb = compute_lloyd_max_codebook(d, bits)
        with open(path, 'w') as f:
            json.dump(cb, f, indent=2)
        print(f"  MSE/coord={cb['mse_per_coord']:.6e}, saved to {path}")

print("Done!")
