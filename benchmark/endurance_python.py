#!/usr/bin/env python3
"""
O-SMC² Endurance — Python reference using Chopin's 'particles' library.
Matches the exact test configs from endurance_bench.exs.

Usage:
    ~/projects/learn_erl/python-env/bin/python benchmark/endurance_python.py
    ~/projects/learn_erl/python-env/bin/python benchmark/endurance_python.py --quick
"""

import sys
import time
import numpy as np
from scipy.special import gammaln

import particles
from particles import distributions as dists
from particles import state_space_models as ssm
from particles import smc_samplers
from particles.core import SMC


# ── SEIR Model for particles library ────────────────────────────────

class SEIRInitDist(dists.ProbDist):
    dtype = np.dtype([("s", "f8"), ("e", "f8"), ("i", "f8"), ("r", "f8")])

    def __init__(self, n_pop=10000):
        self.n_pop = n_pop

    def rvs(self, size=1):
        out = np.zeros(size, dtype=self.dtype)
        out["s"] = self.n_pop - 1
        out["e"] = 1
        out["i"] = 0
        out["r"] = 0
        return out

    def logpdf(self, x):
        return np.zeros(len(x))


class SEIRTransDist(dists.ProbDist):
    dtype = np.dtype([("s", "f8"), ("e", "f8"), ("i", "f8"), ("r", "f8")])

    def __init__(self, xp, beta, sigma, gamma, n_pop):
        self.xp = xp
        self.beta = beta
        self.sigma = sigma
        self.gamma = gamma
        self.n_pop = n_pop

    def rvs(self, size=None):
        xp = self.xp
        n = len(xp) if size is None else size

        p_se = 1.0 - np.exp(-self.beta * np.maximum(xp["i"], 0) / self.n_pop)
        p_ei = 1.0 - np.exp(-self.sigma)
        p_ir = 1.0 - np.exp(-self.gamma)

        s = np.maximum(xp["s"], 0).astype(int)
        e = np.maximum(xp["e"], 0).astype(int)
        i = np.maximum(xp["i"], 0).astype(int)

        y_se = np.random.binomial(s, np.clip(p_se, 0, 1))
        y_ei = np.random.binomial(e, np.clip(p_ei, 0, 1))
        y_ir = np.random.binomial(i, np.clip(p_ir, 0, 1))

        out = np.zeros(n, dtype=self.dtype)
        out["s"] = np.maximum(xp["s"] - y_se, 0)
        out["e"] = np.maximum(xp["e"] + y_se - y_ei, 0)
        out["i"] = np.maximum(xp["i"] + y_ei - y_ir, 0)
        out["r"] = xp["r"] + y_ir
        return out

    def logpdf(self, x):
        return np.zeros(len(x))


class SEIR(ssm.StateSpaceModel):
    default_params = {"n_pop": 10000}

    def PX0(self):
        return SEIRInitDist(n_pop=self.n_pop)

    def PX(self, t, xp):
        return SEIRTransDist(xp, self.beta, self.sigma, self.gamma, self.n_pop)

    def PY(self, t, xp, x):
        lam = np.maximum(x["i"] * self.sigma, 0.1)
        return dists.Poisson(rate=lam)


class SEIRPrior(dists.StructDist):
    def __init__(self):
        d = {
            "beta":  dists.Uniform(a=0.05, b=0.85),
            "sigma": dists.Uniform(a=0.05, b=0.45),
            "gamma": dists.Uniform(a=0.05, b=0.30),
        }
        super().__init__(d)


# ── Data generation (matching Elixir's generate_seir) ────────────────

def generate_seir(n_pop, beta, sigma, gamma, t_max, seed=42):
    rng = np.random.RandomState(seed)
    s, e, i, r = n_pop - 1, 1, 0, 0
    observations = []

    for _ in range(t_max):
        p_se = 1 - np.exp(-beta * i / n_pop)
        p_ei = 1 - np.exp(-sigma)
        p_ir = 1 - np.exp(-gamma)

        y_se = rng.binomial(max(s, 0), np.clip(p_se, 0, 1))
        y_ei = rng.binomial(max(e, 0), np.clip(p_ei, 0, 1))
        y_ir = rng.binomial(max(i, 0), np.clip(p_ir, 0, 1))

        s = max(s - y_se, 0)
        e = max(e + y_se - y_ei, 0)
        i = max(i + y_ei - y_ir, 0)
        r = r + y_ir

        obs = max(0, y_ei + int(rng.normal(0, max(np.sqrt(max(y_ei, 0)), 1))))
        observations.append(obs)

    return observations


def generate_seir_time_varying(n_pop, t_max, seed=42):
    rng = np.random.RandomState(seed)
    s, e, i, r = n_pop - 5, 5, 0, 0
    beta = 0.4
    sigma, gamma = 0.25, 0.15
    observations = []

    for t in range(1, t_max + 1):
        if 30 < t < 50:
            beta = max(beta * 0.95, 0.1)
        elif 50 < t < 70:
            beta = min(beta * 1.03, 0.6)
        elif 80 < t < 100:
            beta = max(beta * 0.96, 0.1)
        else:
            beta = beta + rng.normal() * 0.01
        beta = max(0.05, min(0.8, beta))

        p_se = 1 - np.exp(-beta * i / n_pop)
        p_ei = 1 - np.exp(-sigma)
        p_ir = 1 - np.exp(-gamma)

        y_se = rng.binomial(max(s, 0), np.clip(p_se, 0, 1))
        y_ei = rng.binomial(max(e, 0), np.clip(p_ei, 0, 1))
        y_ir = rng.binomial(max(i, 0), np.clip(p_ir, 0, 1))

        s = max(s - y_se, 0)
        e = max(e + y_se - y_ei, 0)
        i = max(i + y_ei - y_ir, 0)
        r = r + y_ir

        obs = max(0, y_ei + int(rng.normal(0, max(np.sqrt(max(y_ei, 0)), 1))))
        observations.append(obs)

    return observations


# ── Benchmark runner ─────────────────────────────────────────────────

def run_test(name, data, n_pop, n_theta, n_x, n_moves=3):
    fk = smc_samplers.SMC2(
        ssm_cls=SEIR,
        prior=SEIRPrior(),
        data=data,
        init_Nx=n_x,
        ar_to_increase_Nx=-1.0,
        len_chain=n_moves,
        smc_options={"qmc": False},
    )

    t0 = time.time()
    alg = SMC(fk=fk, N=n_theta, verbose=False)
    alg.run()
    elapsed = time.time() - t0

    W = alg.W
    thetas = alg.X.theta
    beta_m = np.average([th["beta"] for th in thetas], weights=W)
    sigma_m = np.average([th["sigma"] for th in thetas], weights=W)
    gamma_m = np.average([th["gamma"] for th in thetas], weights=W)

    ms = int(elapsed * 1000)
    T = len(data)
    print(f"  {name:<30} {ms:>8}ms  T={T}  β={beta_m:.3f} σ={sigma_m:.3f} γ={gamma_m:.3f}")
    return {"name": name, "time_ms": ms, "T": T,
            "beta": round(beta_m, 3), "sigma": round(sigma_m, 3), "gamma": round(gamma_m, 3)}


# ── Test suite matching endurance_bench.exs ──────────────────────────

def main():
    quick = "--quick" in sys.argv

    print("=" * 70)
    print("  O-SMC² Python Endurance (Chopin's 'particles' library)")
    print("  Single-threaded (Python GIL)")
    print("=" * 70)
    print()

    tests = [
        # name, n_pop, beta, sigma, gamma, t_max, n_theta, n_x, n_moves, generator
        ("smoke-seir (Nθ=100,T=40)",
         lambda: (generate_seir(5000, 0.4, 0.25, 0.15, 40), 5000, 100, 50, 2)),
        ("parallel-smoke (Nθ=100,T=40)",
         lambda: (generate_seir(5000, 0.4, 0.25, 0.15, 40), 5000, 100, 50, 2)),
        ("medium-seir (Nθ=200,T=100)",
         lambda: (generate_seir(10000, 0.4, 0.25, 0.15, 100), 10000, 200, 100, 3)),
        ("time-varying (Nθ=200,T=120)",
         lambda: (generate_seir_time_varying(20000, 120), 20000, 200, 100, 3)),
        ("high-rejuv (Nθ=200,T=80)",
         lambda: (generate_seir_time_varying(10000, 80), 10000, 200, 100, 2)),
        ("full-scale (Nθ=400,T=200)",
         lambda: (generate_seir(50000, 0.3, 0.2, 0.1, 200), 50000, 400, 200, 3)),
        ("memory-T500 (Nθ=100,T=500)",
         lambda: (generate_seir(20000, 0.3, 0.2, 0.1, 500), 20000, 100, 50, 2)),
    ]

    if quick:
        tests = tests[:2]

    header = f"{'Test':<30} {'Time':>10}  {'T':>5}  {'β':>6} {'σ':>6} {'γ':>6}"
    print(header)
    print("-" * 75)

    total_ms = 0
    results = []
    for name, gen_fn in tests:
        data, n_pop, n_theta, n_x, n_moves = gen_fn()
        r = run_test(name, data, n_pop, n_theta, n_x, n_moves)
        results.append(r)
        total_ms += r["time_ms"]

    print()
    print("=" * 70)
    print(f"  Total: {total_ms // 1000}s ({total_ms / 60000:.1f} min)")
    print("=" * 70)

    import json
    with open("benchmark/python_endurance.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Saved to benchmark/python_endurance.json")


if __name__ == "__main__":
    main()
