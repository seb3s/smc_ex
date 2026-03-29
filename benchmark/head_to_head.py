#!/usr/bin/env python3
"""
Head-to-head: smc_ex (Elixir) vs particles (Chopin's Python SMC² library).
SEIR epidemic model with binomial transitions.

Uses Nicolas Chopin's 'particles' library — the reference implementation
of the SMC² algorithm from Chopin, Jacob & Papaspiliopoulos (2013).

Usage:
    ~/projects/learn_erl/python-env/bin/python benchmark/head_to_head.py
"""

import numpy as np
import time
import json
from scipy import stats as sp_stats

import particles
from particles import distributions as dists
from particles import state_space_models as ssm
from particles import smc_samplers
from particles.core import SMC


# ── SEIR State-Space Model ──────────────────────────────────────────

class SEIRState:
    """Wrap SEIR compartments as a structured array for particles lib."""
    pass


class SEIR(ssm.StateSpaceModel):
    """
    Stochastic SEIR with binomial transitions.
    Parameters: beta (transmission), sigma (incubation^-1), gamma (recovery^-1).
    Observation: Poisson-distributed case counts ~ Poisson(new_infections).
    """
    default_params = {"n_pop": 10000}

    def PX0(self):
        """Initial state distribution — deterministic."""
        return SEIRInitDist(n_pop=self.n_pop)

    def PX(self, t, xp):
        """Transition: binomial SEIR dynamics."""
        return SEIRTransDist(xp, self.beta, self.sigma, self.gamma, self.n_pop)

    def PY(self, t, xp, x):
        """Observation: Poisson(new_infections)."""
        # x is structured: x['s'], x['e'], x['i'], x['r']
        lam = np.maximum(x["i"] * self.sigma, 0.1)
        return dists.Poisson(rate=lam)


class SEIRInitDist(dists.ProbDist):
    """Deterministic initial state: S=N-5, E=5, I=0, R=0."""
    dtype = np.dtype([("s", "f8"), ("e", "f8"), ("i", "f8"), ("r", "f8")])

    def __init__(self, n_pop=10000):
        self.n_pop = n_pop

    def rvs(self, size=1):
        out = np.zeros(size, dtype=self.dtype)
        out["s"] = self.n_pop - 5
        out["e"] = 5
        out["i"] = 0
        out["r"] = 0
        return out

    def logpdf(self, x):
        return np.zeros(len(x))


class SEIRTransDist(dists.ProbDist):
    """Binomial SEIR transition."""
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
        # Intractable — not needed for bootstrap PF
        return np.zeros(len(x))


# ── Prior ────────────────────────────────────────────────────────────

class SEIRPrior(dists.StructDist):
    """Uniform prior on (beta, sigma, gamma)."""
    def __init__(self):
        d = {
            "beta":  dists.Uniform(a=0.05, b=0.85),
            "sigma": dists.Uniform(a=0.05, b=0.45),
            "gamma": dists.Uniform(a=0.05, b=0.30),
        }
        super().__init__(d)


# ── Data generation ──────────────────────────────────────────────────

def generate_seir(n_pop, true_beta, true_sigma, true_gamma, t_max, seed=42):
    rng = np.random.RandomState(seed)
    s, e, i, r = n_pop - 5, 5, 0, 0
    observations = []

    for _ in range(t_max):
        p_se = 1 - np.exp(-true_beta * i / n_pop)
        p_ei = 1 - np.exp(-true_sigma)
        p_ir = 1 - np.exp(-true_gamma)

        y_se = rng.binomial(max(s, 0), np.clip(p_se, 0, 1))
        y_ei = rng.binomial(max(e, 0), np.clip(p_ei, 0, 1))
        y_ir = rng.binomial(max(i, 0), np.clip(p_ir, 0, 1))

        s = max(s - y_se, 0)
        e = max(e + y_se - y_ei, 0)
        i = max(i + y_ei - y_ir, 0)
        r = r + y_ir

        obs = max(0, y_ei + int(rng.normal(0, max(np.sqrt(y_ei + 1), 1))))
        observations.append(obs)

    return observations


# ── Benchmark runner ─────────────────────────────────────────────────

def run_smc2(name, data, n_pop, n_theta, n_x, n_moves=3):
    """Run SMC² using Chopin's particles library."""
    prior = SEIRPrior()

    fk = smc_samplers.SMC2(
        ssm_cls=SEIR,
        prior=prior,
        data=data,
        init_Nx=n_x,
        ar_to_increase_Nx=-1.0,  # keep Nx fixed
        len_chain=n_moves,
        smc_options={"qmc": False},
    )

    t0 = time.time()
    alg = SMC(fk=fk, N=n_theta, verbose=False)
    alg.run()
    elapsed = time.time() - t0

    # Extract posterior means
    W = alg.W  # normalized weights
    thetas = alg.X.theta
    beta_mean = np.average([th["beta"] for th in thetas], weights=W)
    sigma_mean = np.average([th["sigma"] for th in thetas], weights=W)
    gamma_mean = np.average([th["gamma"] for th in thetas], weights=W)

    return {
        "name": name,
        "time_ms": int(elapsed * 1000),
        "T": len(data),
        "n_theta": n_theta,
        "n_x": n_x,
        "beta": round(beta_mean, 3),
        "sigma": round(sigma_mean, 3),
        "gamma": round(gamma_mean, 3),
    }


def main():
    print("=" * 70)
    print("particles (Chopin's Python SMC²) — Head-to-Head Reference")
    print("=" * 70)
    print()

    tests = [
        ("smoke (T=40, Nθ=50, Nx=50)", 10000, 0.4, 0.25, 0.15, 40, 50, 50, 3),
        ("medium (T=100, Nθ=100, Nx=100)", 10000, 0.4, 0.25, 0.15, 100, 100, 100, 3),
        ("full-scale (T=200, Nθ=200, Nx=100)", 10000, 0.4, 0.25, 0.15, 200, 200, 100, 3),
        ("large (T=200, Nθ=400, Nx=200)", 50000, 0.3, 0.2, 0.1, 200, 400, 200, 3),
    ]

    header = f"{'Test':<45} {'Time':>10} {'β':>8} {'σ':>8} {'γ':>8}"
    print(header)
    print("-" * 85)

    results = []
    for name, n_pop, beta, sigma, gamma, t_max, n_theta, n_x, n_moves in tests:
        data = generate_seir(n_pop, beta, sigma, gamma, t_max, seed=42)
        r = run_smc2(name, data, n_pop, n_theta, n_x, n_moves)
        results.append(r)
        print(f"  {name:<43} {r['time_ms']:>8}ms  {r['beta']:>6}  {r['sigma']:>6}  {r['gamma']:>6}")

    print()
    print("=" * 70)

    with open("benchmark/python_head_to_head.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Saved to benchmark/python_head_to_head.json")


if __name__ == "__main__":
    main()
