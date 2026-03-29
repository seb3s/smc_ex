# O-SMC² Sprint 3: Close the Full-Scale Gap

## Current State

Sprint 1+2 achieved 17.5x on full-scale (92 min → 5.1 min). But Chopin's
`particles` library still wins full-scale: 3.0 min vs our 5.1 min (0.59x).

The reason: **waste-free SMC²**.

## What Waste-Free SMC² Does

Standard SMC² (our approach): run `n_moves` MCMC moves per rejuvenation,
keep only the final accepted state. 3 moves × Nθ particles = 3Nθ PF runs,
but only Nθ samples kept.

Waste-free SMC² (Chopin 2020, Dau & Chopin 2022): run `len_chain` MCMC moves,
keep ALL intermediate states. After 10 moves on Nθ particles:
Nθ × (1 + 10) = 11Nθ weighted samples. No computation discarded.

The `particles` library starts with N0 = N / len_chain particles and
multiplies back to N after the MCMC chain. Each rejuvenation produces the
same number of particles but wastes zero MCMC work.

## Sprint 3 Optimizations

### 3a. Waste-Free MCMC Rejuvenation

Replace the current accept/reject PMCMC with waste-free:

```elixir
# Current: run 3 moves, keep final
Enum.reduce_while(1..n_moves, {theta, log_ev, 0, pf_state}, fn _, acc ->
  # propose, accept/reject, halt on first accept
end)

# Waste-free: run len_chain moves, keep ALL
chain = Enum.scan(1..len_chain, {theta, log_ev, pf_state}, fn _, {th, le, pf} ->
  # propose, always keep the result (accepted or rejected)
  {new_theta, new_log_ev, accepted, new_pf} = pmcmc_step(th, le, pf, ...)
  if accepted, do: {new_theta, new_log_ev, new_pf}, else: {th, le, pf}
end)
# Return all len_chain+1 states as weighted particles
```

At rejuvenation time, start with Nθ / (len_chain + 1) particles. After
the chain, concatenate to get Nθ particles. Each has correct weight.

**Expected impact**: Eliminates the concept of "wasted" MCMC moves entirely.
With len_chain=10, same compute as n_moves=10 but 11x more effective samples.
Should match or beat `particles` on full-scale.

**Effort**: Medium. Restructure `PMCMC.rejuvenate` and the rejuvenation
block in `OnlineSMC2`.

**Risk**: Moderate. Weight computation for waste-free samples requires careful
implementation (Dau & Chopin 2022, Section 2.3). Need to verify ESS
calculation works correctly with the expanded particle set.

### 3b. Eliminate Round 1 (Incremental Evidence Caching)

Round 1 runs 400 PF evaluations to compute `curr_log_ev` for the MH
denominator. But we already computed incremental log-likelihoods during
the normal BPF step.

Track a rolling buffer of the last `window` incremental log-likelihoods
per θ-particle:

```elixir
# In bpf_step, after computing inc_log_lik:
evidence_buffer = Enum.take([inc_log_lik | evidence_buffer], window)
windowed_log_ev = Enum.sum(evidence_buffer)
```

Pass `windowed_log_ev` to PMCMC as the MH denominator. No Round 1 needed.

**Expected impact**: Eliminates 400 PF runs per rejuvenation. At 110
rejuvenations over T=200: 44,000 PF runs saved. ~20% speedup on full-scale.

**Effort**: Low-medium. Add `evidence_buffer` to PF state, sum at
rejuvenation time.

**Risk**: Low. The incremental evidence is mathematically equivalent to the
windowed filter evidence. Just accumulation vs re-computation.

### 3c. Adaptive ESS Threshold

Two-tier threshold:
- Soft threshold (0.5 × Nθ): trigger lightweight rejuvenation (len_chain=3)
- Hard threshold (0.3 × Nθ): trigger full rejuvenation (len_chain=10)

After a rejuvenation where the posterior barely changed (acceptance rate > 60%),
temporarily lower the threshold to 0.3 for the next 5 steps.

**Expected impact**: Reduce rejuvenation frequency from 60% to ~35%.
~40% fewer rejuvenation rounds.

**Effort**: Low. Modify the ESS check in the main loop.

**Risk**: Low. Any positive threshold maintains algorithm validity.

## Implementation Order

1. **3b** (incremental evidence) — low effort, standalone, 20% speedup
2. **3c** (adaptive threshold) — low effort, standalone, ~35% speedup
3. **3a** (waste-free MCMC) — medium effort, the main algorithmic change

Combined target: full-scale from 5.1 min to **< 2 min** (beat Python's 3.0 min).

## References

- Dau, H.D. & Chopin, N. (2022). "Waste-free Sequential Monte Carlo."
  *JRSS-B*, 84(1), 114-148.
- Chopin, N. & Papaspiliopoulos, O. (2020). *An Introduction to Sequential
  Monte Carlo*. Springer.
