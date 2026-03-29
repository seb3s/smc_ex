# Benchmark Results: smc_ex vs particles (Chopin's Python SMC²)

**Date**: 2026-03-27
**Hardware**: Dual Intel Xeon E5-2699 v4 (44 cores / 88 threads), 256GB RAM
**Python**: particles 0.4 (Chopin's library), single-threaded (GIL)
**Elixir**: +sbt tnnps, 88 BEAM schedulers, Task.async_stream
**Each run sequential** — full machine per implementation.

## Wall Time

| Test | Python (1 core) | Elixir (88 cores) | Ratio |
|---|---|---|---|
| smoke (Nθ=100, T=40) | 3,849ms | **584ms** | **6.6x** |
| medium (Nθ=200, T=100) | 44,208ms | **3,619ms** | **12.2x** |
| time-varying (Nθ=200, T=120) | 130,390ms | **13,494ms** | **9.7x** |
| high-rejuv (Nθ=200, T=80) | 33,429ms | **4,619ms** | **7.2x** |
| full-scale (Nθ=400, T=200) | **180,997ms** | 306,759ms | 0.59x |
| memory-T500 (Nθ=100, T=500) | 62,566ms | **15,966ms** | **3.9x** |
| **Total** | **458s (7.6 min)** | **346s (5.8 min)** | **1.3x** |

Elixir wins **5 of 7 tests** and **1.3x overall**.

## Analysis

**BEAM parallelism dominates at moderate scale.** With Nθ=100-200,
Task.async_stream fans out PF runs across 88 cores. Python's GIL limits
it to sequential execution.

**Python wins full-scale (Nθ=400).** The `particles` library implements
Chopin's waste-free SMC² variant, which avoids much of the rejuvenation
overhead. Our Sprint 2 eliminated one redundant PF round and added early
PMCMC termination, but the waste-free approach is algorithmically superior
for high rejuvenation rates (60%+ of steps at Nθ=400).

**Per-core comparison.** On the single-threaded smoke test, Python takes
3,849ms vs Elixir's 1,217ms (sequential, parallel: false). Elixir is
3.2x faster per core — pure Elixir's `:rand` + map operations are faster
than Python + NumPy for this scalar-heavy workload.

## Optimization History

| Version | full-scale (Nθ=400, T=200) | Total suite |
|---|---|---|
| v0 (original) | 5,550,413ms (92 min) | 5,813s (97 min) |
| v1 (Sprint 1+2) | 317,234ms (5.3 min) | 356s (5.9 min) |
| v2 (dedicated run) | 306,759ms (5.1 min) | 346s (5.8 min) |
| particles Python | 180,997ms (3.0 min) | 458s (7.6 min) |

## Sprint 3 Target

Close the full-scale gap. Primary bottleneck: rejuvenation overhead at
high Nθ. The `particles` library's waste-free SMC² avoids redundant PF
computation entirely. Our approach still runs Round 1 (400 PF evaluations
for current log-evidence). Implementing incremental evidence caching +
adaptive ESS threshold should bring full-scale below Python.
