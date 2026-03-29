defmodule SMC.OnlineSMC2 do
  @moduledoc """
  Online Sequential Monte Carlo Squared (O-SMC²).

  Joint online inference of parameters AND latent states for state-space
  models with intractable likelihoods. Implements Algorithm 2 from
  Temfack & Wyse (2025) with the windowed PMCMC rejuvenation from
  Vieira (2018).

  The key advantage: computational cost per observation is O(tk × Nθ × Nx)
  — constant regardless of how long the time series runs. Standard SMC²
  has cost O(t × Nθ × Nx) which grows linearly with time.

  ## Usage

      # Define state-space model (θ-parameterized)
      model = %{
        init: fn(theta, rng) -> {state, rng},
        transition: fn(state, theta, t, rng) -> {new_state, rng},
        observation_logp: fn(state, theta, y_t) -> float
      }

      # Define prior on θ
      prior = %{
        sample: fn(rng) -> {%{beta: 0.3 + noise, ...}, rng},
        logpdf: fn(theta) -> -0.5 * ... end
      }

      # Run
      result = OnlineSMC2.run(model, prior, observations,
        n_theta: 400, n_x: 200, window: 20)

      # result.theta_particles  — weighted θ samples
      # result.theta_weights    — normalized weights
      # result.posterior_history — θ posterior at each time step
      # result.ess_history      — ESS of θ-particles over time
  """

  alias SMC.PMCMC

  require Logger

  @default_opts [
    n_theta: 200,
    n_x: 100,
    window: 20,
    resample_threshold: 0.5,
    n_moves: 3,
    proposal_scale: 2.0,
    seed: 42,
    parallel: true,
    adaptive_threshold: false,
    min_threshold: 0.3,
    waste_free: false,
    len_chain: 10
  ]

  @doc """
  Run O-SMC² on a batch of observations.

  Returns a map with:
  - `:theta_particles` — final θ samples (list of maps)
  - `:theta_weights` — final normalized weights
  - `:posterior_history` — weighted θ mean at each time step
  - `:ess_history` — ESS of θ-particles over time
  - `:rejuvenation_count` — how many times rejuvenation was triggered
  - `:log_evidence` — estimated log marginal likelihood
  """
  def run(model, prior, observations, opts \\ []) do
    opts = Keyword.merge(@default_opts, opts)
    n_theta = opts[:n_theta]
    n_x = opts[:n_x]
    window = opts[:window]
    threshold = opts[:resample_threshold]
    n_moves = opts[:n_moves]
    proposal_scale = opts[:proposal_scale]
    seed = opts[:seed]
    parallel = opts[:parallel]
    adaptive_threshold = opts[:adaptive_threshold]
    min_threshold = opts[:min_threshold]
    waste_free = opts[:waste_free]
    len_chain = opts[:len_chain]

    rng = :rand.seed_s(:exsss, {seed, seed + 1, seed + 2})

    # Initialize θ-particles from prior
    {theta_particles, rng} =
      Enum.map_reduce(1..n_theta, rng, fn _, rng ->
        prior.sample.(rng)
      end)

    # Initialize a particle filter for each θ-particle
    pf_states =
      theta_particles
      |> Enum.with_index()
      |> Enum.map(fn {theta, i} ->
        pf_model = bind_model(model, theta)
        # Initialize x-particles
        pf_rng = :rand.seed_s(:exsss, {seed + i, seed + i + 1, seed + i + 2})

        {particles, _rng} =
          Enum.map_reduce(1..n_x, pf_rng, fn _, rng ->
            pf_model.init.(rng)
          end)

        %{particles: particles, log_weights: List.duplicate(0.0, n_x)}
      end)

    # Uniform weights
    log_omega = List.duplicate(0.0, n_theta)

    # Evidence buffers: one per θ-particle, rolling list of last `window` inc_log_liks
    evidence_buffers = List.duplicate([], n_theta)

    # Observation window (bounded, stored in reverse for O(1) prepend)
    obs_window_rev = []
    # Accumulate histories in reverse (prepend), reverse at the end
    posterior_history_rev = []
    ess_history_rev = []
    rejuvenation_count = 0
    log_evidence = 0.0

    # Adaptive threshold state
    dynamic_threshold = threshold
    cooldown = 0

    # Process observations sequentially
    {theta_particles, _pf_states, log_omega, posterior_history_rev, ess_history_rev,
     rejuvenation_count, log_evidence, _obs_window_rev, _rng, _ev_bufs, _dyn_thresh,
     _cooldown} =
      observations
      |> Enum.with_index(1)
      |> Enum.reduce(
        {theta_particles, pf_states, log_omega, posterior_history_rev, ess_history_rev,
         rejuvenation_count, log_evidence, obs_window_rev, rng, evidence_buffers,
         dynamic_threshold, cooldown},
        fn {y_t, t},
           {thetas, pfs, log_omega, post_hist_rev, ess_hist_rev, rej_count, log_ev, obs_win_rev,
            rng, ev_bufs, dyn_thresh, cooldown} ->
          # Add observation to bounded window (prepend + take last `window`)
          obs_win_rev = [y_t | obs_win_rev]

          obs_win_rev =
            if length(obs_win_rev) > window, do: Enum.take(obs_win_rev, window), else: obs_win_rev

          # Step 1: Run one BPF step for each θ-particle
          # Compute incremental likelihood p̂(y_t | y_{1:t-1}, θ_m)
          {pfs_new, inc_log_liks} =
            if parallel do
              results =
                Enum.zip(thetas, pfs)
                |> Task.async_stream(
                  fn {theta, pf} ->
                    bpf_step(pf, theta, model, y_t, t)
                  end,
                  max_concurrency: System.schedulers_online(),
                  timeout: 30_000
                )
                |> Enum.map(fn {:ok, result} -> result end)

              pfs_new = Enum.map(results, &elem(&1, 0))
              inc_liks = Enum.map(results, &elem(&1, 1))
              {pfs_new, inc_liks}
            else
              {pfs_new, inc_liks} =
                Enum.zip(thetas, pfs)
                |> Enum.map(fn {theta, pf} -> bpf_step(pf, theta, model, y_t, t) end)
                |> Enum.unzip()

              {pfs_new, inc_liks}
            end

          # 3b: Update evidence buffers — prepend inc_log_lik, trim to window size
          ev_bufs =
            Enum.zip(ev_bufs, inc_log_liks)
            |> Enum.map(fn {buf, ill} ->
              new_buf = [ill | buf]
              if length(new_buf) > window, do: Enum.take(new_buf, window), else: new_buf
            end)

          # Step 2: Update θ-particle weights
          new_log_omega =
            Enum.zip(log_omega, inc_log_liks)
            |> Enum.map(fn {lo, ill} -> lo + ill end)

          # Normalize
          max_lo = Enum.max(new_log_omega)
          omega_unnorm = Enum.map(new_log_omega, fn lo -> :math.exp(lo - max_lo) end)
          sum_o = Enum.sum(omega_unnorm)
          omega_norm = Enum.map(omega_unnorm, fn o -> o / sum_o end)

          # Incremental log evidence
          max_ill = Enum.max(inc_log_liks)

          inc_ev =
            max_ill +
              :math.log(
                Enum.sum(Enum.map(inc_log_liks, fn l -> :math.exp(l - max_ill) end)) / n_theta
              )

          log_ev = log_ev + inc_ev

          # Compute ESS
          ess = 1.0 / Enum.sum(Enum.map(omega_norm, fn w -> w * w end))

          # Posterior mean at this step
          post_mean = compute_weighted_mean(thetas, omega_norm)

          # 3c: Adaptive ESS threshold
          effective_threshold =
            if adaptive_threshold do
              dyn_thresh
            else
              threshold
            end

          # Step 3: Rejuvenate if ESS below threshold
          {thetas, pfs_new, new_log_omega, rej_count, rng, ev_bufs, dyn_thresh, cooldown} =
            if ess < effective_threshold * n_theta do
              # 3c: Determine n_moves based on ESS severity
              effective_n_moves =
                if adaptive_threshold and ess >= min_threshold * n_theta do
                  # Soft threshold: lightweight rejuvenation
                  1
                else
                  # Hard threshold (or non-adaptive): full rejuvenation
                  n_moves
                end

              # Resample θ-particles (tuple-based O(1) indexing)
              {resampled_indices, rng} = stratified_resample_indices(omega_norm, n_theta, rng)

              thetas_tuple = List.to_tuple(thetas)
              pfs_tuple = List.to_tuple(pfs_new)
              ev_bufs_tuple = List.to_tuple(ev_bufs)
              thetas_resampled = Enum.map(resampled_indices, &elem(thetas_tuple, &1))
              _pfs_resampled = Enum.map(resampled_indices, &elem(pfs_tuple, &1))
              ev_bufs_resampled = Enum.map(resampled_indices, &elem(ev_bufs_tuple, &1))

              # 3b: Compute windowed log evidence from cached buffers (no Round 1 PF!)
              cached_log_evs = Enum.map(ev_bufs_resampled, fn buf -> Enum.sum(buf) end)

              # Compute proposal distribution (Normal centered on weighted mean/cov)
              proposal = compute_proposal(thetas, omega_norm, proposal_scale)

              # O-SMC² window: reverse the bounded window to chronological order
              obs_window = Enum.reverse(obs_win_rev)

              # Decide between waste-free and standard rejuvenation
              {rejuv_thetas, rejuv_pfs, rejuv_ev_bufs, rejuv_accept_rate, rng} =
                if waste_free do
                  {rt, rp, re, ra} =
                    rejuvenate_waste_free(
                      thetas_resampled,
                      cached_log_evs,
                      ev_bufs_resampled,
                      model,
                      prior,
                      obs_window,
                      proposal,
                      n_theta,
                      n_x,
                      len_chain,
                      seed,
                      t,
                      parallel
                    )

                  {rt, rp, re, ra, rng}
                else
                  # Standard PMCMC rejuvenation with cached evidence (no Round 1)
                  rejuvenate_standard(
                    thetas_resampled,
                    cached_log_evs,
                    ev_bufs_resampled,
                    model,
                    prior,
                    obs_window,
                    proposal,
                    effective_n_moves,
                    n_x,
                    seed,
                    t,
                    n_theta,
                    parallel,
                    rng
                  )
                end

              # 3c: Update dynamic threshold based on acceptance rate
              {dyn_thresh, cooldown} =
                if adaptive_threshold do
                  if rejuv_accept_rate > 0.5 do
                    # Well-explored posterior, lower threshold for 5 steps
                    {min_threshold, 5}
                  else
                    {threshold, 0}
                  end
                else
                  {threshold, 0}
                end

              # Reset weights after resampling
              reset_omega = List.duplicate(0.0, n_theta)

              Logger.debug(
                "[O-SMC²] t=#{t}: rejuvenated (ESS=#{Float.round(ess, 1)}, moves=#{effective_n_moves}, accept=#{Float.round(rejuv_accept_rate, 2)})"
              )

              {rejuv_thetas, rejuv_pfs, reset_omega, rej_count + 1, rng, rejuv_ev_bufs,
               dyn_thresh, cooldown}
            else
              # 3c: Tick down cooldown
              {dyn_thresh, cooldown} =
                if adaptive_threshold and cooldown > 0 do
                  new_cd = cooldown - 1
                  if new_cd == 0, do: {threshold, 0}, else: {dyn_thresh, new_cd}
                else
                  {dyn_thresh, cooldown}
                end

              {thetas, pfs_new, new_log_omega, rej_count, rng, ev_bufs, dyn_thresh, cooldown}
            end

          {thetas, pfs_new, new_log_omega, [post_mean | post_hist_rev], [ess | ess_hist_rev],
           rej_count, log_ev, obs_win_rev, rng, ev_bufs, dyn_thresh, cooldown}
        end
      )

    posterior_history = Enum.reverse(posterior_history_rev)
    ess_history = Enum.reverse(ess_history_rev)

    # Final normalized weights
    max_lo = Enum.max(log_omega)
    omega_unnorm = Enum.map(log_omega, fn lo -> :math.exp(lo - max_lo) end)
    sum_o = Enum.sum(omega_unnorm)
    omega_norm = Enum.map(omega_unnorm, fn o -> o / sum_o end)

    %{
      theta_particles: theta_particles,
      theta_weights: omega_norm,
      posterior_history: posterior_history,
      ess_history: ess_history,
      rejuvenation_count: rejuvenation_count,
      log_evidence: log_evidence,
      n_observations: length(observations)
    }
  end

  # --- Private ---

  # Standard PMCMC rejuvenation with cached evidence (3b: no Round 1 PF)
  defp rejuvenate_standard(
         thetas,
         cached_log_evs,
         ev_bufs,
         model,
         prior,
         obs_window,
         proposal,
         n_moves,
         n_x,
         seed,
         t,
         n_theta,
         parallel,
         rng
       ) do
    indexed = Enum.zip([thetas, cached_log_evs, ev_bufs]) |> Enum.with_index()

    if parallel do
      results =
        indexed
        |> Task.async_stream(
          fn {{theta, curr_log_ev, _ev_buf}, i} ->
            {new_theta, _new_ev, accepted, pf_state} =
              PMCMC.rejuvenate(theta, curr_log_ev, model, prior, obs_window, proposal,
                n_moves: n_moves,
                n_x: n_x,
                seed: seed + t * 1000 + i + n_theta
              )

            {new_theta, pf_state, if(accepted > 0, do: 1, else: 0)}
          end,
          max_concurrency: System.schedulers_online(),
          timeout: 60_000
        )
        |> Enum.map(fn {:ok, result} -> result end)

      rejuv_thetas = Enum.map(results, &elem(&1, 0))
      rejuv_pfs = Enum.map(results, &elem(&1, 1))
      total_accepted = Enum.sum(Enum.map(results, &elem(&1, 2)))
      accept_rate = total_accepted / n_theta

      # After rejuvenation, evidence buffers are reset (new theta may differ)
      # For accepted particles, buffer is stale; reset to empty so next
      # observations rebuild it. For rejected, keep existing buffer.
      rejuv_ev_bufs =
        Enum.zip(results, ev_bufs)
        |> Enum.map(fn {{_th, _pf, acc}, old_buf} ->
          if acc > 0, do: [], else: old_buf
        end)

      {rejuv_thetas, rejuv_pfs, rejuv_ev_bufs, accept_rate, rng}
    else
      {results, rng} =
        Enum.map_reduce(indexed, rng, fn {{theta, curr_log_ev, _ev_buf}, i}, rng ->
          {new_theta, _new_ev, accepted, pf_state} =
            PMCMC.rejuvenate(theta, curr_log_ev, model, prior, obs_window, proposal,
              n_moves: n_moves,
              n_x: n_x,
              seed: seed + t * 1000 + i + n_theta
            )

          {{new_theta, pf_state, if(accepted > 0, do: 1, else: 0)}, rng}
        end)

      rejuv_thetas = Enum.map(results, &elem(&1, 0))
      rejuv_pfs = Enum.map(results, &elem(&1, 1))
      total_accepted = Enum.sum(Enum.map(results, &elem(&1, 2)))
      accept_rate = total_accepted / n_theta

      rejuv_ev_bufs =
        Enum.zip(results, ev_bufs)
        |> Enum.map(fn {{_th, _pf, acc}, old_buf} ->
          if acc > 0, do: [], else: old_buf
        end)

      {rejuv_thetas, rejuv_pfs, rejuv_ev_bufs, accept_rate, rng}
    end
  end

  # 3a: Waste-free MCMC rejuvenation (Dau & Chopin 2022)
  # Thin to n_thin = n_theta / (len_chain + 1), run len_chain MCMC moves,
  # keep ALL intermediate states, expand back to n_theta.
  defp rejuvenate_waste_free(
         thetas,
         cached_log_evs,
         ev_bufs,
         model,
         prior,
         obs_window,
         proposal,
         n_theta,
         n_x,
         len_chain,
         seed,
         t,
         parallel
       ) do
    chain_size = len_chain + 1
    n_thin = max(div(n_theta, chain_size), 1)

    # Thin: keep first n_thin particles
    thetas_thin = Enum.take(thetas, n_thin)
    log_evs_thin = Enum.take(cached_log_evs, n_thin)
    ev_bufs_thin = Enum.take(ev_bufs, n_thin)

    indexed = Enum.zip([thetas_thin, log_evs_thin, ev_bufs_thin]) |> Enum.with_index()

    run_chain = fn {{theta, log_ev, _ev_buf}, i} ->
      PMCMC.rejuvenate_waste_free(theta, log_ev, model, prior, obs_window, proposal,
        len_chain: len_chain,
        n_x: n_x,
        seed: seed + t * 1000 + i + n_theta
      )
    end

    chains =
      if parallel do
        indexed
        |> Task.async_stream(run_chain,
          max_concurrency: System.schedulers_online(),
          timeout: 60_000
        )
        |> Enum.map(fn {:ok, result} -> result end)
      else
        Enum.map(indexed, run_chain)
      end

    # Each chain returns {chain_states, total_accepted} where
    # chain_states = [{theta, log_ev, pf_state}, ...]  of length len_chain + 1
    all_states = Enum.flat_map(chains, fn {chain_states, _acc} -> chain_states end)
    total_accepted = Enum.sum(Enum.map(chains, fn {_states, acc} -> acc end))
    total_moves = n_thin * len_chain
    accept_rate = if total_moves > 0, do: total_accepted / total_moves, else: 0.0

    # Trim or pad to exactly n_theta
    all_states = Enum.take(all_states, n_theta)
    # If we got fewer than n_theta (rounding), pad by repeating last
    all_states =
      if length(all_states) < n_theta do
        last = List.last(all_states)
        all_states ++ List.duplicate(last, n_theta - length(all_states))
      else
        all_states
      end

    rejuv_thetas = Enum.map(all_states, fn {theta, _le, _pf} -> theta end)
    rejuv_pfs = Enum.map(all_states, fn {_theta, _le, pf} -> pf end)
    # Reset evidence buffers for all waste-free particles (theta may have changed)
    rejuv_ev_bufs = List.duplicate([], n_theta)

    {rejuv_thetas, rejuv_pfs, rejuv_ev_bufs, accept_rate}
  end

  # Run one BPF step: propagate x-particles, weight, resample
  defp bpf_step(pf_state, theta, model, y_t, t) do
    n_x = length(pf_state.particles)
    rng = :rand.seed_s(:exsss, {t * 10000 + :erlang.phash2(theta), t, 0})

    # Propagate
    {new_particles, rng} =
      Enum.map_reduce(pf_state.particles, rng, fn state, rng ->
        model.transition.(state, theta, t, rng)
      end)

    # Weight by observation likelihood
    obs_log_weights =
      Enum.map(new_particles, fn state ->
        model.observation_logp.(state, theta, y_t)
      end)

    # Incremental likelihood estimate: mean of weights
    max_olw = Enum.max(obs_log_weights)

    inc_log_lik =
      max_olw +
        :math.log(Enum.sum(Enum.map(obs_log_weights, fn lw -> :math.exp(lw - max_olw) end)) / n_x)

    # Update and normalize
    new_log_weights =
      Enum.zip(pf_state.log_weights, obs_log_weights)
      |> Enum.map(fn {lw, olw} -> lw + olw end)

    max_lw = Enum.max(new_log_weights)
    weights_unnorm = Enum.map(new_log_weights, fn lw -> :math.exp(lw - max_lw) end)
    sum_w = Enum.sum(weights_unnorm)
    norm_weights = Enum.map(weights_unnorm, fn w -> w / sum_w end)

    # ESS-based resampling
    ess = 1.0 / Enum.sum(Enum.map(norm_weights, fn w -> w * w end))

    {particles_out, log_weights_out} =
      if ess < 0.5 * n_x do
        {resampled, _rng} = stratified_resample(new_particles, norm_weights, n_x, rng)
        {resampled, List.duplicate(0.0, n_x)}
      else
        {new_particles, new_log_weights}
      end

    new_pf = %{particles: particles_out, log_weights: log_weights_out}
    {new_pf, inc_log_lik}
  end

  # Compute weighted mean of θ-particles
  defp compute_weighted_mean(thetas, weights) do
    keys = Map.keys(hd(thetas))

    Map.new(keys, fn key ->
      wmean =
        Enum.zip(thetas, weights)
        |> Enum.map(fn {theta, w} -> Map.get(theta, key, 0) * w end)
        |> Enum.sum()

      {key, wmean}
    end)
  end

  # Compute proposal standard deviations from weighted covariance
  defp compute_proposal(thetas, weights, scale) do
    keys = Map.keys(hd(thetas))
    means = compute_weighted_mean(thetas, weights)

    Map.new(keys, fn key ->
      mu = means[key]

      var =
        Enum.zip(thetas, weights)
        |> Enum.map(fn {theta, w} -> w * (Map.get(theta, key, 0) - mu) ** 2 end)
        |> Enum.sum()

      sd = :math.sqrt(max(var, 1.0e-10))
      {key, sd * scale}
    end)
  end

  # Stratified resampling returning indices
  defp stratified_resample_indices(weights, n, rng) do
    cdf = Enum.scan(weights, fn w, acc -> acc + w end)
    {u0, rng} = :rand.uniform_s(rng)

    indices =
      for i <- 0..(n - 1) do
        u = (i + u0) / n
        Enum.find_index(cdf, fn c -> c >= u end) || n - 1
      end

    {indices, rng}
  end

  # Stratified resampling returning particles
  # Uses tuple for O(1) indexing instead of O(N) Enum.at
  defp stratified_resample(particles, weights, n, rng) do
    ptuple = List.to_tuple(particles)
    {indices, rng} = stratified_resample_indices(weights, n, rng)
    resampled = Enum.map(indices, &elem(ptuple, &1))
    {resampled, rng}
  end

  # Bind θ into model functions
  defp bind_model(model, theta) do
    %{
      init: fn rng -> model.init.(theta, rng) end,
      transition: fn state, t, rng -> model.transition.(state, theta, t, rng) end,
      observation_logp: fn state, y_t -> model.observation_logp.(state, theta, y_t) end
    }
  end
end
