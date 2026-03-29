defmodule SMC.ParticleFilter do
  @moduledoc """
  Bootstrap Particle Filter (BPF) for state-space models.

  Given parameters θ and a sequence of observations y_{1:T}, estimates
  the filtering distribution p(x_t | y_{1:t}, θ) via sequential importance
  sampling with resampling.

  This is Algorithm 1 from Temfack & Wyse (2025): for each time step,
  propagate particles through the state transition, weight by the
  observation likelihood, and resample when ESS drops below threshold.

  ## Usage

      # Define a state-space model
      model = %{
        init: fn rng -> {initial_state, rng} end,
        transition: fn state, t, rng -> {new_state, rng} end,
        observation_logp: fn state, y_t -> log_likelihood end
      }

      # Run the filter
      result = ParticleFilter.filter(model, observations, n_particles: 200)

      # result.log_evidence  — log p(y_{1:T} | θ)
      # result.particles     — final particle set
      # result.weights       — final normalized weights
      # result.ess_history   — ESS at each time step
  """

  @default_opts [
    n_particles: 200,
    resample_threshold: 0.5,
    seed: 42
  ]

  @doc """
  Run the bootstrap particle filter on a sequence of observations.

  ## Model specification

  The `model` map must contain:

  - `:init` — `fn(rng) -> {state, rng}` — sample initial state x_0
  - `:transition` — `fn(state, t, rng) -> {new_state, rng}` — state transition f(x_t | x_{t-1})
  - `:observation_logp` — `fn(state, y_t) -> float` — log p(y_t | x_t)

  States can be any Elixir term (maps, tuples, etc.).

  ## Returns

  A map with:
  - `:particles` — list of final particles
  - `:weights` — list of normalized weights
  - `:log_evidence` — log marginal likelihood estimate log p(y_{1:T} | θ)
  - `:ess_history` — ESS at each time step
  - `:filtering_means` — optional: if states are numeric, weighted mean at each step
  """
  def filter(model, observations, opts \\ []) do
    opts = Keyword.merge(@default_opts, opts)
    n = opts[:n_particles]
    threshold = opts[:resample_threshold]
    seed = opts[:seed]

    rng = :rand.seed_s(:exsss, {seed, seed + 1, seed + 2})

    # Initialize particles from prior
    {particles, rng} =
      Enum.map_reduce(1..n, rng, fn _, rng ->
        model.init.(rng)
      end)

    # Initial weights: uniform
    log_weights = List.duplicate(0.0, n)
    log_evidence = 0.0
    # Accumulate in reverse (prepend), reverse at the end
    ess_history_rev = []
    filtering_means_rev = []

    # Sequential filtering
    {particles, log_weights, log_evidence, ess_history_rev, filtering_means_rev, _rng} =
      observations
      |> Enum.with_index(1)
      |> Enum.reduce(
        {particles, log_weights, log_evidence, ess_history_rev, filtering_means_rev, rng},
        fn {y_t, t}, {particles, log_weights, log_evidence, ess_hist_rev, filt_means_rev, rng} ->
          # 1. Propagate particles through transition
          {new_particles, rng} =
            Enum.map_reduce(particles, rng, fn state, rng ->
              model.transition.(state, t, rng)
            end)

          # 2. Weight by observation likelihood
          obs_log_weights =
            Enum.map(new_particles, fn state ->
              model.observation_logp.(state, y_t)
            end)

          # Update log weights
          new_log_weights =
            Enum.zip(log_weights, obs_log_weights)
            |> Enum.map(fn {lw, olw} -> safe_add(lw, olw) end)

          # 3. Normalize weights
          max_lw = Enum.max(new_log_weights)
          weights_unnorm = Enum.map(new_log_weights, fn lw -> :math.exp(lw - max_lw) end)
          sum_w = Enum.sum(weights_unnorm)
          norm_weights = Enum.map(weights_unnorm, fn w -> w / sum_w end)

          # Incremental log evidence: log(mean(exp(obs_log_weights)))
          max_olw = Enum.max(obs_log_weights)

          inc_evidence =
            max_olw +
              :math.log(
                Enum.sum(Enum.map(obs_log_weights, fn lw -> :math.exp(lw - max_olw) end)) / n
              )

          log_evidence = log_evidence + inc_evidence

          # Compute ESS
          ess = 1.0 / Enum.sum(Enum.map(norm_weights, fn w -> w * w end))

          # Optional: compute filtering mean (for numeric states)
          filt_mean = compute_filtering_mean(new_particles, norm_weights)

          # 4. Resample if ESS below threshold
          {particles_out, log_weights_out, rng} =
            if ess < threshold * n do
              {resampled, rng} = stratified_resample(new_particles, norm_weights, n, rng)
              {resampled, List.duplicate(0.0, n), rng}
            else
              {new_particles, new_log_weights, rng}
            end

          {particles_out, log_weights_out, log_evidence, [ess | ess_hist_rev],
           [filt_mean | filt_means_rev], rng}
        end
      )

    ess_history = Enum.reverse(ess_history_rev)
    filtering_means = Enum.reverse(filtering_means_rev)

    # Final normalized weights
    max_lw = Enum.max(log_weights)
    final_weights_unnorm = Enum.map(log_weights, fn lw -> :math.exp(lw - max_lw) end)
    sum_w = Enum.sum(final_weights_unnorm)
    final_weights = Enum.map(final_weights_unnorm, fn w -> w / sum_w end)

    %{
      particles: particles,
      weights: final_weights,
      log_evidence: log_evidence,
      ess_history: ess_history,
      filtering_means: filtering_means,
      n_particles: n,
      n_observations: length(observations)
    }
  end

  @doc """
  Run the filter on a window of observations (for O-SMC² rejuvenation).

  Same as `filter/3` but starts from pre-existing particles instead of
  sampling from the prior. Used by SMC² to evaluate the windowed
  likelihood p(y_{t-tk+1:t} | θ).
  """
  def filter_window(model, observations, initial_particles, opts \\ []) do
    opts = Keyword.merge(@default_opts, opts)
    n = length(initial_particles)
    threshold = opts[:resample_threshold]
    seed = opts[:seed]

    rng = :rand.seed_s(:exsss, {seed, seed + 1, seed + 2})

    log_weights = List.duplicate(0.0, n)
    log_evidence = 0.0

    {particles, _log_weights, log_evidence, _rng} =
      observations
      |> Enum.with_index(1)
      |> Enum.reduce(
        {initial_particles, log_weights, log_evidence, rng},
        fn {y_t, t}, {particles, log_weights, log_evidence, rng} ->
          {new_particles, rng} =
            Enum.map_reduce(particles, rng, fn state, rng ->
              model.transition.(state, t, rng)
            end)

          obs_log_weights =
            Enum.map(new_particles, fn state ->
              model.observation_logp.(state, y_t)
            end)

          new_log_weights =
            Enum.zip(log_weights, obs_log_weights)
            |> Enum.map(fn {lw, olw} -> safe_add(lw, olw) end)

          max_lw = Enum.max(new_log_weights)
          weights_unnorm = Enum.map(new_log_weights, fn lw -> :math.exp(lw - max_lw) end)
          sum_w = Enum.sum(weights_unnorm)
          norm_weights = Enum.map(weights_unnorm, fn w -> w / sum_w end)

          max_olw = Enum.max(obs_log_weights)

          inc_evidence =
            max_olw +
              :math.log(
                Enum.sum(Enum.map(obs_log_weights, fn lw -> :math.exp(lw - max_olw) end)) / n
              )

          log_evidence = log_evidence + inc_evidence

          ess = 1.0 / Enum.sum(Enum.map(norm_weights, fn w -> w * w end))

          {particles_out, log_weights_out, rng} =
            if ess < threshold * n do
              {resampled, rng} = stratified_resample(new_particles, norm_weights, n, rng)
              {resampled, List.duplicate(0.0, n), rng}
            else
              {new_particles, new_log_weights, rng}
            end

          {particles_out, log_weights_out, log_evidence, rng}
        end
      )

    %{
      particles: particles,
      log_evidence: log_evidence
    }
  end

  # --- Private ---

  # Stratified resampling (lower variance than multinomial)
  # Uses tuple for O(1) indexing instead of O(N) Enum.at
  defp stratified_resample(particles, weights, n, rng) do
    ptuple = List.to_tuple(particles)

    # CDF
    cdf =
      weights
      |> Enum.scan(fn w, acc -> acc + w end)

    {u0, rng} = draw_uniform(rng)

    resampled =
      for i <- 0..(n - 1) do
        u = (i + u0) / n
        idx = Enum.find_index(cdf, fn c -> c >= u end) || n - 1
        elem(ptuple, idx)
      end

    {resampled, rng}
  end

  defp draw_uniform(rng) do
    {val, rng} = :rand.uniform_s(rng)
    {val, rng}
  end

  defp safe_add(a, b) when is_number(a) and is_number(b), do: a + b
  # treat non-numeric as -inf
  defp safe_add(_, _), do: -1.0e30

  # Compute weighted mean for numeric map states
  defp compute_filtering_mean(particles, weights) do
    case hd(particles) do
      state when is_map(state) ->
        keys = Map.keys(state)

        Map.new(keys, fn key ->
          vals = Enum.map(particles, fn p -> Map.get(p, key, 0) end)

          if is_number(hd(vals)) do
            wmean = Enum.zip(vals, weights) |> Enum.map(fn {v, w} -> v * w end) |> Enum.sum()
            {key, wmean}
          else
            {key, nil}
          end
        end)

      _ ->
        nil
    end
  end
end
