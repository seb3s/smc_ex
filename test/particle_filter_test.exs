defmodule SMC.ParticleFilterTest do
  use ExUnit.Case

  alias SMC.ParticleFilter

  describe "ParticleFilter.filter/3" do
    test "tracks a random walk with Gaussian observations" do
      # True model: x_t = x_{t-1} + N(0, 1), y_t = x_t + N(0, 0.5)
      :rand.seed(:exsss, {42, 43, 44})
      true_states = Enum.scan(1..50, 0.0, fn _, prev -> prev + :rand.normal() end)
      observations = Enum.map(true_states, fn x -> x + :rand.normal() * 0.5 end)

      model = %{
        init: fn rng ->
          {val, rng} = :rand.normal_s(rng)
          {%{x: val}, rng}
        end,
        transition: fn state, _t, rng ->
          {noise, rng} = :rand.normal_s(rng)
          {%{x: state.x + noise}, rng}
        end,
        observation_logp: fn state, y ->
          z = (y - state.x) / 0.5
          -0.5 * z * z - :math.log(0.5)
        end
      }

      result = ParticleFilter.filter(model, observations, n_particles: 500, seed: 42)

      assert result.n_observations == 50
      assert result.n_particles == 500
      assert length(result.filtering_means) == 50
      assert length(result.ess_history) == 50

      # Log evidence should be finite
      assert is_number(result.log_evidence)
      assert result.log_evidence > -1000

      # Filtering means should track the true states
      errors =
        Enum.zip(result.filtering_means, true_states)
        |> Enum.map(fn {fm, true_x} -> abs(fm[:x] - true_x) end)

      mean_error = Enum.sum(errors) / length(errors)

      assert mean_error < 2.0,
             "Mean tracking error #{Float.round(mean_error, 3)} should be < 2.0"
    end

    test "estimates log evidence for model comparison" do
      # Generate data from a simple model: y ~ N(3, 1)
      observations = Enum.map(1..30, fn _ -> 3.0 + :rand.normal() end)

      # Good model: prior centered near truth
      good_model = %{
        init: fn rng -> {%{mu: 3.0}, rng} end,
        transition: fn state, _t, rng ->
          {noise, rng} = :rand.normal_s(rng)
          {%{mu: state.mu + noise * 0.01}, rng}
        end,
        observation_logp: fn state, y ->
          z = y - state.mu
          -0.5 * z * z
        end
      }

      # Bad model: prior far from truth
      bad_model = %{
        init: fn rng -> {%{mu: 100.0}, rng} end,
        transition: fn state, _t, rng ->
          {noise, rng} = :rand.normal_s(rng)
          {%{mu: state.mu + noise * 0.01}, rng}
        end,
        observation_logp: fn state, y ->
          z = y - state.mu
          -0.5 * z * z
        end
      }

      good_result = ParticleFilter.filter(good_model, observations, n_particles: 200, seed: 1)
      bad_result = ParticleFilter.filter(bad_model, observations, n_particles: 200, seed: 1)

      assert good_result.log_evidence > bad_result.log_evidence,
             "Good model (#{good_result.log_evidence}) should have higher evidence than bad (#{bad_result.log_evidence})"
    end

    test "stochastic SEIR model" do
      # Simple SEIR: N=1000, β=0.3, σ=0.2, γ=0.1
      n_pop = 1000
      beta = 0.3
      sigma = 0.2
      gamma = 0.1

      # Generate true epidemic
      :rand.seed(:exsss, {10, 11, 12})

      {_, _, observations} =
        Enum.reduce(1..60, {%{s: 999, e: 1, i: 0, r: 0}, [], []}, fn _, {state, _states, obs} ->
          p_se = 1 - :math.exp(-beta * state.i / n_pop)
          p_ei = 1 - :math.exp(-sigma)
          p_ir = 1 - :math.exp(-gamma)

          y_se = binomial(state.s, p_se)
          y_ei = binomial(state.e, p_ei)
          y_ir = binomial(state.i, p_ir)

          new_state = %{
            s: state.s - y_se,
            e: state.e + y_se - y_ei,
            i: state.i + y_ei - y_ir,
            r: state.r + y_ir
          }

          # Observe new infections (Poisson noise)
          obs_val = max(0, y_ei + round(:rand.normal() * :math.sqrt(max(y_ei, 1))))
          {new_state, [new_state | _states], obs ++ [obs_val]}
        end)

      # Particle filter for SEIR
      model = %{
        init: fn rng ->
          {%{s: 999, e: 1, i: 0, r: 0}, rng}
        end,
        transition: fn state, _t, rng ->
          p_se = 1 - :math.exp(-beta * state.i / n_pop)
          p_ei = 1 - :math.exp(-sigma)
          p_ir = 1 - :math.exp(-gamma)

          {u1, rng} = :rand.uniform_s(rng)
          {u2, rng} = :rand.uniform_s(rng)
          {u3, rng} = :rand.uniform_s(rng)

          y_se = binomial_approx(state.s, p_se, u1)
          y_ei = binomial_approx(state.e, p_ei, u2)
          y_ir = binomial_approx(state.i, p_ir, u3)

          new = %{
            s: max(state.s - y_se, 0),
            e: max(state.e + y_se - y_ei, 0),
            i: max(state.i + y_ei - y_ir, 0),
            r: state.r + y_ir
          }

          {new, rng}
        end,
        observation_logp: fn state, y_obs ->
          # Poisson log-likelihood: y ~ Poisson(max(new_infections, 0.1))
          lambda = max(state.i * sigma, 0.1)
          y_obs * :math.log(lambda) - lambda - log_factorial(y_obs)
        end
      }

      result = ParticleFilter.filter(model, observations, n_particles: 500, seed: 42)

      assert result.n_observations == 60
      assert is_number(result.log_evidence)

      # ESS should not collapse to 1 (particles shouldn't fully degenerate)
      min_ess = Enum.min(result.ess_history)
      assert min_ess > 1.0, "Min ESS #{min_ess} should be > 1"
    end

    test "filter_window estimates windowed likelihood" do
      observations = Enum.map(1..30, fn _ -> :rand.normal() end)

      model = %{
        init: fn rng -> {%{x: 0.0}, rng} end,
        transition: fn state, _t, rng ->
          {noise, rng} = :rand.normal_s(rng)
          {%{x: state.x + noise * 0.1}, rng}
        end,
        observation_logp: fn state, y ->
          z = y - state.x
          -0.5 * z * z
        end
      }

      # Full filter
      full = ParticleFilter.filter(model, observations, n_particles: 200, seed: 1)

      # Window filter on last 10 observations, starting from full filter's particles
      window_obs = Enum.take(observations, -10)
      window = ParticleFilter.filter_window(model, window_obs, full.particles, seed: 2)

      assert is_number(window.log_evidence)
      assert length(window.particles) == 200
    end
  end

  # --- Helpers ---

  defp binomial(n, p) when n <= 0 or p <= 0, do: 0

  defp binomial(n, p) do
    Enum.count(1..n, fn _ -> :rand.uniform() < p end)
  end

  defp binomial_approx(n, p, u) when n <= 0 or p <= 0, do: 0

  defp binomial_approx(n, p, u) do
    # Quick approximation: use the mean ± noise
    mean = n * p
    round(max(0, min(n, mean + (u - 0.5) * :math.sqrt(max(mean * (1 - p), 0.01)) * 2)))
  end

  defp log_factorial(0), do: 0.0

  defp log_factorial(n) when n > 0 do
    Enum.reduce(1..n, 0.0, fn k, acc -> acc + :math.log(k) end)
  end

  defp log_factorial(_), do: 0.0
end
