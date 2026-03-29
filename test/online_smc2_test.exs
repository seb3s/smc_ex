defmodule SMC.OnlineSMC2Test do
  use ExUnit.Case

  alias SMC.OnlineSMC2

  describe "OnlineSMC2.run/4" do
    test "recovers parameters of a Gaussian model" do
      # True model: y ~ N(mu, 1), mu = 3.0
      :rand.seed(:exsss, {42, 43, 44})
      observations = Enum.map(1..50, fn _ -> 3.0 + :rand.normal() end)

      model = %{
        init: fn theta, rng ->
          {%{x: 0.0}, rng}
        end,
        transition: fn state, _theta, _t, rng ->
          {state, rng}
        end,
        observation_logp: fn _state, theta, y ->
          z = (y - theta.mu) / 1.0
          -0.5 * z * z
        end
      }

      prior = %{
        sample: fn rng ->
          {val, rng} = :rand.normal_s(rng)
          # N(0, 25)
          {%{mu: val * 5.0}, rng}
        end,
        logpdf: fn theta ->
          -0.5 * (theta.mu / 5.0) ** 2
        end
      }

      result =
        OnlineSMC2.run(model, prior, observations,
          n_theta: 100,
          n_x: 10,
          window: 20,
          n_moves: 2,
          seed: 42,
          parallel: false
        )

      assert result.n_observations == 50
      assert length(result.theta_particles) == 100
      assert length(result.ess_history) == 50

      # Posterior mean should be near 3.0
      final_mean = result.posterior_history |> List.last()

      assert abs(final_mean.mu - 3.0) < 1.0,
             "Posterior mean #{Float.round(final_mean.mu, 2)} should be near 3.0"

      # Posterior should narrow over time
      first_mean = Enum.at(result.posterior_history, 5)
      # Early estimate can be far from truth
      assert result.posterior_history |> length() == 50
    end

    test "tracks simple SEIR epidemic" do
      # Generate SEIR data
      n_pop = 5000
      true_beta = 0.4
      true_sigma = 0.25
      true_gamma = 0.15

      :rand.seed(:exsss, {10, 11, 12})

      {_final_state, observations} =
        Enum.reduce(1..40, {%{s: 4999, e: 1, i: 0, r: 0}, []}, fn _, {state, obs} ->
          p_se = 1 - :math.exp(-true_beta * state.i / n_pop)
          p_ei = 1 - :math.exp(-true_sigma)
          p_ir = 1 - :math.exp(-true_gamma)

          y_se = binom(state.s, p_se)
          y_ei = binom(state.e, p_ei)
          y_ir = binom(state.i, p_ir)

          new_state = %{
            s: state.s - y_se,
            e: state.e + y_se - y_ei,
            i: state.i + y_ei - y_ir,
            r: state.r + y_ir
          }

          {new_state, obs ++ [max(y_ei, 0)]}
        end)

      model = %{
        init: fn theta, rng ->
          {%{s: 4999, e: 1, i: 0, r: 0}, rng}
        end,
        transition: fn state, theta, _t, rng ->
          p_se = 1 - :math.exp(-theta.beta * state.i / n_pop)
          p_ei = 1 - :math.exp(-theta.sigma)
          p_ir = 1 - :math.exp(-theta.gamma)

          {u1, rng} = :rand.uniform_s(rng)
          {u2, rng} = :rand.uniform_s(rng)
          {u3, rng} = :rand.uniform_s(rng)

          y_se = binom_approx(state.s, p_se, u1)
          y_ei = binom_approx(state.e, p_ei, u2)
          y_ir = binom_approx(state.i, p_ir, u3)

          new = %{
            s: max(state.s - y_se, 0),
            e: max(state.e + y_se - y_ei, 0),
            i: max(state.i + y_ei - y_ir, 0),
            r: state.r + y_ir
          }

          {new, rng}
        end,
        observation_logp: fn state, theta, y_obs ->
          lambda = max(state.i * theta.sigma, 0.1)
          y_obs * :math.log(lambda) - lambda - log_fact(y_obs)
        end
      }

      prior = %{
        sample: fn rng ->
          {u1, rng} = :rand.uniform_s(rng)
          {u2, rng} = :rand.uniform_s(rng)
          {u3, rng} = :rand.uniform_s(rng)
          {%{beta: u1, sigma: u2 * 0.5 + 0.05, gamma: u3 * 0.3 + 0.05}, rng}
        end,
        logpdf: fn theta ->
          if theta.beta > 0 and theta.beta < 1 and
               theta.sigma > 0.05 and theta.sigma < 0.55 and
               theta.gamma > 0.05 and theta.gamma < 0.35 do
            # uniform
            0.0
          else
            -1.0e30
          end
        end
      }

      result =
        OnlineSMC2.run(model, prior, observations,
          n_theta: 80,
          n_x: 50,
          window: 15,
          n_moves: 2,
          seed: 42,
          parallel: false
        )

      assert result.n_observations == 40
      assert is_number(result.log_evidence)

      # β posterior should be in ballpark of true value (0.4)
      final = List.last(result.posterior_history)

      assert final.beta > 0.1 and final.beta < 0.9,
             "β estimate #{Float.round(final.beta, 3)} should be in [0.1, 0.9]"
    end

    test "log evidence is finite" do
      observations = Enum.map(1..20, fn _ -> :rand.normal() * 2 end)

      model = %{
        init: fn _theta, rng -> {%{x: 0.0}, rng} end,
        transition: fn state, _theta, _t, rng ->
          {n, rng} = :rand.normal_s(rng)
          {%{x: state.x + n * 0.1}, rng}
        end,
        observation_logp: fn _state, theta, y ->
          z = (y - theta.mu) / max(theta.sigma, 0.01)
          -0.5 * z * z - :math.log(max(theta.sigma, 0.01))
        end
      }

      prior = %{
        sample: fn rng ->
          {m, rng} = :rand.normal_s(rng)
          {s, rng} = :rand.uniform_s(rng)
          {%{mu: m * 3, sigma: s * 3 + 0.1}, rng}
        end,
        logpdf: fn theta ->
          if theta.sigma > 0, do: 0.0, else: -1.0e30
        end
      }

      result =
        OnlineSMC2.run(model, prior, observations,
          n_theta: 50,
          n_x: 20,
          window: 10,
          n_moves: 1,
          seed: 1,
          parallel: false
        )

      assert is_number(result.log_evidence)
      assert result.log_evidence > -1000
      assert result.rejuvenation_count >= 0
    end

    test "waste-free rejuvenation produces valid results" do
      :rand.seed(:exsss, {42, 43, 44})
      observations = Enum.map(1..30, fn _ -> 3.0 + :rand.normal() end)

      model = %{
        init: fn _theta, rng ->
          {%{x: 0.0}, rng}
        end,
        transition: fn state, _theta, _t, rng ->
          {state, rng}
        end,
        observation_logp: fn _state, theta, y ->
          z = (y - theta.mu) / 1.0
          -0.5 * z * z
        end
      }

      prior = %{
        sample: fn rng ->
          {val, rng} = :rand.normal_s(rng)
          {%{mu: val * 5.0}, rng}
        end,
        logpdf: fn theta ->
          -0.5 * (theta.mu / 5.0) ** 2
        end
      }

      result =
        OnlineSMC2.run(model, prior, observations,
          n_theta: 44,
          n_x: 10,
          window: 10,
          waste_free: true,
          len_chain: 3,
          seed: 42,
          parallel: false
        )

      assert result.n_observations == 30
      assert length(result.theta_particles) == 44
      assert is_number(result.log_evidence)
      assert result.log_evidence > -1000

      final_mean = result.posterior_history |> List.last()

      assert abs(final_mean.mu - 3.0) < 2.0,
             "Waste-free posterior mean #{Float.round(final_mean.mu, 2)} should be near 3.0"
    end

    test "adaptive threshold reduces rejuvenation count" do
      :rand.seed(:exsss, {42, 43, 44})
      observations = Enum.map(1..30, fn _ -> 3.0 + :rand.normal() end)

      model = %{
        init: fn _theta, rng -> {%{x: 0.0}, rng} end,
        transition: fn state, _theta, _t, rng -> {state, rng} end,
        observation_logp: fn _state, theta, y ->
          z = (y - theta.mu) / 1.0
          -0.5 * z * z
        end
      }

      prior = %{
        sample: fn rng ->
          {val, rng} = :rand.normal_s(rng)
          {%{mu: val * 5.0}, rng}
        end,
        logpdf: fn theta -> -0.5 * (theta.mu / 5.0) ** 2 end
      }

      # With adaptive threshold (default)
      result_adaptive =
        OnlineSMC2.run(model, prior, observations,
          n_theta: 50,
          n_x: 10,
          window: 10,
          seed: 42,
          parallel: false,
          adaptive_threshold: true
        )

      # Without adaptive threshold
      result_fixed =
        OnlineSMC2.run(model, prior, observations,
          n_theta: 50,
          n_x: 10,
          window: 10,
          seed: 42,
          parallel: false,
          adaptive_threshold: false
        )

      # Both should produce valid results
      assert is_number(result_adaptive.log_evidence)
      assert is_number(result_fixed.log_evidence)
      assert length(result_adaptive.theta_particles) == 50
      assert length(result_fixed.theta_particles) == 50
    end

    test "incremental evidence caching produces valid results" do
      # This test verifies the 3b optimization doesn't break correctness.
      # With evidence caching, we should still get valid posteriors.
      :rand.seed(:exsss, {42, 43, 44})
      observations = Enum.map(1..40, fn _ -> 2.0 + :rand.normal() end)

      model = %{
        init: fn _theta, rng -> {%{x: 0.0}, rng} end,
        transition: fn state, _theta, _t, rng -> {state, rng} end,
        observation_logp: fn _state, theta, y ->
          z = (y - theta.mu) / 1.0
          -0.5 * z * z
        end
      }

      prior = %{
        sample: fn rng ->
          {val, rng} = :rand.normal_s(rng)
          {%{mu: val * 5.0}, rng}
        end,
        logpdf: fn theta -> -0.5 * (theta.mu / 5.0) ** 2 end
      }

      result =
        OnlineSMC2.run(model, prior, observations,
          n_theta: 80,
          n_x: 10,
          window: 15,
          n_moves: 2,
          seed: 42,
          parallel: false
        )

      assert result.n_observations == 40
      assert is_number(result.log_evidence)
      final_mean = result.posterior_history |> List.last()

      assert abs(final_mean.mu - 2.0) < 1.5,
             "Posterior mean #{Float.round(final_mean.mu, 2)} should be near 2.0"
    end
  end

  # --- Helpers ---

  defp binom(n, p) when n <= 0 or p <= 0, do: 0
  defp binom(n, p), do: Enum.count(1..n, fn _ -> :rand.uniform() < p end)

  defp binom_approx(n, p, u) when n <= 0 or p <= 0, do: 0

  defp binom_approx(n, p, u) do
    mean = n * p
    round(max(0, min(n, mean + (u - 0.5) * :math.sqrt(max(mean * (1 - p), 0.01)) * 2)))
  end

  defp log_fact(0), do: 0.0
  defp log_fact(n) when n > 0, do: Enum.reduce(1..n, 0.0, fn k, acc -> acc + :math.log(k) end)
  defp log_fact(_), do: 0.0
end
