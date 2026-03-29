# Head-to-head: smc_ex (Elixir) vs particles (Chopin's Python SMC²)
# Same SEIR model, same data, same particle counts.
#
# Usage:
#   cd smc_ex && elixir --erl "+sbt tnnps" -S mix run benchmark/head_to_head.exs
#
# Compare with:
#   ~/projects/learn_erl/python-env/bin/python benchmark/head_to_head.py

defmodule H2H do
  def generate_seir(n_pop, true_beta, true_sigma, true_gamma, t_max) do
    :rand.seed(:exsss, {42, 43, 44})

    {_state, observations} =
      Enum.reduce(1..t_max, {%{s: n_pop - 5, e: 5, i: 0, r: 0}, []}, fn _, {st, obs_acc} ->
        p_se = 1 - :math.exp(-true_beta * st.i / n_pop)
        p_ei = 1 - :math.exp(-true_sigma)
        p_ir = 1 - :math.exp(-true_gamma)

        y_se = binom(max(st.s, 0), p_se)
        y_ei = binom(max(st.e, 0), p_ei)
        y_ir = binom(max(st.i, 0), p_ir)

        new = %{
          s: max(st.s - y_se, 0),
          e: max(st.e + y_se - y_ei, 0),
          i: max(st.i + y_ei - y_ir, 0),
          r: st.r + y_ir
        }

        obs = max(0, y_ei + round(:rand.normal() * max(:math.sqrt(y_ei + 1), 1)))
        {new, [obs | obs_acc]}
      end)

    Enum.reverse(observations)
  end

  defp binom(n, p) when n <= 0 or p <= 0.0, do: 0
  defp binom(n, p) do
    Enum.count(1..n, fn _ -> :rand.uniform() < p end)
  end

  def seir_model(n_pop) do
    %{
      init: fn _theta, rng ->
        {%{s: n_pop - 5, e: 5, i: 0, r: 0}, rng}
      end,
      transition: fn state, theta, _t, rng ->
        p_se = 1 - :math.exp(-theta.beta * state.i / n_pop)
        p_ei = 1 - :math.exp(-theta.sigma)
        p_ir = 1 - :math.exp(-theta.gamma)

        {u1, rng} = :rand.uniform_s(rng)
        {u2, rng} = :rand.uniform_s(rng)
        {u3, rng} = :rand.uniform_s(rng)

        binom_approx = fn nn, pp, u ->
          mean = nn * pp
          round(max(0, min(nn, mean + (u - 0.5) * :math.sqrt(max(mean * (1 - pp), 0.01)) * 2)))
        end

        y_se = binom_approx.(state.s, p_se, u1)
        y_ei = binom_approx.(state.e, p_ei, u2)
        y_ir = binom_approx.(state.i, p_ir, u3)

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
  end

  defp log_fact(0), do: 0.0
  defp log_fact(n) when n > 0, do: Enum.reduce(1..n, 0.0, fn k, acc -> acc + :math.log(k) end)
  defp log_fact(_), do: 0.0

  def seir_prior do
    %{
      sample: fn rng ->
        {u1, rng} = :rand.uniform_s(rng)
        {u2, rng} = :rand.uniform_s(rng)
        {u3, rng} = :rand.uniform_s(rng)
        theta = %{beta: u1 * 0.8 + 0.05, sigma: u2 * 0.4 + 0.05, gamma: u3 * 0.25 + 0.05}
        {theta, rng}
      end,
      logpdf: fn theta ->
        if theta.beta > 0.05 and theta.beta < 0.85 and
           theta.sigma > 0.05 and theta.sigma < 0.45 and
           theta.gamma > 0.05 and theta.gamma < 0.30 do
          0.0
        else
          -1.0e30
        end
      end
    }
  end
end

IO.puts(String.duplicate("=", 70))
IO.puts("smc_ex (Elixir) — Head-to-Head")
IO.puts("#{System.schedulers_online()} schedulers")
IO.puts(String.duplicate("=", 70))
IO.puts("")

tests = [
  {"smoke (T=40, Nθ=50, Nx=50)", 10_000, 0.4, 0.25, 0.15, 40,
   [n_theta: 50, n_x: 50, window: 20, n_moves: 3, seed: 42, parallel: true]},
  {"medium (T=100, Nθ=100, Nx=100)", 10_000, 0.4, 0.25, 0.15, 100,
   [n_theta: 100, n_x: 100, window: 20, n_moves: 3, seed: 42, parallel: true]},
  {"full-scale (T=200, Nθ=200, Nx=100)", 10_000, 0.4, 0.25, 0.15, 200,
   [n_theta: 200, n_x: 100, window: 20, n_moves: 3, seed: 42, parallel: true]},
  {"large (T=200, Nθ=400, Nx=200)", 50_000, 0.3, 0.2, 0.1, 200,
   [n_theta: 400, n_x: 200, window: 30, n_moves: 3, seed: 42, parallel: true]},
]

header = String.pad_trailing("Test", 45) <>
  String.pad_leading("Time", 10) <>
  String.pad_leading("β", 8) <>
  String.pad_leading("σ", 8) <>
  String.pad_leading("γ", 8)
IO.puts(header)
IO.puts(String.duplicate("-", 85))

Enum.each(tests, fn {name, n_pop, beta, sigma, gamma, t_max, opts} ->
  obs = H2H.generate_seir(n_pop, beta, sigma, gamma, t_max)
  model = H2H.seir_model(n_pop)
  prior = H2H.seir_prior()

  t0 = System.monotonic_time(:millisecond)
  result = SMC.run(model, prior, obs, opts)
  elapsed = System.monotonic_time(:millisecond) - t0

  final = List.last(result.posterior_history)

  line = "  " <> String.pad_trailing(name, 43) <>
    String.pad_leading("#{elapsed}ms", 8) <>
    String.pad_leading("#{Float.round(final.beta, 3)}", 8) <>
    String.pad_leading("#{Float.round(final.sigma, 3)}", 8) <>
    String.pad_leading("#{Float.round(final.gamma, 3)}", 8)
  IO.puts(line)
end)

IO.puts("")
IO.puts(String.duplicate("=", 70))
