defmodule SMC do
  @moduledoc """
  Sequential Monte Carlo methods for Elixir.

  Pure Elixir — zero dependencies. No Nx, no EXLA, no NIFs.

  - `SMC.ParticleFilter` — Bootstrap Particle Filter for state estimation
  - `SMC.PMCMC` — Particle Marginal Metropolis-Hastings kernel
  - `SMC.OnlineSMC2` — Online SMC² for joint parameter + state inference

  ## Quick Start

      # Particle filter: track latent states given known parameters
      result = SMC.filter(model, observations, n_particles: 200)

      # O-SMC²: infer parameters AND states online
      result = SMC.run(model, prior, observations,
        n_theta: 400, n_x: 200, window: 20)
  """

  @doc "Run a bootstrap particle filter. See `SMC.ParticleFilter.filter/3`."
  defdelegate filter(model, observations, opts \\ []),
    to: SMC.ParticleFilter

  @doc "Run O-SMC² for joint parameter + state inference. See `SMC.OnlineSMC2.run/4`."
  defdelegate run(model, prior, observations, opts \\ []),
    to: SMC.OnlineSMC2
end
