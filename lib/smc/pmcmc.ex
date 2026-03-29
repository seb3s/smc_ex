defmodule SMC.PMCMC do
  @moduledoc """
  Particle Marginal Metropolis-Hastings (PMMH) kernel.

  Proposes new parameter values and accepts/rejects via the MH ratio
  using particle filter likelihood estimates. Used by O-SMC² for
  rejuvenation when θ-particles degenerate.

  The key insight from Andrieu et al. (2010): replacing the true
  likelihood with a particle filter estimate in the MH ratio still
  targets the correct posterior — the noise in the estimate is absorbed
  by the accept/reject mechanism.
  """

  alias SMC.ParticleFilter

  @doc """
  Run up to M PMMH moves on a single θ-particle.

  Given the current θ, its particle filter state, and a window of
  recent observations, propose θ* from a Normal centered on the
  current weighted mean, evaluate the windowed likelihood via BPF,
  and accept/reject.

  Uses early termination: halts as soon as a move is accepted
  (the particle has been diversified), saving redundant PF evaluations.

  Returns `{new_theta, new_log_evidence, accepted_count, pf_state}` where
  `pf_state` is a `%{particles: ..., log_weights: ...}` map from the last
  accepted PF run (or a fresh run on the original theta if none accepted).
  This allows the caller to skip a redundant post-rejuvenation PF pass.
  """
  def rejuvenate(theta, current_log_evidence, model, prior, obs_window, proposal, opts \\ []) do
    n_moves = Keyword.get(opts, :n_moves, 5)
    n_x = Keyword.get(opts, :n_x, 200)
    seed = Keyword.get(opts, :seed, :rand.uniform(100_000))
    # Caller can pass pre-computed PF state to avoid a redundant PF run
    init_pf_state = Keyword.get(opts, :init_pf_state, nil)

    rng = :rand.seed_s(:exsss, {seed, seed + 1, seed + 2})

    # Use caller-provided PF state, or build one for the current theta
    init_pf_state =
      if init_pf_state do
        init_pf_state
      else
        pf_model_init = bind_model(model, theta)

        init_pf_result =
          ParticleFilter.filter(pf_model_init, obs_window, n_particles: n_x, seed: seed)

        %{particles: init_pf_result.particles, log_weights: List.duplicate(0.0, n_x)}
      end

    {theta, log_ev, accepted, _rng, pf_state} =
      Enum.reduce_while(1..n_moves, {theta, current_log_evidence, 0, rng, init_pf_state}, fn _,
                                                                                             {theta,
                                                                                              log_ev,
                                                                                              acc,
                                                                                              rng,
                                                                                              pf_state} ->
        # Propose θ* from Normal(θ, proposal_cov)
        {theta_star, rng} = propose(theta, proposal, rng)

        # Evaluate prior
        log_prior_star = prior.logpdf.(theta_star)
        log_prior_curr = prior.logpdf.(theta)

        # Skip if prior is -inf (out of support)
        if log_prior_star == :neg_infinity or log_prior_star < -1.0e20 do
          {:cont, {theta, log_ev, acc, rng, pf_state}}
        else
          # Run BPF on the observation window with proposed θ*
          pf_model = bind_model(model, theta_star)

          result_star =
            ParticleFilter.filter(pf_model, obs_window, n_particles: n_x, seed: seed + acc + 1)

          log_ev_star = result_star.log_evidence

          # MH acceptance ratio (Eq. B.7 from Temfack & Wyse)
          # α = min(1, p̂(window|θ*) * p(θ*) / (p̂(window|θ) * p(θ)) * q(θ|θ*)/q(θ*|θ))
          # Symmetric proposal → q ratio = 1
          log_alpha = log_ev_star + log_prior_star - (log_ev + log_prior_curr)
          log_alpha = min(log_alpha, 0.0)

          {u, rng} = :rand.uniform_s(rng)

          if :math.log(u) < log_alpha do
            # Accepted — capture PF state and halt early (particle diversified)
            accepted_pf_state = %{
              particles: result_star.particles,
              log_weights: List.duplicate(0.0, n_x)
            }

            {:halt, {theta_star, log_ev_star, acc + 1, rng, accepted_pf_state}}
          else
            {:cont, {theta, log_ev, acc, rng, pf_state}}
          end
        end
      end)

    {theta, log_ev, accepted, pf_state}
  end

  @doc """
  Waste-free MCMC rejuvenation (Dau & Chopin 2022).

  Instead of running n_moves and keeping only the final accepted state,
  runs `len_chain` MCMC steps and keeps ALL intermediate states (both
  accepted and rejected). Returns a list of `{theta, log_ev, pf_state}`
  tuples of length `len_chain + 1` (original + each move).

  Returns `{chain_states, accepted_count}`.
  """
  def rejuvenate_waste_free(
        theta,
        current_log_evidence,
        model,
        prior,
        obs_window,
        proposal,
        opts \\ []
      ) do
    len_chain = Keyword.get(opts, :len_chain, 10)
    n_x = Keyword.get(opts, :n_x, 200)
    seed = Keyword.get(opts, :seed, :rand.uniform(100_000))

    rng = :rand.seed_s(:exsss, {seed, seed + 1, seed + 2})

    # Initial PF state for the starting theta
    init_pf_state = build_pf_state(model, theta, obs_window, n_x, seed)

    # The original state is the first element in the chain
    initial_entry = {theta, current_log_evidence, init_pf_state}

    {chain_rev, _curr_theta, _curr_log_ev, _curr_pf, accepted, _rng} =
      Enum.reduce(
        1..len_chain,
        {[initial_entry], theta, current_log_evidence, init_pf_state, 0, rng},
        fn move, {chain_rev, curr_theta, curr_log_ev, curr_pf, acc_count, rng} ->
          # Propose theta*
          {theta_star, rng} = propose(curr_theta, proposal, rng)

          # Evaluate prior
          log_prior_star = prior.logpdf.(theta_star)
          log_prior_curr = prior.logpdf.(curr_theta)

          if log_prior_star == :neg_infinity or log_prior_star < -1.0e20 do
            # Reject: keep current state
            entry = {curr_theta, curr_log_ev, curr_pf}
            {[entry | chain_rev], curr_theta, curr_log_ev, curr_pf, acc_count, rng}
          else
            # Run BPF on the observation window with proposed theta*
            pf_model = bind_model(model, theta_star)

            result_star =
              ParticleFilter.filter(pf_model, obs_window, n_particles: n_x, seed: seed + move)

            log_ev_star = result_star.log_evidence

            # MH acceptance ratio
            log_alpha = log_ev_star + log_prior_star - (curr_log_ev + log_prior_curr)
            log_alpha = min(log_alpha, 0.0)

            {u, rng} = :rand.uniform_s(rng)

            if :math.log(u) < log_alpha do
              # Accepted
              pf_state = %{
                particles: result_star.particles,
                log_weights: List.duplicate(0.0, n_x)
              }

              entry = {theta_star, log_ev_star, pf_state}
              {[entry | chain_rev], theta_star, log_ev_star, pf_state, acc_count + 1, rng}
            else
              # Rejected: keep current state
              entry = {curr_theta, curr_log_ev, curr_pf}
              {[entry | chain_rev], curr_theta, curr_log_ev, curr_pf, acc_count, rng}
            end
          end
        end
      )

    chain_states = Enum.reverse(chain_rev)
    {chain_states, accepted}
  end

  # Build initial PF state for a given theta on the observation window
  defp build_pf_state(model, theta, obs_window, n_x, seed) do
    pf_model = bind_model(model, theta)
    result = ParticleFilter.filter(pf_model, obs_window, n_particles: n_x, seed: seed)
    %{particles: result.particles, log_weights: List.duplicate(0.0, n_x)}
  end

  # Propose θ* by adding Gaussian noise to each parameter
  defp propose(theta, proposal, rng) when is_map(theta) do
    {new_theta, rng} =
      Enum.reduce(Map.keys(theta), {%{}, rng}, fn key, {acc, rng} ->
        val = Map.get(theta, key)
        sd = Map.get(proposal, key, 0.01)
        {noise, rng} = :rand.normal_s(rng)
        {Map.put(acc, key, val + noise * sd), rng}
      end)

    {new_theta, rng}
  end

  # Bind θ into the model functions so ParticleFilter sees a θ-free model
  defp bind_model(model, theta) do
    %{
      init: fn rng -> model.init.(theta, rng) end,
      transition: fn state, t, rng -> model.transition.(state, theta, t, rng) end,
      observation_logp: fn state, y_t -> model.observation_logp.(state, theta, y_t) end
    }
  end
end
