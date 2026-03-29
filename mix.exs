defmodule SMC.MixProject do
  use Mix.Project

  def project do
    [
      app: :smc_ex,
      version: "0.1.0",
      elixir: "~> 1.15",
      start_permanent: Mix.env() == :prod,
      deps: deps(),
      description:
        "Sequential Monte Carlo methods for Elixir: particle filters, PMCMC, and O-SMC²",
      package: package()
    ]
  end

  def application do
    [extra_applications: [:logger]]
  end

  defp deps, do: []

  defp package do
    [
      licenses: ["Apache-2.0"],
      links: %{"GitHub" => "https://github.com/borodark/smc_ex"}
    ]
  end
end
