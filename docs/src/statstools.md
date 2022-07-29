# Module `Rimu/StatsTools`

The  module `Rimu/StatsTools` contains helper function for analysis and post
processing of Monte Carlo data.

## Blocking analysis

After equilibration, FCIQMC produces information about observables through
correlated time series. In order to estimate the statistical errors the
time series need to be decorrelated. The main workhorse for achieving this
is the [`blocking_analysis`](@ref), which is based on the paper by Flyvberg
and Peterson [JCP (1989)](http://aip.scitation.org/doi/10.1063/1.457480), and
automated with the M test of Jonsson
[PRE (2018)](https://link.aps.org/doi/10.1103/PhysRevE.98.043304).

Analysing the stochastic errors of observables obtained from the ratio of
sample means is done with [`ratio_of_means`](@ref), where error propagation
of correlated uncertainties is done with the help of the package
[`MonteCarloMeasurements`](https://github.com/baggepinnen/MonteCarloMeasurements.jl).

Many convenience functions are implemented for directly analysing data
obtained from [`lomc!`](@ref) as a `DataFrame`. See, e.g.,
[`shift_estimator`](@ref) and [`projected_energy`](@ref). Asymptotically
unbiased estimators are implemented as [`mixed_estimator`](@ref),
[`growth_estimator`](@ref) and [`rayleigh_quotient_estimator`](@ref).

## Exported
```@autodocs
Modules = [StatsTools]
Pages = ["StatsTools.jl", "blocking.jl", "ratio_of_means.jl", "convenience.jl",
  "variances.jl", "growth_witness.jl", "reweighting.jl", "fidelity.jl"
]
Private = false
```

## Additional docstrings
```@autodocs
Modules = [StatsTools]
Pages = ["StatsTools.jl", "blocking.jl", "ratio_of_means.jl", "convenience.jl",
  "variances.jl", "growth_witness.jl", "reweighting.jl"
]
Public = false
```

## Index
```@index
Pages   = ["statstools.md"]
```
