"""
Tools for the statistical analysis of Monte Carlo data.

### Exports:
- [`blocking_analysis`](@ref)
- [`mean_and_se`](@ref)
- [`ratio_of_means`](@ref)
- [`growth_witness`](@ref)
- [`smoothen`](@ref)
"""
module StatsTools

using Statistics, MonteCarloMeasurements, Distributions, DataFrames
using StrLiterals, StrFormat # for Base.show() methods
import MacroTools
import Measurements

import Statistics: cov
import Measurements: measurement
import MonteCarloMeasurements: Particles
import Base: show

export growth_witness, smoothen
export blocking_analysis, mean_and_se
export ratio_of_means
export replica_fidelity, med_and_errs, ratio_with_errs, to_measurement
export growth_estimator, mixed_estimator, w_lin, w_exp

include("growth_witness.jl")
include("variances.jl")
include("blocking.jl")
include("ratio_of_means.jl")
include("convenience.jl")
include("reweighting.jl")

end  # module StatsTools
