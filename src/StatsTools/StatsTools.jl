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

include("growth_witness.jl")
include("variances.jl")
include("blocking.jl")
include("ratio_of_means.jl")
# TODO
# include("fidelity.jl")

end  # module StatsTools
