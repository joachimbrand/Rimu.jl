"""
Tools for the statistical analysis of Monte Carlo data.
"""
module StatsTools

using Statistics, MonteCarloMeasurements, Distributions, DataFrames
using Parameters
import MacroTools
import Measurements

import Statistics: cov
import Measurements: measurement
import MonteCarloMeasurements: Particles

export growth_witness, smoothen
export block_and_test
export ratio_of_means

include("growth_witness.jl")
include("variances.jl")
include("blocking.jl")
include("ratio_of_means.jl")
# TODO
# include("fidelity.jl")

end  # module StatsTools
