"""
Tools for the statistical analysis of Monte Carlo data.
"""
module StatsTools

using Statistics, MonteCarloMeasurements, Distributions, DataFrames

export growth_witness, smoothen
export mtest, block_and_test

include("growth_witness.jl")
# TODO
# include("variances.jl")
# include("blocking.jl")
# include("fidelity.jl")

end  # module StatsTools
