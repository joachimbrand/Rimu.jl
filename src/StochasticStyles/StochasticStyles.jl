module StochasticStyles

using StaticArrays
using StatsBase
using ..ConsistentRNG
using ..Rimu: MultiScalar, localpart

export deposit!, diagonal_element, offdiagonals, random_offdiagonal
export fciqmc_col!, step_stats, StochasticStyle, update_dvec!, default_style
export
    IsStochasticInteger, IsDeterministic, IsStochasticWithThreshold,
    IsDynamicSemistochastic, IsExplosive, StyleUnknown,
    IsDynamicSemistochasticProjectedSpawns, IsDynamicSemistochasticPlus

include("abstract.jl")
include("spawning.jl")
include("compression.jl")
include("styles.jl")

end
