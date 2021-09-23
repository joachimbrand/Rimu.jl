"""
This module provides concrete implementations of [`StochasticStyle`](@ref)s, which
specify the algorithm used by [`lomc!`](@ref) when
performing stochastic matrix-vector multiplication.

# Stochastic styles:

* [`StochasticStyle`](@ref)
* [`IsStochasticInteger`](@ref)
* [`IsDeterministic`](@ref)
* [`IsStochasticWithThreshold`](@ref)
* [`IsDynamicSemistochastic`](@ref)
* [`IsExplosive`](@ref)
* [`StyleUnknown`](@ref)

The offdiagonal spawning is defined in `spawning.jl` and is controlled by setting a
[`SpawningStrategy`](@ref).

The vector compression strategies are defined in `compression.jl` and are controlled by
setting a [`CompressionStrategy`](@ref).

"""
module StochasticStyles

using StaticArrays
using StatsBase
using ..ConsistentRNG
using ..Rimu: MultiScalar, localpart

using ..Interfaces
import ..Interfaces:
    deposit!, diagonal_element, offdiagonals, random_offdiagonal, default_style,
    fciqmc_col!, step_stats, update_dvec!, compress!
export
    StochasticStyle, IsStochasticInteger, IsDeterministic, IsStochasticWithThreshold,
    IsDynamicSemistochastic, StyleUnknown

include("spawning.jl")
include("compression.jl")
include("styles.jl")

end
