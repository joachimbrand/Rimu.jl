"""
This module defines [`StochasticStyle`](@ref)s, which tell Rimu which algorithm to use when
doing stochastic matrix-vector multiplication.

Exports:

Interface functions for generalized vectors and Hamiltonians:

* [`deposit!`](@ref)
* [`diagonal_element`](@ref)
* [`offdiagonals`](@ref)
* [`random_offdiagonal`](@ref)
* [`default_style`](@ref)

Functions Rimu.jl uses to do FCIQMC:

* [`fciqmc_col!`](@ref)
* [`step_stats`](@ref)
* [`update_dvec!`](@ref)

Stochastic styles:

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
