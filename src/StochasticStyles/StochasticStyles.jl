"""
This module provides concrete implementations of [`StochasticStyle`](@ref)s, which
specify the algorithm used by
[`ProjectorMonteCarloProblem`](@ref Main.ProjectorMonteCarloProblem) when
performing stochastic matrix-vector multiplication.

# Implemented stochastic styles:

* [`StochasticStyle`](@ref): abstract type for stochastic styles
* [`IsStochasticInteger`](@ref)
* [`IsDeterministic`](@ref)
* [`IsStochasticWithThreshold`](@ref)
* [`IsDynamicSemistochastic`](@ref)
* [`StyleUnknown`](@ref)

The offdiagonal spawning is defined in `spawning.jl` and is controlled by setting a
[`SpawningStrategy`](@ref).

The vector compression strategies are defined in `compression.jl` and are controlled by
setting a [`CompressionStrategy`](@ref).
"""
module StochasticStyles

using StaticArrays: StaticArrays
using StatsBase: StatsBase, sample

using ..Rimu: MultiScalar

using ..Interfaces
import ..Interfaces:
    deposit!, diagonal_element, offdiagonals, random_offdiagonal, default_style,
    apply_column!, step_stats, compress!, localpart
export
    StochasticStyle, IsStochasticInteger, IsDeterministic, IsStochasticWithThreshold,
    IsDynamicSemistochastic, StyleUnknown, Exact, WithReplacement, DynamicSemistochastic,
    WithoutReplacement, Bernoulli

include("spawning.jl")
include("compression.jl")
include("styles.jl")

end
