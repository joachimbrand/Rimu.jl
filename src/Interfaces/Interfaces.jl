"""
    module Interfaces

This module contains interfaces that can be used to extend and modify the algorithms and behaviours of `Rimu`.

# Interfaces
Follow the links for the definitions of the interfaces!
* [`AbstractHamiltonian`](@ref) for defining [`Hamiltonians`](@ref Main.Hamiltonians)
* [`AbstractDVec`](@ref) for defining data structures for `Rimu` as in [`DictVectors`](@ref Main.DictVectors)
* [`StochasticStyle`](@ref) for controlling the stochastic algorithms used by [`lomc!`](@ref Main.lomc!) as implemented in [`StochasticStyles`](@ref Main.StochasticStyles)

# Additional exports

## Interface functions for[`AbstractHamiltonian`](@ref)s:
* [`diagonal_element`](@ref)
* [`num_offdiagonals`](@ref)
* [`get_offdiagonal`](@ref)
* [`offdiagonals`](@ref).
* [`random_offdiagonal`](@ref)
* [`starting_address`](@ref)
* [`LOStructure`](@ref)

## working with  [`AbstractDVec`](@ref)s and [`StochasticStyle`](@ref)
* [`deposit!`](@ref)
* [`zero!`](@ref)
* [`default_style`](@ref)
* [`CompressionStrategy`](@ref)

## Functions Rimu.jl uses to do FCIQMC:

* [`fciqmc_col!`](@ref)
* [`fciqmc_step!`](@ref)
* [`sort_into_targets!`](@ref)
* [`working_memory`](@ref)
* [`step_stats`](@ref)
* [`update_dvec!`](@ref)
"""
module Interfaces

using LinearAlgebra
using StaticArrays

import OrderedCollections: freeze

export
    StochasticStyle, default_style, StyleUnknown, fciqmc_col!, step_stats, update_dvec!,
    CompressionStrategy, NoCompression, compress!, move_and_compress!
export
    AbstractDVec, deposit!, storage, localpart, freeze, working_memory,
    fciqmc_step!, sort_into_targets!
export
    AbstractHamiltonian, diagonal_element, num_offdiagonals, get_offdiagonal, offdiagonals,
    random_offdiagonal, starting_address,
    LOStructure, IsDiagonal, IsHermitian, AdjointKnown, AdjointUnknown, has_adjoint

include("stochasticstyles.jl")
include("dictvectors.jl")
include("hamiltonians.jl")

end
