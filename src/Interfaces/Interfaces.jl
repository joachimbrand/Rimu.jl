"""
    module Interfaces

This module contains the bare-bones interfaces needed to implement new types of generalized
vectors and Hamiltonians for use with [`lomc!`](@ref).

For a generalized vector implement what would be needed for the `AbstractDict` interface
(`pairs`, `keys`, `values`, `setindex!, getindex, delete!, length, haskey, empty!, isempty`)
and, in addition:

* [`StochasticStyle`](@ref)
* [`storage`](@ref)
* [`deposit!`](@ref) (optional)
* [`freeze`](@ref) (optional)
* [`localpart`](@ref) (optional)

For a Hamiltonian, implement the following functions:

* [`diagonal_element`](@ref)
* [`num_offdiagonals`](@ref) and [`get_offdiagonal`](@ref), or [`offdiagonals`](@ref).
* [`starting_address`](@ref)
* [`random_offdiagonal`](@ref) (optional)
* [`LOStructure`](@ref) (optional, but recommended)
* [`random_offdiagonal`](@ref) (optional)

A `StochasticStyle` controls how the FCIQMC computation is performed. It follows the
following interface:

* [`fciqmc_col!`](@ref)
* [`step_stats`](@ref)
* [`update_dvec!`](@ref) or [`CompressionStrategy`](@ref) (optional)

"""
module Interfaces

using LinearAlgebra
using StaticArrays

using ..ConsistentRNG
import OrderedCollections: freeze

export
    StochasticStyle, default_style, StyleUnknown, fciqmc_col!, step_stats, update_dvec!,
    CompressionStrategy, NoCompression, compress!

export
    deposit!, storage, localpart, freeze
export
    diagonal_element, num_offdiagonals, get_offdiagonal, offdiagonals, random_offdiagonal,
    starting_address, LOStructure, IsHermitian, AdjointKnown, AdjointUnknown, has_adjoint

include("stochasticstyles.jl")
include("dictvectors.jl")
include("hamiltonians.jl")

end
