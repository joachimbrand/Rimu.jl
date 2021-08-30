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

"""
module Interfaces

using ..ConsistentRNG
using LinearAlgebra
import OrderedCollections: freeze

export StochasticStyle, default_style, StyleUnknown
export deposit!, storage, localpart, freeze
export diagonal_element, num_offdiagonals, get_offdiagonal, offdiagonals, random_offdiagonal, starting_address
export LOStructure, IsHermitian, AdjointKnown, AdjointUnknown, has_adjoint

include("stochasticstyles.jl")
include("dictvectors.jl")
include("hamiltonians.jl")

end
