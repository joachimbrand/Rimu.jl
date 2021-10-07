"""
    module Hamiltonians

This module defines Hamiltonian types and functions for working with
Hamiltonians.

## Exported concrete Hamiltonian types include
- [`MatrixHamiltonian`](@ref)
- [`HubbardReal1D`](@ref)
- [`ExtendedHubbardReal1D`](@ref)
- [`HubbardMom1D`](@ref)
- [`BoseHubbardMom1D2C`](@ref)
- [`BoseHubbardReal1D2C`](@ref)

## Wrappers
- [`GutzwillerSampling`](@ref)
- [`GuidingVectorSampling`](@ref)

## Other
- [`G2Correlator`](@ref)

## Interface for working with Hamiltonians
- [`AbstractHamiltonian`](@ref): defined in the module [`Interfaces`](@ref)
"""
module Hamiltonians

using Parameters, StaticArrays, LinearAlgebra, SparseArrays
using Setfield

using ..StochasticStyles
using ..DictVectors
using ..BitStringAddresses
using ..ConsistentRNG

using ..Interfaces
import ..Interfaces: diagonal_element, num_offdiagonals, get_offdiagonal, starting_address,
    offdiagonals, random_offdiagonal, LOStructure

export AbstractHamiltonian, TwoComponentBosonicHamiltonian
export dimension, rayleigh_quotient, momentum

export MatrixHamiltonian
export HubbardReal1D, HubbardMom1D, ExtendedHubbardReal1D, HubbardRealSpace
export HubbardReal1DEP, shift_lattice, shift_lattice_inv
export BoseHubbardMom1D2C, BoseHubbardReal1D2C
export GutzwillerSampling, GuidingVectorSampling
export Transcorrelated1D
export hubbard_dispersion, continuum_dispersion

export G2Correlator

export LatticeGeometry, PeriodicBoundaries, HardwallBoundaries, LadderBoundaries
export num_neighbours, neighbour_site, difference, add_offset

include("abstract.jl")
include("offdiagonals.jl")
include("operations.jl")
include("geometry.jl")
include("excitations.jl")

include("MatrixHamiltonian.jl")

include("HubbardReal1D.jl")
include("HubbardReal1DEP.jl")
include("HubbardMom1D.jl")
include("HubbardRealSpace.jl")
include("ExtendedHubbardReal1D.jl")

include("BoseHubbardReal1D2C.jl")
include("BoseHubbardMom1D2C.jl")

include("GutzwillerSampling.jl")
include("GuidingVectorSampling.jl")

include("Transcorrelated1D.jl")

include("TwoBodyCorrelation.jl")

end
