"""
    module Hamiltonians

This module defines Hamiltonian types, interfaces, and functions for working with
Hamiltonians.
"""
module Hamiltonians

using Parameters, StaticArrays, LinearAlgebra, SparseArrays
using Setfield

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
export BoseHubbardMom1D2C, BoseHubbardReal1D2C
export GutzwillerSampling, GuidingVectorSampling

export G2Correlator

export LatticeGeometry, PeriodicBoundaries, HardwallBoundaries, LadderBoundaries
export num_neighbours, neighbour_site

include("abstract.jl")
include("offdiagonals.jl")
include("operations.jl")
include("geometry.jl")

include("MatrixHamiltonian.jl")

include("HubbardReal1D.jl")
include("HubbardMom1D.jl")
include("HubbardRealSpace.jl")
include("ExtendedHubbardReal1D.jl")

include("BoseHubbardReal1D2C.jl")
include("BoseHubbardMom1D2C.jl")

include("GutzwillerSampling.jl")
include("GuidingVectorSampling.jl")

include("TwoBodyCorrelation.jl")

end
