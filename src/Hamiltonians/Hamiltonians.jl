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

export AbstractHamiltonian, TwoComponentBosonicHamiltonian, offdiagonals, random_offdiagonal
export diagonal_element, num_offdiagonals, get_offdiagonal, dimension, starting_address
export rayleigh_quotient, momentum

export MatrixHamiltonian
export HubbardReal1D, HubbardMom1D, ExtendedHubbardReal1D, HubbardRealSpace
export BoseHubbardMom1D2C, BoseHubbardReal1D2C
export GutzwillerSampling, GuidingVectorSampling

export BoseHubbardReal1D, ExtendedBHReal1D, BoseHubbardMom1D
export G2Correlator

export PeriodicBoundaries, HardwallBoundaries, LadderBoundaries, num_neighbours, neighbour_site

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

# deprecated:
include("BoseHubbardReal1D.jl")
include("BoseHubbardMom1D.jl")
include("ExtendedBHReal1D.jl")


end
