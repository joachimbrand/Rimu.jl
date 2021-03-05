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

export AbstractHamiltonian, TwoComponentBosonicHamiltonian, hops, generateRandHop
export diagME, numOfHops, hop, dimension, starting_address
export rayleigh_quotient, momentum

export HubbardReal1D, HubbardMom1D, ExtendedHubbardReal1D
export BoseHubbardMom1D2C, BoseHubbardReal1D2C

export BoseHubbardReal1D, ExtendedBHReal1D, BoseHubbardMom1D

include("abstract.jl")
include("hops.jl")
include("operations.jl")

include("HubbardReal1D.jl")
include("HubbardMom1D.jl")
include("ExtendedHubbardReal1D.jl")

include("BoseHubbardReal1D2C.jl")
include("BoseHubbardMom1D2C.jl")

# deprecated:
include("BoseHubbardReal1D.jl")
include("BoseHubbardMom1D.jl")
include("ExtendedBHReal1D.jl")


end
