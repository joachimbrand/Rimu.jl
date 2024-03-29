"""
    module Hamiltonians

This module defines Hamiltonian types and functions for working with
Hamiltonians.

## [Exported concrete Hamiltonian types](#Model-Hamiltonians)

Real space Hubbard models
 - [`HubbardReal1D`](@ref)
 - [`BoseHubbardReal1D2C`](@ref)
 - [`HubbardReal1DEP`](@ref)
 - [`HubbardRealSpace`](@ref)
 - [`ExtendedHubbardReal1D`](@ref)

Momentum space Hubbard models
- [`HubbardMom1D`](@ref)
- [`BoseHubbardMom1D2C`](@ref)
- [`HubbardMom1DEP`](@ref)

Harmonic oscillator models
- [`HOCartesianContactInteractions`](@ref)
- [`HOCartesianEnergyConservedPerDim`](@ref)
- [`HOCartesianCentralImpurity`](@ref)

Other
- [`FroehlichPolaron`](@ref)
- [`MatrixHamiltonian`](@ref)
- [`Transcorrelated1D`](@ref)

## [Wrappers](#Hamiltonian-wrappers)
- [`GutzwillerSampling`](@ref)
- [`GuidingVectorSampling`](@ref)
- [`ParitySymmetry`](@ref)
- [`TimeReversalSymmetry`](@ref)
- [`Stoquastic`](@ref)

## [Observables](#Observables)
- [`G2MomCorrelator`](@ref)
- [`G2RealCorrelator`](@ref)
- [`DensityMatrixDiagonal`](@ref)
- [`Momentum`](@ref)
- [`AxialAngularMomentumHO`](@ref)

## [Interface for working with Hamiltonians](#Hamiltonians-interface)
- [`AbstractHamiltonian`](@ref): defined in the module [`Interfaces`](@ref)
"""
module Hamiltonians

using Combinatorics: Combinatorics, multiset_permutations,
    with_replacement_combinations
using DataFrames: DataFrames, DataFrame, transform
using FFTW: FFTW, fft
using HypergeometricFunctions: HypergeometricFunctions, _₃F₂
using LinearAlgebra: LinearAlgebra, I, diag, dot, ishermitian, issymmetric,
    mul!, norm
using Parameters: Parameters, @unpack
using Setfield: Setfield
using SparseArrays: SparseArrays, nnz, nzrange, sparse
using SpecialFunctions: SpecialFunctions, gamma
using StaticArrays: StaticArrays, SA, SMatrix, SVector
using TupleTools: TupleTools

using ..BitStringAddresses
using ..Interfaces
import ..Interfaces: diagonal_element, num_offdiagonals, get_offdiagonal, starting_address,
    offdiagonals, random_offdiagonal, LOStructure, allowed_address_type

export AbstractHamiltonian
# export TwoComponentHamiltonian
export dimension, rayleigh_quotient, momentum
export BasisSetRep, build_basis

export MatrixHamiltonian
export HubbardReal1D, HubbardMom1D, ExtendedHubbardReal1D, HubbardRealSpace
export HubbardReal1DEP, shift_lattice, shift_lattice_inv
export HubbardMom1DEP
export BoseHubbardMom1D2C, BoseHubbardReal1D2C
export GutzwillerSampling, GuidingVectorSampling
export ParitySymmetry
export TimeReversalSymmetry
export Stoquastic
export Transcorrelated1D
export hubbard_dispersion, continuum_dispersion
export FroehlichPolaron

export G2MomCorrelator, G2RealCorrelator, SuperfluidCorrelator, DensityMatrixDiagonal, Momentum
export StringCorrelator

export LatticeGeometry, PeriodicBoundaries, HardwallBoundaries, LadderBoundaries
export num_neighbours, neighbour_site, num_dimensions

export sparse # from SparseArrays

export HOCartesianContactInteractions, HOCartesianEnergyConservedPerDim, HOCartesianCentralImpurity
export AxialAngularMomentumHO
export get_all_blocks, fock_to_cart

include("abstract.jl")
include("offdiagonals.jl")
include("geometry.jl")
include("excitations.jl")

include("MatrixHamiltonian.jl")

include("HubbardReal1D.jl")
include("HubbardReal1DEP.jl")
include("HubbardMom1D.jl")
include("HubbardMom1DEP.jl")
include("HubbardRealSpace.jl")
include("ExtendedHubbardReal1D.jl")

include("BoseHubbardReal1D2C.jl")
include("BoseHubbardMom1D2C.jl")
include("FroehlichPolaron.jl")

include("GutzwillerSampling.jl")
include("GuidingVectorSampling.jl")
include("ParitySymmetry.jl")
include("TRSymmetry.jl")
include("Stoquastic.jl")

include("Transcorrelated1D.jl")

include("correlation_functions.jl")
include("DensityMatrixDiagonal.jl")
include("Momentum.jl")

include("HOCartesianContactInteractions.jl")
include("HOCartesianEnergyConservedPerDim.jl")
include("HOCartesianCentralImpurity.jl")
include("vertices.jl")
include("ho-cart-tools.jl")
include("angular_momentum.jl")
end
