"""
The module `Rimu.Hamiltonians` defines types and functions for working with
Hamiltonians.

## Exported concrete Hamiltonian types

Real space Hubbard models
 - [`HubbardRealSpace`](@ref)
 - [`HubbardReal1D`](@ref)
 - [`HubbardReal1DEP`](@ref)
 - [`ExtendedHubbardReal1D`](@ref)

Momentum space Hubbard models
- [`HubbardMom1D`](@ref)
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
- [`ParticleNumberOperator`](@ref)
- [`G2RealCorrelator`](@ref)
- [`G2RealSpace`](@ref)
- [`G2MomCorrelator`](@ref)
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
using SparseArrays: SparseArrays, rowvals, nzrange, nonzeros
using SpecialFunctions: SpecialFunctions, gamma
using StaticArrays: StaticArrays, SA, SMatrix, SVector, SArray, setindex
using TupleTools: TupleTools

using ..BitStringAddresses
using ..Interfaces
import ..Interfaces: diagonal_element, num_offdiagonals, get_offdiagonal, starting_address,
    offdiagonals, random_offdiagonal, LOStructure, allowed_address_type

export dimension, rayleigh_quotient, momentum

export MatrixHamiltonian
export HubbardReal1D, HubbardMom1D, ExtendedHubbardReal1D, HubbardRealSpace
export HubbardReal1DEP, shift_lattice, shift_lattice_inv
export HubbardMom1DEP
export GutzwillerSampling, GuidingVectorSampling
export ParitySymmetry
export TimeReversalSymmetry
export Stoquastic
export Transcorrelated1D
export hubbard_dispersion, continuum_dispersion
export FroehlichPolaron
export ParticleNumberOperator

export G2MomCorrelator, G2RealCorrelator, G2RealSpace, SuperfluidCorrelator, DensityMatrixDiagonal, Momentum
export StringCorrelator

export CubicGrid, PeriodicBoundaries, HardwallBoundaries, LadderBoundaries

export HOCartesianContactInteractions, HOCartesianEnergyConservedPerDim, HOCartesianCentralImpurity
export AxialAngularMomentumHO
export get_all_blocks, fock_to_cart

# deprecated and will be removed
export BoseHubbardMom1D2C
export BoseHubbardReal1D2C

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
include("particle_number.jl")

include("HOCartesianContactInteractions.jl")
include("HOCartesianEnergyConservedPerDim.jl")
include("HOCartesianCentralImpurity.jl")
include("vertices.jl")
include("ho-cart-tools.jl")
include("angular_momentum.jl")
end
