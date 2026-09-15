"""
The module `Rimu.Hamiltonians` defines types and functions for working with
Hamiltonians.

## Exported concrete Hamiltonian types

Real space Hubbard models
 - [`HubbardReal1D`](@ref)
 - [`HubbardReal1DEP`](@ref)
 - [`HubbardRealSpace`](@ref)
 - [`ExtendedHubbardReal1D`](@ref)

Momentum space Hubbard models
- [`HubbardMom1D`](@ref)
- [`HubbardMom1DEP`](@ref)

Harmonic oscillator models
- [`HOCartesianContactInteractions`](@ref)
- [`HOCartesianEnergyConservedPerDim`](@ref)
- [`HOCartesianCentralImpurity`](@ref)

Other
- [`FroehlichPolaron1D`](@ref)
- [`MatrixHamiltonian`](@ref)
- [`Transcorrelated1D`](@ref)
- [`HamiltonianProduct`](@ref)
- [`FroehlichPolaron`](@ref)

- [`MolecularHamiltonian`](@ref)

## [Wrappers](#Hamiltonian-wrappers)
- [`GutzwillerSampling`](@ref)
- [`GuidingVectorSampling`](@ref)
- [`ParitySymmetry`](@ref)
- [`TimeReversalSymmetry`](@ref)
- [`Stoquastic`](@ref)
- [`HamiltonianProduct`](@ref)
- [`HamiltonianSum`](@ref)

## [Linear combination helpers](#Linear-combination-helpers)
- [`add`](@ref)
- [`+`](@ref)
- [`scale`](@ref)

## [Observables](#Observables)
- [`ParticleNumberOperator`](@ref)
- [`G2RealCorrelator`](@ref)
- [`G2MomCorrelator`](@ref)
- [`G2RealSpace`](@ref)
- [`DensityMatrixDiagonal`](@ref)
- [`SingleParticleExcitation`](@ref)
- [`TwoParticleExcitation`](@ref)
- [`Momentum`](@ref)
- [`AxialAngularMomentumHO`](@ref)
- [`SignCorrelator`](@ref)

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
using StaticArrays: StaticArrays, SA, SMatrix, SVector, SArray, MVector, setindex
using TupleTools: TupleTools
using VectorInterface: add, scale

using ..BitStringAddresses
import ..BitStringAddresses: ModeMap, FermiFS2CModes, full_mode_maps
using ..Interfaces
using ..Interfaces: sum_mutating!, num_modes_check_equal, num_modes
import ..Interfaces: diagonal_element, num_offdiagonals, get_offdiagonal, starting_address,
    offdiagonals, random_offdiagonal, LOStructure, allows_address_type, operator_column,
    undo_transform, has_random_offdiagonal, has_iterable_offdiagonals, parent_operator

export dimension, rayleigh_quotient, momentum

export IdentityOperator
export MatrixHamiltonian
export HubbardReal1D, HubbardMom1D, ExtendedHubbardReal1D, ExtendedHubbardMom1D, HubbardRealSpace
export HubbardReal1DEP, shift_lattice, shift_lattice_inv
export HubbardMom1DEP
export GutzwillerSampling, GuidingVectorSampling
export ParitySymmetry
export TimeReversalSymmetry
export Stoquastic
export Transcorrelated1D
export hubbard_dispersion, continuum_dispersion
export FroehlichPolaron1D
export FroehlichPolaron
export ParticleNumberOperator

export MolecularHamiltonian

export G2RealCorrelator, G2RealSpace, SuperfluidCorrelator, DensityMatrixDiagonal, Momentum
export SingleParticleExcitation, TwoParticleExcitation, ReducedDensityMatrix
export StringCorrelator, G2MomCorrelator, SignCorrelator

export CubicGrid, PeriodicBoundaries, HardwallBoundaries, LadderBoundaries

export HOCartesianContactInteractions, HOCartesianEnergyConservedPerDim, HOCartesianCentralImpurity
export AxialAngularMomentumHO
export get_all_blocks, fock_to_cart

export ModifiedHamiltonian
export HamiltonianProduct, HamiltonianSum

if VERSION < v"1.10"
    # used for ReducedDensityMatrix
    function hermitianpart!(A)
        A .= (A + A') / 2
        return Hermitian(A)
    end
end

const FermiOrHardcoreBoseFS{N,M,S} = Union{FermiFS{N,M,S},HardcoreBoseFS{N,M,S}}

include("abstract.jl")
include("offdiagonals.jl")
include("geometry.jl")
include("excitations.jl")

include("MatrixHamiltonian.jl")

include("HubbardReal1D.jl")
include("HubbardReal1DEP.jl")
include("ExtendedHubbardMom1D.jl")
include("HubbardMom1D.jl")
include("HubbardMom1DEP.jl")
include("HubbardRealSpace.jl")
include("ExtendedHubbardReal1D.jl")

include("FroehlichPolaron1D.jl")
include("FroehlichPolaron.jl")


include("Transcorrelated1D.jl")

include("Molecular.jl")

include("ModifiedHamiltonian.jl")
include("TransformUndoer.jl")
include("GutzwillerSampling.jl")
include("GuidingVectorSampling.jl")
include("ParitySymmetry.jl")
include("TRSymmetry.jl")
include("Stoquastic.jl")

include("correlation_functions.jl")
include("G2MomCorrelator.jl")
include("DensityMatrixDiagonal.jl")
include("reduced_density_matrix.jl")
include("Momentum.jl")
include("particle_number.jl")

include("HOCartesianContactInteractions.jl")
include("HOCartesianEnergyConservedPerDim.jl")
include("HOCartesianCentralImpurity.jl")
include("vertices.jl")
include("ho-cart-tools.jl")
include("angular_momentum.jl")

include("Product.jl")
include("Sum.jl")
end
