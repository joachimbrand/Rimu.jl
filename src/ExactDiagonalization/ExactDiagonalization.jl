"""
The module `Rimu.ExactDiagonalization` provides a framework for exact diagonalization of
quantum many-body systems defined by an [`AbstractHamiltonian`](@ref) type.

The main usage is through defining an [`ExactDiagonalizationProblem`](@ref) and solving it
with the [`solve`](@ref solve(::ExactDiagonalizationProblem)) function. The module provides
a unified interface for accessing different solver algorithms, which make use of solvers
provided by external packages.

## Exports
- [`ExactDiagonalizationProblem`](@ref)
- [`BasisSetRepresentation`](@ref)
- [`build_basis`](@ref)

- [`KrylovKitSolver`](@ref)
- [`LinearAlgebraSolver`](@ref)
- [`ArpackSolver`](@ref)
- [`LOBPCGSolver`](@ref)
"""
module ExactDiagonalization

using LinearAlgebra: LinearAlgebra, eigen!, issymmetric, ishermitian, Matrix, dot, ⋅
using LinearMaps: LinearMaps, LinearMap
using SparseArrays: SparseArrays, nnz, nzrange, sparse
using CommonSolve: CommonSolve, solve, init
using VectorInterface: VectorInterface, add
using OrderedCollections: freeze
using NamedTupleTools: delete
using StaticArrays: setindex
import Folds

using Rimu: Rimu, DictVectors, Hamiltonians, Interfaces, BitStringAddresses, replace_keys,
    clean_and_warn_if_others_present, split_keys, HubbardMomSpace, ExtendedHubbardMom1D
using ..Interfaces: AbstractDVec, AbstractHamiltonian, AbstractOperator, AdjointUnknown,
    diagonal_element, offdiagonals, starting_address, LOStructure, IsHermitian,
    operator_column
using ..BitStringAddresses: AbstractFockAddress, BoseFS, FermiFS, HardcoreBoseFS,
    CompositeFS, near_uniform, BitString
using ..DictVectors: FrozenDVec, PDVec, DVec
using ..Hamiltonians: allows_address_type, check_address_type, dimension,
    ParitySymmetry, TimeReversalSymmetry, AbstractOperator

export ExactDiagonalizationProblem, KrylovKitSolver, LinearAlgebraSolver
export ArpackSolver, LOBPCGSolver
export BasisSetRepresentation, build_basis

export LinearMap

export sparse # from SparseArrays

const FermiOrHardcoreBoseFS{N,M,S} = Union{FermiFS{N,M,S},HardcoreBoseFS{N,M,S}}

include("basis_breadth_first_search.jl")
include("basis_fock.jl")
include("basis_set_representation.jl")
include("operator_as_map.jl")
include("algorithms.jl")
include("exact_diagonalization_problem.jl")
include("init_and_solvers.jl")
include("solve.jl")

end # module
