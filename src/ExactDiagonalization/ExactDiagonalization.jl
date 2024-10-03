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

using LinearAlgebra: LinearAlgebra, eigen!, ishermitian, Matrix
using SparseArrays: SparseArrays, nnz, nzrange, sparse
using CommonSolve: CommonSolve, solve, init
using VectorInterface: VectorInterface, add
using OrderedCollections: freeze
using NamedTupleTools: delete
using StaticArrays: setindex

using Rimu: Rimu, DictVectors, Hamiltonians, Interfaces, BitStringAddresses, replace_keys,
    clean_and_warn_if_others_present
using ..Interfaces: AbstractDVec, AbstractHamiltonian, AdjointUnknown,
    diagonal_element, offdiagonals, starting_address, LOStructure, IsHermitian
using ..BitStringAddresses: AbstractFockAddress, BoseFS, FermiFS, CompositeFS, near_uniform
using ..DictVectors: FrozenDVec, PDVec, DVec
using ..Hamiltonians: allows_address_type, check_address_type, dimension,
    ParitySymmetry, TimeReversalSymmetry


export ExactDiagonalizationProblem, KrylovKitSolver, LinearAlgebraSolver
export ArpackSolver, LOBPCGSolver
export BasisSetRepresentation, build_basis

export sparse # from SparseArrays


include("basis_bfs.jl")
include("basis_fock.jl")
include("basis_set_representation.jl")
include("algorithms.jl")
include("exact_diagonalization_problem.jl")
include("init_and_solvers.jl")
include("solve.jl")

include("deprecated.jl")

end # module
