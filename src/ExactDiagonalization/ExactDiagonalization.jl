module ExactDiagonalization

using LinearAlgebra: LinearAlgebra, eigen!, ishermitian, Matrix
using SparseArrays: SparseArrays, nnz, nzrange, sparse
using CommonSolve: CommonSolve, solve, init
using VectorInterface: VectorInterface, add
using OrderedCollections: freeze
using NamedTupleTools: delete

using Rimu: Rimu, DictVectors, Hamiltonians, Interfaces, BitStringAddresses, replace_keys,
    clean_and_warn_if_others_present
using ..Interfaces: AbstractDVec, AbstractHamiltonian, AdjointUnknown,
    diagonal_element, offdiagonals, starting_address, LOStructure, IsHermitian
using ..BitStringAddresses: AbstractFockAddress
using ..DictVectors: FrozenDVec, PDVec, DVec
using ..Hamiltonians: check_address_type, dimension, ParitySymmetry, TimeReversalSymmetry


export ExactDiagonalizationProblem, KrylovKitSolver, LinearAlgebraSolver
export ArpackSolver, LOBPCGSolver
export BasisSetRep, build_basis

export sparse # from SparseArrays


include("BasisSetRep.jl")
include("algorithms.jl")
include("exact_diagonalization_problem.jl")
include("init_and_solvers.jl")
include("solve.jl")

end # module
