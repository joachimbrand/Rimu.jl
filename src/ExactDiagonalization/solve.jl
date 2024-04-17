# result types for `solve(::ExactDiagonalizationProblem; ...)`
abstract type AbstractEDResult end
# iteration for destructuring into components
Base.iterate(S::AbstractEDResult) = (S.values, Val(:vectors))
Base.iterate(S::AbstractEDResult, ::Val{:vectors}) = (S.vectors, Val(:success))
Base.iterate(S::AbstractEDResult, ::Val{:success}) = (S.info, Val(:done))
Base.iterate(::AbstractEDResult, ::Val{:done}) = nothing

# a lazy type for iterating over coefficient vectors
struct LazyCoefficientVectors{VT,T} <: AbstractVector{VT}
    m::Matrix{T}
end
function LazyCoefficientVectors(m::Matrix{T}) where {T}
    return LazyCoefficientVectors{typeof(view(m, :, 1)),T}(m)
end
Base.size(lcv::LazyCoefficientVectors) = (size(lcv.m, 2), 1)
Base.getindex(lcv::LazyCoefficientVectors, i::Int) = view(lcv.m, :, i)

# lazy type for iterating over constructed DVecs
struct LazyDVecs{DV,LCV,B} <: AbstractVector{DV}
    vs::LCV
    basis::B
end
function LazyDVecs(vs::LCV, basis::B) where {LCV,B}
    return LazyDVecs{typeof(DVec(zip(basis, vs[1]))),LCV,B}(vs, basis)
end
Base.size(lv::LazyDVecs) = size(lv.vs)
Base.getindex(lv::LazyDVecs, i::Int) = DVec(zip(lv.basis, lv.vs[i]))

# a generic result type for ExactDiagonalizationProblem
struct EDResult{A,P,VA<:AbstractVector,VE<:AbstractVector,CV,B,I,R} <: AbstractEDResult
    algorithm::A
    problem::P
    values::VA
    vectors::VE
    coefficient_vectors::CV
    basis::B
    info::I
    howmany::Int
    raw::R # algorithm-specific raw result, e.g. the matrix of eigenvectors
    success::Bool
end

function Base.show(io::IO, r::EDResult)
    io = IOContext(io, :compact => true)
    n = length(r.values)
    println(io, "EDResult for algorithm $(r.algorithm) with $n eigenvalue(s),")
    print(io, "  values = ")
    show(io, r.values)
    print(io, ",\n  and vectors of length $(length(r.vectors[1])).")
    print(io, "\n  Convergence info: ")
    show(io, r.info)
    print(io, ", with howmany = $(r.howmany) eigenvalues requested.")
    print(io, "\n  success = $(r.success).")
end

# solve directly on the ExactDiagonalizationProblem
"""
    solve(p::ExactDiagonalizationProblem, [algorithm]; kwargs...)

Solve an [`ExactDiagonalizationProblem`](@ref) `p` directly. Optionally specify an
`algorithm.` Returns a result type with the eigenvalues, eigenvectors, and convergence
information.

For a description of the keyword arguments, see the documentation for
[`ExactDiagonalizationProblem`](@ref).
"""
function CommonSolve.solve(p::ExactDiagonalizationProblem; kwargs...)
    s = init(p; kwargs...)
    return solve(s)
end
function CommonSolve.solve(p::ExactDiagonalizationProblem, algorithm; kwargs...)
    s = init(p, algorithm; kwargs...)
    return solve(s)
end

# The code for `CommonSolve.solve(::MatrixEDSolver{<:ALG}; ...)` for
# - ALG<:KrylovKitMatrix is part of the `KrylovKitExt.jl` extension
# - ALG<:ArpackSolver is part of the `ArpackExt.jl` extension
# - ALG<:LOBPCGSolver is part of the `IterativeSolversExt.jl` extension

# The code for `CommonSolve.solve(::KrylovKitDirectEDSolver; ...)` is part of the
# `KrylovKitExt.jl` extension.

function CommonSolve.solve(s::MatrixEDSolver{<:LinearAlgebraSolver};
    kwargs...
)
    # combine keyword arguments
    kw_nt = (; s.kw_nt..., kwargs...)
    howmany = get(kw_nt, :howmany, dimension(s.basissetrep))

    # extract relevant keyword arguments
    permute = get(kw_nt, :permute, true)
    scale = get(kw_nt, :scale, true)

    eigen_factorization = if isdefined(kw_nt, :sortby)
        sortby = kw_nt.sortby
        eigen(Matrix(s.basissetrep.sm); permute, scale, sortby)
    else
        eigen(Matrix(s.basissetrep.sm); permute, scale)
    end
    nt = delete(kw_nt, (:permute, :scale, :sortby))
    !isempty(nt) && @warn "Unused keyword arguments in `solve`: $nt"
    # eigen_factorization = eigen(Matrix(s.basissetrep.sm); kw_nt...)

    coefficient_vectors = LazyCoefficientVectors(eigen_factorization.vectors)
    vectors = LazyDVecs(coefficient_vectors, s.basissetrep.basis)
    return EDResult(
        s.algorithm,
        s.problem,
        eigen_factorization.values,
        vectors,
        coefficient_vectors,
        s.basissetrep.basis,
        "Dense matrix eigensolver solution from `LinearAlgebra.eigen`",
        howmany,
        eigen_factorization.vectors,
        true # successful if no exception was thrown
    )
end
