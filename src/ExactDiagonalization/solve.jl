# a lazy type for iterating over coefficient vectors starting from a matrix
struct LazyCoefficientVectors{VT,T} <: AbstractVector{VT}
    m::Matrix{T}
end
function LazyCoefficientVectors(m::Matrix{T}) where {T}
    return LazyCoefficientVectors{typeof(view(m, :, 1)),T}(m)
end
Base.size(lcv::LazyCoefficientVectors) = (size(lcv.m, 2), 1)
Base.getindex(lcv::LazyCoefficientVectors, i::Int) = view(lcv.m, :, i)

# a lazy type for iterating over coefficient vectors starting from a vector of DVecs
struct LazyCoefficientVectorsDVecs{T,VDV<:Vector{<:AbstractDVec},B} <: AbstractVector{T}
    vecs::VDV
    basis::B
end
function LazyCoefficientVectorsDVecs(vecs, basis)
    T = valtype(vecs[1])
    return LazyCoefficientVectorsDVecs{T,typeof(vecs),typeof(basis)}(vecs, basis)
end
Base.size(v::LazyCoefficientVectorsDVecs) = (length(v.vecs),)
function Base.getindex(lcv::LazyCoefficientVectorsDVecs{T}, i::Int) where {T}
    return T[lcv.vecs[i][address] for address in lcv.basis]
end

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
struct EDResult{A,P,VA<:AbstractVector,VE<:AbstractVector,CV,B,I,R}
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
# iteration for destructuring into components
Base.iterate(S::EDResult) = (S.values, Val(:vectors))
Base.iterate(S::EDResult, ::Val{:vectors}) = (S.vectors, Val(:success))
Base.iterate(S::EDResult, ::Val{:success}) = (S.info, Val(:done))
Base.iterate(::EDResult, ::Val{:done}) = nothing

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
    kw_nt = clean_and_warn_if_others_present(kw_nt, (:permute, :scale, :sortby))

    eigen_factorization = eigen(Matrix(s.basissetrep.sm); kw_nt...)

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
        dimension(s.basissetrep),
        eigen_factorization.vectors,
        true # successful if no exception was thrown
    )
end
