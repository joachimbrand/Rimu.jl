# result types for `solve(::ExactDiagonalizationProblem; ...)`
abstract type AbstractEDResult end
# iteration for destructuring into components
Base.iterate(S::AbstractEDResult) = (S.values, Val(:vectors))
Base.iterate(S::AbstractEDResult, ::Val{:vectors}) = (S.vectors, Val(:success))
Base.iterate(S::AbstractEDResult, ::Val{:success}) = (S.info, Val(:done))
Base.iterate(::AbstractEDResult, ::Val{:done}) = nothing

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
# - ALG<:ArpackEigs is part of the `ArpackExt.jl` extension
# - ALG<:LOBPCG is part of the `IterativeSolversExt.jl` extension

# The code for `CommonSolve.solve(::KrylovKitDirectEDSolver; ...)` is part of the
# `KrylovKitExt.jl` extension.

function CommonSolve.solve(s::MatrixEDSolver{<:LinearAlgebraEigen};
    kwargs...
)
    # combine keyword arguments
    kw_nt = (; s.kw_nt..., kwargs...)

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

    return MatrixEDEigenResult(s.problem, s.basissetrep.basis, eigen_factorization, true)
end

struct MatrixEDEigenResult{P,B,F} <: AbstractEDResult
    problem::P
    basis::B
    eigen_factorization::F
    success::Bool
end
function Base.show(io::IO, r::MatrixEDEigenResult)
    io = IOContext(io, :compact => true)
    n = length(r.values)
    print(io, "MatrixEDEigenResult with $n eigenvalue(s),\n  values = ")
    show(io, r.values)
    print(io, ",\n  and vectors of length $n.")
end
function Base.getproperty(r::MatrixEDEigenResult, key::Symbol)
    vs = getfield(r, :eigen_factorization).vectors
    n = size(vs, 2)
    if key === :values
        return getfield(r, :eigen_factorization).values
    elseif key === :vectors
        return [DVec(zip(getfield(r, :basis), view(vs, :, i))) for i in 1:n]
    elseif key === :info
        return "Dense matrix eigensolver solution from `LinearAlgebra.eigen`."
    else
        return getfield(r, key)
    end
end
