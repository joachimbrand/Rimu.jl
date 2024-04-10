module KrylovKitExt

using KrylovKit: KrylovKit, EigSorter, eigsolve
using LinearAlgebra: LinearAlgebra, mul!, ishermitian, issymmetric
using Rimu: Rimu, AbstractDVec, AbstractHamiltonian, IsDeterministic, PDVec, DVec,
    PDWorkingMemory, scale!!, working_memory, zerovector, KrylovKitMatrixEDSolver,
    delete
using CommonSolve: CommonSolve

const U = Union{Symbol,EigSorter}

"""
    OperatorMultiplier

A struct that holds the working memory for repeatedly multiplying vectors with an operator.
"""
struct OperatorMultiplier{H,W<:PDWorkingMemory}
    hamiltonian::H
    working_memory::W
end
function OperatorMultiplier(hamiltonian, vector::PDVec)
    return OperatorMultiplier(hamiltonian, PDWorkingMemory(vector; style=IsDeterministic()))
end

function (o::OperatorMultiplier)(v)
    result = zerovector(v)
    return mul!(result, o.hamiltonian, v, o.working_memory)
end

function KrylovKit.eigsolve(
    ham::AbstractHamiltonian, vec::PDVec, howmany::Int=1, which::U=:LR; kwargs...
)
    # Change the type of `vec` to float, if needed.
    v = scale!!(vec, 1.0)
    prop = OperatorMultiplier(ham, v)
    return eigsolve(
        prop, v, howmany, which;
        ishermitian=ishermitian(ham), issymmetric=issymmetric(ham), kwargs...
    )
end

# This method only exists to detect whether a Hamiltonian is Hermitian or not.
function KrylovKit.eigsolve(
    ham::AbstractHamiltonian, vec::AbstractDVec, howmany::Int=1, which::U=:LR; kwargs...
)
    return @invoke eigsolve(
        ham::Any, vec::Any, howmany, which;
        ishermitian=ishermitian(ham), issymmetric=issymmetric(ham), kwargs...
    )
end


"""
    KrylovKitResult

A struct that holds the results of an "ExactDiagonalizationProblem" solved with KrylovKit.

# Fields
- `vals`: The eigenvalues.
- `vecs`: The eigenvectors as `DVec`s.
- `info`: The convergence information.
"""
struct KrylovKitResult{T,DV,I}
    vals::Vector{T}
    vecs::Vector{DV}
    info::I
end

function Base.show(io::IO, r::KrylovKitResult)
    n = length(r.vals)
    info = r.info
    print(io, "Rimu.KrylovKitResult with $n eigenvalue(s),\n  vals = ")
    show(io, r.vals)
    print(io, ",\n  and vecs of length $(length(r.vecs[1])).")
    print(io, "\n  Convergence info: ")
    info.converged == 0 && print(io, "no converged values ")
    info.converged == 1 && print(io, "one converged value ")
    info.converged > 1 && print(io, "$(info.converged) converged values ")
    println(io,
        "after ",
        info.numiter,
        " iteration(s) and ",
        info.numops,
        " applications of the linear map.")
    print(io, "  The norms of the residuals are ≤ $(maximum(info.normres)).")
end

function CommonSolve.solve(s::Rimu.KrylovKitMatrixEDSolver;
    verbose=nothing,
    abstol=nothing,
    reltol=nothing,
    maxiters=nothing,
    howmany=1,
    which=:SR,
    kwargs...
)
    kwargs = (s.kwargs..., kwargs..., :howmany=>howmany, :which=>which)
    if !isnothing(verbose)
        if verbose
            kwargs = (kwargs..., :verbosity => 1)
        else
            kwargs = (kwargs..., :verbosity => 0)
        end
    end
    if !isnothing(abstol)
        kwargs = (kwargs..., :tol => abstol)
    elseif !isnothing(reltol)
        kwargs = (kwargs..., :tol => reltol)
    end
    if !isnothing(maxiters)
        kwargs = (kwargs..., :maxiter => maxiters)
    end

    # Remove duplicates and the `howmany` and `which` keys from the kwargs.
    nt = NamedTuple(kwargs) # remove duplicates, only the last value is kept
    howmany, which = nt.howmany, nt.which
    kwargs = delete(nt, (:howmany, :which))

    T = eltype(s.basissetrep.sm)
    x0 = if isnothing(s.v0)
        rand(T, dimension(s.basissetrep)) # random initial guess
    else
        # convert v0 to a DVec to use it like a dictionary
        [DVec(s.v0)[a] for a in s.basissetrep.basis]
    end
    vals, vecs, info = eigsolve(s.basissetrep.sm, x0, howmany, which; kwargs...)
    dvecs = [DVec(zip(s.basissetrep.basis, v)) for v in vecs]
    return KrylovKitResult{eltype(vals),eltype(dvecs),typeof(info)}(vals, dvecs, info)
end

end
