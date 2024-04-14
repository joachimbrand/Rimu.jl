module KrylovKitExt

using KrylovKit: KrylovKit, EigSorter, eigsolve
using LinearAlgebra: LinearAlgebra, mul!, ishermitian, issymmetric
using CommonSolve: CommonSolve
using Setfield: Setfield, @set

using Rimu: Rimu, AbstractDVec, AbstractHamiltonian, IsDeterministic, PDVec, DVec,
    PDWorkingMemory, scale!!, working_memory, zerovector, MatrixEDSolver,
    delete, KrylovKitMatrix, dimension, AbstractEDResult

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
- `values`: The eigenvalues.
- `vectors`: The eigenvectors as `DVec`s.
- `info`: The convergence information.
"""
struct KrylovKitResult{P,VA<:Vector,VDV<:Vector,I} <: AbstractEDResult
    problem::P
    values::VA
    vectors::VDV
    info::I
    howmany::Int
    success::Bool
end


struct MatrixEDKrylovKitResult{P,VA<:Vector,VE<:Vector,B,I} <: AbstractEDResult
    problem::P
    values::VA
    vecs::VE
    basis::B
    info::I
    howmany::Int
    success::Bool
end
function Base.getproperty(r::MatrixEDKrylovKitResult, key::Symbol)
    vecs = getfield(r, :vecs)
    if key === :vectors
        return [DVec(zip(getfield(r, :basis), v)) for v in vecs]
    else
        return getfield(r, key)
    end
end

function Base.show(io::IO, r::R) where R<:Union{KrylovKitResult,MatrixEDKrylovKitResult}
    io = IOContext(io, :compact => true)
    ts = R isa KrylovKitResult ? "KrylovKitResult" : "MatrixEDKrylovKitResult"
    n = length(r.values)
    info = r.info
    print(io, "$ts with $n eigenvalue(s),\n  values = ")
    show(io, r.values)
    print(io, ",\n  and vectors of length $(length(r.vectors[1])).")
    print(io, "\n  Convergence info: ")
    print(io, "$(info.converged) converged out of howmany = $(r.howmany) requested value(s) ")
    println(io,
        "after ",
        info.numiter,
        " iteration(s) and ",
        info.numops,
        " applications of the linear map.")
    print(io, "  The norms of the residuals are ≤ ")
    show(io, maximum(info.normres))
    print(io, ".\n  success = $(r.success).")
end

# solve for KrylovKit solvers: prepare arguments for `KrylovKit.eigsolve`
function CommonSolve.solve(s::S; kwargs...
) where {S<:Union{Rimu.MatrixEDSolver{<:KrylovKitMatrix}, Rimu.KrylovKitDirectEDSolver}}
    # combine keyword arguments and set defaults for `howmany` and `which`
    kw_nt = (; howmany = 1, which = :SR, s.kw_nt..., kwargs...)
    # check if universal keyword arguments are present
    if isdefined(kw_nt, :verbose)
        if kw_nt.verbose
            kw_nt = (; kw_nt..., verbosity = 1)
        else
            kw_nt = (; kw_nt..., verbosity = 0)
        end
    end
    if isdefined(kw_nt, :reltol)
        kw_nt = (; kw_nt..., tol = kw_nt.reltol)
    end
    if isdefined(kw_nt, :abstol) # abstol has precedence over reltol
        kw_nt = (; kw_nt..., tol = kw_nt.abstol)
    end
    if isdefined(kw_nt, :maxiters)
        kw_nt = (; kw_nt..., maxiter = kw_nt.maxiters)
    end
    kw_nt = delete(kw_nt, (:verbose, :reltol, :abstol, :maxiters))

    # Remove the `howmany` and `which` keys from the kwargs.
    howmany, which = kw_nt.howmany, kw_nt.which
    kw_nt = delete(kw_nt, (:howmany, :which))

    return _kk_eigsolve(s, howmany, which, kw_nt)
end

# solve with KrylovKit and matrix
function _kk_eigsolve(s::Rimu.MatrixEDSolver{<:KrylovKitMatrix}, howmany, which, kw_nt)
    # set up the starting vector
    T = eltype(s.basissetrep.sm)
    x0 = if isnothing(s.v0)
            rand(T, dimension(s.basissetrep)) # random initial guess
        else
            # convert v0 to a DVec to use it like a dictionary
            [DVec(s.v0)[a] for a in s.basissetrep.basis]
    end
    # solve the problem
    vals, vecs, info = eigsolve(s.basissetrep.sm, x0, howmany, which; kw_nt...)
    success = info.converged ≥ howmany
    if !success
        @warn "KrylovKit.eigsolve did not converge for all requested eigenvalues:" *
              " $(info.converged) converged out of $howmany requested value(s)."
    end
    return MatrixEDKrylovKitResult(
        s.problem, vals, vecs, s.basissetrep.basis, info, howmany, success
    )
end

# solve with KrylovKit direct
function _kk_eigsolve(s::Rimu.KrylovKitDirectEDSolver, howmany, which, kw_nt)
    vals, vecs, info = eigsolve(s.problem.h, s.v0, howmany, which; kw_nt...)
    success = info.converged ≥ howmany
    if !success
        @warn "KrylovKit.eigsolve did not converge for all requested eigenvalues:" *
              " $(info.converged) converged out of $howmany requested value(s)."
    end
    return KrylovKitResult(s.problem, vals, vecs, info, howmany, success)
end

end
