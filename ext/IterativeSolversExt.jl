module IterativeSolversExt

using IterativeSolvers: IterativeSolvers, lobpcg, LOBPCGResults
using CommonSolve: CommonSolve, solve
using Rimu: Rimu, MatrixEDSolver, DVec, AbstractEDResult, delete, LOBPCG

struct LOBPCGEDResult{A,P,R <: LOBPCGResults,B} <: AbstractEDResult
    algorithm::A
    problem::P
    results::R
    basis::B
    howmany::Int
    success::Bool
end
function Base.getproperty(r::LOBPCGEDResult, key::Symbol)
    results = getfield(r, :results)
    if key === :values
        return results.λ
    elseif key === :vectors
        basis = getfield(r, :basis)
        vec_matrix = results.X
        return [DVec(zip(basis, @view vec_matrix[:, i])) for i in 1:size(vec_matrix, 2)]
    elseif key === :info
        return (;
            tolerance = results.tolerance,
            residual_norms = results.residual_norms,
            iterations = results.iterations,
            maxiter = results.maxiter,
            converged = results.converged,
            trace = results.trace
        )
    else
        return getfield(r, key)
    end
end
function Base.show(io::IO, r::LOBPCGEDResult)
    io = IOContext(io, :compact => true)
    n = length(r.values)
    println(io, "LOBPCGEDResult for algorithm $(r.algorithm) with $n eigenvalue(s),")
    print(io, "  values = ")
    show(io, r.values)
    print(io, ",\n  and vectors of length $(length(r.vectors[1])).")
    print(io, "\n  Convergence info: ")
    print(io, " iterations = ", r.info.iterations)
    print(io, ", converged = ", all(r.info.converged))
    print(io, ", maxiter = ", r.info.maxiter)
    print(io, ", residual_norms ≤ ")
    show(io, maximum(r.info.residual_norms))
    print(io, ", with howmany = $(r.howmany) eigenvalues requested.")
    print(io, "\n  success = $(r.success).")
end

function CommonSolve.solve(s::S; kwargs...) where {S<:Rimu.MatrixEDSolver{<:LOBPCG}}
    # combine keyword arguments and set defaults for `howmany` and `which`
    kw_nt = (; howmany=1, which=:SR, s.kw_nt..., kwargs...)
    # check if universal keyword arguments are present
    if isdefined(kw_nt, :reltol)
        kw_nt = (; kw_nt..., tol=kw_nt.reltol)
    end
    if isdefined(kw_nt, :abstol) # abstol has precedence over reltol
        kw_nt = (; kw_nt..., tol=kw_nt.abstol)
    end
    if isdefined(kw_nt, :maxiters)
        kw_nt = (; kw_nt..., maxiter=kw_nt.maxiters)
    end
    kw_nt = delete(kw_nt, (:verbose, :reltol, :abstol, :maxiters))

    # Remove the `howmany` and `which` keys from the kwargs.
    largest = (kw_nt.which == :SR) ? false : true
    kw_nt = (; nev=kw_nt.howmany, kw_nt...) # if nev was passed, it will overwrite howmany
    nev = kw_nt.nev # number of eigenvalues
    kw_nt = delete(kw_nt, (:howmany, :which, :nev))

    # solve the problem
    results = lobpcg(s.basissetrep.sm, largest, nev; kw_nt...)

    success = all(results.converged)
    if !success
        @warn "IterativeSolvers.lobpcg did not converge for all requested eigenvalues:" *
              " $(sum(results.converged)) converged out of $nev requested value(s)."
    end

    # create the result object
    return LOBPCGEDResult(s.algorithm, s.problem, results, s.basissetrep.basis, nev, success)
end

end # module
