module IterativeSolversExt

using IterativeSolvers: IterativeSolvers, lobpcg, LOBPCGResults
using CommonSolve: CommonSolve, solve
using NamedTupleTools: delete
using Rimu: Rimu, DVec, replace_keys, delete_and_warn_if_present
using Rimu.ExactDiagonalization: MatrixEDSolver, LOBPCGSolver,
    LazyCoefficientVectors, LazyDVecs, EDResult

struct LOBPCGConvergenceInfo
    tolerance::Float64
    iterations::Int
    converged::Bool
    maxiter::Int
    residual_norms::Vector{Float64}
end
function Base.show(io::IO, info::LOBPCGConvergenceInfo)
    print(io, "tolerance = $(info.tolerance), ")
    print(io, "iterations = $(info.iterations), ")
    print(io, "converged = $(info.converged), ")
    print(io, "maxiter = $(info.maxiter), ")
    print(io, "residual_norms ≤ ")
    show(io, maximum(info.residual_norms))
end

function CommonSolve.solve(s::S; kwargs...) where {S<:MatrixEDSolver{<:LOBPCGSolver}}
    # combine keyword arguments and set defaults for `howmany` and `which`
    kw_nt = (; howmany=1, which=:SR, s.kw_nt..., kwargs...)
    # check if universal keyword arguments are present
    kw_nt = replace_keys(kw_nt, (:abstol=>:tol, :maxiters=>:maxiter))
    kw_nt = delete_and_warn_if_present(kw_nt, (:verbose, :reltol))

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

    coefficient_vectors = LazyCoefficientVectors(results.X)
    vectors = LazyDVecs(coefficient_vectors, s.basissetrep.basis)
    info = LOBPCGConvergenceInfo(
        results.tolerance,
        results.iterations,
        results.converged,
        results.maxiter,
        results.residual_norms
    )
    # create the result object
    return EDResult(
        s.algorithm,
        s.problem,
        results.λ,
        vectors,
        coefficient_vectors,
        s.basissetrep.basis,
        info,
        nev,
        results,
        success
    )
end

end # module
