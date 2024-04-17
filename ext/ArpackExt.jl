module ArpackExt

using Arpack: Arpack, eigs
using CommonSolve: CommonSolve, solve
using Rimu: Rimu, DVec, delete
using Rimu.ExactDiagonalization: ArpackSolver, MatrixEDSolver,
    LazyCoefficientVectors, LazyDVecs, EDResult

struct ArpackConvergenceInfo
    converged::Int
    numiter::Int
    numops::Int
    residuals::Vector{Float64}
end
function Base.show(io::IO, info::ArpackConvergenceInfo)
    print(io, "converged = $(info.converged), ")
    print(io, "numiter = $(info.numiter), ")
    print(io, "numops = $(info.numops), ")
    print(io, "residuals ≤ ")
    show(io, maximum(info.residuals))
end

function CommonSolve.solve(s::S; kwargs...
) where {S<:MatrixEDSolver{<:ArpackSolver}}
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
    verbose = get(kw_nt, :verbose, false)
    kw_nt = delete(kw_nt, (:verbose, :reltol, :abstol, :maxiters))

    # Remove the `howmany` key from the kwargs.
    kw_nt = (; nev=kw_nt.howmany, kw_nt..., ritzvec=true)
    kw_nt = delete(kw_nt, (:howmany,))
    howmany = kw_nt.nev

    # set up the starting vector
    v0 = if isnothing(s.v0)
        zeros((0,))
    else
        # convert v0 to a DVec to use it like a dictionary
        [DVec(s.v0)[a] for a in s.basissetrep.basis]
    end
    # solve the problem
    vals, vec_matrix, nconv, niter, nmult, resid = eigs(s.basissetrep.sm; v0, kw_nt...)

    verbose && @info "Arpack.eigs: $nconv converged out of $howmany requested eigenvalues,"*
        " $niter iterations," *
        " $nmult matrix vector multiplications, norm of residuals ≤ $(maximum(resid))"
    success = nconv ≥ howmany
    # vecs = [view(vec_matrix, :, i) for i in 1:length(vals)] # convert to array of vectors
    coefficient_vectors = LazyCoefficientVectors(vec_matrix)
    vectors = LazyDVecs(coefficient_vectors, s.basissetrep.basis)
    info = ArpackConvergenceInfo(nconv, niter, nmult, resid)
    if !success
        @warn "Arpack.eigs did not converge for all requested eigenvalues:" *
              " $nconv converged out of $howmany requested value(s)."
    end
    return EDResult(
        s.algorithm,
        s.problem,
        vals,
        vectors,
        coefficient_vectors,
        s.basissetrep.basis,
        info,
        howmany,
        vec_matrix,
        success
    )
end

end # module
