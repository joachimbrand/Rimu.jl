module ArpackExt

using Arpack: Arpack, eigs
using CommonSolve: CommonSolve, solve
using Rimu: Rimu, ArpackEigs, MatrixEDSolver, DVec, AbstractEDResult, delete

struct MatrixEDResult{A,P,VA<:Vector,VE<:Vector,B,I} <: AbstractEDResult
    algorithm::A
    problem::P
    values::VA
    vecs::VE
    basis::B
    info::I
end
function Base.getproperty(r::MatrixEDResult, key::Symbol)
    vecs = getfield(r, :vecs)
    if key === :vectors
        return [DVec(zip(getfield(r, :basis), v)) for v in vecs]
    else
        return getfield(r, key)
    end
end

function Base.show(io::IO, r::MatrixEDResult)
    # algs = string(Base.typename(typeof(r.algorithm)).type)
    n = length(r.values)
    println(io, "MatrixEDResult for algorithm $(r.algorithm) with $n eigenvalue(s),")
    print(io, "  `values` = ")
    show(io, r.values)
    print(io, ",\n  and `vectors` of length $(length(r.vectors[1])).")
    print(io, "\n  Convergence `info`: ")
    show(io, r.info)
end

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
    println(io, "norm of residuals ≤ $(maximum(info.residuals))")
    info.converged < 1 && println(io, "Arpack.eigs did not converge.")
end

function CommonSolve.solve(s::S; kwargs...
) where {S<:Rimu.MatrixEDSolver{<:ArpackEigs}}
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

    # set up the starting vector
    v0 = if isnothing(s.v0)
        zeros((0,))
    else
        # convert v0 to a DVec to use it like a dictionary
        [DVec(s.v0)[a] for a in s.basissetrep.basis]
    end
    # solve the problem
    vals, vec_matrix, nconv, niter, nmult, resid = eigs(s.basissetrep.sm; v0, kw_nt...)

    verbose && @info "Arpack.eigs: $nconv converged eigenvalues, $niter iterations," *
        " $nmult matrix vector multiplications, norm of residuals ≤ $(maximum(resid))"
    vecs = [view(vec_matrix, :, i) for i in 1:length(vals)] # convert to array of vectors
    info = ArpackConvergenceInfo(nconv, niter, nmult, resid)
    return MatrixEDResult(s.algorithm, s.problem, vals, vecs, s.basissetrep.basis, info)
end

end # module