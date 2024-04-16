module ArpackExt

using Arpack: Arpack, eigs
using CommonSolve: CommonSolve, solve
using Rimu: Rimu, DVec, delete
using Rimu.ExactDiagonalization: ArpackEigs, MatrixEDSolver, AbstractEDResult

struct MatrixEDResult{A,P,VA<:Vector,VE<:Vector,B,I} <: AbstractEDResult
    algorithm::A
    problem::P
    values::VA
    vecs::VE
    basis::B
    info::I
    howmany::Int
    success::Bool
end
function Base.getproperty(r::MatrixEDResult, key::Symbol)
    if key === :vectors
        vecs = getfield(r, :vecs)
        basis = getfield(r, :basis)
        return [DVec(zip(basis, v)) for v in vecs]
    else
        return getfield(r, key)
    end
end

function Base.show(io::IO, r::MatrixEDResult)
    io = IOContext(io, :compact => true)
    n = length(r.values)
    println(io, "MatrixEDResult for algorithm $(r.algorithm) with $n eigenvalue(s),")
    print(io, "  values = ")
    show(io, r.values)
    print(io, ",\n  and vectors of length $(length(r.vectors[1])).")
    print(io, "\n  Convergence info: ")
    show(io, r.info)
    print(io, ", with howmany = $(r.howmany) eigenvalues requested.")
    print(io, "\n  sucess = $(r.success).")
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
    print(io, "residuals ≤ ")
    show(io, maximum(info.residuals))
end

function CommonSolve.solve(s::S; kwargs...
) where {S<:MatrixEDSolver{<:ArpackEigs}}
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
    vecs = [view(vec_matrix, :, i) for i in 1:length(vals)] # convert to array of vectors
    info = ArpackConvergenceInfo(nconv, niter, nmult, resid)
    if !success
        @warn "Arpack.eigs did not converge for all requested eigenvalues:" *
              " $nconv converged out of $howmany requested value(s)."
    end
    return MatrixEDResult(
        s.algorithm, s.problem, vals, vecs, s.basissetrep.basis, info, howmany, success
    )
end

end # module
