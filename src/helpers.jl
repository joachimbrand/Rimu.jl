# small functions supporting fciqmc!()
# versions without dependence on MPI.jl
using Base.Threads: nthreads

threadedWorkingMemory(dv) = threadedWorkingMemory(localpart(dv))
function threadedWorkingMemory(v::AbstractDVec)
    return Tuple(similar(v) for _ in 1:nthreads())
end
function threadedWorkingMemory(v::AbstractVector)
    return Tuple(similar(v) for _ in 1:nthreads())
end

"""
    Rimu.refine_r_strat(r_strat::ReportingStrategy, ham)
Refine the reporting strategy by replacing `Symbol`s in the keyword argument
`hproj` by the appropriate value. See [`ReportingStrategy`](@ref)
"""
refine_r_strat(r_strat::ReportingStrategy, ham) = r_strat # default

function refine_r_strat(r_strat::ReportingStrategy{P1,P2}, ham) where
                                                {P1 <: Nothing, P2 <: Symbol}
    # return ReportingStrategy(r_strat, hproj = nothing) # ignore `hproj`
    return @set r_strat.hproj = nothing # ignore `hproj`
    # using @set macro from the Setfield.jl package
end

function refine_r_strat(r_strat::ReportingStrategy{P1,P2}, ham) where
                                                {P1, P2 <: Symbol}
    if r_strat.hproj == :lazy
        @info "`hproj = :lazy` may slow down the code"
        return @set r_strat.hproj = missing
    elseif r_strat.hproj == :not
        return @set r_strat.hproj = nothing
    elseif r_strat.hproj == :eager
        return @set r_strat.hproj = copy(ham'*r_strat.projector)
    elseif r_strat.hproj == :auto
        if P1  <: AbstractProjector # for projectors don't compute `df.hproj`
            return @set r_strat.hproj = nothing
        elseif Hamiltonians.has_adjoint(ham) # eager is possible
            hpv = ham'*r_strat.projector # pre-calculate left vector with adjoint Hamiltonian
            # use smaller container to save memory
            return @set r_strat.hproj = copy(hpv)
        else # lazy is default
            return @set r_strat.hproj = missing
        end
    end
    @error "Value $(r_strat.hproj) for keyword `hproj` is not recognized. See documentation of [`ReportingStrategy`](@doc)."
end

# three-argument version
"""
    sort_into_targets!(target, source, stats) -> agg, wm, agg_stats
Aggregate coefficients from `source` to `agg` and from `stats` to `agg_stats`
according to thread- or MPI-level parallelism. `wm` passes back a reference to
working memory.
"""
sort_into_targets!(target, w, stats) =  w, target, stats
# default serial (single thread, no MPI) version: don't copy just swap

combine_stats(stats) = sum(stats)
combine_stats(stats::SArray) = stats # special case for fciqmc_step!() using ThreadsX

function sort_into_targets!(target, ws::NTuple{NT,W}, statss) where {NT,W}
    # multi-threaded non-MPI version
    zero!(target)
    for w in ws # combine new walkers generated from different threads
        add!(target, w)
    end
    return target, ws, combine_stats(statss)
end
# three argument version for MPIData to be found in RMPI.jl
