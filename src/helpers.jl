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
