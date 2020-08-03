# small functions supporting fciqmc!()
# versions without dependence on MPI.jl

using Base.Threads: nthreads

localpart(dv) = dv # default for local data

function threadedWorkingMemory(dv)
    v = localpart(dv)
    cws = capacity(v)÷nthreads()+1
    return Tuple(similar(v,cws) for i=1:nthreads())
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

function sort_into_targets!(target, ws::NTuple{NT,W}, statss) where {NT,W}
    # multi-threaded non-MPI version
    empty!(target)
    for w in ws # combine new walkers generated from different threads
        add!(target, w)
    end
    return target, ws, sum(statss)
end
# three argument version for MPIData to be found in mpi_helpers.jl
