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

combine_stats(stats) = sum(stats)
combine_stats(stats::SArray) = stats # special case for fciqmc_step!() using ThreadsX

function sort_into_targets!(target, ws::NTuple{NT,W}, statss) where {NT,W}
    # multi-threaded non-MPI version
    empty!(target)
    for w in ws # combine new walkers generated from different threads
        add!(target, w)
    end
    return target, ws, combine_stats(statss)
end
# three argument version for MPIData to be found in RMPI.jl


"""
    walkernumber(w)
Compute the number of walkers in `w`. It is used for updating the shift.
Overload this function for modifying population control.

In most cases `walkernumber(w)` is identical to
`norm(w,1)`. For `AbstractDVec`s with complex coefficients
it reports the one norm
separately for the real and the imaginary part as a `ComplexF64`.
See [`Norm1ProjectorPPop`](@ref).
"""
walkernumber(w) = walkernumber(StochasticStyle(w), w)
# use StochasticStyle trait for dispatch
# the following default is fast and generic enough to be good for real walkers and
# complex walkers as two populations
# For `MPIData` the dot product returns a reduction from all ranks # MPI sync
walkernumber(::StochasticStyle, w) = Norm1ProjectorPPop() ⋅ w
# Here just sum the numbers
walkernumber(::DictVectors.IsStochastic2PopStoquastic, w) = float(UniformProjector() ⋅ w)
