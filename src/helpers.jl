# small functions supporting fciqmc!()
# versions without dependence on MPI.jl

using Base.Threads: nthreads

localpart(dv) = dv # default for local data

threadedWorkingMemory(dv) = threadedWorkingMemory(localpart(dv))
function threadedWorkingMemory(v::AbstractDVec)
    cws = capacity(v) ÷ nthreads() + 1
    return Tuple(similar(v, cws) for _ in 1:nthreads())
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


"""
    walkernumber(w)
Compute the number of walkers in `w`. In most cases this is identical to
`norm(w,1)`. For coefficient vectors with
`StochasticStyle(w) == IsStochastic2Pop` it reports the one norm
separately for the real and the imaginary part as a `ComplexF64`.
"""
walkernumber(w) = norm(w,1) # generic fallback
# use StochasticStyle trait for dispatch
walkernumber(w::AbstractDVec) = walkernumber(StochasticStyle(w), w)
walkernumber(::StochasticStyle, w) = norm(w,1)
# for AbstractDVec with complex walkers
function walkernumber(::T, w) where T <: Union{DictVectors.IsStochastic2Pop,
                                               DictVectors.IsStochastic2PopInitiator,
                                               DictVectors.IsStochastic2PopWithThreshold
                                               }
    return isempty(w) ? 0.0+0.0im : sum(p->abs(real(p)) + abs(imag(p))*im, w)|>ComplexF64
end
