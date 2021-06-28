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
    MultiScalar

Wrapper over a tuple that supports `+`, `-`, `min`, and `max`. Used with MPI communication
because `SVector`s are treated as arrays by `MPI.Allreduce` and `Tuples` do not support
scalar operations.

# Example

Suppose you want to compute the sum of a vector `dv` and also get the number of positive
elements it has in a single pass. You can use `MultiScalar`:

```julia
julia> dv = DVec(:a => 1, :b => -2, :c => 1);

julia> s, p = mapreduce(+, values(dv)) do v
    Rimu.MultiScalar(v, Int(sign(v) == 1))
end;

julia> s, p
(0, 2)
```

This will work with `MPIData`.

Note that only `MultiScalar`s with the same types can be operated on. This is a feature, as it
forces type stability.
"""
struct MultiScalar{T<:Tuple}
    tuple::T
end
MultiScalar(args...) = MultiScalar(args)
MultiScalar(v::SVector) = MultiScalar(Tuple(v))
MultiScalar(arg) = MultiScalar((arg,))

for op in (:+, :*, :max, :min)
    @eval function Base.$op(a::MultiScalar{T}, b::MultiScalar{T}) where {T}
        return MultiScalar($op.(a.tuple, b.tuple))
    end
end

Base.iterate(m::MultiScalar, args...) = iterate(m.tuple, args...)


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
combine_stats(stats::MultiScalar) = stats # special case for fciqmc_step!() using ThreadsX

function sort_into_targets!(target, ws::NTuple{NT,W}, statss) where {NT,W}
    # multi-threaded non-MPI version
    zero!(target)
    for w in ws # combine new walkers generated from different threads
        add!(target, w)
    end
    return target, ws, combine_stats(statss)
end
# three argument version for MPIData to be found in RMPI.jl
