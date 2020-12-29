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

# function setup_lomc(H::Type; n =6, m = 6, targetwalkers = )

# function ini_state_vector(address, nwalkers, capacity, Style)
#     if Style ∈ Union{IsDeterministic,IsStochasticWithThreshold,IsSemistochastic}
#         nwalkers /= 1 # make it floating point
#     end
#     dv = DVec(Dict(address=>nwalkers), capacity)
#     StochasticStyle(::Type{typeof(dv)}) = Style(1.0)

function purge_negative_walkers!(w::AbstractDVec{K,V}) where {K,V <:Real}
    for (k,v) in pairs(w)
        if v < 0
            delete!(w,k)
        end
    end
    return w
end
function purge_negative_walkers!(w::AbstractDVec{K,V}) where {K,V <:Complex}
    for (k,v) in pairs(w)
        if real(v) < 0
            v = 0 + im*imag(v)
        end
        if imag(v) < 0
            v = real(v)
        end
        w[k] = convert(V,v)
    end
    return w
end

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
function walkernumber(::IsStochastic2Pop, w)
    return isempty(w) ? 0.0+0.0im : mapreduce(p->abs(real(p)) + abs(imag(p))*im, +, w)|>ComplexF64
end
