# second level: `fciqmc_step!()`


"""
    fciqmc_step!(Ĥ, v, shift, dτ, pnorm, w;
                          m_strat::MemoryStrategy = NoMemory()) -> ṽ, w̃, stats
Perform a single matrix(/operator)-vector multiplication:
```math
\\tilde{v} = [1 - dτ(\\hat{H} - S)]⋅v ,
```
where `Ĥ == ham` and `S == shift`. Whether the operation is performed in
stochastic, semistochastic, or determistic way is controlled by the trait
`StochasticStyle(w)`. See [`StochasticStyle`](@ref). `w` is a local data
structure with the same size and type as `v` and used for working. Both `v` and
`w` are modified.

Returns the result `ṽ`, a (possibly changed) reference to working memory `w̃`,
 and the array
`stats = [spawns, deaths, clones, antiparticles, annihilations]`. Stats will
contain zeros when running in deterministic mode.
"""
function fciqmc_step!(Ĥ, dv, shift, dτ, pnorm, w, m = 1.0;
                      m_strat::MemoryStrategy = NoMemory())
    # single-threaded version suitable for MPI
    # m is an optional dummy argument that can be removed later
    v = localpart(dv)
    @assert w ≢ v "`w` and `v` must not be the same object"
    zero!(w) # clear working memory
    # call fciqmc_col!() on every entry of `v` and add the stats returned by
    # this function:
    stat_names, stats = step_stats(v, Val(1))
    for (k, v) in pairs(v)
        stats += SVector(fciqmc_col!(w, Ĥ, k, v, shift, dτ))
    end
    r = apply_memory_noise!(w, v, shift, dτ, pnorm, m_strat) # memory noise
    return (sort_into_targets!(dv, w, stats)..., stat_names, r) # MPI syncronizing
    # stats == [spawns, deaths, clones, antiparticles, annihilations]
end # fciqmc_step!

# function fciqmc_step!(Ĥ, v::D, shift, dτ, pnorm, w::D;
#                       m_strat::MemoryStrategy = NoMemory()) where D
#     # serial version
#     @assert w ≢ v "`w` and `v` must not be the same object"
#     zero!(w) # clear working memory
#     # call fciqmc_col!() on every entry of `v` and add the stats returned by
#     # this function:
#     stats = mapreduce(p-> SVector(fciqmc_col!(w, Ĥ, p.first, p.second, shift, dτ)), +,
#       pairs(v))
#     r = apply_memory_noise!(w, v, shift, dτ, pnorm, m_strat) # memory noise
#     return w, v, stats, r
#     # stats == [spawns, deaths, clones, antiparticles, annihilations]
# end # fciqmc_step!

# Below follow multiple implementations of `fciqmc_step!` using multithreading.
# This is for testing purposes only and eventually all but one should be removed.
# The active version is selected by dispatch on the 7th positional argument
# and by modifying the call from fciqmc!() in line 294.

# previous default but slower than the other versions
function fciqmc_step!(Ĥ, dv, shift, dτ, pnorm, ws::NTuple{NT,W};
      m_strat::MemoryStrategy = NoMemory(),
      batchsize = max(100, min(length(dv)÷Threads.nthreads(), round(Int,sqrt(length(dv))*10)))
    ) where {NT,W}
    # println("batchsize ",batchsize)
    # multithreaded version; should also work with MPI
    @assert NT == Threads.nthreads() "`nthreads()` not matching dimension of `ws`"
    v = localpart(dv)
    stat_names, stats = step_stats(v, Val(NT))
    # [zeros(valtype(v), 5), for i=1:NT] # pre-allocate array for stats
    zero!.(ws) # clear working memory
    @sync for btr in Iterators.partition(pairs(v), batchsize)
        Threads.@spawn for (add, num) in btr
            stats[Threads.threadid()] .+= fciqmc_col!(ws[Threads.threadid()], Ĥ, add, num, shift, dτ)
        end
    end # all threads have returned; now running on single thread again
    r = apply_memory_noise!(ws, v, shift, dτ, pnorm, m_strat) # memory noise
    return (sort_into_targets!(dv, ws, stats)..., stat_names, r) # MPI syncronizing
end # fciqmc_step!

import SplittablesBase.halve, SplittablesBase.amount
import Base.Threads.@spawn, Base.Threads.nthreads, Base.Threads.threadid

# This version seems to be faster but have slightly more allocations
function fciqmc_step!(Ĥ, dv, shift, dτ, pnorm, ws::NTuple{NT,W}, f::Float64;
      m_strat::MemoryStrategy = NoMemory()
    ) where {NT,W}
    # multithreaded version; should also work with MPI
    @assert NT == nthreads() "`nthreads()` not matching dimension of `ws`"
    @assert NT > 1 "attempted to run threaded code with one thread"
    v = localpart(dv)
    stat_names, stats = step_stats(v, Val(NT))

    batchsize = max(100.0, min(amount(pairs(v))/NT, sqrt(amount(pairs(v)))*10))

    # define recursive dispatch function that loops two halves in parallel
    function loop_configs!(ps) # recursively spawn threads
        if amount(ps) > batchsize
            two_halves = halve(ps) #
            fh = @spawn loop_configs!(two_halves[1]) # runs in parallel
            loop_configs!(two_halves[2])           # with second half
            wait(fh)                             # wait for fist half to finish
        else # run serial
            # id = threadid() # specialise to which thread we are running on here
            # serial_loop_configs!(ps, ws[id], statss[id], trng())
            for (add, num) in ps
                ss = fciqmc_col!(ws[threadid()], Ĥ, add, num, shift, dτ)
                stats[threadid()] += SVector(ss)
            end
        end
        return nothing
    end

    # function serial_loop_configs!(ps, w, rng, stats)
    #     @inbounds for (add, num) in ps
    #         ss = fciqmc_col!(w, Ĥ, add, num, shift, dτ, rng)
    #         # @show threadid(), ss
    #         stats .+= ss
    #     end
    #     return nothing
    # end

    zero!.(ws) # clear working memory
    loop_configs!(pairs(v))

    r = apply_memory_noise!(ws, v, shift, dτ, pnorm, m_strat) # memory noise
    return (sort_into_targets!(dv, ws, stats)... , stat_names, r) # MPI syncronizing
end

# using ThreadsX
# new attempt at threaded version: This one is type-unstable but has
# the lowest memory allocations for large walker numbers and is fast
function fciqmc_step!(Ĥ, dv, shift, dτ, pnorm, ws::NTuple{NT,W}, f::Bool;
      m_strat::MemoryStrategy = NoMemory()
    ) where {NT,W}
    # multithreaded version; should also work with MPI
    @assert NT == Threads.nthreads() "`nthreads()` not matching dimension of `ws`"
    v = localpart(dv)
    # statss = [zeros(Int,5) for i=1:NT]
    # [zeros(valtype(v), 5), for i=1:NT] # pre-allocate array for stats
    zero!.(ws) # clear working memory
    # stats = mapreduce(p-> SVector(fciqmc_col!(ws[Threads.threadid()], Ĥ, p.first, p.second, shift, dτ)), +,
    #   pairs(v))

    stats = ThreadsX.sum(SVector(fciqmc_col!(ws[Threads.threadid()], Ĥ, p.first, p.second, shift, dτ)) for p in pairs(v))
    # return ws, stats
    r = apply_memory_noise!(ws, v, shift, dτ, pnorm, m_strat) # memory noise
    return (sort_into_targets!(dv, ws, stats)... , r) # MPI syncronizing
end # fciqmc_step!

# new attempt at threaded version: type stable but slower and more allocs
function fciqmc_step!(Ĥ, dv, shift, dτ, pnorm, ws::NTuple{NT,W}, f::Int;
      m_strat::MemoryStrategy = NoMemory()
    ) where {NT,W}
    # multithreaded version; should also work with MPI
    @assert NT == Threads.nthreads() "`nthreads()` not matching dimension of `ws`"
    v = localpart(dv)

    statss = allocate_statss(v, NT)
    zero!.(ws) # clear working memory

    function col!(p) # take a pair address -> value and run `fciqmc_col!()` on it
        statss[threadid()] .+= fciqmc_col!(ws[threadid()], Ĥ, p.first, p.second, shift, dτ)
        return nothing
    end

    # parallel execution happens here:
    ThreadsX.map(col!, pairs(v))

    # return ws, stats
    r = apply_memory_noise!(ws, v, shift, dτ, pnorm, m_strat) # memory noise
    return (sort_into_targets!(dv, ws, statss)... , r) # MPI syncronizing
end # fciqmc_step!

#  ## old version for single-thread MPI. No longer needed
# function Rimu.fciqmc_step!(Ĥ, dv::MPIData{D,S}, shift, dτ, pnorm, w::D;
#                            m_strat::MemoryStrategy = NoMemory()) where {D,S}
#     # MPI version, single thread
#     v = localpart(dv)
#     @assert w ≢ v "`w` and `v` must not be the same object"
#     empty!(w)
#     stats = zeros(Int, 5) # pre-allocate array for stats
#     for (add, num) in pairs(v)
#         res = Rimu.fciqmc_col!(w, Ĥ, add, num, shift, dτ)
#         stats .+= res # just add all stats together
#     end
#     r = apply_memory_noise!(w, v, shift, dτ, pnorm, m_strat) # memory noise
#     # thresholdProject!(w, v, shift, dτ, m_strat) # apply walker threshold if applicable
#     sort_into_targets!(dv, w)
#     MPI.Allreduce!(stats, +, dv.comm) # add stats of all ranks
#     return dv, w, stats, r
#     # returns the structure with the correctly distributed end
#     # result `dv` and cumulative `stats` as an array on all ranks
#     # stats == (spawns, deaths, clones, antiparticles, annihilations)
# end # fciqmc_step!
