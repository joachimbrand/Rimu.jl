# second level: `fciqmc_step!()`
# `prep_shift()` - helper function for `fciqmc_step!()`

"""
    prep_shift(::StochasticStyle, shift, pnorm)
Prepare shift according to `StochasticStyle`. Passes through `shift` except for
`IsStochastic2PopRealShift` where a named tuple of suitable values for each walker
population is computed from a complex shift argument.
"""
prep_shift(::StochasticStyle, shift, pnorm) = shift # default
function prep_shift(::DictVectors.IsStochastic2PopRealShift, shift, pnorm)
    return if abs(real(pnorm)*imag(pnorm)) > 1
        (
        r = real(shift) - imag(shift)*imag(pnorm)/real(pnorm),
        i = real(shift) + imag(shift)*real(pnorm)/imag(pnorm),
        )
    else # ignore imaginary part of shift if population is small to avoid division by zero
        (r = real(shift), i = real(shift))
    end
end # returns named tuple of real (Float64) numbers
function prep_shift(iss::DictVectors.IsStochastic2PopRealShiftScaled, shift, pnorm)
    return if abs(real(pnorm)*imag(pnorm)) > 1
        (
        r = real(shift) - imag(shift)*imag(pnorm)/real(pnorm)*iss.scale,
        i = real(shift) + imag(shift)*real(pnorm)/imag(pnorm)*iss.scale,
        )
    else # ignore imaginary part of shift if population is small to avoid division by zero
        (r = real(shift), i = real(shift))
    end
end # returns named tuple of real (Float64) numbers


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
    empty!(w) # clear working memory
    shift = prep_shift(StochasticStyle(v), shift, pnorm)
    # call fciqmc_col!() on every entry of `v` and add the stats returned by
    # this function:
    stats = allocate_statss(v, nothing)
    for (k, v) in pairs(v)
        stats += SVector(fciqmc_col!(w, Ĥ, k, v, shift, dτ))
    end
    r = apply_memory_noise!(w, v, shift, dτ, pnorm, m_strat) # memory noise
    return (sort_into_targets!(dv, w, stats)... , r) # MPI syncronizing
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

# Provide allocation of `statss` array for multithreading as a separate function in order to
# achive type stability.
# `statss` is a Vector with `nt` slots, each collecting the `stats` returned
# by `fciqmc_col`.
# `nt` is the number of threads such that each thread can accummulate data
# avoiding race conditions
allocate_statss(v,nt) = allocate_statss(StochasticStyle(v), v, nt)
allocate_statss(::StochasticStyle, v, nt) = [zeros(Int,5) for i=1:nt]
function allocate_statss(::SS, v, nt) where SS <: Union{DictVectors.IsStochastic2Pop,
    DictVectors.IsStochastic2PopInitiator,DictVectors.IsStochastic2PopWithThreshold
}
    return [zeros(Complex{Int},5) for i=1:nt]
end
# Nothing signals no threading is used
allocate_statss(::StochasticStyle, _, ::Nothing) = SVector(0, 0, 0, 0, 0)
function allocate_statss(::SS, v, ::Nothing) where SS <: Union{DictVectors.IsStochastic2Pop,
    DictVectors.IsStochastic2PopInitiator,DictVectors.IsStochastic2PopWithThreshold
}
    return zeros(SVector{5,Complex{Int}})
end

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
    statss = allocate_statss(v, NT) # [zeros(Int,5) for i=1:NT]
    # [zeros(valtype(v), 5), for i=1:NT] # pre-allocate array for stats
    zero!.(ws) # clear working memory
    shift = prep_shift(StochasticStyle(v), shift, pnorm)
    @sync for btr in Iterators.partition(pairs(v), batchsize)
        Threads.@spawn for (add, num) in btr
            statss[Threads.threadid()] .+= fciqmc_col!(ws[Threads.threadid()], Ĥ, add, num, shift, dτ)
        end
    end # all threads have returned; now running on single thread again
    r = apply_memory_noise!(ws, v, shift, dτ, pnorm, m_strat) # memory noise
    return (sort_into_targets!(dv, ws, statss)... , r) # MPI syncronizing
end # fciqmc_step!

import SplittablesBase.halve, SplittablesBase.amount
import Base.Threads.@spawn, Base.Threads.nthreads, Base.Threads.threadid

# This version seems to be faster but have slightly more allocations
function fciqmc_step!(Ĥ, dv, shift, dτ, pnorm, ws::NTuple{NT,W}, f::Float64;
      m_strat::MemoryStrategy = NoMemory()
    ) where {NT,W}
    # multithreaded version; should also work with MPI
    @assert NT == nthreads() "`nthreads()` not matching dimension of `ws`"
    v = localpart(dv)
    statss = allocate_statss(v, NT)

    batchsize = max(100.0, min(amount(pairs(v))/NT, sqrt(amount(pairs(v)))*10))
    shift = prep_shift(StochasticStyle(v), shift, pnorm)

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
                # @show threadid(), ss
                statss[threadid()] .+= ss
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
    return (sort_into_targets!(dv, ws, statss)... , r) # MPI syncronizing
    #
    # return statss
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
    shift = prep_shift(StochasticStyle(v), shift, pnorm)
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
    shift = prep_shift(StochasticStyle(v), shift, pnorm)

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
