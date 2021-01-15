

"""
    lomc!(ham, v; kwargs...)
    -> nt::NamedTuple

Linear operator Monte Carlo:
Perform the FCIQMC algorithm for determining the lowest eigenvalue of `ham`.
`v` can be a single starting vector of (wrapped) type `:<AbstractDVec`,
or a vector of such structures for a replica simulation.
Returns a named tuple containg all information required for continuation runs.
In particular, `nt.df` is a `DataFrame` with statistics about the run, or a
tuple of `DataFrame`s for a replica run.

### Keyword arguments, defaults, and precedence:
* `laststep` - can be used to override information otherwise contained in `params`
* `threading = :auto` - can be used to control the use of multithreading (overridden by `wm`)
  * `:auto` - use multithreading if `s_strat.targetwalkers ≥ 500`
  * `true` - use multithreading if available (set shell variable `JULIA_NUM_THREADS`!)
  * `false` - run on single thread
* `wm` - working memory; if set, it controls the use of multithreading and overrides `threading`; is mutated
* `params::FciqmcRunStrategy = RunTillLastStep(laststep = 100)` - contains basic parameters of simulation state, see [`FciqmcRunStrategy`](@ref); is mutated
* `s_strat::ShiftStrategy = DoubleLogUpdate(targetwalkers = 1000)` - see [`ShiftStrategy`](@ref)
* `r_strat::ReportingStrategy = EveryTimeStep()` - see [`ReportingStrategy`](@ref)
* `τ_strat::TimeStepStrategy = ConstantTimeStep()` - see [`TimeStepStrategy`](@ref)
* `m_strat::MemoryStrategy = NoMemory()` - see [`MemoryStrategy`](@ref)
* `p_strat::ProjectStrategy = NoProjection()` - see [`ProjectStrategy`](@ref)

### Return values
```julia
nt = lomc!(args...)
```
The named tuple `nt` contains the following fields:
```julia
nt = (
    ham = ham, # the linear operator, from input
    v = v, # the current coefficient vector, mutated from input
    params = params, # struct with state parameters, mutated from input
    df = df, # DataFrame with statistics per time step
    wm = wm, # working memory, mutated from input
    s_strat = s_strat, # from input
    r_strat = r_strat, # from input
    τ_strat = τ_strat, # from input
    m_strat = m_strat, # from input
    p_strat = p_strat, # from input
)
```
"""
function lomc!(ham, v;
    laststep = nothing,
    threading = :auto,
    df = nothing,
    wm = nothing,
    params::FciqmcRunStrategy{T} = RunTillLastStep(),
    s_strat::ShiftStrategy = DoubleLogUpdate(),
    r_strat::ReportingStrategy = EveryTimeStep(),
    τ_strat::TimeStepStrategy = ConstantTimeStep(),
    m_strat::MemoryStrategy = NoMemory(),
    p_strat::ProjectStrategy = NoProjection()
) where T # type for shift and walkernumber
    r_strat = refine_r_strat(r_strat, ham)
    if !isnothing(laststep)
        params.laststep = laststep
    end
    if isnothing(wm)
        if threading == :auto
            threading = max(real(s_strat.targetwalkers),imag(s_strat.targetwalkers)) ≥ 500 ? true : false
        end
        # now threading is a Bool
        if threading
            wm = threadedWorkingMemory(v)
        else
            wm = similar(localpart(v))
        end
    end
    if isnothing(df)
        # unpack the parameters:
        @unpack step, laststep, shiftMode, shift, dτ = params
        len = length(v) # MPIsync
        nor = T(walkernumber(v)) # MPIsync
        v_proj, h_proj = compute_proj_observables(v, ham, r_strat) # MPIsync

        # just for getting the type of step_statsd do a step with zero time:
        vd, wd, step_statsd, rd = fciqmc_step!(ham, copytight(localpart(v)),
                                            shift, 0.0, nor,
                                            wm; m_strat=m_strat)
        TS = eltype(step_statsd)
        # prepare df for recording data
        df = DataFrame(steps=Int[], dτ=Float64[], shift=T[],
                            shiftMode=Bool[],len=Int[], norm=T[],
                            vproj=typeof(v_proj)[], hproj=typeof(h_proj)[],
                            spawns=TS[], deaths=TS[], clones=TS[],
                            antiparticles=TS[], annihilations=TS[],
                            shiftnoise=Float64[])
        # Note the row structure defined here (currently 13 columns)
        # When changing the structure of `df`, it has to be changed in all places
        # where data is pushed into `df`.
        @assert Symbol.(names(df)) == [:steps, :dτ, :shift, :shiftMode, :len, :norm,
                                :vproj, :hproj,
                                :spawns, :deaths, :clones, :antiparticles,
                                :annihilations, :shiftnoise
                             ] "Column names in `df` not as expected."
        # Push first row of df to show starting point
        push!(df, (step, dτ, shift, shiftMode, len, nor, v_proj, h_proj,
                    0, 0, 0, 0, 0, 0.0))
    end
    # set up the named tuple of lomc!() return values
    # nt = (;ham, v, params, df, wm, s_strat, r_strat, τ_strat, m_strat, p_strat)
    nt = (
        ham = ham,
        v = v,
        params = params,
        df = df,
        wm = wm,
        s_strat = s_strat,
        r_strat = r_strat,
        τ_strat = τ_strat,
        m_strat = m_strat,
        p_strat = p_strat,
    )
    return lomc!(nt) # call lomc!() with prepared arguments parameters
end

"""
    lomc!(nt::NamedTuple, laststep::Int = nt.params.laststep)
Linear operator Monte Carlo:
Call signature for a continuation run.

`nt` should have the same structure as the return value of `lomc!()`.
The optional argument `laststep` can be used to set a new last step.
If `laststep > nt.params.step`, additional time steps will be computed and
the statistics in the `DataFrame` `nt.df` will be appended.
"""
function lomc!(a::NamedTuple) # should be type stable
    @unpack ham, v, params, df, wm, s_strat, r_strat, τ_strat, m_strat, p_strat = a
    ConsistentRNG.check_crng_independence(v) # sanity check of RNGs

    rr_strat = refine_r_strat(r_strat, ham) # set up r_strat for fciqmc!()

    fciqmc!(v, params, df, ham, s_strat, rr_strat, τ_strat, wm;
        m_strat = m_strat, p_strat = p_strat
    )
    nt = (a..., r_strat = rr_strat)
    return nt
end

function lomc!(a::NamedTuple, laststep::Int) # should be type stable
    a.params.laststep = laststep
    return lomc!(a)
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
        return @set r_strat.hproj = copytight(ham'*r_strat.projector)
    elseif r_strat.hproj == :auto
        if P1  <: AbstractProjector # for projectors don't compute `df.hproj`
            return @set r_strat.hproj = nothing
        elseif Hamiltonians.LOStructure(ham) == Hamiltonians.HermitianLO() # eager is possible
            hpv = ham'*r_strat.projector # pre-calculate left vector with adjoint Hamiltonian
            # use smaller container to save memory
            return @set r_strat.hproj = copytight(hpv)
        else # lazy is default
            return @set r_strat.hproj = missing
        end
    end
    @error "Value $(r_strat.hproj) for keyword `hproj` is not recognized. See documentation of [`ReportingStrategy`](@doc)."
end


"""
    fciqmc!(v, pa::FciqmcRunStrategy, [df,]
             ham, s_strat::ShiftStrategy,
             [r_strat::ReportingStrategy, τ_strat::TimeStepStrategy, w])
    -> df

Perform the FCIQMC algorithm for determining the lowest eigenvalue of `ham`.
`v` can be a single starting vector of type `:<AbstractDVec` or a vector
of such structures. In the latter case, independent replicas are constructed.
Returns a `DataFrame` `df` with statistics about the run, or a tuple of `DataFrame`s
for a replica run.
Strategies can be given for updating the shift (see [`ShiftStrategy`](@ref))
and (optionally), for reporting (see [`ReportingStrategy`](@ref)),
and for updating the time step `dτ` (see [`TimeStepStrategy`](@ref)).

A pre-allocated data structure `w` for working memory can be passed as argument,
and controls multi-threading behaviour. By default multi-threading is turned
on. To turn multi-threading off, pass `similar(localpart(v))` for w.

This function mutates `v`, the parameter struct `pa` as well as
`df`, and `w`.

NOTE: The function `fciqmc!()` may be deprecated soon. Change all scripts to
call `lomc!()` instead!
"""
function fciqmc!(svec, pa::FciqmcRunStrategy{T},
                 ham,
                 s_strat::ShiftStrategy,
                 r_strat::ReportingStrategy = EveryTimeStep(),
                 τ_strat::TimeStepStrategy = ConstantTimeStep(),
                 w = threadedWorkingMemory(svec); kwargs...) where T
    # unpack the parameters:
    @unpack step, laststep, shiftMode, shift, dτ = pa
    len = length(svec) # MPIsync
    nor = T(walkernumber(svec)) # MPIsync

    # should not be necessary if we do all calls from lomc!()
    r_strat = refine_r_strat(r_strat, ham)

    v_proj, h_proj = compute_proj_observables(svec, ham, r_strat) # MPIsync

    # prepare df for recording data
    df = DataFrame(steps=Int[], dτ=Float64[], shift=T[],
                        shiftMode=Bool[],len=Int[], norm=T[],
                        vproj=typeof(v_proj)[], hproj=typeof(h_proj)[],
                        spawns=Int[], deaths=Int[], clones=Int[],
                        antiparticles=Int[], annihilations=Int[],
                        shiftnoise=Float64[])
    # Note the row structure defined here (currently 13 columns)
    # When changing the structure of `df`, it has to be changed in all places
    # where data is pushed into `df`.
    @assert Symbol.(names(df)) == [:steps, :dτ, :shift, :shiftMode, :len, :norm,
                            :vproj, :hproj,
                            :spawns, :deaths, :clones, :antiparticles,
                            :annihilations, :shiftnoise
                         ] "Column names in `df` not as expected."
    # Push first row of df to show starting point
    push!(df, (step, dτ, shift, shiftMode, len, nor, v_proj, h_proj,
                0, 0, 0, 0, 0, 0.0))
    # println("DataFrame is set up")
    # # (DD <: MPIData) && println("$(svec.s.id): arrived at barrier; before")
    # (DD <: MPIData) && MPI.Barrier(svec.s.comm)
    # # println("after barrier")
    rdf =  fciqmc!(svec, pa, df, ham, s_strat, r_strat, τ_strat, w
                    # )
                    ; kwargs...)
    # # (DD <: MPIData) && println("$(svec.s.id): arrived at barrier; after")
    # (DD <: MPIData) && MPI.Barrier(svec.s.comm)
    return rdf
end

# for continuation runs we can also pass a DataFrame
function fciqmc!(v, pa::RunTillLastStep{T}, df::DataFrame,
                 ham,
                 s_strat::ShiftStrategy,
                 r_strat::ReportingStrategy = EveryTimeStep(),
                 τ_strat::TimeStepStrategy = ConstantTimeStep(),
                 w = threadedWorkingMemory(v)
                 ; m_strat::MemoryStrategy = NoMemory(),
                 p_strat::ProjectStrategy = NoProjection()
                 ) where T # type for shift and walkernumber
    # unpack the parameters:
    @unpack step, laststep, shiftMode, shift, dτ = pa

    # should not be necessary if we do all calls from lomc!()
    r_strat = refine_r_strat(r_strat, ham)

    # check `df` for consistency
    @assert Symbol.(names(df)) == [:steps, :dτ, :shift, :shiftMode, :len, :norm,
                            :vproj, :hproj,
                            :spawns, :deaths, :clones, :antiparticles,
                            :annihilations, :shiftnoise
                         ] "Column names in `df` not as expected."

    svec = v # keep around a reference to the starting data container
    pnorm = tnorm = T(walkernumber(v)) # norm of "previous" vector
    maxlength = capacity(localpart(v))
    @assert maxlength ≤ capacity(w) "`w` needs to have at least `capacity(v)`"

    while step < laststep
        step += 1
        # println("Step: ",step)
        # perform one complete stochastic vector matrix multiplication
        v, w, step_stats, r = fciqmc_step!(ham, v, shift, dτ, pnorm, w
                                            , 1.0 # for selecting multithreaded version
                                            ; m_strat=m_strat)
        tnorm = norm_project!(p_strat, v, shift, pnorm, dτ) |> T  # MPIsync
        # project coefficients of `w` to threshold

        v_proj, h_proj = compute_proj_observables(v, ham, r_strat)  # MPIsync

        # update shift and mode if necessary
        shift, shiftMode, pnorm = update_shift(s_strat,
                                    shift, shiftMode,
                                    tnorm, pnorm, dτ, step, df, v, w)
        # the updated "previous" norm pnorm is returned from `update_shift()`
        # in order to allow delaying the update, e.g. with `DelayedLogUpdate`
        # pnorm = tnorm # remember norm of this step for next step (previous norm)
        dτ = update_dτ(τ_strat, dτ, tnorm) # will need to pass more information later
        # when we add different stratgies
        len = length(v) # MPI sycncronising: total number of configs
        # record results according to ReportingStrategy r_strat
        report!(df, (step, dτ, shift, shiftMode, len, tnorm, v_proj, h_proj,
                        step_stats..., r), r_strat)
        # DF ≠ Nothing && push!(df, (step, dτ, shift, shiftMode, len, tnorm,
        #                 step_stats...))
        # housekeeping: avoid overflow of dvecs
        len_local = length(localpart(v))
        len_local > 0.8*maxlength && if len_local > maxlength
            @error "`maxlength` exceeded" len_local maxlength
            break
        else
            @warn "`maxlength` nearly reached" len_local maxlength
        end
    end
    # make sure that `svec` contains the current population:
    if !(v === svec)
        copy!(svec, v)
    end
    # pack up parameters for continuation runs
    # note that this modifes the struct pa
    @pack! pa = step, shiftMode, shift, dτ
    return  df
    # note that `svec` and `pa` are modified but not returned explicitly
end # fciqmc

# replica version
function fciqmc!(vv::Vector, pa::RunTillLastStep{T}, ham::AbstractHamiltonian,
                 s_strat::ShiftStrategy,
                 r_strat::ReportingStrategy = EveryTimeStep(),
                 τ_strat::TimeStepStrategy = ConstantTimeStep(),
                 wv = threadedWorkingMemory.(vv) # wv = similar.(localpart.(vv))
                 ; m_strat::MemoryStrategy = NoMemory(),
                 p_strat::ProjectStrategy = NoProjection()) where T
    # τ_strat is currently ignored in the replica version
    # unpack the parameters:
    @unpack step, laststep, shiftMode, shift, dτ = pa
    V = valtype(vv[1])
    N = length(vv) # number of replicas to propagate
    # keep references to the passed data vectors around
    vv_orig = similar(vv) # empty vector
    vv_orig .= vv # fill it with references to the coefficient DVecs

    # should not be necessary if we do all calls from lomc!()
    r_strat = refine_r_strat(r_strat, ham)

    maxlength = minimum(capacity.(vv))
    reduce(&, capacity.(wv) .≥ maxlength) || error("replica containers `wv` have insufficient capacity")

    shifts = [shift for i = 1:N] # Vector because it needs to be mutable
    vShiftModes = [shiftMode for i = 1:N] # separate for each replica
    pnorms = zeros(T, N) # initialise as vector
    pnorms .= walkernumber.(vv) # 1-norm i.e. number of psips as Tuple (of previous step)

    # initalise df for storing results of each replica separately
    dfs = Tuple(DataFrame(steps=Int[], shift=T[], shiftMode=Bool[],
                         len=Int[], norm=T[], spawns=V[], deaths=V[],
                         clones=V[], antiparticles=V[],
                         annihilations=V[], shiftnoise=Float64[]) for i in 1:N)
    # dfs is thus an NTuple of DataFrames
    for i in 1:N
        push!(dfs[i], (step, shifts[i], vShiftModes[i], length(vv[i]),
                      pnorms[i], 0, 0, 0, 0, 0, 0.0))
    end

    # prepare `DataFrame` for variational ground state estimator
    # we are assuming that N ≥ 2, otherwise this will fail
    PType = promote_type(V,eltype(ham)) # type of scalar product
    RType = promote_type(PType,Float64) # for division
    mixed_df= DataFrame(steps =Int[], xdoty =V[], xHy =PType[], aveH =RType[])
    dp = vv[1]⋅vv[2] # <v_1 | v_2>
    expval = dot(vv[1], ham, vv[2]) # vv[1]⋅ham(vv[2]) # <v_1 | ham | v_2>
    push!(mixed_df,(step, dp, expval, expval/dp))

    norms = zeros(T, N)
    mstats = [zeros(Int,5) for i=1:N]
    rs = zeros(N)
    while step < laststep
        step += 1
        for (i, v) in enumerate(vv) # loop over replicas
            # perform one complete stochastic vector matrix multiplication
            vv[i], wv[i], stats, rs[i] = fciqmc_step!(ham, v, shifts[i],
                dτ, pnorms[i], wv[i]; m_strat = m_strat
            )
            mstats[i] .= stats
            norms[i] = norm_project!(p_strat, vv[i], shifts[i], pnorms[i], dτ) |> T # MPIsync
            shifts[i], vShiftModes[i], pnorms[i] = update_shift(
                s_strat, shifts[i], vShiftModes[i],
                norms[i], pnorms[i], dτ, step, dfs[i], vv[i], wv[i]
            )
        end #loop over replicas
        lengths = length.(vv)
        # update time step
        dτ = update_dτ(τ_strat, dτ) # will need to pass more information
        # later when we add different stratgies
        # record results
        for i = 1:N
            push!(dfs[i], (step, shifts[i], vShiftModes[i], lengths[i],
                  norms[i], mstats[i]..., rs[i]))
        end
        v1Dv2 = vv[1]⋅vv[2] # <v_1 | v_2> overlap
        v2Dhv2 =  vv[1]⋅ham(vv[2]) # <v_1 | ham | v_2>
        push!(mixed_df,(step, v1Dv2, v2Dhv2, v2Dhv2/v1Dv2))

        # prepare for next step:
        # pnorms .= norms # remember norm of this step for next step (previous norm)
        llength = maximum(lengths)
        llength > 0.8*maxlength && if llength > maxlength
            @error "`maxlength` exceeded" llength maxlength
            break
        else
            @warn "`maxlength` nearly reached" llength maxlength
        end

    end # while step
    # make sure that `svecs` contains the current population:
    for i = 1:N
        if vv[i] ≢ vv_orig[i]
            copyto!(vv_orig[i], vv[i])
        end
    end
    # pack up and parameters for continuation runs
    # note that this modifes the struct pa
    shiftMode = reduce(&,vShiftModes) # only true if all are in vShiftMode
    shift = reduce(+,shifts)/N # return average value of shift
    @pack! pa = step, shiftMode, shift, dτ

    return mixed_df, dfs # return dataframes with stats
    # note that `vv_orig` and `pa` are modified but not returned explicitly
end # fciqmc

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
    # call fciqmc_col!() on every entry of `v` and add the stats returned by
    # this function:
    stats = mapreduce(p-> SVector(fciqmc_col!(w, Ĥ, p.first, p.second, shift, dτ)), +,
      pairs(v))
    r = applyMemoryNoise!(w, v, shift, dτ, pnorm, m_strat) # memory noise
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
#     r = applyMemoryNoise!(w, v, shift, dτ, pnorm, m_strat) # memory noise
#     return w, v, stats, r
#     # stats == [spawns, deaths, clones, antiparticles, annihilations]
# end # fciqmc_step!

# provide allocation of statss array for multithreading as a separate function in order to
# achive type stability
# `nt` is the number of threads such that each thread can accummulate data
# avoiding race conditions
allocate_statss(v,nt) = allocate_statss(StochasticStyle(v), v, nt)
allocate_statss(::StochasticStyle, v, nt) = [zeros(Int,5) for i=1:nt]
function allocate_statss(::SS, v, nt) where SS <: Union{IsStochastic2Pop,IsStochastic2PopInitiator,IsStochastic2PopWithThreshold}
    return [zeros(Complex{Int},5) for i=1:nt]
end

# Below follow multiple implementations of `fciqmc_step!` using multithreading.
# This is for testing purposes only and eventually all but one should be removed.
# The active version is selected by dispatch on the 7th positional argument
# and by modifying the call from fciqmc!() in line 290.

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
    @sync for btr in Iterators.partition(pairs(v), batchsize)
        Threads.@spawn for (add, num) in btr
            statss[Threads.threadid()] .+= fciqmc_col!(ws[Threads.threadid()], Ĥ, add, num, shift, dτ)
        end
    end # all threads have returned; now running on single thread again
    r = applyMemoryNoise!(ws, v, shift, dτ, pnorm, m_strat) # memory noise
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

    r = applyMemoryNoise!(ws, v, shift, dτ, pnorm, m_strat) # memory noise
    return (sort_into_targets!(dv, ws, statss)... , r) # MPI syncronizing
    #
    # return statss
end

using ThreadsX
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
    r = applyMemoryNoise!(ws, v, shift, dτ, pnorm, m_strat) # memory noise
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
    r = applyMemoryNoise!(ws, v, shift, dτ, pnorm, m_strat) # memory noise
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
#     r = applyMemoryNoise!(w, v, shift, dτ, pnorm, m_strat) # memory noise
#     # thresholdProject!(w, v, shift, dτ, m_strat) # apply walker threshold if applicable
#     sort_into_targets!(dv, w)
#     MPI.Allreduce!(stats, +, dv.comm) # add stats of all ranks
#     return dv, w, stats, r
#     # returns the structure with the correctly distributed end
#     # result `dv` and cumulative `stats` as an array on all ranks
#     # stats == (spawns, deaths, clones, antiparticles, annihilations)
# end # fciqmc_step!


"""
    norm_project!(p_strat::ProjectStrategy, w, shift, pnorm) -> walkernumber
Compute the walkernumber of `w` and update the coefficient vector `w` according to
`p_strat`.

This may include stochastic projection of the coefficients
to `s.threshold` preserving the sign depending on [`StochasticStyle(w)`](@ref)
and `p_strat`. See [`ProjectStrategy`](@ref).
"""
norm_project!(p::ProjectStrategy, w, args...) = norm_project!(StochasticStyle(w), p, w, args...)

norm_project!(::StochasticStyle, p, w, args...) = walkernumber(w) # MPIsync
# default, compute 1-norm
# e.g. triggered with the `NoProjection` strategy

norm_project!(::StochasticStyle, p::NoProjectionTwoNorm, w, args...) = norm(w, 2) # MPIsync
# compute 2-norm but do not perform projection

function norm_project!(s::S, p::ThresholdProject, w, args...) where S<:Union{IsStochasticWithThreshold}
    return norm_project_threshold!(w, p.threshold) # MPIsync
end

function norm_project_threshold!(w, threshold)
    # MPIsync
    # perform projection if below threshold preserving the sign
    lw = localpart(w)
    for (add, val) in kvpairs(lw)
        pprob = abs(val)/threshold
        if pprob < 1 # projection is only necessary if abs(val) < s.threshold
            lw[add] = (pprob > cRand()) ? threshold*sign(val) : zero(val)
        end
    end
    return walkernumber(w) # MPIsync
end

function norm_project_threshold!(w::AbstractDVec{K,V}, threshold) where {K,V<:Union{Integer,Complex{Int}}}
    @error "Trying to scale integer based walker vector. Use float walkers!"
end

function norm_project!(s::S, p::ScaledThresholdProject, w, args...) where S<:Union{IsStochasticWithThreshold}
    f_norm = norm(w, 1) # MPIsync
    proj_norm = norm_project_threshold!(w, p.threshold)
    # MPI sycncronising
    rmul!(localpart(w), f_norm/proj_norm) # scale in order to remedy projection noise
    return f_norm
end

function norm_project!(s::IsStochasticWithThreshold,
                        p::ComplexNoiseCancellation, w,
                        shift::T, pnorm::T, dτ
    ) where T <: Complex
    f_norm = norm(w, 1)::Real # MPIsync
    im_factor = dτ*imag(shift) + p.κ*√dτ*sync_cRandn(w) # MPIsync
    # Wiener increment
    # do we need to synchronize such that we add the same noise on each MPI rank?
    # or thread ? - not thread, as threading is done inside fciqmc_step!()
    scale_factor = 1 - im_factor*imag(pnorm)/f_norm
    rmul!(localpart(w), scale_factor) # scale coefficient vector
    c_im = f_norm/real(pnorm)*imag(pnorm) + im_factor*real(pnorm)
    return complex(f_norm*scale_factor, c_im) |> T # return complex norm
end

function norm_project!(s::StochasticStyle,
                        p::ComplexNoiseCancellation, args...
    )
    throw(ErrorException("`ComplexNoiseCancellation` requires complex shift in `FciqmcRunStrategy` and  `IsStochasticWithThreshold`."))
end

"""
    r = applyMemoryNoise!(w, v, shift, dτ, pnorm, m_strat::MemoryStrategy)
Apply memory noise to `w`, i.e. `w .+= r.*v`, computing the noise `r` according
to `m_strat`. Note that `m_strat`
needs to be compatible with `StochasticStyle(w)`. Otherwise, an
error exception is thrown. See [`MemoryStrategy`](@ref).

`w` is the walker array after fciqmc step, `v` the previous one, `pnorm` the
norm of `v`, and `r` the instantaneously applied noise.
"""
function applyMemoryNoise!(w::Union{AbstractArray{T},AbstractDVec{K,T}},
         v, shift, dτ, pnorm, m
    ) where  {K,T<:Real}
    applyMemoryNoise!(StochasticStyle(w), w, v, real(shift), dτ, real(pnorm), m)
end
# only use real part of the shift and norm if the coefficients are real

# otherwise, pass on complex shift in generic method
function applyMemoryNoise!(w::Union{AbstractArray,AbstractDVec}, args...)
    applyMemoryNoise!(StochasticStyle(w), w, args...)
end

function applyMemoryNoise!(ws::NTuple{NT,W}, args...) where {NT,W}
    applyMemoryNoise!(StochasticStyle(W), ws, args...)
end

function applyMemoryNoise!(s::StochasticStyle, w, v, shift, dτ, pnorm, m::NoMemory)
    return 0.0 # does nothing
end

function applyMemoryNoise!(s::StochasticStyle, w, v, shift, dτ, pnorm, m::MemoryStrategy)
    throw(ErrorException("MemoryStrategy `$(typeof(m))` does not work with StochasticStyle `$(typeof(s))`. Ignoring memory noise for now."))
    # @error "MemoryStrategy `$(typeof(m))` does not work with StochasticStyle `$(typeof(s))`. Ignoring memory noise for now." maxlog=2
    return 0.0 # default prints an error message
end

function applyMemoryNoise!(s::IsStochasticWithThreshold,
                           w, v, shift, dτ, pnorm, m::DeltaMemory)
    tnorm = norm(w, 1) # MPIsync
    # current norm of `w` after FCIQMC step
    # compute memory noise
    r̃ = (pnorm - tnorm)/(dτ*pnorm) + shift
    push!(m.noiseBuffer, r̃) # add current value to buffer
    # Buffer only remembers up to `Δ` values. Average over whole buffer.
    r = r̃ - sum(m.noiseBuffer)/length(m.noiseBuffer)

    # apply `r` noise to current state vector
    axpy!(dτ*r, v, w) # w .+= dτ*r .* v
    # nnorm = norm(w, 1) # new norm after applying noise

    return dτ*r
end

function applyMemoryNoise!(s::IsStochasticWithThreshold,
                           w, v, shift, dτ, pnorm, m::DeltaMemory2)
    tnorm = norm(w, 1) # MPIsync
    # current norm of `w` after FCIQMC step
    # compute memory noise
    r̃ = pnorm - tnorm + shift*dτ*pnorm
    push!(m.noiseBuffer, r̃) # add current value to buffer
    # Buffer only remembers up to `Δ` values. Average over whole buffer.
    r = (r̃ - sum(m.noiseBuffer)/length(m.noiseBuffer))/(dτ*pnorm)

    # apply `r` noise to current state vector
    axpy!(dτ*r, v, w) # w .+= dτ*r .* v
    # nnorm = norm(w, 1) # new norm after applying noise

    return dτ*r
end

function applyMemoryNoise!(s::IsStochasticWithThreshold,
    w, v, shift, dτ, pnorm, m::DeltaMemory3)
tnorm = norm(w, 1) # MPIsync
# current norm of `w` after FCIQMC step
# compute memory noise
r̃ = (pnorm - tnorm)/pnorm + dτ*shift
push!(m.noiseBuffer, r̃) # add current value to buffer
# Buffer only remembers up to `Δ` values. Average over whole buffer.
r = r̃ - sum(m.noiseBuffer)/length(m.noiseBuffer)

# apply `r` noise to current state vector
rmul!(w, 1 + m.level * r) # w = w * (1 + level*r)

return r
end

function applyMemoryNoise!(s::IsStochasticWithThreshold,
                           w, v, shift, dτ, pnorm, m::ShiftMemory)
    push!(m.noiseBuffer, shift) # add current value of `shift` to buffer
    # Buffer only remembers up to `Δ` values. Average over whole buffer.
    r = - shift + sum(m.noiseBuffer)/length(m.noiseBuffer)

    # apply `r` noise to current state vector
    axpy!(dτ*r, v, w) # w .+= dτ*r .* v
    # nnorm = norm(w, 1) # new norm after applying noise

    return dτ*r
end

function applyMemoryNoise!(s::IsStochasticWithThreshold,
                           w, v, shift, dτ, pnorm, m::ProjectedMemory)
    tp = m.projector⋅w # w  may be a tuple for multithreading
    # TODO: make this work with multithreading and MPI

    # current projection of `w` after FCIQMC step
    pp  = m.pp
    # projection of `v`, i.e. before FCIQMC step
    # compute memory noise
    r̃ = (pp - tp)/pp + shift*dτ
    push!(m.noiseBuffer, r̃) # add current value to buffer
    # Buffer only remembers up to `Δ` values. Average over whole buffer.
    r = r̃ - sum(m.noiseBuffer)/length(m.noiseBuffer)

    # apply `r` noise to current state vector
    axpy!(r, v, w) # w .+= r .* v
    # TODO: make this work with multithreading
    m.pp = tp + r*pp # update previous projection
    return r
end

# This one works to remove the bias when projection is done with exact
# eigenvector
function applyMemoryNoise!(s::IsStochasticWithThreshold,
                           w, v, shift, dτ, pnorm, m::ProjectedMemory2)
    tp = m.projector⋅w # w  may be a tuple for multithreading
    # TODO: make this work with multithreading and MPI
    # current projection of `w` after FCIQMC step

    pp  = m.projector⋅v
    # projection of `v`, i.e. before FCIQMC step
    # compute memory noise
    r̃ = (pp - tp)/pp + shift*dτ
    push!(m.noiseBuffer, r̃) # add current value to buffer
    # Buffer only remembers up to `Δ` values. Average over whole buffer.
    r = r̃ - sum(m.noiseBuffer)/length(m.noiseBuffer)

    # apply `r` noise to current state vector
    axpy!(r, v, w) # w .+= r .* v
    # TODO: make this work with multithreading
    m.pp = tp + r*pp # update previous projection
    return r
end

# seems to not be effective
function applyMemoryNoise!(s::IsStochasticWithThreshold,
                           w, v, shift, dτ, pnorm, m::ProjectedMemory3)
    tp = m.projector⋅w # w  may be a tuple for multithreading
    # TODO: make this work with multithreading and MPI
    # current projection of `w` after FCIQMC step

    pp  = m.projector⋅v
    # projection of `v`, i.e. before FCIQMC step
    # compute memory noise
    r̃ = (pp - tp) + shift*dτ*pp
    push!(m.noiseBuffer, r̃) # add current value to buffer
    # Buffer only remembers up to `Δ` values. Average over whole buffer.
    r = r̃ - sum(m.noiseBuffer)/length(m.noiseBuffer)
    if true # abs(pp) > 0.01
        r = r/pp
        # apply `r` noise to current state vector
        axpy!(r, v, w) # w .+= r .* v
        # TODO: make this work with multithreading
        m.pp = tp + r*pp # update previous projection
    else
        r = 0.0
        m.pp = tp
    end
    return r
end

# this one does not work well - no bias correction achieved
function applyMemoryNoise!(s::IsStochasticWithThreshold,
                           w, v, shift, dτ, pnorm, m::ProjectedMemory4)
    tp = m.projector⋅w # w  may be a tuple for multithreading
    # TODO: make this work with multithreading and MPI
    # current projection of `w` after FCIQMC step

    pp  = m.projector⋅v
    # projection of `v`, i.e. before FCIQMC step
    # compute memory noise
    r̃ = (pp - tp) + shift*dτ*pp
    push!(m.noiseBuffer, r̃) # add current value to buffer
    # Buffer only remembers up to `Δ` values. Average over whole buffer.
    r = r̃ - sum(m.noiseBuffer)/length(m.noiseBuffer)
    sf = 0.2
    r = sf*tanh(r/pp/sf)
    # apply `r` noise to current state vector
    axpy!(r, v, w) # w .+= r .* v
    # TODO: make this work with multithreading
    m.pp = tp + r*pp # update previous projection
    return r
end

function applyMemoryNoise!(s::StochasticStyle, w, v, shift, dτ, pnorm, m::PurgeNegatives)
    purge_negative_walkers!(w)
    return 0.0
end

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

# to do: implement parallel version
# function fciqmc_step!(w::D, ham::AbstractHamiltonian, v::D, shift, dτ) where D<:DArray
#   check that v and w are compatible
#   for each worker
#      call fciqmc_step!()  on respective local parts
#      sort and consolidate configurations to where they belong
#      communicate via RemoteChannels
#   end
#   return statistics
# end


# struct MySSVec{T} <: AbstractVector{T}
#     v::Vector{T}
#     sssize::Int
# end
# Base.size(mv::MySSVec) = size(mv.v)
# Base.getindex(mv::MySSVec, I...) = getindex(mv.v, I...)
#
# StochasticStyle(::Type{<:MySSVec}) = IsSemistochastic()

# Here is a simple function example to demonstrate that functions can
# dispatch on the trait and that computation of the type is done at compiler
# level.
#
# ```julia
# tfun(v) = tfun(v, StochasticStyle(v))
# tfun(v, ::IsDeterministic) = 1
# tfun(v, ::IsStochastic) = 2
# tfun(v, ::IsSemistochastic) = 3
# tfun(v, ::Any) = 4
#
# b = [1, 2, 3]
# StochasticStyle(b)
# IsStochastic()

# julia> @code_llvm tfun(b)
#
# ;  @ /Users/brand/git/juliamc/scripts/fciqmc.jl:448 within `tfun'
# define i64 @julia_tfun_13561(%jl_value_t addrspace(10)* nonnull align 16 dereferenceable(40)) {
# top:
#   ret i64 2
# }
# ```

"""
    fciqmc_col!(w, ham, add, num, shift, dτ)
    fciqmc_col!(::Type{T}, args...)
    -> spawns, deaths, clones, antiparticles, annihilations
Spawning and diagonal step of FCIQMC for single column of `ham`. In essence it
computes

`w .+= (1 .+ dτ.*(shift .- ham[:,add])).*num`.

Depending on `T == `[`StochasticStyle(w)`](@ref), a stochastic or deterministic algorithm will
be chosen. The possible values for `T` are:

- [`IsDeterministic()`](@ref) deteministic algorithm
- [`IsStochastic()`](@ref) stochastic version where the changes added to `w` are purely integer, according to the FCIQMC algorithm
- [`IsStochasticNonlinear(c)`](@ref) stochastic algorithm with nonlinear diagonal
- [`IsSemistochastic()`](@ref) semistochastic version: TODO
"""
function fciqmc_col!(w::Union{AbstractArray{T},AbstractDVec{K,T}},
    ham, add, num, shift, dτ
) where  {K,T<:Real}
    return fciqmc_col!(StochasticStyle(w), w, ham, add, num, real(shift), dτ)
end
# only use real part of the shift if the coefficients are real

# otherwise, pass on complex shift in generic method
fciqmc_col!(w::Union{AbstractArray,AbstractDVec}, args...) = fciqmc_col!(StochasticStyle(w), w, args...)

# generic method for unknown trait: throw error
fciqmc_col!(::Type{T}, args...) where T = throw(TypeError(:fciqmc_col!,
    "first argument: trait not recognised",StochasticStyle,T))

function fciqmc_col!(::IsDeterministic, w, ham::AbstractMatrix, add, num, shift, dτ)
    w .+= (1 .+ dτ.*(shift .- view(ham,:,add))).*num
    # todo: return something sensible
    return (0, 0, 0, 0, 0)
end

function fciqmc_col!(::IsDeterministic, w, ham::AbstractHamiltonian, add, num, shift, dτ)
    # off-diagonal: spawning psips
    for (nadd, elem) in Hops(ham, add)
        w[nadd] += -dτ * elem * num
    end
    # diagonal death or clone
    w[add] += (1 + dτ*(shift - diagME(ham,add)))*num
    return (0, 0, 0, 0, 0)
end

function fciqmc_col!(::IsStochastic, w, ham::AbstractHamiltonian, add, num::Real,
                        shift, dτ)
    # version for single population of integer psips
    # off-diagonal: spawning psips
    spawns = deaths = clones = antiparticles = annihilations = zero(num)
    hops = Hops(ham,add)
    for n in 1:abs(num) # for each psip attempt to spawn once
        naddress, pgen, matelem = generateRandHop(hops)
        pspawn = dτ * abs(matelem) /pgen # non-negative Float64
        nspawn = floor(pspawn) # deal with integer part separately
        cRand() < (pspawn - nspawn) && (nspawn += 1) # random spawn
        # at this point, nspawn is non-negative
        # now converted to correct type and compute sign
        nspawns = convert(typeof(num), -nspawn * sign(num) * sign(matelem))
        # - because Hamiltonian appears with - sign in iteration equation
        if sign(w[naddress]) * sign(nspawns) < 0 # record annihilations
            annihilations += min(abs(w[naddress]),abs(nspawns))
        end
        if !iszero(nspawns)
            w[naddress] += nspawns
            # perform spawn (if nonzero): add walkers with correct sign
            spawns += abs(nspawns)
        end
    end
    # diagonal death / clone
    dME = diagME(ham,add)
    pd = dτ * (dME - shift)
    newdiagpop = (1-pd)*num
    ndiag = trunc(newdiagpop)
    abs(newdiagpop-ndiag)>cRand() && (ndiag += sign(newdiagpop))
    # only treat non-integer part stochastically
    ndiags = convert(typeof(num),ndiag) # now integer type
    if sign(w[add]) ≠ sign(ndiags) # record annihilations
        annihilations += min(abs(w[add]),abs(ndiags))
    end
    w[add] += ndiags # should carry to correct sign
    if  pd < 0 # record event statistics
        clones += abs(ndiags - num)
    elseif pd < 1
        deaths += abs(ndiags - num)
    else
        antiparticles += abs(ndiags)
    end
    return (spawns, deaths, clones, antiparticles, annihilations)
    # note that w is not returned
end

function fciqmc_col!(::IsStochastic2Pop, w, ham::AbstractHamiltonian, add, cnum::Complex,
                        cshift, dτ)
    # version for complex integer psips
    # off-diagonal: spawning psips
    spawns = deaths = clones = antiparticles = annihilations = zero(cnum)
    # stats reported are complex, for each component separately
    hops = Hops(ham,add)
    # real psips first
    num = real(cnum)
    for n in 1:abs(num) # for each psip attempt to spawn once
        naddress, pgen, matelem = generateRandHop(hops)
        pspawn = dτ * abs(matelem) /pgen # non-negative Float64
        nspawn = floor(pspawn) # deal with integer part separately
        cRand() < (pspawn - nspawn) && (nspawn += 1) # random spawn
        # at this point, nspawn is non-negative
        # now converted to correct type and compute sign
        nspawns = convert(typeof(num), -nspawn * sign(num) * sign(matelem))
        # - because Hamiltonian appears with - sign in iteration equation
        if sign(real(w[naddress])) * sign(nspawns) < 0 # record annihilations
            annihilations += min(abs(real(w[naddress])),abs(nspawns))
        end
        if !iszero(nspawns)
            w[naddress] += nspawns
            # perform spawn (if nonzero): add walkers with correct sign
            spawns += abs(nspawns)
        end
    end
    # now imaginary psips
    num = imag(cnum)
    for n in 1:abs(num) # for each psip attempt to spawn once
        naddress, pgen, matelem = generateRandHop(hops)
        pspawn = dτ * abs(matelem) /pgen # non-negative Float64
        nspawn = floor(pspawn) # deal with integer part separately
        cRand() < (pspawn - nspawn) && (nspawn += 1) # random spawn
        # at this point, nspawn is non-negative
        # now converted to correct type and compute sign
        nspawns = im*convert(typeof(num), -nspawn * sign(num) * sign(matelem))
        # - because Hamiltonian appears with - sign in iteration equation
        if sign(imag(w[naddress])) * sign(imag(nspawns)) < 0 # record annihilations
            annihilations += min(abs(imag(w[naddress])),abs(nspawns))
        end
        if !iszero(nspawns)
            w[naddress] += nspawns
            # perform spawn (if nonzero): add walkers with correct sign
            spawns += im*abs(nspawns)
        end
    end

    # diagonal death / clone
    shift = real(cshift) # use only real part of shift for now
    dME = diagME(ham,add)
    pd = dτ * (dME - shift) # real valued so far
    cnewdiagpop = (1-pd)*cnum # now it's complex
    # treat real part
    newdiagpop = real(cnewdiagpop)
    num = real(cnum)
    ndiag = trunc(newdiagpop)
    abs(newdiagpop-ndiag)>cRand() && (ndiag += sign(newdiagpop))
    # only treat non-integer part stochastically
    ndiags = convert(typeof(num),ndiag) # now complex integer type
    if sign(real(w[add])) ≠ sign(ndiag) # record annihilations
        annihilations += min(abs(real(w[add])),abs(real(ndiags)))
    end
    w[add] += ndiags # should carry the correct sign
    if  pd < 0 # record event statistics
        clones += abs(real(ndiags) - num)
    elseif pd < 1
        deaths += abs(real(ndiags) - num)
    else
        antiparticles += abs(real(ndiags))
    end
    # treat imaginary part
    newdiagpop = imag(cnewdiagpop)
    num = imag(cnum)
    ndiag = trunc(newdiagpop)
    abs(newdiagpop-ndiag)>cRand() && (ndiag += sign(newdiagpop))
    # only treat non-integer part stochastically
    ndiags = im*convert(typeof(num),ndiag) # now complex integer type
    if sign(imag(w[add])) ≠ sign(ndiag) # record annihilations
        annihilations += min(abs(imag(w[add])),abs(imag(ndiags)))
    end
    w[add] += ndiags # should carry the correct sign
    if  pd < 0 # record event statistics
        clones += im*abs(imag(ndiags) - num)
    elseif pd < 1
        deaths += im*abs(imag(ndiags) - num)
    else
        antiparticles += im*abs(imag(ndiags))
    end

    # imaginary part of shift leads to spawns across populations
    cspawn = im*dτ*imag(cshift)*cnum # to be spawned as complex number with signs

    # real part - to be spawned into real walkers
    rspawn = real(cspawn) # float with sign
    nspawn = trunc(rspawn) # deal with integer part separately
    cRand() < abs(rspawn - nspawn) && (nspawn += sign(rspawn)) # random spawn
    # at this point, nspawn has correct sign
    # now convert to correct type
    cnspawn = convert(typeof(cnum), nspawn)
    if sign(real(w[add])) * sign(nspawn) < 0 # record annihilations
        annihilations += min(abs(real(w[add])),abs(nspawn))
    end
    w[add] += cnspawn
    # perform spawn (if nonzero): add walkers with correct sign
    spawns += abs(nspawn)

    # imag part - to be spawned into imaginary walkers
    ispawn = imag(cspawn) # float with sign
    nspawn = trunc(ispawn) # deal with integer part separately
    cRand() < abs(ispawn - nspawn) && (nspawn += sign(ispawn)) # random spawn
    # at this point, nspawn has correct sign
    # now convert to correct type
    cnspawn = convert(typeof(cnum), nspawn*im)# imaginary spawns!
    if sign(imag(w[add])) * sign(nspawn) < 0 # record annihilations
        annihilations += min(abs(imag(w[add])),abs(nspawn))
    end
    w[add] += cnspawn
    # perform spawn (if nonzero): add walkers with correct sign
    spawns += abs(nspawn)

    return (spawns, deaths, clones, antiparticles, annihilations)
    # note that w is not returned
end

function fciqmc_col!(::IsStochastic2PopInitiator, w, ham::AbstractHamiltonian,
                        add, cnum::Complex, cshift, dτ)
    # version for complex integer psips with initiator approximation
    # off-diagonal: spawning psips
    spawns = deaths = clones = antiparticles = annihilations = zero(cnum)
    # stats reported are complex, for each component separately
    hops = Hops(ham,add)
    # real psips first
    num = real(cnum)
    for n in 1:abs(num) # for each psip attempt to spawn once
        naddress, pgen, matelem = generateRandHop(hops)
        pspawn = dτ * abs(matelem) /pgen # non-negative Float64
        nspawn = floor(pspawn) # deal with integer part separately
        cRand() < (pspawn - nspawn) && (nspawn += 1) # random spawn
        # at this point, nspawn is non-negative
        # now converted to correct type and compute sign
        nspawns = convert(typeof(num), -nspawn * sign(num) * sign(matelem))
        # - because Hamiltonian appears with - sign in iteration equation
        if sign(real(w[naddress])) * sign(nspawns) < 0 # record annihilations
            annihilations += min(abs(real(w[naddress])),abs(nspawns))
        end
        if !iszero(nspawns)
            w[naddress] += nspawns
            # perform spawn (if nonzero): add walkers with correct sign
            spawns += abs(nspawns)
        end
    end
    # now imaginary psips
    num = imag(cnum)
    for n in 1:abs(num) # for each psip attempt to spawn once
        naddress, pgen, matelem = generateRandHop(hops)
        pspawn = dτ * abs(matelem) /pgen # non-negative Float64
        nspawn = floor(pspawn) # deal with integer part separately
        cRand() < (pspawn - nspawn) && (nspawn += 1) # random spawn
        # at this point, nspawn is non-negative
        # now converted to correct type and compute sign
        nspawns = im*convert(typeof(num), -nspawn * sign(num) * sign(matelem))
        # - because Hamiltonian appears with - sign in iteration equation
        if sign(imag(w[naddress])) * sign(imag(nspawns)) < 0 # record annihilations
            annihilations += min(abs(imag(w[naddress])),abs(nspawns))
        end
        if !iszero(nspawns)
            w[naddress] += nspawns
            # perform spawn (if nonzero): add walkers with correct sign
            spawns += im*abs(nspawns)
        end
    end

    # diagonal death / clone
    shift = real(cshift) # use only real part of shift for now
    dME = diagME(ham,add)
    pd = dτ * (dME - shift) # real valued so far
    cnewdiagpop = (1-pd)*cnum # now it's complex
    # treat real part
    newdiagpop = real(cnewdiagpop)
    num = real(cnum)
    ndiag = trunc(newdiagpop)
    abs(newdiagpop-ndiag)>cRand() && (ndiag += sign(newdiagpop))
    # only treat non-integer part stochastically
    ndiags = convert(typeof(num),ndiag) # now complex integer type
    if sign(real(w[add])) ≠ sign(ndiag) # record annihilations
        annihilations += min(abs(real(w[add])),abs(real(ndiags)))
    end
    w[add] += ndiags # should carry the correct sign
    if  pd < 0 # record event statistics
        clones += abs(real(ndiags) - num)
    elseif pd < 1
        deaths += abs(real(ndiags) - num)
    else
        antiparticles += abs(real(ndiags))
    end
    # treat imaginary part
    newdiagpop = imag(cnewdiagpop)
    num = imag(cnum)
    ndiag = trunc(newdiagpop)
    abs(newdiagpop-ndiag)>cRand() && (ndiag += sign(newdiagpop))
    # only treat non-integer part stochastically
    ndiags = im*convert(typeof(num),ndiag) # now complex integer type
    if sign(imag(w[add])) ≠ sign(ndiag) # record annihilations
        annihilations += min(abs(imag(w[add])),abs(imag(ndiags)))
    end
    w[add] += ndiags # should carry the correct sign
    if  pd < 0 # record event statistics
        clones += im*abs(imag(ndiags) - num)
    elseif pd < 1
        deaths += im*abs(imag(ndiags) - num)
    else
        antiparticles += im*abs(imag(ndiags))
    end

    # imaginary part of shift leads to spawns across populations
    cspawn = im*dτ*imag(cshift)*cnum # to be spawned as complex number with signs

    # real part - to be spawned into real walkers
    if real(w[add]) ≠ 0 # only spawn into occupied sites (initiator approximation)
        rspawn = real(cspawn) # float with sign
        nspawn = trunc(rspawn) # deal with integer part separately
        cRand() < abs(rspawn - nspawn) && (nspawn += sign(rspawn)) # random spawn
        # at this point, nspawn has correct sign
        # now convert to correct type
        cnspawn = convert(typeof(cnum), nspawn)
        if sign(real(w[add])) * sign(nspawn) < 0 # record annihilations
            annihilations += min(abs(real(w[add])),abs(nspawn))
        end
        w[add] += cnspawn
        # perform spawn (if nonzero): add walkers with correct sign
        spawns += abs(nspawn)
    end

    # imag part - to be spawned into imaginary walkers
    if imag(w[add]) ≠ 0 # only spawn into occupied sites (initiator approximation)
        ispawn = imag(cspawn) # float with sign
        nspawn = trunc(ispawn) # deal with integer part separately
        cRand() < abs(ispawn - nspawn) && (nspawn += sign(ispawn)) # random spawn
        # at this point, nspawn has correct sign
        # now convert to correct type
        cnspawn = convert(typeof(cnum), nspawn*im)# imaginary spawns!
        if sign(imag(w[add])) * sign(nspawn) < 0 # record annihilations
            annihilations += min(abs(imag(w[add])),abs(nspawn))
        end
        w[add] += cnspawn
        # perform spawn (if nonzero): add walkers with correct sign
        spawns += abs(nspawn)
    end

    return (spawns, deaths, clones, antiparticles, annihilations)
    # note that w is not returned
end

function fciqmc_col!(nl::IsStochasticNonlinear, w, ham::AbstractHamiltonian, add, num::Real,
                        shift, dτ)
    # version for single population of integer psips
    # Nonlinearity in diagonal death step according to Ali's suggestion
    # off-diagonal: spawning psips
    spawns = deaths = clones = antiparticles = annihilations = zero(num)
    hops = Hops(ham,add)
    for n in 1:abs(num) # for each psip attempt to spawn once
        naddress, pgen, matelem = generateRandHop(hops)
        pspawn = dτ * abs(matelem) /pgen # non-negative Float64
        nspawn = floor(pspawn) # deal with integer part separately
        cRand() < (pspawn - nspawn) && (nspawn += 1) # random spawn
        # at this point, nspawn is non-negative
        # now converted to correct type and compute sign
        nspawns = convert(typeof(num), -nspawn * sign(num) * sign(matelem))
        # - because Hamiltonian appears with - sign in iteration equation
        if sign(w[naddress]) * sign(nspawns) < 0 # record annihilations
            annihilations += min(abs(w[naddress]),abs(nspawns))
        end
        if !iszero(nspawns)
            w[naddress] += nspawns
            # perform spawn (if nonzero): add walkers with correct sign
            spawns += abs(nspawns)
        end
    end
    # diagonal death / clone
    dME = diagME(ham,add)
    shifteff = shift*(1 - exp(-num/nl.c))
    pd = dτ * (dME - shifteff)
    newdiagpop = (1-pd)*num
    ndiag = trunc(newdiagpop)
    abs(newdiagpop-ndiag)>cRand() && (ndiag += sign(newdiagpop))
    # only treat non-integer part stochastically
    ndiags = convert(typeof(num),ndiag) # now integer type
    if sign(w[add]) ≠ sign(ndiags) # record annihilations
        annihilations += min(abs(w[add]),abs(ndiags))
    end
    w[add] += ndiags # should carry to correct sign
    if  pd < 0 # record event statistics
        clones += abs(ndiags - num)
    elseif pd < 1
        deaths += abs(ndiags - num)
    else
        antiparticles += abs(ndiags)
    end
    return (spawns, deaths, clones, antiparticles, annihilations)
    # note that w is not returned
end # inner_step!

function fciqmc_col!(::IsStochastic, w, ham::AbstractHamiltonian, add,
                        tup::Tuple{Real,Real},
                        shift, dτ)
    # trying out Ali's suggestion with occupation ratio of neighbours
    # off-diagonal: spawning psips
    num = tup[1] # number of psips on configuration
    occ_ratio= tup[2] # ratio of occupied vs total number of neighbours
    spawns = deaths = clones = antiparticles = annihilations = zero(num)
    hops = Hops(ham,add)
    for n in 1:abs(num) # for each psip attempt to spawn once
        naddress, pgen, matelem = generateRandHop(hops)
        pspawn = dτ * abs(matelem) /pgen # non-negative Float64
        nspawn = floor(pspawn) # deal with integer part separately
        cRand() < (pspawn - nspawn) && (nspawn += 1) # random spawn
        # at this point, nspawn is non-negative
        # now converted to correct type and compute sign
        nspawns = convert(typeof(num), -nspawn * sign(num) * sign(matelem))
        # - because Hamiltonian appears with - sign in iteration equation
        wnapsips, wnaflag = w[naddress]
        if sign(wnapsips) * sign(nspawns) < 0 # record annihilations
            annihilations += min(abs(wnapsips),abs(nspawns))
        end
        if !iszero(nspawns)
            w[naddress] = (wnapsips+nspawns, wnaflag)
            # perform spawn (if nonzero): add walkers with correct sign
            spawns += abs(nspawns)
        end
    end
    # diagonal death / clone
    dME = diagME(ham,add)
    # modify shift locally according to occupation ratio of neighbouring configs
    mshift = occ_ratio > 0 ? shift*occ_ratio : shift
    pd = dτ * (dME - mshift) # modified
    newdiagpop = (1-pd)*num
    ndiag = trunc(newdiagpop)
    abs(newdiagpop-ndiag)>cRand() && (ndiag += sign(newdiagpop))
    # only treat non-integer part stochastically
    ndiags = convert(typeof(num),ndiag) # now appropriate type
    wapsips, waflag = w[add]
    if sign(wapsips) ≠ sign(ndiags) # record annihilations
        annihilations += min(abs(wapsips),abs(ndiags))
    end
    w[add] = (wapsips + ndiags, waflag) # should carry to correct sign
    if  pd < 0 # record event statistics
        clones += abs(ndiags - num)
    elseif pd < 1
        deaths += abs(ndiags - num)
    else
        antiparticles += abs(ndiags)
    end
    return (spawns, deaths, clones, antiparticles, annihilations)
    # note that w is not returned
end # inner_step!

function fciqmc_col!(s::IsSemistochastic, w, ham::AbstractHamiltonian, add,
         val_flag_tuple::Tuple{N, F}, shift, dτ) where {N<:Number, F<:Integer}
    (val, flag) = val_flag_tuple
    deterministic = flag & one(F) # extract deterministic flag
    # diagonal death or clone
    new_val = w[add][1] + (1 + dτ*(shift - diagME(ham,add)))*val
    if deterministic
        w[add] = (new_val, flag) # new tuple
    else
        if new_val < s.threshold
            if new_val/s.threshold > cRand()
                new_val = convert(N,s.threshold)
                w[add] = (new_val, flag) # new tuple
            end
            # else # do nothing, stochastic space and rounded to zero
        else
            w[add] = (new_val, flag) # new tuple
        end
    end
    # off-diagonal: spawning psips
    if deterministic
        for (nadd, elem) in Hops(ham, add)
            wnapsips, wnaflag = w[nadd]
            if wnaflag & one(F) # new address `nadd` is also in deterministic space
                w[nadd] = (wnapsips - dτ * elem * val, wnaflag)  # new tuple
            else
                # TODO: det -> sto
                pspawn = abs(val * dτ * matelem) # non-negative Float64, pgen = 1
                nspawn = floor(pspawn) # deal with integer part separately
                cRand() < (pspawn - nspawn) && (nspawn += 1) # random spawn
                # at this point, nspawn is non-negative
                # now converted to correct type and compute sign
                nspawns = convert(N, -nspawn * sign(val) * sign(matelem))
                # - because Hamiltonian appears with - sign in iteration equation
                if sign(wnapsips) * sign(nspawns) < 0 # record annihilations
                    annihilations += min(abs(wnapsips),abs(nspawns))
                end
                if !iszero(nspawns) # successful attempt to spawn
                    w[naddress] = (wnapsips+nspawns, wnaflag)
                    # perform spawn (if nonzero): add walkers with correct sign
                    spawns += abs(nspawns)
                end
            end
        end
    else
        # TODO: stochastic
        hops = Hops(ham, add)
        for n in 1:floor(abs(val)) # abs(val÷s.threshold) # for each psip attempt to spawn once
            naddress, pgen, matelem = generateRandHop(hops)
            pspawn = dτ * abs(matelem) /pgen # non-negative Float64
            nspawn = floor(pspawn) # deal with integer part separately
            cRand() < (pspawn - nspawn) && (nspawn += 1) # random spawn
            # at this point, nspawn is non-negative
            # now converted to correct type and compute sign
            nspawns = convert(typeof(val), -nspawn * sign(val) * sign(matelem))
            # - because Hamiltonian appears with - sign in iteration equation
            wnapsips, wnaflag = w[naddress]
            if sign(wnapsips) * sign(nspawns) < 0 # record annihilations
                annihilations += min(abs(wnapsips),abs(nspawns))
            end
            if !iszero(nspawns)
                w[naddress] = (wnapsips+nspawns, wnaflag)
                # perform spawn (if nonzero): add walkers with correct sign
                spawns += abs(nspawns)
            end
        end
        # deal with non-integer remainder
        rval =  abs(val%1) # abs(val%threshold)
        naddress, pgen, matelem = generateRandHop(hops)
        pspawn = rval * dτ * abs(matelem) /pgen # non-negative Float64
        nspawn = floor(pspawn) # deal with integer part separately
        cRand() < (pspawn - nspawn) && (nspawn += 1) # random spawn
        # at this point, nspawn is non-negative
        # now converted to correct type and compute sign
        nspawns = convert(typeof(val), -nspawn * sign(val) * sign(matelem))
        # - because Hamiltonian appears with - sign in iteration equation
        wnapsips, wnaflag = w[naddress]
        if sign(wnapsips) * sign(nspawns) < 0 # record annihilations
            annihilations += min(abs(wnapsips),abs(nspawns))
        end
        if !iszero(nspawns)
            w[naddress] = (wnapsips+nspawns, wnaflag)
            # perform spawn (if nonzero): add walkers with correct sign
            spawns += abs(nspawns)
        end
        # done with stochastic spawning
    end
    return (0, 0, 0, 0, 0)
end

function fciqmc_col!(s::IsStochasticWithThreshold, w, ham::AbstractHamiltonian,
        add, val::N, shift, dτ) where N <: Real

    # diagonal death or clone: deterministic fomula
    # w[add] += (1 + dτ*(shift - diagME(ham,add)))*val
    # projection to threshold should be applied after all colums are evaluated
    new_val = (1 + dτ*(shift - diagME(ham,add)))*val
    # apply threshold if necessary
    if abs(new_val) < s.threshold
        # project stochastically to threshold
        # w[add] += (abs(new_val)/s.threshold > cRand()) ? sign(new_val)*s.threshold : 0
        w[add] += ifelse(cRand() < abs(new_val)/s.threshold, sign(new_val)*s.threshold, 0)
    else
        w[add] += new_val
    end

    # off-diagonal: spawning psips stochastically
    # only integers are spawned!!
    hops = Hops(ham, add)
    # first deal with integer psips
    for n in 1:floor(abs(val)) # for each psip attempt to spawn once
        naddress, pgen, matelem = generateRandHop(hops)
        pspawn = dτ * abs(matelem) /pgen # non-negative Float64
        nspawn = floor(pspawn) # deal with integer part separately
        cRand() < (pspawn - nspawn) && (nspawn += 1) # random spawn
        # at this point, nspawn is non-negative
        # now converted to correct type and compute sign
        nspawns = convert(N, -nspawn * sign(val) * sign(matelem))
        # - because Hamiltonian appears with - sign in iteration equation
        if !iszero(nspawns)
            w[naddress] += nspawns
            # perform spawn (if nonzero): add walkers with correct sign
        end
    end
    # deal with non-integer remainder: attempt to spawn
    rval =  abs(val%1) # non-integer part reduces probability for spawning
    naddress, pgen, matelem = generateRandHop(hops)
    pspawn = rval * dτ * abs(matelem) /pgen # non-negative Float64
    nspawn = floor(pspawn) # deal with integer part separately
    cRand() < (pspawn - nspawn) && (nspawn += 1) # random spawn
    # at this point, nspawn is non-negative
    # now converted to correct type and compute sign
    nspawns = convert(N, -nspawn * sign(val) * sign(matelem))
    # - because Hamiltonian appears with - sign in iteration equation
    if !iszero(nspawns)
        w[naddress] += nspawns
        # perform spawn (if nonzero): add walkers with correct sign
    end
    # done with stochastic spawning
    return (0, 0, 0, 0, 0)
end
