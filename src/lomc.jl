# top level functions `lomc!()` and `fciqmc!()`
# `refine_r_strat`: helper function for `lomc!()`

# TODO: deprecate `fciqmc!`

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
        _, _, step_statsd, _ = fciqmc_step!(ham, copy(localpart(v)),
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
    # nt = (;ham, v, params, df, wm, s_strat, r_strat, τ_strat, m_strat)
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
    @unpack ham, v, params, df, wm, s_strat, r_strat, τ_strat, m_strat = a
    ConsistentRNG.check_crng_independence(v) # sanity check of RNGs

    rr_strat = refine_r_strat(r_strat, ham) # set up r_strat for fciqmc!()

    fciqmc!(v, params, df, ham, s_strat, rr_strat, τ_strat, wm;
        m_strat = m_strat
    )
    nt = (a..., r_strat = rr_strat)
    return nt
end

function lomc!(a::NamedTuple, laststep::Int) # should be type stable
    a.params.laststep = laststep
    return lomc!(a)
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
                 maxlength = 0,
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
    if iszero(maxlength)
        maxlength = 2 * max(real(s_strat.targetwalkers), imag(s_strat.targetwalkers))
    end

    while step < laststep
        step += 1
        # println("Step: ",step)
        # perform one complete stochastic vector matrix multiplication
        v, w, step_stats, r = fciqmc_step!(ham, v, shift, dτ, pnorm, w
                                            , 1.0 # for selecting multithreaded version
                                            ; m_strat=m_strat)
        v = update_dvec!(v, shift)
        tnorm = walkernumber(v)
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
            @warn "`maxlength` nearly reached" len_local maxlength once=true
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
function fciqmc!(vv::AbstractVector, pa::RunTillLastStep{T}, ham::AbstractHamiltonian,
                 s_strat::ShiftStrategy,
                 r_strat::ReportingStrategy = EveryTimeStep(),
                 τ_strat::TimeStepStrategy = ConstantTimeStep(),
                 wv = threadedWorkingMemory.(vv) # wv = similar.(localpart.(vv))
                 ; m_strat::MemoryStrategy = NoMemory(),
                 report_xHy = false,
                 maxlength = 0,
                 ) where T
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

    if iszero(maxlength)
        maxlength = 2 * max(real(s_strat.targetwalkers), imag(s_strat.targetwalkers))
    end

    shifts = [shift for i = 1:N] # Vector because it needs to be mutable
    vShiftModes = [shiftMode for i = 1:N] # separate for each replica
    pnorms = zeros(T, N) # initialise as vector
    pnorms .= walkernumber.(vv) # 1-norm i.e. number of psips as Tuple (of previous step)
    v_proj, h_proj = compute_proj_observables(vv[1], ham, r_strat)

    # initalise df for storing results of each replica separately
    dfs = Tuple(DataFrame(steps=Int[], shift=T[], shiftMode=Bool[],
        len=Int[], norm=T[],
        vproj=typeof(v_proj)[], hproj=typeof(h_proj)[],
        spawns=V[], deaths=V[],
        clones=V[], antiparticles=V[],
        annihilations=V[], shiftnoise=Float64[]) for i in 1:N
    )
    # dfs is thus an NTuple of DataFrames
    for i in 1:N
        v_proj, h_proj = compute_proj_observables(vv[i], ham, r_strat)
        push!(dfs[i], (step, shifts[i], vShiftModes[i], length(vv[i]),
            pnorms[i], v_proj, h_proj, 0, 0, 0, 0, 0, 0.0)
        )
    end

    # prepare `DataFrame` for variational ground state estimator
    # we are assuming that N ≥ 2, otherwise this will fail
    PType = Union{Missing, promote_type(V,eltype(ham))} # type of scalar product
    RType = Union{Missing, promote_type(PType,Float64)} # for division
    mixed_df= DataFrame(steps =Int[], xdoty =V[], xHy =PType[], aveH =RType[])
    dp = vv[1]⋅vv[2] # <v_1 | v_2>
    # vv[1]⋅ham(vv[2]) # <v_1 | ham | v_2>
    expval = report_xHy ? dot(vv[1], ham, vv[2]) : missing
    push!(mixed_df,(step, dp, expval, expval/dp))

    norms = zeros(T, N)
    mstats = [zeros(Complex{Int},5) for i=1:N]
    rs = zeros(N)
    while step < laststep
        step += 1
        for (i, v) in enumerate(vv) # loop over replicas
            # perform one complete stochastic vector matrix multiplication
            vv[i], wv[i], stats, rs[i] = fciqmc_step!(ham, v, shifts[i],
                dτ, pnorms[i], wv[i]; m_strat = m_strat
            )
            mstats[i] .= stats
            vv[i] = update_dvec!(vv[i], shifts[i])
            norms[i] = walkernumber(vv[i])
            shifts[i], vShiftModes[i], pnorms[i] = update_shift(
                s_strat, shifts[i], vShiftModes[i],
                norms[i], pnorms[i], dτ, step, dfs[i], vv[i], wv[i]
            )
            v_proj, h_proj = compute_proj_observables(v, ham, r_strat) # MPIsync

            # record results
            push!(dfs[i], (step, shifts[i], vShiftModes[i], length(vv[i]),
                  norms[i], v_proj, h_proj, mstats[i]..., rs[i]))
        end #loop over replicas
        # lengths = length.(vv)
        # update time step
        dτ = update_dτ(τ_strat, dτ) # will need to pass more information
        # later when we add different stratgies
        # # record results
        # for i = 1:N
        #     push!(dfs[i], (step, shifts[i], vShiftModes[i], lengths[i],
        #           norms[i], mstats[i]..., rs[i]))
        # end
        v1Dv2 = vv[1]⋅vv[2] # <v_1 | v_2> overlap
        # <v_1 | ham | v_2>
        v2Dhv2 =  report_xHy ? vv[1]⋅ham(vv[2]) : missing
        push!(mixed_df,(step, v1Dv2, v2Dhv2, v2Dhv2/v1Dv2))

        # prepare for next step:
        # pnorms .= norms # remember norm of this step for next step (previous norm)
        lengths = [dfs[i].len[end] for i in 1:N]
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
