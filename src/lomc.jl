"""
    ReplicaState(v, w, pnorm, params, id)

Struct that holds all information needed for an independent run of the algorithm.

Can be advanced a step forward with [`advance!`](@ref).

# Fields

* `hamiltonian`: the model Hamiltonian.
* `v`: vector.
* `w`: working memory.
* `pnorm`: previous walker number (see [`walkernumber`](@ref)).
* `params`: the [`FCIQMCRunStrategy`](@ref).
* `id`: appended to reported columns.
"""
mutable struct ReplicaState{H,T,V,W,R<:FciqmcRunStrategy{T}}
    hamiltonian::H
    v::V       # vector
    w::W       # working memory. Maybe working memories could be shared among replicas?
    pnorm::T   # previous walker number - used to control the shift
    params::R  # params: step, laststep, dτ...
    id::String # id is appended to column names
end

function ReplicaState(h, v, w, params, id="")
    if isnothing(w)
        w = similar(v)
    end
    return ReplicaState(h, v, w, walkernumber(v), params, id)
end

function Base.show(io::IO, r::ReplicaState)
    print(
        io,
        "ReplicaState(v: ", length(r.v), "-element ", nameof(typeof(r.v)),
        ", w: ", length(r.w), "-element ", nameof(typeof(r.w)), ")"
    )
end

"""
    QMCState

Holds all information needed to run FCIQMC, except the data frame. Holds a `NTuple` of
`ReplicaState`s and various strategies that control the algorithm.
"""
struct QMCState{
    H,
    N,
    R<:ReplicaState,
    MS<:MemoryStrategy,
    RS<:ReportingStrategy,
    SS<:ShiftStrategy,
    TS<:TimeStepStrategy,
    RRS<:ReplicaStrategy,
    PS<:NTuple{<:Any,PostStepStrategy},
}
    hamiltonian::H
    replicas::NTuple{N,R}
    maxlength::Ref{Int}

    m_strat::MS
    r_strat::RS
    s_strat::SS
    τ_strat::TS
    post_step::PS
    replica::RRS
end

"""
    _n_walkers(v, s_strat)
Returns an estimate of the expected number of walkers as an integer.
"""
function _n_walkers(v, s_strat)
    n = if hasfield(typeof(s_strat), :targetwalkers)
        s_strat.targetwalkers
    else # e.g. for LogUpdate()
        walkernumber(v)
    end
    return ceil(Int, max(real(n), imag(n)))
end

function QMCState(
    hamiltonian, v;
    laststep=nothing,
    dτ=nothing,
    threading=:auto,
    wm=nothing,
    params::FciqmcRunStrategy=RunTillLastStep(
        laststep = 100,
        shift = diagonal_element(
            hamiltonian,
            starting_address(hamiltonian)
        )/one(valtype(v))
    ),
    s_strat::ShiftStrategy=DoubleLogUpdate(),
    r_strat::ReportingStrategy=ReportDFAndInfo(),
    τ_strat::TimeStepStrategy=ConstantTimeStep(),
    m_strat::MemoryStrategy=NoMemory(),
    replica::ReplicaStrategy=NoStats(),
    post_step=(),
    maxlength= 2 * _n_walkers(v, s_strat) + 100, # padding for small walker numbers
)
    # Set up default arguments
    r_strat = refine_r_strat(r_strat)
    if !isnothing(laststep)
        params.laststep = laststep
    end
    if !isnothing(dτ)
        params.dτ = dτ
    end
    wm = default_working_memory(threading, v, _n_walkers(v, s_strat))
    if !(post_step isa Tuple)
        post_step = (post_step,)
    end
    # Set up replicas
    nreplicas = num_replicas(replica)
    if nreplicas > 1
        replicas = ntuple(nreplicas) do i
            ReplicaState(hamiltonian, deepcopy(v), deepcopy(wm), deepcopy(params), "_$i")
        end
    else
        replicas = (ReplicaState(hamiltonian, v, wm, params, ""),)
    end

    return QMCState(
        hamiltonian, replicas, Ref(maxlength),
        m_strat, r_strat, s_strat, τ_strat, post_step, replica
    )
end

# Allow setting step and laststep from QMCState.
function Base.getproperty(state::QMCState, key::Symbol)
    if key == :step
        step = state.replicas[1].params.step
        return step
    elseif key == :laststep
        laststep = state.replicas[1].params.laststep
        return laststep
    elseif key == :maxlength
        return getfield(state, :maxlength)[]
    else
        return getfield(state, key)
    end
end
function Base.setproperty!(state::QMCState, key::Symbol, value)
    if key == :step
        for r in state.replicas
            r.params.step = value
        end
        return value
    elseif key == :laststep
        for r in state.replicas
            r.params.laststep = value
        end
        return value
    elseif key == :maxlength
        getfield(state, :maxlength)[] = value
        return value
    else
        # This will error
        return setfield!(state, key, value)
    end
end

function default_working_memory(threading, v, targetwalkers::Real)
    if threading == :auto
        threading = targetwalkers ≥ 500
    end
    if threading && Threads.nthreads() > 1
        return threadedWorkingMemory(v)
    else
        return similar(localpart(v))
    end
end

function Base.show(io::IO, st::QMCState)
    print(io, "QMCState")
    if length(st.replicas) > 1
        print(io, " with ", length(st.replicas), " replicas")
    end
    print(io, "\n  H:    ", st.hamiltonian)
    print(io, "\n  step: ", st.step, " / ", st.laststep)
    print(io, "\n  replicas: ")
    for (i, r) in enumerate(st.replicas)
        print(io, "\n    $i: ", r)
    end
end

"""
    lomc!(ham::AbstractHamiltonian, [v]; kwargs...) -> df, state
    lomc!(state::QMCState, [df]; kwargs...) -> df, state

Linear operator Monte Carlo: Perform a projector quantum Monte Carlo simulation for
determining the lowest eigenvalue of `ham`. `v` can be a single starting vector. The default
choice is
```julia
v = DVec(starting_address(ham) => 10; style=IsStochasticInteger())
```
and triggers the integer walker FCIQMC algorithm. See [`DVec`](@ref) and
[`StochasticStyle`](@ref).

# Keyword arguments, defaults, and precedence:

* `params::FciqmcRunStrategy = RunTillLastStep(laststep = 100, dτ = 0.01, shift = diagonal_element(ham, starting_address(ham)))` - basic parameters of simulation state, see [`FciqmcRunStrategy`](@ref); is mutated
* `laststep` - can be used to override information otherwise contained in `params`
* `s_strat::ShiftStrategy = DoubleLogUpdate(targetwalkers = 100, ζ = 0.08, ξ = ζ^2/4)` - how to update the `shift`, see [`ShiftStrategy`](@ref)
* `maxlength = 2 * s_strat.targetwalkers + 100` - upper limit on the length of `v`; when reached, `lomc!` will abort
* `style = IsStochasticInteger()` - set [`StochasticStyle`](@ref) for default `v`; unused if `v` is specified.
* `post_step::NTuple{N,<:PostStepStrategy} = ()` - extract observables (e.g. [`ProjectedEnergy`](@ref)), see [`PostStepStrategy`](@ref).
* `replica::ReplicaStrategy = NoStats(1)` - run several synchronised simulation, see [`ReplicaStrategy`](@ref).
* `r_strat::ReportingStrategy = ReportDFAndInfo()` - how and when to report results, see [`ReportingStrategy`](@ref)
* `τ_strat::TimeStepStrategy = ConstantTimeStep()` - adjust time step dynamically, see [`TimeStepStrategy`](@ref)
* `m_strat::MemoryStrategy = NoMemory()` - experimental: inject memory noise, see [`MemoryStrategy`](@ref)
* `threading = :auto` - can be used to control the use of multithreading (overridden by `wm`)
  * `:auto` - use multithreading if `s_strat.targetwalkers ≥ 500`
  * `true` - use multithreading if available (set shell variable `JULIA_NUM_THREADS`!)
  * `false` - run on single thread
* `wm` - working memory; if set, it controls the use of multithreading and overrides `threading`; is mutated
* `df = DataFrame()` - when called with `AbstractHamiltonian` argument, a `DataFrame` can be passed into `lomc!` that will be pushed into.

# Return values

`lomc!` returns a named tuple with the following fields:

* `df`: a `DataFrame` with all statistics being reported.
* `state`: a `QMCState` that can be used for continuations.

# Example

```jldoctest
julia> add = BoseFS((1,2,3));

julia> hamiltonian = HubbardReal1D(add);

julia> df1, state = lomc!(hamiltonian);

julia> df2, _ = lomc!(state, df1; laststep=200); # Continuation run

julia> size(df1)
(100, 12)

julia> size(df2)
(200, 12)
```
"""
function lomc!(ham, v; df=DataFrame(), kwargs...)
    state = QMCState(ham, v; kwargs...)
    return lomc!(state, df)
end
function lomc!(ham; style=IsStochasticInteger(), kwargs...)
    v = DVec(starting_address(ham)=>10; style)
    return lomc!(ham, v; kwargs...)
end
# For continuation, you can pass a QMCState and a DataFrame
function lomc!(state::QMCState, df=DataFrame(); laststep=0)
    report = Report()
    if !iszero(laststep)
        state.laststep = laststep
    end

    # Sanity checks.
    step, laststep = state.step, state.laststep
    for replica in state.replicas
        ConsistentRNG.check_crng_independence(replica.v)
        @assert replica.params.step == step
        @assert replica.params.laststep == laststep
    end
    # Get we will use the first replica's step to keep track of the step. Perhaps step should
    # be moved to QMCState?
    while step < laststep
        step += 1
        report!(state.r_strat, step, report, :steps, step)
        # This loop could be threaded if num_threads() == num_replicas? MPIData would need
        # to be aware of the replica's id and use that in communication.
        success = true
        for replica in state.replicas
            success &= advance!(report, state, replica)
        end
        replica_names, replica_values = replica_stats(state.replica, state.replicas)
        report!(state.r_strat, step, report, replica_names, replica_values)
        report_after_step(state.r_strat, step, report, state)
        ensure_correct_lengths(report)
        !success && break
    end

    # Put report into DataFrame and merge with `df`. We are assuming the previous `df` is
    # compatible, which should be the case if the run is an actual continuation. Maybe the
    # DataFrames should be merged in a more permissive manner?
    result_df = finalize_report!(state.r_strat, report)
    if !isempty(df)
        return (; df=vcat(df, result_df), state)
    else
        return (; df=result_df, state)
    end
end

"""
    advance!(report::Report, state::QMCState, replica::ReplicaState)

Advance the `replica` by one step. The `state` is used only to access the various strategies
involved. Steps, stats, and computed quantities are written to the `report`.

Returns `true` if the step was successful.
"""
function advance!(
    report, state::QMCState, replica::ReplicaState{T}
) where {T}
    @unpack hamiltonian, m_strat, r_strat, s_strat, τ_strat = state
    @unpack v, w, pnorm, params, id = replica
    @unpack step, shiftMode, shift, dτ = params
    step += 1

    # Step
    v, w, step_stat_values, step_stat_names, shift_noise = fciqmc_step!(
        hamiltonian, v, shift, dτ, pnorm, w, 1.0; m_strat
    )
    v, update_dvec_stats = update_dvec!(v)

    # Stats
    tnorm = walkernumber(v)
    len = length(v)

    # Updates
    shift, shiftMode, pnorm = update_shift(
        s_strat, shift, shiftMode, tnorm, pnorm, dτ, step, nothing, v, w
    )
    dτ = update_dτ(τ_strat, dτ, tnorm)

    @pack! params = step, shiftMode, shift, dτ
    @pack! replica = v, w, pnorm, params

    # Note: post_step must be called after packing the values.
    post_step_stats = post_step(state.post_step, replica)

    # Reporting
    report!(
        r_strat, step, report,
        (dτ, shift, shiftMode, len, norm=tnorm), id,
    )
    report!(r_strat, step, report, step_stat_names, step_stat_values, id)
    report!(r_strat, step, report, update_dvec_stats, id)
    report!(r_strat, step, report, (;shift_noise), id)
    report!(state.r_strat, step, report, post_step_stats, id)

    if len == 0
        if length(state.replicas) > 1
            @error "population in replica$(replica.id) is dead. Aborting."
        else
            @error "population is dead. Aborting."
        end
        return false
    end
    if len > state.maxlength[]
        if length(state.replicas) > 1
            @error "`maxlength` reached in replica$(replica.id). Aborting."
        else
            @error "`maxlength` reached. Aborting."
        end
        return false
    end
    return true
end
