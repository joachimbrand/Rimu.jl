"""
    ReplicaState(hamiltonian, v, w, pnorm, r_strat)

Struct that holds all information needed for an independent run of the algorithm.

Can be advanced a step forward with [`advance!`](@ref).
"""
mutable struct ReplicaState{T,V,W,R<:FciqmcRunStrategy{T}}
    v::V       # vector
    w::W       # working memory. Maybe working memories could be shared among replicas?
    pnorm::T   # previous walker number - used to control the shift
    params::R  # params: step, laststep, dτ...
    id::String # id is appended to column names
end

function ReplicaState(v, w, params, id="")
    if isnothing(w)
        w = similar(v)
    end
    return ReplicaState(v, w, walkernumber(v), params, id)
end

function Base.show(io::IO, r::ReplicaState)
    print(io, "ReplicaState(v::", nameof(typeof(r.v)), ", w::", nameof(typeof(r.w)), ")")
end

"""
    QMCState

Holds all inforamtion needed to run FCIQMC, except the data frame. Holds a `NTuple` of
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
    O,
}
    hamiltonian::H
    replicas::NTuple{N,R}
    maxlength::Int

    m_strat::MS
    r_strat::RS
    s_strat::SS
    τ_strat::TS

    # Report x†⋅operator⋅y
    # In the future, this should be replaced with a struct a la ReplicaStrategy.
    operator::O
end

function QMCState(
    hamiltonian, v;
    laststep=nothing,
    dτ=nothing,
    threading=:auto,
    wm=nothing,
    params::FciqmcRunStrategy=RunTillLastStep{float(valtype(v))}(),
    s_strat::ShiftStrategy=DoubleLogUpdate(),
    r_strat::ReportingStrategy=EveryTimeStep(),
    τ_strat::TimeStepStrategy=ConstantTimeStep(),
    m_strat::MemoryStrategy=NoMemory(),
    maxlength=2 * max(real(s_strat.targetwalkers), imag(s_strat.targetwalkers)),
    num_replicas=1,
    report_xHy=false,
    operator=report_xHy ? hamiltonian : nothing,
)
    # Checks
    if num_replicas < 1
        error("need at least one replica.")
    elseif !isnothing(operator) && num_replicas < 2
        error(
            "operator $operator needs at least two replicas. ",
            "Set the number of replicas with the `num_replicas` keyowrd argument."
        )
    end
    # Set up default arguments
    r_strat = refine_r_strat(r_strat, hamiltonian)
    if !isnothing(laststep)
        params.laststep = laststep
    end
    if !isnothing(dτ)
        params.dτ = dτ
    end
    wm = default_working_memory(threading, v, s_strat)
    # Set up replicas
    if num_replicas > 1
        replicas = ntuple(num_replicas) do i
            ReplicaState(deepcopy(v), deepcopy(wm), deepcopy(params), "_$i")
        end
    else
        replicas = (ReplicaState(v, wm, params, ""),)
    end

    return QMCState(
        hamiltonian, replicas, maxlength, m_strat, r_strat, s_strat, τ_strat, operator
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
    else
        # This will error
        return setfield!(state, key, value)
    end
end

function default_working_memory(threading, v, s_strat)
    if threading == :auto
        threading = max(real(s_strat.targetwalkers),imag(s_strat.targetwalkers)) ≥ 500
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
    for r in st.replicas
        print(io, "\n  ⋅ ", r)
    end
end

"""
    lomc!(ham, v; kwargs...)

Linear operator Monte Carlo: Perform the FCIQMC algorithm for determining the lowest
eigenvalue of `ham`. `v` can be a single starting vector of (wrapped) type
`:<AbstractDVec`.

Returns a `DataFrame` with various statistics and a `QMCState` containing all information
required for continuation runs.

# Keyword arguments, defaults, and precedence:

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
* `num_replicas` - set the number of replica runs to run.
* `operator` - set an operator to use with replicas.

# Return values

`lomc!` returns a named tuple with the following fields:

* `df`: a `DataFrame` with all statistics being reported.
* `state`: a `QMCState` that can be used for continuations.

# Example

```jldoctest
julia> add = BoseFS((1,2,3));

julia> H = HubbardReal1D(add);

julia> dv = DVec(add => 1);

julia> df1, state = lomc!(H, dv);

julia> df2, _ = lomc!(state, df1; laststep=200); # Contuniation run

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
# For continuation, you can pass a QMCState and a DataFrame
function lomc!(state::QMCState, df=DataFrame(); laststep=0)
    report = Report()
    if !iszero(laststep)
        for replica in state.replicas
            replica.params.laststep = laststep
        end
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
        # This loop could be threaded if num_threads() == num_replicas? MPIData would need
        # to be aware of the replica's id and use that in communication.
        success = true
        for replica in state.replicas
            success &= advance!(report, state, replica)
        end
        # The following should probably be replaced with some kind of ReplicaStrategy.
        # Right now it is designed to work similarly to the old implementation.
        if length(state.replicas) ≥ 2
            c1 = state.replicas[1].v
            c2 = state.replicas[2].v
            xdoty = c1 ⋅ c2
            report!(state.r_strat, step, report, :xdoty, xdoty)
            if !isnothing(state.operator)
                xHy = dot(c1, state.operator, c2)
                report!(state.r_strat, step, report, :xHy, xHy)
            end
        end
        print_report(state.r_strat, step, report, state)
        !success && break
    end

    # Put report into DataFrame and merge with `df`. We are assuming the previous `df` is
    # compatible, which should be the case if the run is an actual continuation. Maybe the
    # DataFrames should be merged in a more permissive manner?
    df = vcat(df, DataFrame(report))
    return (; df, state)
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
    @unpack step, laststep, shiftMode, shift, dτ = params
    step += 1

    v, w, step_stats, stat_names, shift_noise = fciqmc_step!(
        hamiltonian, v, shift, dτ, pnorm, w, 1.0; m_strat
    )
    v, update_stats = update_dvec!(v)
    tnorm = walkernumber(v)
    len = length(v)

    proj_observables = compute_proj_observables(v, hamiltonian, r_strat)

    shift, shiftMode, pnorm = update_shift(
        s_strat, shift, shiftMode, tnorm, pnorm, dτ, step, nothing, v, w
    )
    dτ = update_dτ(τ_strat, dτ, tnorm)

    @pack! params = step, laststep, shiftMode, shift, dτ
    @pack! replica = v, w, pnorm, params

    report!(
        r_strat, step, report,
        (steps=step, dτ, shift, shiftMode, len, norm=tnorm), id,
    )
    report!(r_strat, step, report, proj_observables, id)
    report!(r_strat, step, report, stat_names, step_stats, id)
    report!(r_strat, step, report, update_stats, id)
    report!(r_strat, step, report, (;shift_noise), id)

    if len == 0
        if length(state.replicas) > 1
            @error "population in replica$(replica.id) is dead. Aborting."
        else
            @error "population is dead. Aborting."
        end
        return false
    end
    if len > state.maxlength
        if length(state.replicas) > 1
            @error "`maxlength` reached in replica$(replica.id). Aborting."
        else
            @error "`maxlength` reached. Aborting."
        end
        return false
    end
    return true
end
