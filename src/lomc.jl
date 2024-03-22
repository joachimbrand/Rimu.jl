"""
    FirstOrderTransitionOperator(hamiltonian, shift, dτ) <: AbstractHamiltonian

First order transition operator
```math
𝐓 = 1 + dτ(S - 𝐇)
```
where ``𝐇`` is the `hamiltonian` and ``S`` is the `shift`.

``𝐓`` represents the first order expansion of the exponential evolution operator
of the imaginary-time Schrödinger equation (Euler step) and repeated application
will project out the ground state eigenvector of the `hamiltonian`.  It is the
transition operator used in FCIQMC.
"""
struct FirstOrderTransitionOperator{T,S,H} <: AbstractHamiltonian{T}
    hamiltonian::H
    shift::S
    dτ::Float64

    function FirstOrderTransitionOperator(hamiltonian::H, shift::S, dτ) where {H,S}
        T = eltype(hamiltonian)
        return new{T,S,H}(hamiltonian, shift, Float64(dτ))
    end
end

function Hamiltonians.diagonal_element(t::FirstOrderTransitionOperator, add)
    diag = diagonal_element(t.hamiltonian, add)
    return 1 - t.dτ * (diag - t.shift)
end

struct FirstOrderOffdiagonals{
    A,V,O<:AbstractVector{Tuple{A,V}}
} <: AbstractVector{Tuple{A,V}}
    dτ::Float64
    offdiagonals::O
end
function Hamiltonians.offdiagonals(t::FirstOrderTransitionOperator, add)
    return FirstOrderOffdiagonals(t.dτ, offdiagonals(t.hamiltonian, add))
end
Base.size(o::FirstOrderOffdiagonals) = size(o.offdiagonals)

function Base.getindex(o::FirstOrderOffdiagonals, i)
    add, val = o.offdiagonals[i]
    return add, -val * o.dτ
end


"""
    ReplicaState(v, wm, pnorm, params, id)

Struct that holds all information needed for an independent run of the algorithm.

Can be advanced a step forward with [`advance!`](@ref).

# Fields

* `hamiltonian`: the model Hamiltonian.
* `v`: vector.
* `pv`: vector from the previous step.
* `wm`: working memory.
* `pnorm`: previous walker number (see [`walkernumber`](@ref)).
* `shift`: shift to control the walker number.
* `dτ`: time step size.
* `id`: string ID appended to reported column names.

See also [`QMCState`](@ref), [`ReplicaStrategy`](@ref), [`replica_stats`](@ref),
[`lomc!`](@ref).
"""
mutable struct ReplicaState{H,T,V,W}
    # Future TODO: rename these fields, add interface for accessing them.
    hamiltonian::H
    v::V       # vector
    pv::V      # previous vector.
    wm::W      # working memory. Maybe working memories could be shared among replicas?
    pnorm::T   # previous walker number - used to control the shift
    shift::T   # shift to control the walker number
    dτ::Float64 # time step size
    id::String # id is appended to column names
end

function ReplicaState(h, v, wm, shift, dτ, id="")
    if isnothing(wm)
        wm = similar(v)
    end
    pv = zerovector(v)
    return ReplicaState(h, v, pv, wm, walkernumber(v), shift, dτ, id)
end

function Base.show(io::IO, r::ReplicaState)
    print(
        io,
        "ReplicaState(v: ", length(r.v), "-element ", nameof(typeof(r.v)),
        ", wm: ", length(r.wm), "-element ", nameof(typeof(r.wm)), ")"
    )
end

"""
    QMCState

Holds all information needed to run [`lomc!`](@ref), except the dataframe. Holds an
`NTuple` of [`ReplicaState`](@ref)s, the Hamiltonian, and various strategies that control
the algorithm. Constructed and returned by [`lomc!`](@ref).
"""
struct QMCState{
    H,
    N,
    R<:NTuple{N,<:ReplicaState},
    RS<:ReportingStrategy,
    SS<:ShiftStrategy,
    TS<:TimeStepStrategy,
    RRS<:ReplicaStrategy,
    PS<:NTuple{<:Any,PostStepStrategy},
}
    hamiltonian::H
    replicas::R
    maxlength::Ref{Int}
    step::Ref{Int}
    laststep::Ref{Int}
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
    step=nothing,
    laststep=nothing,
    dτ=nothing,
    shift=nothing,
    wm=nothing,
    style=nothing,
    targetwalkers=1000,
    address=starting_address(hamiltonian),
    params::FciqmcRunStrategy=RunTillLastStep(
        laststep = 100,
        shift = float(valtype(v))(diagonal_element(hamiltonian, address))
    ),
    s_strat::ShiftStrategy=DoubleLogUpdate(; targetwalkers),
    r_strat::ReportingStrategy=ReportDFAndInfo(),
    τ_strat::TimeStepStrategy=ConstantTimeStep(),
    threading=nothing,
    replica::ReplicaStrategy=NoStats(),
    post_step=(),
    maxlength= 2 * _n_walkers(v, s_strat) + 100, # padding for small walker numbers
)
    Hamiltonians.check_address_type(hamiltonian, keytype(v))
    # Set up r_strat and params
    r_strat = refine_r_strat(r_strat)

    # eventually we want to deprecate the use of params
    if !isnothing(params)
        if !isnothing(step)
            params.step = step
        end
        if !isnothing(laststep)
            params.laststep = laststep
        end
        if !isnothing(dτ)
            params.dτ = dτ
        end
        if !isnothing(shift)
            params.shift = shift
        end
        step = params.step
        dτ = params.dτ
        shift = params.shift
        laststep = params.laststep
    else
        if isnothing(step)
            step = 0
        end
        if isnothing(laststep)
            laststep = 100
        end
        if isnothing(dτ)
            dτ = 0.01
        end
        if isnothing(shift)
            shift = float(valtype(v))(diagonal_element(hamiltonian, address))
        end
    end

    if threading ≠ nothing
        @warn "Starting vector is provided. Ignoring `threading=$threading`."
    end
    if style ≠ nothing
        @warn "Starting vector is provided. Ignoring `style=$style`."
    end
    wm = isnothing(wm) ? working_memory(v) : wm

    # Set up post_step
    if !(post_step isa Tuple)
        post_step = (post_step,)
    end

    # Set up replicas
    nreplicas = num_replicas(replica)
    if nreplicas > 1
        replicas = ntuple(nreplicas) do i
            ReplicaState(
                hamiltonian,
                deepcopy(v),
                deepcopy(wm),
                shift,
                dτ,
                "_$i")
        end
    else
        replicas = (ReplicaState(hamiltonian, v, wm, shift, dτ),)
    end

    return QMCState(
        hamiltonian, replicas, Ref(Int(maxlength)), Ref(Int(step)),
        Ref(Int(laststep)),
        r_strat, s_strat, τ_strat, post_step, replica
    )
end

function Base.show(io::IO, st::QMCState)
    print(io, "QMCState")
    if length(st.replicas) > 1
        print(io, " with ", length(st.replicas), " replicas")
    end
    print(io, "\n  H:    ", st.hamiltonian)
    print(io, "\n  step: ", st.step[], " / ", st.laststep[])
    print(io, "\n  replicas: ")
    for (i, r) in enumerate(st.replicas)
        print(io, "\n    $i: ", r)
    end
end

function report_default_metadata!(report::Report, state::QMCState)
    report_metadata!(report, "Rimu.PACKAGE_VERSION", Rimu.PACKAGE_VERSION)
    # add metadata from state
    report_metadata!(report, "laststep", state.laststep[])
    report_metadata!(report, "num_replicas", length(state.replicas))
    report_metadata!(report, "hamiltonian", state.hamiltonian)
    report_metadata!(report, "r_strat", state.r_strat)
    report_metadata!(report, "s_strat", state.s_strat)
    report_metadata!(report, "τ_strat", state.τ_strat)
    params = state.replicas[1]
    report_metadata!(report, "dτ", params.dτ)
    report_metadata!(report, "step", state.step[])
    report_metadata!(report, "shift", params.shift)
    report_metadata!(report, "maxlength", state.maxlength[])
    report_metadata!(report, "post_step", state.post_step)
    report_metadata!(report, "v_summary", summary(state.replicas[1].v))
    report_metadata!(report, "v_type", typeof(state.replicas[1].v))
    return report
end

"""
    default_starting_vector(hamiltonian::AbstractHamiltonian; kwargs...)
    default_starting_vector(
        address=starting_address(hamiltonian);
        style=IsStochasticInteger(),
        initiator=NonInitiator(),
        threading=nothing
    )
Return a default starting vector for [`lomc!`](@ref). The default choice for the starting
vector is
```julia
v = PDVec(address => 10; style, initiator)
```
if threading is available, or otherwise
```julia
v = DVec(address => 10; style)
```
if `initiator == NonInitiator()`, and
```julia
v = InitiatorDVec(address => 10; style, initiator)
```
if not. See [`PDVec`](@ref), [`DVec`](@ref), [`InitiatorDVec`](@ref),
[`StochasticStyle`](@ref), and [`InitiatorRule`].
"""
function default_starting_vector(
    hamiltonian::AbstractHamiltonian;
    address=starting_address(hamiltonian), kwargs...
)
    return default_starting_vector(address; kwargs...)
end
function default_starting_vector(address;
    style=IsStochasticInteger(),
    threading=nothing,
    initiator=NonInitiator(),
)
    if isnothing(threading)
        threading = Threads.nthreads() > 1
    end
    if threading
        v = PDVec(address => 10; style, initiator)
    elseif initiator isa NonInitiator
        v = DVec(address => 10; style)
    else
        v = InitiatorDVec(address => 10; style, initiator)
    end
    return v
end

"""
    lomc!(ham::AbstractHamiltonian, [v]; kwargs...) -> df, state
    lomc!(state::QMCState, [df]; kwargs...) -> df, state

Linear operator Monte Carlo: Perform a projector quantum Monte Carlo simulation for
determining the lowest eigenvalue of `ham`. The details of the simulation are controlled by
the optional keyword arguments and by the type of the optional starting vector `v`.
Alternatively, a `QMCState` can be passed in to continue a previous simulation.

# Common keyword arguments and defaults:

* `laststep = 100` - controls the number of steps.
* `dτ = 0.01` - time step.
* `targetwalkers = 1000` - target for the 1-norm of the coefficient vector.
* `address = starting_address(ham)` - set starting address for default `v` and `shift`.
* `style = IsStochasticInteger()` - set [`StochasticStyle`](@ref) for default `v`; unused
  if `v` is specified.
* `initiator = NonInitiator()` - set [`InitiatorRule`](@ref) for default `v`; unused if `v`
  is specified.
* `threading` - default is to use multithreading and
  [MPI](https://juliaparallel.org/MPI.jl/latest/) if multiple threads are available. Set to
  `true` to force [`PDVec`](@ref) for the starting vector, `false` for serial computation;
  unused if `v` is specified.
* `shift = diagonal_element(ham, address)` - initial value of shift.
* `post_step::NTuple{N,<:PostStepStrategy} = ()` - extract observables (e.g.
  [`ProjectedEnergy`](@ref)), see [`PostStepStrategy`](@ref).
* `replica::ReplicaStrategy = NoStats(1)` - run several synchronised simulations, see
  [`ReplicaStrategy`](@ref).
* `r_strat::ReportingStrategy = ReportDFAndInfo()` - how and when to report results, see
  [`ReportingStrategy`](@ref)
* `name = "lomc!"` - name displayed in progress bar (via `ProgressLogging`)
* `metadata` - user-supplied metadata to be added to the report `df`. Must be an iterable of
  pairs or a `NamedTuple`, e.g. `metadata = ("key1" => "value1", "key2" => "value2")`.
  All metadata is converted to strings.

Some metadata is automatically added to the report `df` including
[`Rimu.PACKAGE_VERSION`](@ref) and data from `state`.

# Return values

`lomc!` returns a named tuple with the following fields:

* `df`: a `DataFrame` with all statistics being reported.
* `state`: a `QMCState` that can be used for continuations.

# Example

```jldoctest
julia> add = BoseFS(1,2,3);

julia> hamiltonian = HubbardReal1D(add);

julia> df1, state = lomc!(hamiltonian; targetwalkers=500, laststep=100);

julia> df2, _ = lomc!(state, df1; laststep=200, metadata=(;info="cont")); # Continuation run

julia> size(df1)
(100, 11)

julia> size(df2)
(200, 11)

julia> using DataFrames; metadata(df2, "info") # retrieve custom metadata
"cont"

julia> metadata(df2, "hamiltonian") # some metadata is automatically added
"HubbardReal1D(BoseFS{6,3}(1, 2, 3); u=1.0, t=1.0)"
```

# Further keyword arguments and defaults:

* `τ_strat::TimeStepStrategy = ConstantTimeStep()` - adjust time step or not, see
  [`TimeStepStrategy`](@ref)
* `s_strat::ShiftStrategy = DoubleLogUpdate(; targetwalkers, ζ = 0.08, ξ = ζ^2/4)` -
  how to update the `shift`, see [`ShiftStrategy`](@ref).
* `maxlength = 2 * s_strat.targetwalkers + 100` - upper limit on the length of `v`; when
  reached, `lomc!` will abort
* `wm` - working memory for re-use in subsequent calculations; is mutated.
* `df = DataFrame()` - when called with `AbstractHamiltonian` argument, a `DataFrame` can
  be passed for merging with the report `df`.

The default choice for the starting vector is
`v = default_starting_vector(; address, style, threading, initiator)`.
See [`default_starting_vector`](@ref), [`PDVec`](@ref), [`DVec`](@ref),
[`StochasticStyle`](@ref), and [`InitiatorRule`](@ref).
"""
function lomc!(ham, v; df=DataFrame(), name="lomc!", metadata=nothing, kwargs...)
    state = QMCState(ham, v; kwargs...)
    return lomc!(state, df; name, metadata)
end
function lomc!(
    ham;
    style=IsStochasticInteger(),
    threading=nothing,
    address=starting_address(ham),
    initiator=NonInitiator(),
    kwargs...
)
    v = default_starting_vector(address; style, threading, initiator)
    return lomc!(ham, v; address, kwargs...) # pass address for setting the default shift
end
# For continuation, you can pass a QMCState and a DataFrame
function lomc!(state::QMCState, df=DataFrame(); laststep=0, name="lomc!", metadata=nothing)
    if !iszero(laststep)
        state.laststep[] = laststep
    end

    # initialise report
    report = Report()
    report_default_metadata!(report, state)
    isnothing(metadata) || report_metadata!(report, metadata) # add user metadata

    # Sanity checks.
    check_transform(state.replica, state.hamiltonian)

    # main loop
    laststep = state.laststep[]
    step = initial_step = state.step[]
    update_steps = max((laststep - initial_step) ÷ 200, 100) # log often but not too often

    @withprogress name=name while step < laststep
        step = state.step[] += 1
        if step % reporting_interval(state.r_strat) == 0
            report!(state.r_strat, step, report, :steps, step)
        end
        # This loop could be threaded if num_threads() == num_replicas? MPIData would
        # need to be aware of the replica's id and use that in communication.
        success = true
        for replica in state.replicas
            success &= advance!(report, state, replica)
        end
        if step % reporting_interval(state.r_strat) == 0
            replica_names, replica_values = replica_stats(state.replica, state.replicas)
            report!(state.r_strat, step, report, replica_names, replica_values)
            report_after_step(state.r_strat, step, report, state)
            ensure_correct_lengths(report)
        end
        if step % update_steps == 0 # for updating progress bars
            @logprogress (step-initial_step)/(laststep-initial_step)
        end
        !success && break
    end

    # Put report into DataFrame and merge with `df`. We are assuming the previous `df` is
    # compatible, which should be the case if the run is an actual continuation. Maybe the
    # DataFrames should be merged in a more permissive manner?
    result_df = finalize_report!(state.r_strat, report)
    if !isempty(df)
        df = vcat(df, result_df) # metadata is not propagated
        for (key, val) in get_metadata(report) # add metadata
            DataFrames.metadata!(df, key, val)
        end
        return (; df, state)
    else
        return (; df=result_df, state)
    end
end

"""
    advance!(report::Report, state::QMCState, replica::ReplicaState)

Advance the `replica` by one step. The `state` is used only to access the various strategies
involved. Steps, stats, and computed quantities are written to the `report`.

Returns `true` if the step was successful and calculation should proceed, `false` when
it should terminate.
"""
function advance!(report, state::QMCState, replica::ReplicaState)

    @unpack hamiltonian, r_strat, s_strat, τ_strat = state
    @unpack v, pv, wm, pnorm, dτ, shift, id = replica
    step = state.step[]

    ### PROPAGATOR ACTS
    ### FROM HERE
    transition_op = FirstOrderTransitionOperator(hamiltonian, shift, dτ)

    # Step
    step_stat_names, step_stat_values, wm, pv = apply_operator!(wm, pv, v, transition_op)
    # pv was mutated and now contains the new vector.
    v, pv = (pv, v)

    # Stats
    tnorm = walkernumber(v)
    len = length(v)

    # Updates
    shift, pnorm, proceed = update_shift(
        s_strat, shift, tnorm, pnorm, dτ, step, nothing, v, pv
    )
    dτ = update_dτ(τ_strat, dτ, tnorm)

    @pack! replica = dτ, shift, v, pv, wm, pnorm
    ### TO HERE

    if step % reporting_interval(state.r_strat) == 0
        # Note: post_step must be called after packing the values.
        post_step_stats = post_step_action(state.post_step, replica, step)

        # Reporting
        report!(
            r_strat, step, report,
            (dτ, shift, len, norm=tnorm), id,
        )
        if hasfield(typeof(s_strat), :shift_mode)
            report!(r_strat, step, report, (; shift_mode=s_strat.shift_mode[]), id)
        end
        report!(r_strat, step, report, step_stat_names, step_stat_values, id)
        report!(state.r_strat, step, report, post_step_stats, id)
    end

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
    return proceed # Bool
end
