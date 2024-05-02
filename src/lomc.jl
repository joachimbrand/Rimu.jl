"""
    FirstOrderTransitionOperator(hamiltonian, shift, dτ) <: AbstractHamiltonian
    FirstOrderTransitionOperator(sp::DefaultShiftParameters, hamiltonian)

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

function FirstOrderTransitionOperator(sp::DefaultShiftParameters, hamiltonian)
    return FirstOrderTransitionOperator(hamiltonian, sp.shift, sp.time_step)
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
    SingleState(v, wm, pnorm, params, id)

Struct that holds all information needed for an independent run of the algorithm.

Can be advanced a step forward with [`advance!`](@ref).

# Fields

* `hamiltonian`: the model Hamiltonian.
* `v`: vector.
* `pv`: vector from the previous step.
* `wm`: working memory.
* `pnorm`: previous walker number (see [`walkernumber`](@ref)).
* `s_strat`: shift strategy.
* `τ_strat`: time step strategy.
* `shift`: shift to control the walker number.
* `dτ`: time step size.
* `id`: string ID appended to reported column names.

See also [`QMCState`](@ref), [`ReplicaStrategy`](@ref), [`replica_stats`](@ref),
[`lomc!`](@ref).
"""
mutable struct SingleState{H,V,W,SS<:ShiftStrategy,TS<:TimeStepStrategy,SP}
    # Future TODO: rename these fields, add interface for accessing them.
    hamiltonian::H
    v::V       # vector
    pv::V      # previous vector.
    wm::W      # working memory. Maybe working memories could be shared among replicas?
    s_strat::SS # shift strategy
    τ_strat::TS # time step strategy
    shift_parameters::SP # norm, shift, time_step; is mutable
    id::String # id is appended to column names
end

function SingleState(h, v, wm, s_strat, τ_strat, shift, dτ::Float64, id="")
    if isnothing(wm)
        wm = similar(v)
    end
    pv = zerovector(v)
    sp = initialise_shift_parameters(s_strat, shift, walkernumber(v), dτ)
    return SingleState(h, v, pv, wm, s_strat, τ_strat, sp, id)
    # return SingleState{
    #     typeof(h),typeof(v),typeof(wm),typeof(s_strat),typeof(τ_strat),typeof(sp)
    # }(h, v, pv, wm, s_strat, τ_strat, sp, id)
end

function Base.show(io::IO, r::SingleState)
    print(
        io,
        "SingleState(v: ", length(r.v), "-element ", nameof(typeof(r.v)),
        ", wm: ", length(r.wm), "-element ", nameof(typeof(r.wm)), ")"
    )
end

"""
    QMCState

Holds all information needed to run [`lomc!`](@ref), except the dataframe. Holds an
`NTuple` of [`SingleState`](@ref)s, the Hamiltonian, and various strategies that control
the algorithm. Constructed and returned by [`lomc!`](@ref).
"""
struct QMCState{
    H,
    N,
    R<:NTuple{N,<:SingleState},
    RS<:ReportingStrategy,
    RRS<:ReplicaStrategy,
    PS<:NTuple{<:Any,PostStepStrategy},
}
    hamiltonian::H
    replica_states::R
    maxlength::Ref{Int}
    step::Ref{Int}
    # laststep::Ref{Int}
    simulation_plan::SimulationPlan
    reporting_strategy::RS
    post_step::PS
    replica_strategy::RRS
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
    simulation_plan=nothing,
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
    reporting_strategy::ReportingStrategy=ReportDFAndInfo(),
    τ_strat::TimeStepStrategy=ConstantTimeStep(),
    threading=nothing,
    replica_strategy::ReplicaStrategy=NoStats(),
    post_step=(),
    maxlength= 2 * _n_walkers(v, s_strat) + 100, # padding for small walker numbers
)
    Hamiltonians.check_address_type(hamiltonian, keytype(v))
    # Set up reporting_strategy and params
    reporting_strategy = refine_r_strat(reporting_strategy)

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

    if isnothing(simulation_plan)
        simulation_plan = SimulationPlan(;
            starting_step = step,
            last_step = laststep
        )
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

    # Set up replica_states
    nreplicas = num_replicas(replica_strategy)
    if nreplicas > 1
        replica_states = ntuple(nreplicas) do i
            SingleState(
                hamiltonian,
                deepcopy(v),
                deepcopy(wm),
                deepcopy(s_strat),
                deepcopy(τ_strat),
                shift,
                dτ,
                "_$i")
        end
    else
        replica_states = (SingleState(hamiltonian, v, wm, s_strat, τ_strat, shift, dτ),)
    end

    return QMCState(
        hamiltonian, replica_states, Ref(Int(maxlength)),
        Ref(simulation_plan.starting_step), # step
        simulation_plan,
        # Ref(Int(laststep)),
        reporting_strategy, post_step, replica_strategy
    )
end

function Base.show(io::IO, st::QMCState)
    print(io, "QMCState")
    if length(st.replica_states) > 1
        print(io, " with ", length(st.replica_states), " replicas")
    end
    print(io, "\n  H:    ", st.hamiltonian)
    print(io, "\n  step: ", st.step[], " / ", st.simulation_plan.last_step)
    print(io, "\n  replicas: ")
    for (i, r) in enumerate(st.replica_states)
        print(io, "\n    $i: ", r)
    end
end

function report_default_metadata!(report::Report, state::QMCState)
    report_metadata!(report, "Rimu.PACKAGE_VERSION", Rimu.PACKAGE_VERSION)
    # add metadata from state
    replica = state.replica_states[1]
    shift_parameters=replica.shift_parameters
    report_metadata!(report, "laststep", state.simulation_plan.last_step)
    report_metadata!(report, "num_replicas", length(state.replica_states))
    report_metadata!(report, "hamiltonian", state.hamiltonian)
    report_metadata!(report, "reporting_strategy", state.reporting_strategy)
    report_metadata!(report, "s_strat", replica.s_strat)
    report_metadata!(report, "τ_strat", replica.τ_strat)
    report_metadata!(report, "dτ", shift_parameters.time_step)
    report_metadata!(report, "step", state.step[])
    report_metadata!(report, "shift", shift_parameters.shift)
    report_metadata!(report, "maxlength", state.maxlength[])
    report_metadata!(report, "post_step", state.post_step)
    report_metadata!(report, "v_summary", summary(state.replica_states[1].v))
    report_metadata!(report, "v_type", typeof(state.replica_states[1].v))
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
* `replica_strategy::ReplicaStrategy = NoStats(1)` - run several synchronised simulations, see
  [`ReplicaStrategy`](@ref). (Deprecated: `replica` is accepted as an alias for
  `replica_strategy`.)
* `reporting_strategy::ReportingStrategy = ReportDFAndInfo()` - how and when to report
  results, see [`ReportingStrategy`](@ref). (Deprecated: `r_strat` is accepted as an alias
  for `reporting_strategy`.)
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
(100, 10)

julia> size(df2)
(200, 10)

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
function lomc!(
    ham, v;
    df=DataFrame(),
    name="lomc!",
    metadata=nothing,
    r_strat=ReportDFAndInfo(),
    replica = NoStats(),
    kwargs...
)
    state = QMCState(
        ham, v;
        reporting_strategy=r_strat, replica_strategy=replica, kwargs...
    )
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

function lomc!(::AbstractMatrix, v=nothing; kwargs...)
    throw(ArgumentError("Using lomc! with a matrix is no longer supported. Use `MatrixHamiltonian` instead."))
end


"""
    advance!(report::Report, state::QMCState, replica::SingleState)

Advance the `replica` by one step. The `state` is used only to access the various strategies
involved. Steps, stats, and computed quantities are written to the `report`.

Returns `true` if the step was successful and calculation should proceed, `false` when
it should terminate.
"""
function advance!(report, state::QMCState, replica::SingleState)

    @unpack hamiltonian, reporting_strategy = state
    @unpack v, pv, wm, id, s_strat, τ_strat, shift_parameters = replica
    @unpack shift, pnorm, time_step = shift_parameters
    step = state.step[]

    ### PROPAGATOR ACTS
    ### FROM HERE
    transition_op = FirstOrderTransitionOperator(shift_parameters, hamiltonian)

    # Step
    step_stat_names, step_stat_values, wm, pv = apply_operator!(wm, pv, v, transition_op)
    # pv was mutated and now contains the new vector.
    v, pv = (pv, v)

    # Stats
    tnorm = walkernumber(v)
    len = length(v)

    # Updates
    time_step = update_dτ(τ_strat, time_step, tnorm)

    shift_stats, proceed = update_shift_parameters!(
        s_strat, shift_parameters, tnorm, v, pv, step, report
    )

    @pack! replica = v, pv, wm
    ### TO HERE

    if step % reporting_interval(state.reporting_strategy) == 0
        # Note: post_step must be called after packing the values.
        post_step_stats = post_step_action(state.post_step, replica, step)

        # Reporting
        report!(reporting_strategy, step, report, (; dτ=time_step, len), id)
        report!(reporting_strategy, step, report, shift_stats, id) # shift, norm, shift_mode

        report!(reporting_strategy, step, report, step_stat_names, step_stat_values, id)
        report!(reporting_strategy, step, report, post_step_stats, id)
    end

    if len == 0
        if length(state.replica_states) > 1
            @error "population in replica $(replica.id) is dead. Aborting."
        else
            @error "population is dead. Aborting."
        end
        return false
    end
    if len > state.maxlength[]
        if length(state.replica_states) > 1
            @error "`maxlength` reached in replica $(replica.id). Aborting."
        else
            @error "`maxlength` reached. Aborting."
        end
        return false
    end
    return proceed # Bool
end
