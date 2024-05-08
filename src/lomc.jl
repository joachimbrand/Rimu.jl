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
    lomc!(ham::AbstractHamiltonian, [v]; kwargs...) -> df, state
    lomc!(state::ReplicaState, [df]; kwargs...) -> df, state

Linear operator Monte Carlo: Perform a projector quantum Monte Carlo simulation for
determining the lowest eigenvalue of `ham`. The details of the simulation are controlled by
the optional keyword arguments and by the type of the optional starting vector `v`.
Alternatively, a `ReplicaState` can be passed in to continue a previous simulation.

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
* `post_step_strategy::NTuple{N,<:PostStepStrategy} = ()` - extract observables (e.g.
  [`ProjectedEnergy`](@ref)), see [`PostStepStrategy`](@ref). (Deprecated: `post_step` is
  accepted as an alias for `post_step_strategy`.)
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
* `state`: a `ReplicaState` that can be used for continuations.

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
!!! warning
    The use of this `lomc!` is deprecated. Use
    [`ProjectorMonteCarloProblem`](@ref) and [`solve`](@ref) instead.
"""
function lomc!(
    ham, v;
    df=DataFrame(),
    name="lomc!",
    metadata=nothing,
    r_strat=ReportDFAndInfo(),
    replica = NoStats(),
    post_step = (),
    kwargs...
)
    state = ReplicaState(
        ham, v;
        reporting_strategy = r_strat, # deprecations
        replica_strategy = replica,
        post_step_strategy = post_step,
        kwargs...
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
    advance!(algorithm, report::Report, state::ReplicaState, replica::SingleState)

Advance the `replica` by one step according to the `algorithm`. The `state` is used only
to access the various strategies involved. Steps, stats, and computed quantities are written
to the `report`.

Returns `true` if the step was successful and calculation should proceed, `false` when
it should terminate.
"""
function advance!(::FCIQMC, report, state::ReplicaState, replica::SingleState)

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
        # Note: post_step_stats must be called after packing the values.
        post_step_stats = post_step_action(state.post_step_strategy, replica, step)

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

function advance!(algorithm, report, state::ReplicaState, replica::SpectralState{1})
    return advance!(algorithm, report, state, only(replica.spectral_states))
end
# TODO: add advance! for SpectralState{N} where N > 1
