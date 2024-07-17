# This file contains deprecated code. This code will be removed in a future release.

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
julia> address = BoseFS(1,2,3);

julia> hamiltonian = HubbardReal1D(address);

julia> df1, state = lomc!(hamiltonian; targetwalkers=500, laststep=100);

julia> df2, _ = lomc!(state, df1; laststep=200, metadata=(;info="cont")); # Continuation run

julia> size(df1)
(100, 9)

julia> size(df2)
(200, 9)

julia> using DataFrames; metadata(df2, "info") # retrieve custom metadata
"cont"

julia> metadata(df2, "hamiltonian") # some metadata is automatically added
"HubbardReal1D(fs\\"|1 2 3⟩\\"; u=1.0, t=1.0)"
```

# Further keyword arguments and defaults:

* `τ_strat::TimeStepStrategy = ConstantTimeStep()` - adjust time step or not, see
  [`TimeStepStrategy`](@ref)
* `s_strat::ShiftStrategy = DoubleLogUpdate(; target_walkers=targetwalkers, ζ = 0.08, ξ = ζ^2/4)` -
  how to update the `shift`, see [`ShiftStrategy`](@ref).
* `maxlength = 2 * s_strat.target_walkers + 100` - upper limit on the length of `v`; when
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
    df = DataFrame(),
    name = "lomc!",
    metadata = nothing,
    reporting_strategy = ReportDFAndInfo(),
    r_strat = reporting_strategy,
    targetwalkers = 1000,
    s_strat = DoubleLogUpdate(; target_walkers=targetwalkers),
    τ_strat = ConstantTimeStep(),
    replica_strategy = NoStats(),
    replica = replica_strategy,
    post_step_strategy = (),
    post_step = post_step_strategy,
    step=nothing,
    laststep=nothing,
    dτ=nothing,
    shift=nothing,
    address=starting_address(ham),
    params::FciqmcRunStrategy=RunTillLastStep(
        laststep=100,
        shift=float(valtype(v))(diagonal_element(ham, address))
    ),
    maxlength = nothing,
    wm = nothing
)
    if !isnothing(wm)
        @warn "The `wm` argument has been removed and will be ignored."
    end
    if hasfield(typeof(s_strat), :target_walkers)
        targetwalkers = s_strat.target_walkers
    end
    if isnothing(maxlength)
        maxlength = round(Int, 2 * abs(targetwalkers) + 100)
    end

    # eventually we want to deprecate the use of params
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
    starting_step = params.step
    time_step = params.dτ
    shift = params.shift
    last_step = params.laststep

    p = ProjectorMonteCarloProblem(ham;
        algorithm=FCIQMC(s_strat, τ_strat),
        start_at = v,
        starting_step,
        last_step,
        time_step,
        shift,
        replica_strategy = replica,
        reporting_strategy = r_strat,
        post_step_strategy = post_step,
        maxlength,
        metadata,
        display_name = name,
        random_seed = false
    )
    simulation = init(p; copy_vectors=false)
    solve!(simulation)

    # Put report into DataFrame and merge with `df`. We are assuming the previous `df` is
    # compatible, which should be the case if the run is an actual continuation. Maybe the
    # DataFrames should be merged in a more permissive manner?
    result_df = DataFrame(simulation)

    if !isempty(df)
        df = vcat(df, result_df) # metadata is not propagated
        for (key, val) in get_metadata(simulation.report) # add metadata
            DataFrames.metadata!(df, key, val)
        end
        return (; df, state=simulation.state)
    else
        return (; df=result_df, state=simulation.state)
    end
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

# methods for backward compatibility
function lomc!(state::ReplicaState, df=DataFrame(); laststep=0, name="lomc!", metadata=nothing)
    if !iszero(laststep)
        state = @set state.simulation_plan.last_step = laststep
    end
    @unpack spectral_states, maxlength, step, simulation_plan,
    reporting_strategy, post_step_strategy, replica_strategy = state
    first_replica = first(state) # SingleState
    @unpack hamiltonian = first_replica
    @assert step[] ≥ simulation_plan.starting_step
    problem = ProjectorMonteCarloProblem(hamiltonian;
        algorithm=first_replica.algorithm,
        start_at=first_replica.v,
        initial_shift_parameters=first_replica.shift_parameters,
        replica_strategy,
        reporting_strategy,
        post_step_strategy,
        maxlength=maxlength[],
        simulation_plan,
        metadata,
        display_name=name,
        random_seed=false
    )
    report = Report()
    report_default_metadata!(report, state)
    report_metadata!(report, problem.metadata) # add user metadata
    # Sanity checks.
    check_transform(state.replica_strategy, hamiltonian)

    simulation = PMCSimulation(
        problem, state, report, false, false, false, "", 0.0
    )
    solve!(simulation)

    # Put report into DataFrame and merge with `df`. We are assuming the previous `df` is
    # compatible, which should be the case if the run is an actual continuation. Maybe the
    # DataFrames should be merged in a more permissive manner?
    result_df = DataFrame(simulation)

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
# This constructor is currently only used by lomc! and should not be used for new code.
function SingleState(h, v, wm, shift_strategy, time_step_strategy, shift, dτ::Float64, id="")
    if isnothing(wm)
        wm = similar(v)
    end
    pv = zerovector(v)
    sp = initialise_shift_parameters(shift_strategy, shift, walkernumber(v), dτ)
    alg = FCIQMC(; shift_strategy, time_step_strategy)
    return SingleState(h, alg, v, pv, wm, sp, id)
end
