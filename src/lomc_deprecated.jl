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
* `params::FciqmcRunStrategy = RunTillLastStep(laststep = 100, dτ = 0.01, shift =
  diagonal_element(ham, address)` -
  basic parameters of simulation state, see [`FciqmcRunStrategy`](@ref). Parameter values
  are overridden by explicit keyword arguments `laststep`, `dτ`, `shift`; is mutated.
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
    iter = _init(state, df; laststep, name, metadata)
    return CommonSolve.solve!(iter)
end

# old constructor for QMCState
function QMCState(
    hamiltonian, v;
    laststep=nothing,
    dτ=nothing,
    shift=nothing,
    wm=nothing,
    style=nothing,
    targetwalkers=1000,
    address=starting_address(hamiltonian),
    params::FciqmcRunStrategy=RunTillLastStep(
        laststep=100,
        shift=float(valtype(v))(diagonal_element(hamiltonian, address))
    ),
    s_strat::ShiftStrategy=DoubleLogUpdate(; targetwalkers),
    r_strat::ReportingStrategy=ReportDFAndInfo(),
    τ_strat::TimeStepStrategy=ConstantTimeStep(),
    threading=nothing,
    replica::ReplicaStrategy=NoStats(),
    post_step=(),
    maxlength=2 * _n_walkers(v, s_strat) + 100, # padding for small walker numbers
)
    Hamiltonians.check_address_type(hamiltonian, keytype(v))
    # Set up r_strat and params
    r_strat = refine_r_strat(r_strat)
    if !isnothing(laststep)
        params.laststep = laststep
    end
    if !isnothing(dτ)
        params.dτ = dτ
    end
    if !isnothing(shift)
        params.shift = shift
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
            ReplicaState(hamiltonian, deepcopy(v), deepcopy(wm), deepcopy(params), "_$i")
        end
    else
        replicas = (ReplicaState(hamiltonian, v, wm, params, ""),)
    end

    return QMCState(
        hamiltonian, replicas, Ref(Int(maxlength)),
        r_strat, s_strat, τ_strat, post_step, replica
    )
end
