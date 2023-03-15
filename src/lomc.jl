"""
    FirstOrderTransitionOperator(hamiltonian, shift, dτ) <: AbstractHamiltonian

The transition operator used in FCIQMC.
```math
\\hat{\\mathbf{T} = 1 - dτ(H - S)
```
where ``H`` is the `hamiltonian` and ``S`` is the `shift`.
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
    return FCIQMCOffdiagonals(t.dτ, offdiagonals(t.hamiltonian, add))
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
* `params`: the [`FciqmcRunStrategy`](@ref).
* `id`: string ID appended to reported column names.

See also [`QMCState`](@ref), [`ReplicaStrategy`](@ref), [`replica_stats`](@ref),
[`lomc!`](@ref).
"""
mutable struct ReplicaState{H,T,V,W,R<:FciqmcRunStrategy{T}}
    # Future TODO: rename these fields, add interface for accessing them.
    hamiltonian::H
    v::V       # vector
    pv::V      # previous vector.
    wm::W      # working memory. Maybe working memories could be shared among replicas?
    pnorm::T   # previous walker number - used to control the shift
    params::R  # params: step, laststep, dτ...
    id::String # id is appended to column names
end

function ReplicaState(h, v, wm, params, id="")
    if isnothing(wm)
        wm = similar(v)
    end
    pv = zerovector(v)
    return ReplicaState(h, v, pv, wm, walkernumber(v), params, id)
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
    R<:ReplicaState,
    RS<:ReportingStrategy,
    SS<:ShiftStrategy,
    TS<:TimeStepStrategy,
    RRS<:ReplicaStrategy,
    PS<:NTuple{<:Any,PostStepStrategy},
}
    hamiltonian::H
    replicas::NTuple{N,R}
    maxlength::Ref{Int}

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
    wm=nothing,
    params::FciqmcRunStrategy=RunTillLastStep(
        laststep = 100,
        shift = float(valtype(v))(diagonal_element(
            hamiltonian,
            starting_address(hamiltonian)
        ))
    ),
    s_strat::ShiftStrategy=DoubleLogUpdate(),
    r_strat::ReportingStrategy=ReportDFAndInfo(),
    τ_strat::TimeStepStrategy=ConstantTimeStep(),
    threading=nothing,
    replica::ReplicaStrategy=NoStats(),
    post_step=(),
    maxlength= 2 * _n_walkers(v, s_strat) + 100, # padding for small walker numbers
)
    # Set up r_strat and params
    r_strat = refine_r_strat(r_strat)
    if !isnothing(laststep)
        params.laststep = laststep
    end
    if !isnothing(dτ)
        params.dτ = dτ
    end

    if threading ≠ nothing
        @warn "Threading has been removed. Ignoring `threading=$threading`."
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

* `params::FciqmcRunStrategy = RunTillLastStep(laststep = 100, dτ = 0.01, shift =
  diagonal_element(ham, starting_address(ham)))` -
  basic parameters of simulation state, see [`FciqmcRunStrategy`](@ref); is mutated
* `laststep` - can be used to override information otherwise contained in `params`
* `s_strat::ShiftStrategy = DoubleLogUpdate(targetwalkers = 1000, ζ = 0.08, ξ = ζ^2/4)` -
  how to update the `shift`, see [`ShiftStrategy`](@ref)
* `maxlength = 2 * s_strat.targetwalkers + 100` - upper limit on the length of `v`; when
  reached, `lomc!` will abort
* `style = IsStochasticInteger()` - set [`StochasticStyle`](@ref) for default `v`; unused
  if `v` is specified.
* `post_step::NTuple{N,<:PostStepStrategy} = ()` - extract observables (e.g.
  [`ProjectedEnergy`](@ref)), see [`PostStepStrategy`](@ref).
* `replica::ReplicaStrategy = NoStats(1)` - run several synchronised simulation, see
  [`ReplicaStrategy`](@ref).
* `r_strat::ReportingStrategy = ReportDFAndInfo()` - how and when to report results, see
  [`ReportingStrategy`](@ref)
* `τ_strat::TimeStepStrategy = ConstantTimeStep()` - adjust time step dynamically, see
  [`TimeStepStrategy`](@ref)
* `wm` - working memory; if set, it controls the use of multithreading and overrides
  `threading`; is mutated
* `df = DataFrame()` - when called with `AbstractHamiltonian` argument, a `DataFrame` can
  be passed into `lomc!` that will be pushed into
* `name = "lomc!"` - name displayed in progress bar (via `ProgressLogging`)

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
(100, 11)

julia> size(df2)
(200, 11)
```
"""
function lomc!(ham, v; df=DataFrame(), name="lomc!", kwargs...)
    state = QMCState(ham, v; kwargs...)
    return lomc!(state, df; name)
end
function lomc!(ham; style=IsStochasticInteger(), kwargs...)
    v = DVec(starting_address(ham)=>10; style)
    return lomc!(ham, v; kwargs...)
end
# For continuation, you can pass a QMCState and a DataFrame
function lomc!(state::QMCState, df=DataFrame(); laststep=0, name="lomc!")
    report = Report()
    if !iszero(laststep)
        state.laststep = laststep
    end

    # Sanity checks.
    step, laststep = state.step, state.laststep
    for replica in state.replicas
        @assert replica.params.step == step
        @assert replica.params.laststep == laststep
    end
    check_transform(state.replica, state.hamiltonian)

    # main loop
    initial_step = step
    update_steps = max((laststep - initial_step) ÷ 200, 100) # log often but not too often
    # update_steps = 400
    @withprogress name=name while step < laststep
        step += 1
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
        return (; df=vcat(df, result_df), state)
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
    @unpack v, pv, wm, pnorm, params, id = replica
    @unpack step, shiftMode, shift, dτ = params
    step += 1

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
    shift, shiftMode, pnorm, proceed = update_shift(
        s_strat, shift, shiftMode, tnorm, pnorm, dτ, step, nothing, v, pv
    )
    dτ = update_dτ(τ_strat, dτ, tnorm)

    @pack! params = step, shiftMode, shift, dτ
    @pack! replica = v, pv, wm, pnorm, params
    ### TO HERE

    if step % reporting_interval(state.r_strat) == 0
        # Note: post_step must be called after packing the values.
        post_step_stats = post_step(state.post_step, replica)

        # Reporting
        report!(
            r_strat, step, report,
            (dτ, shift, shiftMode, len, norm=tnorm), id,
        )
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
