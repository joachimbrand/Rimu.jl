"""
    PMCSimulation
Holds the state and the results of a projector quantum Monte Carlo (PMC) simulation.
Initialise with [`init(::ProjectorMonteCarloProblem)`](@ref) and solve with
[`solve!(::PMCSimulation)`](@ref).

Obtain the results of a simulation `sm` as a DataFrame with `DataFrame(sm)`.

See also [`state_vectors`](@ref), [`single_states`](@ref),
[`ProjectorMonteCarloProblem`](@ref), [`init`](@ref), [`solve!`](@ref).
"""
mutable struct PMCSimulation
    problem::ProjectorMonteCarloProblem
    algorithm # currently only FCIQMC() is implemented
    state::ReplicaState
    report::Report
    modified::Bool
    aborted::Bool
    success::Bool
    message::String
    elapsed_time::Float64
end

function _set_up_v(dv::AbstractDVec, copy_vectors, _...)
    # we are not changing type, style or initiator because an AbstractDVec is passed
    if copy_vectors
        return deepcopy(dv)
    else
        return dv
    end
end

function _set_up_v(fdv::FrozenDVec, _, style, initiator, threading)
    # we are allocating new memory
    if threading
        v = PDVec(fdv; style, initiator)
    elseif initiator isa NonInitiator
        v = DVec(fdv; style)
    else
        v = InitiatorDVec(fdv; style, initiator)
    end
    return v
end

function PMCSimulation(problem::ProjectorMonteCarloProblem; copy_vectors=true)
    @unpack algorithm, hamiltonian, starting_vectors, style, threading, simulation_plan,
        replica_strategy, shift_strategy, initial_shift_parameters,
        reporting_strategy, post_step_strategy, time_step_strategy,
        maxlength, metadata, initiator, random_seed, spectral_strategy = problem

    n_replicas = num_replicas(replica_strategy)

    # seed the random number generator
    if !isnothing(random_seed)
        Random.seed!(random_seed + hash(RMPI.mpi_rank()))
    end

    # set up the starting vectors and shift parameters
    if length(starting_vectors) == 1 && n_replicas > 1
        vectors = ntuple(n_replicas) do i
            v = _set_up_v(only(starting_vectors), copy_vectors, style, initiator, threading)
            sizehint!(v, maxlength)
            return v
        end
        shift_parameters = ntuple(n_replicas) do i
            deepcopy(only(initial_shift_parameters))
        end
    else
        @assert length(starting_vectors) == n_replicas == length(initial_shift_parameters)
        vectors = map(starting_vectors) do dv
            v = _set_up_v(dv, copy_vectors, style, initiator, threading)
            sizehint!(v, maxlength)
            return v
        end
        shift_parameters = deepcopy(initial_shift_parameters)
    end
    wm = working_memory(first(vectors))

    # set up the spectral_states
    if n_replicas == 1
        spectral_states =  (SpectralState(
            (SingleState(
                hamiltonian,
                only(vectors),
                zerovector(only(vectors)),
                wm,
                deepcopy(shift_strategy),
                deepcopy(time_step_strategy),
                only(shift_parameters),
                ""),
            ),
            spectral_strategy
        ),)
    else
        spectral_states = ntuple(n_replicas) do i
            v, sp = vectors[i], shift_parameters[i]
            rwm = (typeof(v) == typeof(first(vectors))) ? wm : working_memory(v)
            SpectralState(
                (SingleState(
                    hamiltonian,
                    v,
                    zerovector(v),
                    rwm,
                    deepcopy(shift_strategy),
                    deepcopy(time_step_strategy),
                    sp,
                    "_$i"),
                ),
                spectral_strategy
            )
        end
    end
    @assert spectral_states isa NTuple{n_replicas, <:SpectralState}

    # set up the initial state
    state = ReplicaState(
        hamiltonian, # hamiltonian
        spectral_states, # spectral_states
        Ref(maxlength), # maxlength
        Ref(simulation_plan.starting_step), # step
        simulation_plan, # simulation_plan
        reporting_strategy, # reporting_strategy
        post_step_strategy, # post_step_strategy
        replica_strategy # replica_strategy
    )
    report = Report()
    report_metadata!(report, "algorithm", algorithm)
    report_default_metadata!(report, state)
    report_metadata!(report, metadata) # add user metadata
    # Sanity checks.
    check_transform(state.replica_strategy, state.hamiltonian)

    return PMCSimulation(
        problem, algorithm, state, report, false, false, false, "", 0.0
    )
end

function Base.show(io::IO, sm::PMCSimulation)
    print(io, "PMCSimulation")
    st = sm.state
    print(io, " with ", num_replicas(st), " replica(s) and ")
    print(io, num_spectral_states(st), " spectral state(s).")
    print(io, "\n  Algorithm:   ", sm.algorithm)
    print(io, "\n  Hamiltonian: ", st.hamiltonian)
    print(io, "\n  Step:        ", st.step[], " / ", st.simulation_plan.last_step)
    print(io, "\n  modified = $(sm.modified), aborted = $(sm.aborted), success = $(sm.success)")
    sm.message == "" || print(io, "\n  message: ", sm.message)
end

num_spectral_states(sm::PMCSimulation) = num_spectral_states(sm.state)
num_replicas(sm::PMCSimulation) = num_replicas(sm.state)

function report_simulation_status_metadata!(report::Report, sm::PMCSimulation)
    @unpack modified, aborted, success, message, elapsed_time = sm

    report_metadata!(report, "modified", modified)
    report_metadata!(report, "aborted", aborted)
    report_metadata!(report, "success", success)
    report_metadata!(report, "message", message)
    report_metadata!(report, "elapsed_time", elapsed_time)
    return report
end

# iteration for backward compatibility
# undocumented; may be removed in the future
function Base.iterate(sm::PMCSimulation, state=1)
    if state == 1
        return DataFrame(sm), 2
    elseif state == 2
        return sm.state, 3
    else
        return nothing
    end
end
# getproperty access to :df and :state fields for backward compatibility
# undocumented; may be removed in the future
function Base.getproperty(sm::PMCSimulation, key::Symbol)
    if key == :df
        return DataFrame(sm)
    else
        return getfield(sm, key)
    end
end

# Tables.jl integration: provide access to report.data
# Note that using the `Tables` interface will not include metadata in the output.
# To include metadata, use the `DataFrame` constructor.
Tables.istable(::Type{<:PMCSimulation}) = true
Tables.columnaccess(::Type{<:PMCSimulation}) = true
Tables.columns(sm::PMCSimulation) = Tables.columns(sm.report.data)
Tables.schema(sm::PMCSimulation) = Tables.schema(sm.report.data)

state_vectors(sim::PMCSimulation) = state_vectors(sim.state)
single_states(sim::PMCSimulation) = single_states(sim.state)

# TODO: interface for reading results

DataFrames.DataFrame(s::PMCSimulation) = DataFrame(s.report)

"""
    init(problem::ProjectorMonteCarloProblem; copy_vectors=true)::PMCSimulation

Initialise a [`Rimu.PMCSimulation`](@ref).

See also [`ProjectorMonteCarloProblem`](@ref), [`solve!`](@ref), [`solve`](@ref),
[`step!`](@ref), [`Rimu.PMCSimulation`](@ref).
"""
function CommonSolve.init(problem::ProjectorMonteCarloProblem; copy_vectors=true)
    return PMCSimulation(problem; copy_vectors)
end

"""
    step!(sm::PMCSimulation)::PMCSimulation

Advance the simulation by one step.

Calling [`solve!`](@ref) will advance the simulation until the last step or the walltime is
exceeded. When completing the simulation without calling [`solve!`](@ref), the simulation
report needs to be finalised by calling [`Rimu.finalize_report!`](@ref).

See also [`ProjectorMonteCarloProblem`](@ref), [`init`](@ref), [`solve!`](@ref), [`solve`](@ref),
[`Rimu.PMCSimulation`](@ref).
"""
function CommonSolve.step!(sm::PMCSimulation)
    @unpack state, report, algorithm = sm
    @unpack spectral_states, simulation_plan, step, reporting_strategy,
        replica_strategy = state

    if sm.aborted || sm.success
        @warn "Simulation is already aborted or finished."
        return sm
    end
    if step[] >= simulation_plan.last_step
        @warn "Simulation has already reached the last step."
        return sm
    end

    step[] += 1

    # report step number
    if step[] % reporting_interval(reporting_strategy) == 0
        report!(reporting_strategy, step[], report, :steps, step[])
    end

    proceed = true
    # advance all spectral_states
    for replica in spectral_states
        proceed &= advance!(algorithm, report, state, replica)
    end
    sm.modified = true

    # report replica stats
    if step[] % reporting_interval(reporting_strategy) == 0
        replica_names, replica_values = replica_stats(replica_strategy, spectral_states)
        report!(reporting_strategy, step[], report, replica_names, replica_values)
        report_after_step!(reporting_strategy, step[], report, state)
        ensure_correct_lengths(report)
    end

    if !proceed
        sm.aborted = true
        sm.message = "Aborted in step $(step[])."
        return sm
    end
    if step[] == simulation_plan.last_step
        sm.success = true
    end
    return sm
end

"""
    solve(::ProjectorMonteCarloProblem)::PMCSimulation

Initialize and solve a [`ProjectorMonteCarloProblem`](@ref) until the last step is completed
or the walltime limit is reached.

See also [`init`](@ref), [`solve!`](@ref), [`step!`](@ref), [`Rimu.PMCSimulation`](@ref).
"""
CommonSolve.solve

"""
    solve!(sm::PMCSimulation; kwargs...)::PMCSimulation

Solve a [`Rimu.PMCSimulation`](@ref) until the last step is completed or the walltime limit
is reached.

To continue a previously completed simulation, set a new `last_step` or `walltime` using the
keyword arguments. Optionally, changes can be made to the `replica_strategy`, the
`post_step_strategy`, or the `reporting_strategy`.

# Optional keyword arguments:
* `last_step = nothing`: Set the last step to a new value and continue the simulation.
* `walltime = nothing`: Set the allowed walltime to a new value and continue the simulation.
* `reset_time = false`: Reset the `elapsed_time` counter and continue the simulation.
* `empty_report = false`: Empty the report before continuing the simulation.
* `replica_strategy = nothing`: Change the replica strategy. Requires the number of replicas
    to match the number of replicas in the simulation `sm`. Implies `empty_report = true`.
* `post_step_strategy = nothing`: Change the post-step strategy. Implies
    `empty_report = true`.
* `reporting_strategy = nothing`: Change the reporting strategy. Implies
    `empty_report = true`.
* `metadata = nothing`: Add metadata to the report.

See also [`ProjectorMonteCarloProblem`](@ref), [`init`](@ref), [`solve`](@ref),
[`step!`](@ref), [`Rimu.PMCSimulation`](@ref).
"""
function CommonSolve.solve!(sm::PMCSimulation;
    last_step = nothing,
    walltime = nothing,
    reset_time = false,
    replica_strategy=nothing,
    post_step_strategy=nothing,
    reporting_strategy=nothing,
    empty_report=false,
    metadata=nothing,
)
    reset_flags = reset_time # reset flags if resetting time
    if !isnothing(last_step)
        state = sm.state
        sm.state = @set state.simulation_plan.last_step = last_step
        report_metadata!(sm.report, "laststep", last_step)
        reset_flags = true
    end
    if !isnothing(walltime)
        state = sm.state
        sm.state = @set state.simulation_plan.walltime = walltime
        reset_flags = true
    end
    if !isnothing(replica_strategy)
        if num_replicas(sm) ≠ num_replicas(replica_strategy)
            throw(ArgumentError("Number of replicas in the strategy must match the number of replicas in the simulation."))
        end
        state = sm.state
        sm.state = @set state.replica_strategy = replica_strategy
        reset_flags = true
        empty_report = true
    end
    if !isnothing(post_step_strategy)
        # set up post_step_strategy as a tuple
        if post_step_strategy isa PostStepStrategy
            post_step_strategy = (post_step_strategy,)
        end
        state = sm.state
        sm.state = @set state.post_step_strategy = post_step_strategy
        reset_flags = true
        empty_report = true
    end
    if !isnothing(reporting_strategy)
        state = sm.state
        sm.state = @set state.reporting_strategy = reporting_strategy
        reset_flags = true
    end

    @unpack report = sm
    if empty_report
        empty!(report)
        report_default_metadata!(report, sm.state)
    end
    isnothing(metadata) || report_metadata!(report, metadata) # add user metadata

    @unpack simulation_plan, step, reporting_strategy = sm.state

    last_step = simulation_plan.last_step
    initial_step = step[]

    if step[] >= last_step
        @warn "Simulation has already reached the last step."
        return sm
    end

    if reset_flags # reset the flags
        sm.aborted = false
        sm.success = false
        sm.message = ""
    end
    if reset_time # reset the elapsed time
        sm.elapsed_time = 0.0
    end

    if sm.aborted || sm.success
        @warn "Simulation is already aborted or finished."
        return sm
    end
    un_finalize!(report)

    starting_time = time() + sm.elapsed_time # simulation time accumulates
    update_steps = max((last_step - initial_step) ÷ 200, 100) # log often but not too often
    name = get_metadata(sm.report, "display_name")

    @withprogress name = while !sm.aborted && !sm.success
        if time() - starting_time > simulation_plan.walltime
            sm.aborted = true
            sm.message = "Walltime limit reached."
            @warn "Walltime limit reached. Aborting simulation."
        else
            step!(sm)
        end
        if step[] % update_steps == 0 # for updating progress bars
            @logprogress (step[] - initial_step) / (last_step - initial_step)
        end

    end
    sm.elapsed_time = time() - starting_time
    report_simulation_status_metadata!(report, sm) # potentially overwrite values
    finalize_report!(reporting_strategy, report)
    return sm
end

# methods for backward compatibility
function lomc!(state::ReplicaState, df=DataFrame(); laststep=0, name="lomc!", metadata=nothing)
    if !iszero(laststep)
        state = @set state.simulation_plan.last_step = laststep
    end
    @unpack hamiltonian, spectral_states, maxlength, step, simulation_plan,
        reporting_strategy, post_step_strategy, replica_strategy = state
    first_replica = only(first(spectral_states).single_states) # SingleState
    @assert step[] ≥ simulation_plan.starting_step
    problem = ProjectorMonteCarloProblem(hamiltonian;
        start_at = first_replica.v,
        initial_shift_parameters = first_replica.shift_parameters,
        shift_strategy = first_replica.shift_strategy,
        time_step_strategy = first_replica.time_step_strategy,
        replica_strategy ,
        reporting_strategy,
        post_step_strategy,
        maxlength = maxlength[],
        simulation_plan,
        metadata,
        display_name = name,
        random_seed = false
    )
    report = Report()
    report_default_metadata!(report, state)
    report_metadata!(report, problem.metadata) # add user metadata
    # Sanity checks.
    check_transform(state.replica_strategy, state.hamiltonian)

    simulation = PMCSimulation(
        problem, FCIQMC(), state, report, false, false, false, "", 0.0
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
