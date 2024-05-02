"""
    QMCSimulation
Holds the state and the results of a QMC simulation. Initialise with
[`init(::FCIQMCProblem)`](@ref) and solve with [`solve!(::QMCSimulation)`](@ref).

Obtain the results of a simulation `sm` as a DataFrame with `DataFrame(sm)`.
"""
struct QMCSimulation
    qmc_problem::FCIQMCProblem
    qmc_state::QMCState
    report::Report
    modified::Ref{Bool}
    aborted::Ref{Bool}
    success::Ref{Bool}
    message::Ref{String}
    elapsed_time::Ref{Float64}
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

function QMCSimulation(problem::FCIQMCProblem; copy_vectors=true)
    @unpack hamiltonian, starting_vectors, style, threading, simulation_plan,
        replica_strategy, shift_strategy, initial_shift_parameters,
        reporting_strategy, post_step_strategy, time_step_strategy,
        maxlength, metadata, initiator, random_seed = problem

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

    # set up the replica_states
    if n_replicas == 1
        replica_states = (SingleState(
            hamiltonian,
            only(vectors),
            zerovector(only(vectors)),
            wm,
            deepcopy(shift_strategy),
            deepcopy(time_step_strategy),
            only(shift_parameters),
            ""),)
    else
        replica_states = ntuple(n_replicas) do i
            v, sp = vectors[i], shift_parameters[i]
            rwm = (typeof(v) == typeof(first(vectors))) ? wm : working_memory(v)
            SingleState(
                hamiltonian,
                v,
                zerovector(v),
                rwm,
                deepcopy(shift_strategy),
                deepcopy(time_step_strategy),
                sp,
                "_$i")
        end
    end
    @assert replica_states isa NTuple{n_replicas, <:SingleState}

    # set up the initial state
    qmc_state = QMCState(
        hamiltonian, # hamiltonian
        replica_states, # replica_states
        Ref(maxlength), # maxlength
        Ref(simulation_plan.starting_step), # step
        simulation_plan, # simulation_plan
        reporting_strategy, # reporting_strategy
        post_step_strategy, # post_step
        replica_strategy # replica_strategy
    )
    report = Report()
    report_default_metadata!(report, qmc_state)
    report_metadata!(report, metadata) # add user metadata
    # Sanity checks.
    check_transform(qmc_state.replica_strategy, qmc_state.hamiltonian)

    return QMCSimulation(
        problem, qmc_state, report, Ref(false), Ref(false), Ref(false), Ref(""), Ref(0.0)
    )
end

function Base.show(io::IO, sm::QMCSimulation)
    print(io, "QMCSimulation")
    st = sm.qmc_state
    if length(st.replica_states) > 1
        print(io, " with ", length(st.replica_states), " replicas")
    end
    print(io, "\n  H:    ", st.hamiltonian)
    print(io, "\n  step: ", st.step[], " / ", st.simulation_plan.last_step)
    print(io, "\n  modified = $(sm.modified[]), aborted = $(sm.aborted[]), success = $(sm.success[])")
    sm.message[] == "" || print(io, "\n  message: ", sm.message[])
    print(io, "\n  replicas: ")
    for (i, r) in enumerate(st.replica_states)
        print(io, "\n    $i: ", r)
    end
end

function report_simulation_status_metadata!(report::Report, sm::QMCSimulation)
    @unpack modified, aborted, success, message, elapsed_time = sm

    report_metadata!(report, "modified", modified[])
    report_metadata!(report, "aborted", aborted[])
    report_metadata!(report, "success", success[])
    report_metadata!(report, "message", message[])
    report_metadata!(report, "elapsed_time", elapsed_time[])
    return report
end

# iteration for backward compatibility
# undocumented; may be removed in the future
function Base.iterate(sm::QMCSimulation, state=1)
    if state == 1
        return DataFrame(sm), 2
    elseif state == 2
        return sm.qmc_state, 3
    else
        return nothing
    end
end
# getproperty access to :df and :state fields for backward compatibility
# undocumented; may be removed in the future
function Base.getproperty(sm::QMCSimulation, key::Symbol)
    if key == :df
        return DataFrame(sm)
    elseif key == :state
        return sm.qmc_state
    else
        return getfield(sm, key)
    end
end

# Tables.jl integration: provide access to report.data
# Note that using the `Tables` interface will not include metadata in the output.
# To include metadata, use the `DataFrame` constructor.
Tables.istable(::Type{<:QMCSimulation}) = true
Tables.columnaccess(::Type{<:QMCSimulation}) = true
Tables.columns(sm::QMCSimulation) = Tables.columns(sm.report.data)
Tables.schema(sm::QMCSimulation) = Tables.schema(sm.report.data)


# TODO: interface for reading results

num_replicas(s::QMCSimulation) = num_replicas(s.qmc_problem)
DataFrames.DataFrame(s::QMCSimulation) = DataFrame(s.report)

"""
    init(problem::FCIQMCProblem; copy_vectors=true)::QMCSimulation

Initialise a [`Rimu.QMCSimulation`](@ref).

See also [`FCIQMCProblem`](@ref), [`solve!`](@ref), [`solve`](@ref), [`step!`](@ref),
[`Rimu.QMCSimulation`](@ref).
"""
function CommonSolve.init(problem::FCIQMCProblem; copy_vectors=true)
    return QMCSimulation(problem; copy_vectors)
end

"""
    step!(sm::QMCSimulation)::QMCSimulation

Advance the simulation by one step.

Calling [`solve!`](@ref) will advance the simulation until the last step or the walltime is
exceeded. When completing the simulation without calling [`solve!`](@ref), the simulation
report needs to be finalised by calling [`Rimu.finalize_report!`](@ref).

See also [`FCIQMCProblem`](@ref), [`init`](@ref), [`solve!`](@ref), [`solve`](@ref),
[`Rimu.QMCSimulation`](@ref).
"""
function CommonSolve.step!(sm::QMCSimulation)
    @unpack qmc_state, report, modified, aborted, success, message = sm
    @unpack replica_states, simulation_plan, step, reporting_strategy,
        replica_strategy = qmc_state

    if aborted[] || success[]
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
    # advance all replica_states
    for replica in replica_states
        proceed &= advance!(report, qmc_state, replica)
    end
    modified[] = true

    # report replica stats
    if step[] % reporting_interval(reporting_strategy) == 0
        replica_names, replica_values = replica_stats(replica_strategy, replica_states)
        report!(reporting_strategy, step[], report, replica_names, replica_values)
        report_after_step!(reporting_strategy, step[], report, qmc_state)
        ensure_correct_lengths(report)
    end

    if !proceed
        aborted[] = true
        message[] = "Aborted in step $(step[])."
        return sm
    end
    if step[] == simulation_plan.last_step
        success[] = true
    end
    return sm
end

"""
    CommonSolve.solve(::FCIQMCProblem)::QMCSimulation

Initialize and solve the simulation until the last step or the walltime is exceeded.

See also [`FCIQMCProblem`](@ref), [`init`](@ref), [`solve!`](@ref), [`step!`](@ref),
[`Rimu.QMCSimulation`](@ref).
"""
CommonSolve.solve

"""
    CommonSolve.solve!(sm::QMCSimulation; kwargs...)::QMCSimulation

Solve the simulation until the last step or the walltime is exceeded.

# Keyword arguments:
* `last_step = nothing`: Set the last step to a new value and continue the simulation.
* `walltime = nothing`: Set the allowed walltime to a new value and continue the simulation.
* `reset_time = false`: Reset the `elapsed_time` counter and continue the simulation.

See also [`FCIQMCProblem`](@ref), [`init`](@ref), [`solve`](@ref), [`step!`](@ref),
[`Rimu.QMCSimulation`](@ref).
"""
function CommonSolve.solve!(sm::QMCSimulation;
    last_step = nothing,
    walltime = nothing,
    reset_time = false,
)
    reset_flags = reset_time # reset flags if resetting time
    if !isnothing(last_step)
        sm = @set sm.qmc_state.simulation_plan.last_step = last_step
        report_metadata!(sm.report, "laststep", last_step)
        reset_flags = true
    end
    if !isnothing(walltime)
        sm = @set sm.qmc_state.simulation_plan.walltime = walltime
        reset_flags = true
    end

    @unpack aborted, success, message, elapsed_time, report = sm
    @unpack simulation_plan, step, reporting_strategy = sm.qmc_state

    last_step = simulation_plan.last_step
    initial_step = step[]

    if step[] >= last_step
        @warn "Simulation has already reached the last step."
        return sm
    end

    if reset_flags # reset the flags
        aborted[] = false
        success[] = false
        message[] = ""
    end
    if reset_time # reset the elapsed time
        elapsed_time[] = 0.0
    end

    if aborted[] || success[]
        @warn "Simulation is already aborted or finished."
        return sm
    end
    un_finalize!(report)

    starting_time = time() + elapsed_time[] # simulation time accumulates
    update_steps = max((last_step - initial_step) ÷ 200, 100) # log often but not too often
    name = get_metadata(sm.report, "display_name")

    @withprogress name = while !aborted[] && !success[]
        if time() - starting_time > simulation_plan.walltime
            aborted[] = true
            message[] = "Walltime limit reached."
            @warn "Walltime limit reached. Aborting simulation."
        else
            step!(sm)
        end
        if step[] % update_steps == 0 # for updating progress bars
            @logprogress (step[] - initial_step) / (last_step - initial_step)
        end

    end
    elapsed_time[] = time() - starting_time
    report_simulation_status_metadata!(report, sm) # potentially overwrite values
    finalize_report!(reporting_strategy, report)
    return sm
end

# methods for backward compatibility
function lomc!(state::QMCState, df=DataFrame(); laststep=0, name="lomc!", metadata=nothing)
    if !iszero(laststep)
        state = @set state.simulation_plan.last_step = laststep
    end
    @unpack hamiltonian, replica_states, maxlength, step, simulation_plan,
        reporting_strategy, post_step, replica_strategy = state
    first_replica = first(replica_states)
    @assert step[] ≥ simulation_plan.starting_step
    problem = FCIQMCProblem(hamiltonian;
        start_at = first_replica.v,
        initial_shift_parameters = first_replica.shift_parameters,
        shift_strategy = first_replica.s_strat,
        time_step_strategy = first_replica.τ_strat,
        replica_strategy ,
        reporting_strategy,
        post_step_strategy = post_step,
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

    simulation = QMCSimulation(
        problem, state, report, Ref(false), Ref(false), Ref(false), "", Ref(0.0)
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
