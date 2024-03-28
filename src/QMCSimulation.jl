"""
    QMCSimulation
Holds the state and the results of a QMC simulation. Initialise with
[`init(::QMCProblem)`](@ref) amd solve with [`solve(::QMCSimulation)`].

## Fields
- `qmc_problem::QMCProblem`: the problem to be solved.
- `qmc_state::QMCState`: the state of the simulation.
- `report::Report`: the report of the simulation.
"""
struct QMCSimulation
    qmc_problem::QMCProblem
    qmc_state::QMCState
    report::Report
    modified::Ref{Bool}
    aborted::Ref{Bool}
    success::Ref{Bool}
    message::String
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

function QMCSimulation(problem::QMCProblem; copy_vectors=true)
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

    # set up the replicas
    if n_replicas == 1
        replicas = (ReplicaState(
            hamiltonian,
            only(vectors),
            zerovector(only(vectors)),
            wm,
            deepcopy(shift_strategy),
            deepcopy(time_step_strategy),
            only(initial_shift_parameters),
            ""),)
    else
        replicas = ntuple(n_replicas) do i
            v, sp = vectors[i], shift_parameters[i]
            rwm = (typeof(v) == typeof(first(vectors))) ? wm : working_memory(v)
            ReplicaState(
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
    @assert replicas isa NTuple{n_replicas, <:ReplicaState}

    # set up the initial state
    qmc_state = QMCState(
        hamiltonian, # hamiltonian
        replicas, # replicas
        Ref(maxlength), # maxlength
        Ref(simulation_plan.starting_step), # step
        simulation_plan, # simulation_plan
        reporting_strategy, # r_strat
        post_step_strategy, # post_step
        replica_strategy # replica
    )
    report = Report()
    report_default_metadata!(report, qmc_state)
    report_metadata!(report, metadata) # add user metadata
    # Sanity checks.
    check_transform(qmc_state.replica, qmc_state.hamiltonian)

    return QMCSimulation(
        problem, qmc_state, report, Ref(false), Ref(false), Ref(false), "", Ref(0.0)
    )
end

function Base.show(io::IO, sm::QMCSimulation)
    print(io, "QMCSimulation")
    st = sm.qmc_state
    if length(st.replicas) > 1
        print(io, " with ", length(st.replicas), " replicas")
    end
    print(io, "\n  H:    ", st.hamiltonian)
    print(io, "\n  step: ", st.step[], " / ", st.simulation_plan.last_step)
    print(io, "\n  modified = $(sm.modified[]), aborted = $(sm.aborted[]), success = $(sm.success[])")
    sm.message == "" || print(io, "\n  message: ", sm.message)
    print(io, "\n  replicas: ")
    for (i, r) in enumerate(st.replicas)
        print(io, "\n    $i: ", r)
    end
end

# TODO: interface for reading results

num_replicas(s::QMCSimulation) = num_replicas(s.qmc_problem)
DataFrames.DataFrame(s::QMCSimulation) = DataFrame(s.report)

"""
    init(problem::QMCProblem; copy_vectors=true)::QMCSimulation

Initialise a [`Rimu.QMCSimulation`](@ref).

See also [`QMCProblem`](@ref).
"""
function CommonSolve.init(problem::QMCProblem; copy_vectors=true)
    return QMCSimulation(problem; copy_vectors)
end

"""
    step!(sm::QMCSimulation)::QMCSimulation

Advance the simulation by one step.

See also [`QMCProblem`](@ref), [`init`](@ref), [`Rimu.QMCSimulation`](@ref).
"""
function CommonSolve.step!(sm::QMCSimulation)
    @unpack qmc_state, report, modified, aborted, success = sm
    @unpack replicas, simulation_plan, step, r_strat,
        replica = qmc_state

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
    if step[] % reporting_interval(r_strat) == 0
        report!(r_strat, step[], report, :steps, step[])
    end

    proceed = true
    # advance all replicas
    for replica in replicas
        proceed &= advance!(report, qmc_state, replica)
    end
    modified[] = true

    # report replica stats
    if step[] % reporting_interval(r_strat) == 0
        replica_names, replica_values = replica_stats(replica, replicas)
        report!(r_strat, step[], report, replica_names, replica_values)
        report_after_step(r_strat, step[], report, step[])
        ensure_correct_lengths(report)
    end

    if !proceed
        aborted[] = true
        return sm
    end
    if step[] == simulation_plan.last_step
        success[] = true
    end
    return sm
end

"""
    CommonSolve.solve!(sm::QMCSimulation)::QMCSimulation

Solve the simulation until the last step or the walltime is exceeded.
"""
function CommonSolve.solve!(sm::QMCSimulation)
    @unpack aborted, success, message, elapsed_time = sm
    @unpack simulation_plan, step = sm.qmc_state

    last_step = simulation_plan.last_step
    initial_step = step[]

    if aborted[] || success[]
        @warn "Simulation is already aborted or finished."
        return sm
    end
    if step[] >= simulation_plan.last_step
        @warn "Simulation has already reached the last step."
        return sm
    end

    starting_time = time()
    update_steps = max((last_step - initial_step) ÷ 200, 100) # log often but not too often
    name = get_metadata(sm.report, "display_name")

    @withprogress name = while !sm.aborted[] && !sm.success[]
        step!(sm)
        if time() - starting_time > simulation_plan.walltime
            aborted[] = true
            message = "Walltime limit reached."
        end
        if step[] % update_steps == 0 # for updating progress bars
            @logprogress (step[] - initial_step) / (last_step - initial_step)
        end

    end
    elapsed_time[] = time() - starting_time
    return sm
end
