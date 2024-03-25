"""
    QMCSimulation
Holds the state and the results of a QMC simulation.

## Fields
- `qmc_problem::QMCProblem`: the problem to be solved.
- `qmc_state::QMCState`: the state of the simulation.
- `report::Report`: the report of the simulation.
"""
struct QMCSimulation
    qmc_problem::QMCProblem
    qmc_state::QMCState
    report::Report
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

"""
    init(problem::QMCProblem; copy_vectors=true)

TBW
"""
function CommonSolve.init(problem::QMCProblem; copy_vectors=true)
    @unpack hamiltonian, starting_vectors, style, threading, simulation_plan,
        replica_strategy, shift_strategy, initial_shift_parameters,
        reporting_strategy, post_step_strategy, time_step_strategy,
        maxlength, metadata, initiator = problem

    n_replicas = num_replicas(replica_strategy)

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

    return QMCSimulation(problem, qmc_state, report)
end

# TODO: add docstring, show method, and tests

num_replicas(s::QMCSimulation) = num_replicas(s.qmc_problem)
