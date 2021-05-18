"""
    ReplicaState(hamiltonian, v, w, pnorm, r_strat)

Struct that holds all information needed for an independent run of the algorithm.

Can be advanced a step forward with [`advance!`](@ref).
"""
mutable struct ReplicaState{T,V,W,R<:FciqmcRunStrategy{T}}
    v::V       # vector
    w::W       # working memory. Maybe working memories could be shared among replicas?
    pnorm::T   # previous walker number - used to control the shift
    params::R  # params: step, laststep, dτ...
    id::String # id is appended to column names
end

function ReplicaState(v, w, params, id="")
    if isnothing(w)
        w = similar(v)
    end
    return ReplicaState(v, w, walkernumber(v), params, id)
end

"""
    QMCState

Holds all inforamtion needed to run FCIQMC, except the data frame. Holds a `NTuple` of
`ReplicaState`s and various strategies that control the algorithm.
"""
struct QMCState{
    H,
    N,
    R<:ReplicaState,
    MS<:MemoryStrategy,
    RS<:ReportingStrategy,
    SS<:ShiftStrategy,
    TS<:TimeStepStrategy,
    O,
}
    hamiltonian::H
    replicas::NTuple{N,R}

    m_strat::MS
    r_strat::RS
    s_strat::SS
    τ_strat::TS

    # Report x†⋅operator⋅y
    # In the future, this should be replaced with a strict a la ReplicaStrategy.
    operator::O
end

function QMCState(
    hamiltonian, v;
    laststep=nothing,
    dτ=nothing,
    threading=:auto,
    wm=nothing,
    params::FciqmcRunStrategy=RunTillLastStep(),
    s_strat::ShiftStrategy=DoubleLogUpdate(),
    r_strat::ReportingStrategy=EveryTimeStep(),
    τ_strat::TimeStepStrategy=ConstantTimeStep(),
    m_strat::MemoryStrategy=NoMemory(),
    num_replicas=1,
    report_xHy=false,
    operator=report_xHy ? hamiltonian : nothing,
)
    # Checks
    if num_replicas < 1
        error("need at least one replica.")
    elseif !isnothing(operator) && num_replicas < 2
        error(
            "operator $operator needs at least two replicas. ",
            "Set the number of replicas with the `num_replicas` keyowrd argument."
        )
    end
    # Set up default arguments
    r_strat = refine_r_strat(r_strat, hamiltonian)
    if !isnothing(laststep)
        params.laststep = laststep
    end
    if !isnothing(dτ)
        params.dτ = dτ
    end
    wm = default_working_memory(threading, v, s_strat)

    # Set up replicas
    if num_replicas > 1
        replicas = ntuple(num_replicas) do i
            ReplicaState(deepcopy(v), deepcopy(wm), deepcopy(params), "_$i")
        end
    else
        replicas = (ReplicaState(v, wm, params, ""),)
    end

    return QMCState(hamiltonian, replicas, m_strat, r_strat, s_strat, τ_strat, operator)
end

function default_working_memory(threading, v, s_strat)
    if threading == :auto
        threading = max(real(s_strat.targetwalkers),imag(s_strat.targetwalkers)) ≥ 500
    end
    if threading
        return threadedWorkingMemory(v)
    else
        return similar(localpart(v))
    end
end

function Base.show(io::IO, st::QMCState)
    print(io, "QMCState")
    if length(st.replicas) > 1
        print(io, " with ", length(st.replicas), " replicas")
    end
    println(io, "\n  H:    ", st.hamiltonian)
end

function lomc!(ham, v; df=DataFrame(), kwargs...)
    state = QMCState(ham, v; kwargs...)
    return lomc!(state, df)
end
# For continuation, you can pass a QMCState and a DataFrame
function lomc!(state::QMCState, df=DataFrame())
    report = Report()

    # Sanity checks.
    for replica in state.replicas
        ConsistentRNG.check_crng_independence(replica.v)
    end
    # Get we will use the first replica's step to keep track of the step. Perhaps step should
    # be moved to QMCState?
    first_replica = first(state.replicas)
    step, laststep = first_replica.params.step, first_replica.params.laststep
    while step < laststep
        step += 1
        # This loop could be threaded if num_threads() == num_replicas? MPIData would need
        # to be aware of the replica's id and use that in communication.
        for replica in state.replicas
            advance!(report, state, replica)
        end
        # The following should probably be replaced with some kind of ReplicaStrategy.
        # Right now it is designed to work similarly to the old implementation.
        if length(state.replicas) ≥ 2
            c1 = state.replicas[1].v
            c2 = state.replicas[2].v
            xdoty = c1 ⋅ c2
            report!(report, :xdoty, xdoty)
            if !isnothing(state.operator)
                xHy = dot(c1, state.operator, c2)
                report!(report, :xHy, xHy)
            end
        end
    end

    # Put report into DataFrame and merge with `df`. We are assuming the previous `df` is
    # compatible, which should be the case if the run is an actual continuation. Maybe the
    # DataFrames should be merged in a more permissive manner?
    df = vcat(df, DataFrame(report))
    return (; df, state)
end

"""
    advance!(report::Report, state::QMCState, replica::ReplicaState)

Advance the `replica` by one step. The `state` is used only to access the various strategies
involved. Steps, stats, and computed quantities are written to the `report`.
"""
function advance!(
    report, state::QMCState, replica::ReplicaState{T}
) where {T}
    @unpack hamiltonian, m_strat, r_strat, s_strat, τ_strat = state
    @unpack v, w, pnorm, params, id = replica
    @unpack step, laststep, shiftMode, shift, dτ = params
    step += 1

    v, w, step_stats, stat_names, shift_noise = fciqmc_step!(
        hamiltonian, v, shift, dτ, pnorm, w, 1.0; m_strat
    )
    v, update_stats = update_dvec!(v, shift)
    tnorm = walkernumber(v)
    len = length(v)

    v_proj, h_proj = compute_proj_observables(v, hamiltonian, r_strat)

    shift, shiftMode, pnorm = update_shift(
        s_strat, shift, shiftMode, tnorm, pnorm, dτ, step, nothing, v, w
    )
    dτ = update_dτ(τ_strat, dτ, tnorm)

    @pack! params = step, laststep, shiftMode, shift, dτ
    @pack! replica = v, w, pnorm, params

    colnames = (
        "steps", "dτ", "shift", "shiftMode",
        "len", "norm", "vproj", "hproj",
        stat_names..., "shiftnoise"
    )
    values = (
        step, dτ, shift, shiftMode, len, tnorm, v_proj, h_proj,
        step_stats..., shift_noise,
    )
    report!(report, colnames, values, id)
    report!(report, update_stats, id)
    # TODO add maxlength and check if it was reached.
end
