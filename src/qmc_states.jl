"""
    SingleState(hamiltonian, algorithm, v, wm, pnorm, params, id)

Struct that holds a single state vector and all information needed for an independent run
of the algorithm. Can be advanced a step forward with [`advance!`](@ref).

See also [`SpectralState`](@ref), [`SpectralStrategy`](@ref),
[`ReplicaState`](@ref), [`ReplicaStrategy`](@ref), [`replica_stats`](@ref),
[`PMCSimulation`](@ref).
"""
mutable struct SingleState{H,A,V,W,SP}
    # Future TODO: rename these fields, add interface for accessing them.
    hamiltonian::H
    algorithm::A
    v::V       # vector
    pv::V      # previous vector.
    wm::W      # working memory. Maybe working memories could be shared among replicas?
    shift_parameters::SP # norm, shift, time_step; is mutable
    id::String # id is appended to column names
end

function Base.show(io::IO, r::SingleState)
    print(
        io,
        "SingleState(v: ", length(r.v), "-element ", nameof(typeof(r.v)),
        ", wm: ", length(r.wm), "-element ", nameof(typeof(r.wm)), ")"
    )
end

"""
    SpectralState <: AbstractVector{SingleState}
Holds one or several [`SingleState`](@ref)s representing the ground state and excited
states of a single replica.

Indexing the `SpectralState` `state[i]` returns the `i`th `SingleState`.

See also [`SpectralStrategy`](@ref), [`ReplicaState`](@ref), [`SingleState`](@ref),
[`PMCSimulation`](@ref).
"""
struct SpectralState{
    N,
    NS<:NTuple{N,SingleState},
    SS<:SpectralStrategy{N}
} <: AbstractVector{SingleState}
    single_states::NS # Tuple of SingleState
    spectral_strategy::SS # SpectralStrategy
    id::String # id is appended to column names
end
function SpectralState(t::Tuple, ss::SpectralStrategy, id="")
    return SpectralState(t, ss, id)
end
num_spectral_states(::SpectralState{N}) where {N} = N

Base.size(s::SpectralState) = (num_spectral_states(s),)
Base.getindex(s::SpectralState, i::Int) = s.single_states[i]

function Base.show(io::IO, s::SpectralState)
    print(io, "SpectralState")
    print(io, " with ", num_spectral_states(s), " spectral states")
    print(io, "\n    spectral_strategy: ", s.spectral_strategy)
    for (i, r) in enumerate(s.single_states)
        print(io, "\n      $i: ", r)
    end
end

function state_vectors(state::SpectralState)
    return SMatrix{1, num_spectral_states(state)}(s.v for s in state.single_states)
end

"""
    _n_walkers(v, shift_strategy)
Returns an estimate of the expected number of walkers as an integer.
"""
function _n_walkers(v, shift_strategy)
    n = if hasfield(typeof(shift_strategy), :target_walkers)
        shift_strategy.target_walkers
    else # e.g. for LogUpdate()
        walkernumber(v)
    end
    return ceil(Int, max(real(n), imag(n)))
end

"""
    ReplicaState <: AbstractMatrix{SingleState}

Holds information about multiple replicas of [`SpectralState`](@ref)s.

Indexing the `ReplicaState` `state[i, j]` returns a `SingleState` from the `i`th replica
and `j`th spectral state.

See also [`ReplicaStrategy`](@ref), [`SpectralState`](@ref), [`SingleState`](@ref),
[`PMCSimulation`](@ref).
"""
struct ReplicaState{
    N, # number of replicas
    S, # number of spectral states
    R<:NTuple{N,<:SpectralState{S}},
    RS<:ReportingStrategy,
    RRS<:ReplicaStrategy,
    PS<:NTuple{<:Any,PostStepStrategy},
} <: AbstractMatrix{SingleState}
    spectral_states::R
    maxlength::Ref{Int}
    step::Ref{Int}
    simulation_plan::SimulationPlan
    reporting_strategy::RS
    post_step_strategy::PS
    replica_strategy::RRS
end


num_replicas(::ReplicaState{N}) where {N} = N
num_spectral_states(::ReplicaState{<:Any, S}) where {S} = S

function Base.show(io::IO, st::ReplicaState)
    print(io, "ReplicaState")
    print(io, " with ", num_replicas(st), " replicas  and ")
    print(io, num_spectral_states(st), " spectral states")
    print(io, "\n  H:    ", first(st).hamiltonian)
    print(io, "\n  step: ", st.step[], " / ", st.simulation_plan.last_step)
    print(io, "\n  replicas: ")
    for (i, r) in enumerate(st.spectral_states)
        print(io, "\n    $i: ", r)
    end
end

Base.size(st::ReplicaState) = (num_replicas(st), num_spectral_states(st))
Base.getindex(st::ReplicaState, i::Int, j::Int) = st.spectral_states[i].single_states[j]
Base.IndexStyle(::Type{<:ReplicaState}) = IndexCartesian()

"""
    StateVectors <: AbstractMatrix{V}
Represents a matrix of configuration vectors from the `state`.
Construct this object with [`state_vectors`](@ref).
"""
struct StateVectors{V,R} <: AbstractMatrix{V}
    state::R
end
Base.size(sv::StateVectors) = size(sv.state)
Base.getindex(sv::StateVectors, i::Int, j::Int) = sv.state[i, j].v

"""
    state_vectors(state::ReplicaState)
    state_vectors(sim::PMCSimulation)
Return an `AbstractMatrix` of configuration vectors from the `state`.
The vectors can be accessed by indexing the resulting collection, where the row index
corresponds to the replica index and the column index corresponds to the spectral state
index.

See also [`SingleState`](@ref), [`ReplicaState`](@ref), [`SpectralState`](@ref),
[`PMCSimulation`](@ref).
"""
function state_vectors(state::R) where {R<:ReplicaState}
    V = typeof(first(state).v)
    return StateVectors{V,R}(state)
end

function report_default_metadata!(report::Report, state::ReplicaState)
    report_metadata!(report, "Rimu.PACKAGE_VERSION", Rimu.PACKAGE_VERSION)
    # add metadata from state
    s_state = first(state)
    algorithm = s_state.algorithm
    shift_parameters = s_state.shift_parameters
    report_metadata!(report, "algorithm", algorithm)
    report_metadata!(report, "laststep", state.simulation_plan.last_step)
    report_metadata!(report, "num_replicas", num_replicas(state))
    report_metadata!(report, "num_spectral_states", num_spectral_states(state))
    report_metadata!(report, "hamiltonian", s_state.hamiltonian)
    report_metadata!(report, "reporting_strategy", state.reporting_strategy)
    report_metadata!(report, "shift_strategy", algorithm.shift_strategy)
    report_metadata!(report, "time_step_strategy", algorithm.time_step_strategy)
    report_metadata!(report, "time_step", shift_parameters.time_step)
    report_metadata!(report, "step", state.step[])
    report_metadata!(report, "shift", shift_parameters.shift)
    report_metadata!(report, "maxlength", state.maxlength[])
    report_metadata!(report, "post_step_strategy", state.post_step_strategy)
    report_metadata!(report, "v_summary", summary(s_state.v))
    report_metadata!(report, "v_type", typeof(s_state.v))
    return report
end

"""
    default_starting_vector(hamiltonian::AbstractHamiltonian; kwargs...)
    default_starting_vector(
        address=starting_address(hamiltonian);
        style=IsDynamicSemistochastic(),
        initiator=NonInitiator(),
        threading=nothing,
        population=10
    )
Return a default starting vector for [`ProjectorMonteCarloProblem`](@ref). The default
choice for the starting vector is
```julia
v = PDVec(address => population; style, initiator)
```
if threading is available, or otherwise
```julia
v = DVec(address => population; style)
```
if `initiator == NonInitiator()`, and
```julia
v = InitiatorDVec(address => population; style, initiator)
```
if not. See [`PDVec`](@ref), [`DVec`](@ref), [`InitiatorDVec`](@ref),
[`StochasticStyle`](@ref), and [`InitiatorRule`](@ref).
"""
function default_starting_vector(
    hamiltonian::AbstractHamiltonian;
    address=starting_address(hamiltonian), kwargs...
)
    return default_starting_vector(address; kwargs...)
end

function default_starting_vector(address::AbstractFockAddress; population=10, kwargs...)
    return default_starting_vector(address=>population; kwargs...)
end

function default_starting_vector(fdv::Union{FrozenDVec,Pair};
    style = IsDynamicSemistochastic(),
    threading = nothing,
    initiator = NonInitiator(),
)
    threading === nothing && (threading = Threads.nthreads() > 1)
    return _setup_dvec(fdv, style, initiator, threading)
end

function _setup_dvec(fdv::Union{FrozenDVec,Pair}, style, initiator, threading)
    # we are allocating new memory
    if threading
        return PDVec(fdv; style, initiator)
    end
    if initiator isa NonInitiator
        v = DVec(fdv; style)
    else
        v = InitiatorDVec(fdv; style, initiator)
    end
    return v
end
