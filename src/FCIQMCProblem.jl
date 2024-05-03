"""
    SimulationPlan(; starting_step = 1, last_step = 100, walltime = Inf)
Defines the duration of the simulation. The simulation ends when the `last_step` is reached
or the `walltime` is exceeded.

See [`FCIQMCProblem`](@ref), [`QMCSimulation`](@ref).
"""
Base.@kwdef struct SimulationPlan
    starting_step::Int = 0
    last_step::Int = 100
    walltime::Float64 = Inf
end

"""
    FCIQMCProblem(hamiltonian::AbstractHamiltonian; kwargs...)
Defines a problem to be solved by the FCIQMC algorithm.

# Common keyword arguments and defaults:
- `time_step = 0.01`: Initial time step size.
- `last_step = 100`: Controls the number of steps.
- `targetwalkers = 1_000`: Target for the 1-norm of the coefficient vector.
- `start_at = starting_address(hamiltonian)`: Define the initial state vector. This can be a
    single address, a collection of addresses, a single starting vector, or a collection of
    starting vectors. If multiple starting vectors are passed, their number must match the
    number of replicas. If (a collection of) [`AbstractDVec`](@ref)(s) is passed, the
    keyword arguments `style`, `initiator`, and `threading` are ignored.
- `style = IsDynamicSemistochastic()`: The [`StochasticStyle`](@ref) of the simulation.
- `initiator = NonInitiator()`: Whether to use initiators. See [`InitiatorRule`](@ref).
- `threading`: Default is to use multithreading and
  [MPI](https://juliaparallel.org/MPI.jl/latest/) if multiple threads are available. Set to
  `true` to force [`PDVec`](@ref) for the starting vector, `false` for serial computation;
  may be overridden by `start_at`.
- `reporting_strategy = ReportDFAndInfo()`: How and when to report results, see
  [`ReportingStrategy`](@ref).
- `post_step_strategy = ()`: Extract observables (e.g.
  [`ProjectedEnergy`](@ref)), see [`PostStepStrategy`](@ref).
- `n_replicas = 1`: Number of synchronised independent simulations.
- `replica_strategy = NoStats(n_replicas)`: Which results to report from replica
  simulations, see [`ReplicaStrategy`](@ref).

# Example

```jldoctest
julia> hamiltonian = HubbardReal1D(BoseFS(1,2,3));

julia> problem = FCIQMCProblem(hamiltonian; targetwalkers = 500, last_step = 100);

julia> simulation = solve(problem);

julia> simulation.success[]
true

julia> size(DataFrame(simulation))
(100, 10)
```

# Further keyword arguments:
- `starting_step = 1`: Starting step of the simulation.
- `walltime = Inf`: Maximum time allowed for the simulation.
- `simulation_plan = SimulationPlan(; starting_step, last_step, walltime)`: Defines the
    duration of the simulation. Takes precedence over `last_step` and `walltime`.
- `ζ = 0.08`: Damping parameter for the shift update.
- `ξ = ζ^2/4`: Forcing parameter for the shift update.
- `shift_strategy = DoubleLogUpdate(; targetwalkers, ζ, ξ)`: How to update the `shift`,
    see [`ShiftStrategy`](@ref).
- `shift`: Initial shift value or collection of shift values. Determined by default from the
    Hamiltonian and the starting vectors.
- `initial_shift_parameters`: Initial shift parameters or collection of initial shift
    parameters. Overrides `shift` if provided.
- `time_step_strategy = ConstantTimeStep()`: Adjust time step or not, see
  [`TimeStepStrategy`](@ref).
- `maxlength = 2 * targetwalkers + 100`: Maximum length of the vectors.
- `display_name = "QMCSimulation"`: Name displayed in progress bar (via `ProgressLogging`).
- `metadata`: User-supplied metadata to be added to the report. Must be an iterable of
  pairs or a `NamedTuple`, e.g. `metadata = ("key1" => "value1", "key2" => "value2")`.
  All metadata is converted to strings.
- `random_seed = true`: Provide and store a seed for the random number generator.

See also [`init`](@ref), [`solve`](@ref).
"""
struct FCIQMCProblem{N} # is not type stable but does not matter
    hamiltonian::AbstractHamiltonian
    starting_vectors
    style::StochasticStyle
    initiator::InitiatorRule
    threading::Bool
    simulation_plan::SimulationPlan
    replica_strategy::ReplicaStrategy{N}
    shift_strategy::ShiftStrategy
    initial_shift_parameters::Tuple
    reporting_strategy::ReportingStrategy
    post_step_strategy::Tuple
    time_step_strategy::TimeStepStrategy
    spectral_strategy::SpectralStrategy
    maxlength::Int
    metadata::LittleDict{String,String} # user-supplied metadata + display_name
    random_seed::Union{Nothing,UInt64}
end
# could be extended later with
# - `spectral_strategy = NoStats(n_spectral_states)`: strategy for handling excited states

function FCIQMCProblem(
    hamiltonian::AbstractHamiltonian;
    n_replicas = 1,
    start_at = starting_address(hamiltonian),
    shift = nothing,
    style = IsDynamicSemistochastic(),
    initiator = NonInitiator(),
    threading = Threads.nthreads()>1,
    time_step = 0.01,
    starting_step = 0,
    last_step = 100,
    walltime = Inf,
    simulation_plan = SimulationPlan(starting_step, last_step, walltime),
    replica_strategy = NoStats(n_replicas),
    targetwalkers = 1_000,
    ζ = 0.08,
    ξ = ζ^2/4,
    shift_strategy = DoubleLogUpdate(; targetwalkers, ζ, ξ),
    initial_shift_parameters = nothing,
    reporting_strategy = ReportDFAndInfo(),
    post_step_strategy = (),
    time_step_strategy = ConstantTimeStep(),
    spectral_strategy=GramSchmidt(),
    maxlength = nothing,
    metadata = nothing,
    display_name = "QMCSimulation",
    random_seed = true
)
    n_replicas = num_replicas(replica_strategy) # replica_strategy may override n_replicas

    if random_seed == true
        random_seed = rand(RandomDevice(),UInt64)
    elseif random_seed == false
        random_seed = nothing
    elseif !isnothing(random_seed)
        random_seed = UInt64(random_seed)
    end

    # set up starting_vectors
    if start_at isa AbstractFockAddress # single address
        starting_vectors = (freeze(DVec(start_at => 1,)),) # tuple of length 1
    elseif eltype(start_at) <: AbstractFockAddress # multiple addresses
        starting_vectors = (freeze(DVec(address => 1 for address in start_at)),)
    elseif start_at isa Union{AbstractDVec, RMPI.MPIData} # single starting vector
        starting_vectors = (start_at,) # tuple of length 1
    elseif eltype(start_at) <: Pair{<:AbstractFockAddress} # single starting vector
        starting_vectors = (freeze(DVec(start_at)),) # tuple of length 1
    elseif eltype(start_at) <: AbstractDVec # multiple starting vectors
        starting_vectors = Tuple(sv for sv in start_at)
    else
        throw(ArgumentError("`start_at` has invalid format."))
    end
    @assert starting_vectors isa NTuple{<:Any,<:Union{AbstractDVec,FrozenDVec,RMPI.MPIData}}
    length(starting_vectors) == 1 || length(starting_vectors) == n_replicas ||
        throw(ArgumentError("The number of starting vectors must match the number of replicas."))
    all(v -> keytype(v) <: allowed_address_type(hamiltonian), starting_vectors) ||
        throw(ArgumentError("The address type is not allowed for the Hamiltonian."))

    # set up initial_shift_parameters
    if isnothing(initial_shift_parameters)
        if shift === nothing
            initial_shifts = _determine_initial_shift(hamiltonian, starting_vectors)
        elseif shift isa Number
            initial_shifts = [float(shift) for _ in 1:length(starting_vectors)]
        elseif length(shift) == length(starting_vectors)
            initial_shifts = float.(shift)
        else
            throw(ArgumentError("The number of shifts must match the number of starting vectors."))
        end
        initial_shift_parameters = Tuple(map(zip(starting_vectors, initial_shifts)) do (sv, s)
            initialise_shift_parameters(shift_strategy, s, walkernumber(sv), time_step)
        end)
    elseif !(initial_shift_parameters isa Tuple)
        initial_shift_parameters = Tuple(initial_shift_parameters for _ in 1:length(starting_vectors))
    end
    @assert length(initial_shift_parameters) == length(starting_vectors)

    if isnothing(maxlength)
        maxlength = 2 * shift_strategy.targetwalkers + 100 # padding for small walkernumbers
    end

    # convert metadata to LittleDict
    report = Report()
    report_metadata!(report, "display_name", display_name)
    isnothing(metadata) || report_metadata!(report, metadata) # add user metadata
    metadata = report.meta::LittleDict{String, String}

    # set up post_step_strategy as a tuple
    if post_step_strategy isa PostStepStrategy
        post_step_strategy = (post_step_strategy,)
    end

    return FCIQMCProblem{n_replicas}(
        hamiltonian,
        starting_vectors,
        style,
        initiator,
        threading,
        simulation_plan,
        replica_strategy,
        shift_strategy,
        initial_shift_parameters,
        reporting_strategy,
        post_step_strategy,
        time_step_strategy,
        spectral_strategy,
        maxlength,
        metadata,
        random_seed
    )
end

function _determine_initial_shift(hamiltonian, starting_vectors)
    shifts = map(starting_vectors) do v
        if v isa FrozenDVec
            v = DVec(v)
        end
        dot(v, hamiltonian, v) / (v ⋅ v)
        ## or
        # minimum(a -> diagonal_element(hamiltonian, a), keys(v))
    end
    return shifts
end

num_replicas(::FCIQMCProblem{N}) where N = N
