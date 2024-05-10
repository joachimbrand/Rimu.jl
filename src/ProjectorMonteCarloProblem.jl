"""
    FCIQMC(; kwargs...)

Algorithm for the full configuration interaction quantum Monte Carlo (FCIQMC) method.
The default algorithm for [`ProjectorMonteCarloProblem`](@ref).

# Keyword arguments and defaults:
- `shift_strategy = DoubleLogUpdate(; targetwalkers = 1_000, ζ = 0.08,
    ξ = ζ^2/4)`: How to update the `shift`.
- `time_step_strategy = ConstantTimeStep()`: Adjust time step or not.

See also [`ProjectorMonteCarloProblem`](@ref), [`ShiftStrategy`](@ref),
[`TimeStepStrategy`](@ref), [`DoubleLogUpdate`](@ref), [`ConstantTimeStep`](@ref).
"""
Base.@kwdef struct FCIQMC{SS<:ShiftStrategy,TS<:TimeStepStrategy}
    shift_strategy::SS = DoubleLogUpdate()
    time_step_strategy::TS = ConstantTimeStep()
end
function Base.show(io::IO, a::FCIQMC)
    print(io, "FCIQMC($(a.shift_strategy), $(a.time_step_strategy))")
end

"""
    set_up_initial_shift_parameters(algorithm::FCIQMC, hamiltonian,
    starting_vectors, shift, time_step, initial_shift_parameters
)

Set up the initial shift parameters for the FCIQMC algorithm.
"""
function set_up_initial_shift_parameters(algorithm::FCIQMC, hamiltonian,
    starting_vectors, shift, time_step, initial_shift_parameters
)
    shift_strategy = algorithm.shift_strategy
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
    return initial_shift_parameters
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

"""
    SimulationPlan(; starting_step = 1, last_step = 100, walltime = Inf)
Defines the duration of the simulation. The simulation ends when the `last_step` is reached
or the `walltime` is exceeded.

See [`ProjectorMonteCarloProblem`](@ref), [`PMCSimulation`](@ref).
"""
Base.@kwdef struct SimulationPlan
    starting_step::Int = 0
    last_step::Int = 100
    walltime::Float64 = Inf
end

"""
    ProjectorMonteCarloProblem(hamiltonian::AbstractHamiltonian; kwargs...)
Defines a problem to be solved by projector quantum Monte Carlo (QMC) methods, such as the
the [`FCIQMC`](@ref) algorithm.

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

julia> problem = ProjectorMonteCarloProblem(hamiltonian; targetwalkers = 500, last_step = 100);

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
- `time_step_strategy = ConstantTimeStep()``: Adjust time step or not, see
    `TimeStepStrategy`.
- `algorithm = FCIQMC(; shift_strategy, time_step_strategy)`: The algorithm to use.
    Currenlty only [`FCIQMC`](@ref) is implemented.
- `shift`: Initial shift value or collection of shift values. Determined by default from the
    Hamiltonian and the starting vectors.
- `initial_shift_parameters`: Initial shift parameters or collection of initial shift
    parameters. Overrides `shift` if provided.
- `maxlength = 2 * targetwalkers + 100`: Maximum length of the vectors.
- `display_name = "PMCSimulation"`: Name displayed in progress bar (via `ProgressLogging`).
- `metadata`: User-supplied metadata to be added to the report. Must be an iterable of
  pairs or a `NamedTuple`, e.g. `metadata = ("key1" => "value1", "key2" => "value2")`.
  All metadata is converted to strings.
- `random_seed = true`: Provide and store a seed for the random number generator.

See also [`init`](@ref), [`solve`](@ref).
"""
struct ProjectorMonteCarloProblem{N,S} # is not type stable but does not matter
    # N is the number of replicas, S is the number of spectral states
    algorithm
    hamiltonian::AbstractHamiltonian
    starting_vectors
    style::StochasticStyle
    initiator::InitiatorRule
    threading::Bool
    simulation_plan::SimulationPlan
    replica_strategy::ReplicaStrategy{N}
    initial_shift_parameters::Tuple
    reporting_strategy::ReportingStrategy
    post_step_strategy::Tuple
    spectral_strategy::SpectralStrategy{S}
    maxlength::Int
    metadata::LittleDict{String,String} # user-supplied metadata + display_name
    random_seed::Union{Nothing,UInt64}
end

function Base.show(io::IO, p::ProjectorMonteCarloProblem)
    nr = num_replicas(p)
    ns = num_spectral_states(p)
    println(io, "ProjectorMonteCarloProblem with $nr replica(s) and $ns spectral state(s):")
    isnothing(p.algorithm) || println(io, "  Algorithm: ", p.algorithm)
    println(io, "  Hamiltonian: ", p.hamiltonian)
    println(io, "  Style: ", p.style)
    println(io, "  Initiator: ", p.initiator)
    println(io, "  Threading: ", p.threading)
    println(io, "  Simulation Plan: ", p.simulation_plan)
    println(io, "  Replica Strategy: ", p.replica_strategy)
    print(io, "  Reporting Strategy: ", p.reporting_strategy)
    println(io, "  Post Step Strategy: ", p.post_step_strategy)
    println(io, "  Spectral Strategy: ", p.spectral_strategy)
    println(io, "  Maxlength: ", p.maxlength)
    println(io, "  Metadata: ", p.metadata)
    print(io, "  Random Seed: ", p.random_seed)
end


function ProjectorMonteCarloProblem(
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
    time_step_strategy=ConstantTimeStep(),
    algorithm=FCIQMC(; shift_strategy, time_step_strategy),
    initial_shift_parameters=nothing,
    reporting_strategy = ReportDFAndInfo(),
    post_step_strategy = (),
    spectral_strategy = GramSchmidt(),
    maxlength = nothing,
    metadata = nothing,
    display_name = "PMCSimulation",
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
    initial_shift_parameters = set_up_initial_shift_parameters(algorithm, hamiltonian,
        starting_vectors, shift, time_step, initial_shift_parameters)

    @assert length(initial_shift_parameters) == length(starting_vectors)

    shift_strategy = algorithm.shift_strategy
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

    return ProjectorMonteCarloProblem{n_replicas,num_spectral_states(spectral_strategy)}(
        algorithm,
        hamiltonian,
        starting_vectors,
        style,
        initiator,
        threading,
        simulation_plan,
        replica_strategy,
        initial_shift_parameters,
        reporting_strategy,
        post_step_strategy,
        spectral_strategy,
        maxlength,
        metadata,
        random_seed
    )
end

num_replicas(::ProjectorMonteCarloProblem{N}) where N = N
num_spectral_states(::ProjectorMonteCarloProblem{<:Any,S}) where {S} = S
