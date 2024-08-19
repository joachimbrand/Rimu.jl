"""
    PMCAlgorithm
Abstract type for projector Monte Carlo algorithms.

See [`ProjectorMonteCarloProblem`](@ref), [`FCIQMC`](@ref).
"""
abstract type PMCAlgorithm end

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
function Base.show(io::IO, plan::SimulationPlan)
    print(
        io, "SimulationPlan(starting_step=", plan.starting_step,
        ", last_step=", plan.last_step, ", walltime=", plan.walltime, ")"
    )
end

"""
    ProjectorMonteCarloProblem(hamiltonian::AbstractHamiltonian; kwargs...)
Defines a problem to be solved by projector quantum Monte Carlo (QMC) methods, such as the
the [`FCIQMC`](@ref) algorithm.

# Common keyword arguments and defaults:
- `time_step = 0.01`: Initial time step size.
- `last_step = 100`: Controls the number of steps.
- `target_walkers = 1_000`: Target for the 1-norm of the coefficient vector.
- `start_at = starting_address(hamiltonian)`: Define the initial state vector(s).
    An ``r × s`` matrix of state vectors can be passed where ``r`` is the
    number of replicas and ``s`` the number of spectral states. See also
    [`default_starting_vector`](@ref).
- `style = IsDynamicSemistochastic()`: The [`StochasticStyle`](@ref) of the simulation.
- `initiator = false`: Whether to use initiators. Can be `true`, `false`, or a valid
    [`InitiatorRule`](@ref).
- `threading`: Default is to use multithreading and/or
  [MPI](https://juliaparallel.org/MPI.jl/latest/) if available. Set to
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

julia> problem = ProjectorMonteCarloProblem(hamiltonian; target_walkers = 500, last_step = 100);

julia> simulation = solve(problem);

julia> simulation.success[]
true

julia> size(DataFrame(simulation))
(100, 9)
```

# Further keyword arguments:
- `starting_step = 1`: Starting step of the simulation.
- `walltime = Inf`: Maximum time allowed for the simulation.
- `simulation_plan = SimulationPlan(; starting_step, last_step, walltime)`: Defines the
    duration of the simulation. Takes precedence over `last_step` and `walltime`.
- `ζ = 0.08`: Damping parameter for the shift update.
- `ξ = ζ^2/4`: Forcing parameter for the shift update.
- `shift_strategy = DoubleLogUpdate(; target_walkers, ζ, ξ)`: How to update the `shift`,
    see [`ShiftStrategy`](@ref).
- `time_step_strategy = ConstantTimeStep()`: Adjust time step or not, see
    `TimeStepStrategy`.
- `algorithm = FCIQMC(; shift_strategy, time_step_strategy)`: The algorithm to use.
    Currenlty only [`FCIQMC`](@ref) is implemented.
- `shift`: Initial shift value or collection of shift values. Determined by default from the
    Hamiltonian and the starting vectors.
- `initial_shift_parameters`: Initial shift parameters or collection of initial shift
    parameters. Overrides `shift` if provided.
- `maxlength = 2 * target_walkers + 100`: Maximum length of the vectors.
- `display_name = "PMCSimulation"`: Name displayed in progress bar (via `ProgressLogging`).
- `metadata`: User-supplied metadata to be added to the report. Must be an iterable of
  pairs or a `NamedTuple`, e.g. `metadata = ("key1" => "value1", "key2" => "value2")`.
  All metadata is converted to strings.
- `random_seed = true`: Provide and store a seed for the random number generator. If set to
    `true`, a random seed is generated. If set to number, this number is used as the seed.
    The seed is used by `solve` such that `solve`ing the problem twice will yield identical
    results. If set to `false`, no seed is used and results are not reproducible.

See also [`init`](@ref), [`solve`](@ref).
"""
struct ProjectorMonteCarloProblem{N,S} # is not type stable but does not matter
    # N is the number of replicas, S is the number of spectral states
    algorithm::PMCAlgorithm
    hamiltonian::AbstractHamiltonian
    start_at  # starting_vectors
    style::StochasticStyle
    initiator::InitiatorRule
    threading::Bool
    simulation_plan::SimulationPlan
    replica_strategy::ReplicaStrategy{N}
    initial_shift_parameters
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
    initiator = false,
    threading = nothing,
    time_step = 0.01,
    starting_step = 0,
    last_step = 100,
    walltime = Inf,
    simulation_plan = SimulationPlan(starting_step, last_step, walltime),
    replica_strategy = NoStats(n_replicas),
    targetwalkers = nothing, # deprecated
    target_walkers = 1_000,
    ζ = 0.08,
    ξ = ζ^2/4,
    shift_strategy = DoubleLogUpdate(; target_walkers, ζ, ξ),
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
    if !isnothing(targetwalkers)
        @warn "The keyword argument `targetwalkers` is deprecated. Use `target_walkers` instead."
        target_walkers = targetwalkers
    end

    n_replicas = num_replicas(replica_strategy) # replica_strategy may override n_replicas

    if random_seed == true
        random_seed = rand(RandomDevice(),UInt64)
    elseif random_seed == false
        random_seed = nothing
    elseif !isnothing(random_seed)
        random_seed = UInt64(random_seed)
    end

    if initiator isa Bool
        initiator = initiator ? Initiator() : NonInitiator()
    end

    if isnothing(threading)
        s_strat = algorithm.shift_strategy
        if !hasfield(typeof(s_strat), :target_walkers) || abs(s_strat.target_walkers) > 1_000
            threading = Threads.nthreads() > 1
        else
            threading = false
        end
    end

    # a proper setup of initial_shift_parameters is done in PMCSimulation
    # here we just store the initial shift and time_step if initial_shift_parameters is not
    # provided
    if isnothing(initial_shift_parameters)
        initial_shift_parameters = (; shift, time_step)
    end

    shift_strategy = algorithm.shift_strategy
    if isnothing(maxlength)
        maxlength = round(Int, 2 * abs(shift_strategy.target_walkers) + 100)
        # padding for small walkernumbers
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

    if !(eltype(hamiltonian)<: Real)
        throw(ArgumentError("Only real-valued Hamiltonians are currently supported "*
            "for ProjectorMonteCarloProblem. Please get in touch with the Rimu.jl " *
            "developers if you need a complex-valued Hamiltonian!"))
    end

    return ProjectorMonteCarloProblem{n_replicas,num_spectral_states(spectral_strategy)}(
        algorithm,
        hamiltonian,
        start_at, # starting_vectors,
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
