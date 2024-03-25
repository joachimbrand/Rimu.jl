"""
    SimulationPlan(; starting_step = 1, last_step = 100, walltime = Inf)
Defines the duration of the simulation. The simulation ends when the `last_step` is reached
or the `walltime` is exceeded.

See [`QMCProblem`](@ref).
"""
Base.@kwdef struct SimulationPlan
    starting_step::Int = 0
    last_step::Int = 100
    walltime::Float64 = Inf
end

"""
    QMCProblem(hamiltonian::AbstractHamiltonian; kwargs...)
Defines the problem to be solved by the QMC algorithm.

# Keyword arguments and defaults:
- `targetwalkers = 1_000`: target for the 1-norm of the coefficient vector.
- `time_step = 0.01`: initial time step size.
- `last_step = 100`: controls the number of steps.
- `walltime = Inf`: maximum time allowed for the simulation.
- `reporting_strategy = ReportDFAndInfo()`: strategy for reporting the results.
- `post_step_strategy = ()`: strategy to be executed after each step.
- `replica_strategy = NoStats(n_replicas)`: run several synchronised simulations, see
  [`ReplicaStrategy`](@ref).
- `start_at = starting_address(hamiltonian)`: Define the initial state vector. This can be a
    single address, a collection of addresses, a single starting vector, or a collection of
    starting vectors. If multiple starting vectors are passed, their number must match the
    number of replicas.
- `shift`: initial shift value or collection of shift values. Determined by default from the
    Hamiltonian and the starting vectors.
- `style = IsDynamicSemistochastic()`: The [`StochasticStyle`](@ref) of the simulation.
- `threads = Threads.nthreads()`: number of threads to use.
- `starting_step = 1`: starting step of the simulation.
- `simulation_plan = SimulationPlan(; starting_step, last_step, walltime)`: defines the
    duration of the simulation. Takes precedence over `last_step` and `walltime`.
- `n_replicas = 1`: number of replicas
- `ζ = 0.08`: damping parameter for the shift update.
- `ξ = ζ^2/4`: forcing parameter for the shift update.
- `shift_strategy = DoubleLogUpdate(; targetwalkers, ζ, ξ)`: strategy for updating the shift.
- `initial_shift_parameters`: initial shift parameters or collection of initial shift
    parameters. Overrides `shift` if provided.
- `time_step_strategy = ConstantTimeStep()`: strategy for updating the time step.
- `maxlength = 2 * targetwalkers + 100`: maximum length of the vectors.
"""
struct QMCProblem{N} # is not type stable but does not matter
    hamiltonian::AbstractHamiltonian
    starting_vectors
    style::StochasticStyle
    threads::Int
    simulation_plan::SimulationPlan
    replica_strategy::ReplicaStrategy{N}
    shift_strategy::ShiftStrategy
    initial_shift_parameters::Tuple
    reporting_strategy::ReportingStrategy
    post_step_strategy::Tuple
    time_step_strategy::TimeStepStrategy
    maxlength::Int
end
# could be extended later with
# - `spectral_strategy = NoStats(n_spectral_states)`: strategy for handling excited states

function QMCProblem(
    hamiltonian::AbstractHamiltonian;
    n_replicas = 1,
    start_at = starting_address(hamiltonian),
    shift = nothing,
    style = IsDynamicSemistochastic(),
    threads = Threads.nthreads(),
    time_step = 0.01,
    starting_step = 1,
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
    maxlength = nothing
)
    n_replicas = num_replicas(replica_strategy) # replica_strategy may override n_replicas

    # set up starting_vectors
    if start_at isa AbstractFockAddress # single address
        starting_vectors = (DVec(start_at => 1,),) # tuple of length 1
    elseif eltype(start_at) <: AbstractFockAddress # multiple addresses
        starting_vectors = (DVec(address => 1 for address in start_at),)
    elseif start_at isa AbstractDVec # single starting vector
        starting_vectors = (start_at,) # tuple of length 1
    elseif eltype(start_at) <: Pair{<:AbstractFockAddress} # single starting vector
        starting_vectors = (DVec(start_at),) # tuple of length 1
    elseif eltype(start_at) <: AbstractDVec # multiple starting vectors
        starting_vectors = Tuple(sv for sv in start_at)
    else
        throw(ArgumentError("`start_at` has invalid format."))
    end
    @assert starting_vectors isa NTuple{<:Any, <:AbstractDVec}
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

    return QMCProblem{n_replicas}(
        hamiltonian,
        starting_vectors,
        style,
        threads,
        simulation_plan,
        replica_strategy,
        shift_strategy,
        initial_shift_parameters,
        reporting_strategy,
        post_step_strategy,
        time_step_strategy,
        maxlength
    )
end

function _determine_initial_shift(hamiltonian, starting_vectors)
    shifts = map(starting_vectors) do v
        dot(v, hamiltonian, v) / (v ⋅ v)
        ## or
        # minimum(a -> diagonal_element(hamiltonian, a), keys(v))
    end
    return shifts
end

num_replicas(::QMCProblem{N}) where N = N

# TODO: define method for init(::QMCProblem) to set up the simulation
