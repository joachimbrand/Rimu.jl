"""
    FCIQMC(; kwargs...) <: PMCAlgorithm

Algorithm for the full configuration interaction quantum Monte Carlo (FCIQMC) method.
The default algorithm for [`ProjectorMonteCarloProblem`](@ref).

# Keyword arguments and defaults:
- `shift_strategy = DoubleLogUpdate(; targetwalkers = 1_000, ζ = 0.08,
    ξ = ζ^2/4)`: How to update the `shift`.
- `time_step_strategy = ConstantTimeStep()`: Adjust time step or not.

See also [`ProjectorMonteCarloProblem`](@ref), [`ShiftStrategy`](@ref),
[`TimeStepStrategy`](@ref), [`DoubleLogUpdate`](@ref), [`ConstantTimeStep`](@ref).
"""
Base.@kwdef struct FCIQMC{SS<:ShiftStrategy,TS<:TimeStepStrategy} <: PMCAlgorithm
    shift_strategy::SS = DoubleLogUpdate()
    time_step_strategy::TS = ConstantTimeStep()
end
function Base.show(io::IO, a::FCIQMC)
    print(io, "FCIQMC($(a.shift_strategy), $(a.time_step_strategy))")
end

"""
    set_up_initial_shift_parameters(
        algorithm::FCIQMC, hamiltonian, starting_vectors, shift, time_step, initial_shift_parameters
    )

Set up the initial shift parameters for the [`FCIQMC`](@ref) algorithm.
"""
function set_up_initial_shift_parameters(algorithm::FCIQMC, hamiltonian,
    starting_vectors::SMatrix{R,S}, shift, time_step
) where {R,S}
    shift_strategy = algorithm.shift_strategy

    if shift === nothing
        initial_shifts = _determine_initial_shift(hamiltonian, starting_vectors)
    elseif shift isa Number
        initial_shifts = [float(shift) for _ in 1:(S * R)] # not great for excited states
    elseif length(shift) == S * R
        initial_shifts = float.(shift)
    else
        throw(ArgumentError("The number of shifts must match the number of starting vectors."))
    end
    initial_shift_parameters = Tuple(map(zip(starting_vectors, initial_shifts)) do (sv, s)
        initialise_shift_parameters(shift_strategy, s, walkernumber(sv), time_step)
    end)
    @assert length(initial_shift_parameters) == S * R
    return SMatrix{R,S}(initial_shift_parameters)
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
    FirstOrderTransitionOperator(hamiltonian, shift, time_step) <: AbstractHamiltonian
    FirstOrderTransitionOperator(sp::DefaultShiftParameters, hamiltonian)

First order transition operator
```math
𝐓 = 1 + dτ(S - 𝐇)
```
where ``𝐇`` is the `hamiltonian`, ``dτ`` the `time_step` and ``S`` is the `shift`.

``𝐓`` represents the first order expansion of the exponential evolution operator
of the imaginary-time Schrödinger equation (Euler step) and repeated application
will project out the ground state eigenvector of the `hamiltonian`.  It is the
transition operator used in [`FCIQMC`](@ref).
"""
struct FirstOrderTransitionOperator{T,S,H} <: AbstractHamiltonian{T}
    hamiltonian::H
    shift::S
    time_step::Float64

    function FirstOrderTransitionOperator(hamiltonian::H, shift::S, time_step) where {H,S}
        T = eltype(hamiltonian)
        return new{T,S,H}(hamiltonian, shift, Float64(time_step))
    end
end

function FirstOrderTransitionOperator(sp::DefaultShiftParameters, hamiltonian)
    return FirstOrderTransitionOperator(hamiltonian, sp.shift, sp.time_step)
end

function Hamiltonians.diagonal_element(t::FirstOrderTransitionOperator, add)
    diag = diagonal_element(t.hamiltonian, add)
    return 1 - t.time_step * (diag - t.shift)
end

struct FirstOrderOffdiagonals{
    A,V,O<:AbstractVector{Tuple{A,V}}
} <: AbstractVector{Tuple{A,V}}
    time_step::Float64
    offdiagonals::O
end
function Hamiltonians.offdiagonals(t::FirstOrderTransitionOperator, add)
    return FirstOrderOffdiagonals(t.time_step, offdiagonals(t.hamiltonian, add))
end
Base.size(o::FirstOrderOffdiagonals) = size(o.offdiagonals)

function Base.getindex(o::FirstOrderOffdiagonals, i)
    add, val = o.offdiagonals[i]
    return add, -val * o.time_step
end

"""
    advance!(algorithm::PMCAlgorithm, report::Report, state::ReplicaState, s_state::SingleState)

Advance the `s_state` by one step according to the `algorithm`. The `state` is used only to
access the various strategies involved. Steps, stats, and computed quantities are written
to the `report`.

Returns `true` if the step was successful and calculation should proceed, `false` when
it should terminate.

See also [`solve!`](@ref), [`step!`](@ref).
"""
function advance!(algorithm::FCIQMC, report, state::ReplicaState, s_state::SingleState)

    @unpack reporting_strategy = state
    @unpack hamiltonian, v, pv, wm, id, shift_parameters = s_state
    @unpack shift, pnorm, time_step = shift_parameters
    @unpack shift_strategy, time_step_strategy = algorithm
    step = state.step[]

    ### PROPAGATOR ACTS
    ### FROM HERE
    transition_op = FirstOrderTransitionOperator(shift_parameters, hamiltonian)

    # Step
    step_stat_names, step_stat_values, wm, pv = apply_operator!(wm, pv, v, transition_op)
    # pv was mutated and now contains the new vector.
    v, pv = (pv, v)

    # Stats:
    tnorm, len = walkernumber_and_length(v)

    # Updates
    time_step = update_time_step(time_step_strategy, time_step, tnorm)

    shift_stats, proceed = update_shift_parameters!(
        shift_strategy, shift_parameters, tnorm, v, pv, step, report
    )

    @pack! s_state = v, pv, wm
    ### TO HERE

    if step % reporting_interval(state.reporting_strategy) == 0
        # Note: post_step_stats must be called after packing the values.
        post_step_stats = post_step_action(state.post_step_strategy, s_state, step)

        # Reporting
        if !(time_step_strategy isa ConstantTimeStep) # report time_step unless it is constant
            report!(reporting_strategy, step, report, (; time_step), id)
        end

        report!(reporting_strategy, step, report, (; len), id)
        report!(reporting_strategy, step, report, shift_stats, id) # shift, norm, shift_mode

        report!(reporting_strategy, step, report, step_stat_names, step_stat_values, id)
        report!(reporting_strategy, step, report, post_step_stats, id)
    end

    if len == 0
        if length(state.spectral_states) > 1
            @error "population in single state $(s_state.id) is dead. Aborting."
        else
            @error "population is dead. Aborting."
        end
        return false
    end
    if len > state.maxlength[]
        if length(state.spectral_states) > 1
            @error "`maxlength` reached in single state $(s_state.id). Aborting."
        else
            @error "`maxlength` reached. Aborting."
        end
        return false
    end
    return proceed # Bool
end

function advance!(algorithm, report, state::ReplicaState, replica::SpectralState{1})
    return advance!(algorithm, report, state, only(replica.single_states))
end

function advance!(algorithm, report, state::ReplicaState, replica::SpectralState{N, <:Any, GramSchmidt{N}}) where {N}
    proceed = true
    if state.step[] % replica.spectral_strategy.orthogonalization_interval == 0
        for i in 1:N
            for j in 1:i-1
                u = replica[i].v
                v = replica[j].v
                add!(u, v, -dot(u, v) / norm(v)^2)
            end
        end
    end
    for i in 1:N
        proceed &= advance!(algorithm, report, state, replica[i])
    end
    return proceed
end
