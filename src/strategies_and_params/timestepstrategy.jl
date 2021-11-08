"""
    TimeStepStrategy

Abstract type for strategies for updating the time step with
[`update_dτ()`](@ref). Implemented strategies:

   * [`ConstantTimeStep`](@ref)
"""
abstract type TimeStepStrategy end

"""
    ConstantTimeStep <: TimeStepStrategy

Keep `dτ` constant.
"""
struct ConstantTimeStep <: TimeStepStrategy end

"""
    update_dτ(s<:TimeStepStrategy, dτ, tnorm) -> new dτ
Update the time step according to the strategy `s`.
"""
update_dτ(::ConstantTimeStep, dτ, args...) = dτ

struct AcceleratingTimeStep{T} <: TimeStepStrategy
    target_norm::T
    low::Float64
    high::Float64
end

function update_dτ(strat::AcceleratingTimeStep, dτ, tnorm)
    rel_norm = abs(strat.target_norm - tnorm) / strat.target_norm
    return rel_norm * strat.low + (1 - rel_norm) * strat.high
end
