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
