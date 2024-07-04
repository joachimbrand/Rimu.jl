"""
    TimeStepStrategy

Abstract type for strategies for updating the time step with
[`update_dτ()`](@ref). Implemented strategies:

   * [`ConstantTimeStep`](@ref)
"""
abstract type TimeStepStrategy end

"""
    ConstantTimeStep <: TimeStepStrategy

Keep `time_step` constant.
"""
struct ConstantTimeStep <: TimeStepStrategy end

"""
    update_dτ(s<:TimeStepStrategy, time_step, tnorm) -> new_time_step
Update the time step according to the strategy `s`.
"""
update_dτ(::ConstantTimeStep, time_step, args...) = time_step
