# define parameters and strategies for fciqmc as well as methods that use them
#
# FciqmcRunStrategy, RunTillLastStep
#
# TimeStepStrategy, update_dτ()
#
# ShiftStrategy, update_shift()

"""
Abstract type representing the strategy for running and terminating
[`fciqmc!()`](@ref). Implemented strategies:

   * [`RunTillLastStep`](@ref)
"""
abstract type FciqmcRunStrategy end

"""
    RunTillLastStep(step::Int = 0 # number of current/starting timestep
                 laststep::Int = 50 # number of final timestep
                 shiftMode::Bool = false # whether to adjust shift
                 shift::Float64 = 0.0 # starting/current value of shift
                 dτ::Float64 = 0.01 # current value of time step
    ) <: FciqmcRunStrategy
Parameters for running [`fciqmc!()`](@ref) for a fixed number of time steps.
"""
@with_kw mutable struct RunTillLastStep <: FciqmcRunStrategy
    step::Int = 0 # number of current/starting timestep
    laststep::Int = 50 # number of final timestep
    shiftMode::Bool = false # whether to adjust shift
    shift::Float64 = 0.0 # starting/current value of shift
    dτ::Float64 = 0.01 # time step
end

"""
Abstract type for defining the strategy for updating the time step with
[`update_dτ()`](@ref). Implemented
strategies:

   * [`ConstantTimeStep`](@ref)
"""
abstract type TimeStepStrategy end

"Keep `dτ` constant."
struct ConstantTimeStep <: TimeStepStrategy end

"""
    update_dτ(s<:TimeStepStrategy, dτ, args...) -> new dτ
Update the time step according to the strategy `s`.
"""
update_dτ(::ConstantTimeStep, dτ, args...) = dτ
# here we implement the trivial strategy: don't change dτ

"""
Abstract type for defining the strategy for updating the `shift` with
[`update_shift()`](@ref). Implemented strategies:

   * [`DontUpdate`](@ref)
   * [`LogUpdate`](@ref)
   * [`DelayedLogUpdate`](@ref)
   * [`LogUpdateAfterTargetWalkers`](@ref)
   * [`DelayedLogUpdateAfterTargetWalkers`](@ref)
"""
abstract type ShiftStrategy end

"""
    LogUpdateAfterTargetWalkers(targetwalkers, ζ = 0.3) <: ShiftStrategy
Strategy for updating the shift: After `targetwalkers` is reached, update the
shift according to the log formula with damping parameter `ζ`.
See [`LogUpdate`](@ref).
"""
@with_kw struct LogUpdateAfterTargetWalkers <: ShiftStrategy
    targetwalkers::Int
    ζ::Float64 = 0.3 # damping parameter, best left at value of 0.3
end

"""
    DelayedLogUpdateAfterTargetWalkers(targetwalkers, ζ = 0.3, a = 10) <: ShiftStrategy
Strategy for updating the shift: After `targetwalkers` is reached, update the
shift according to the log formula with damping parameter `ζ` and delay of
`a` steps. See [`DelayedLogUpdate`](@ref).
"""
@with_kw struct DelayedLogUpdateAfterTargetWalkers <: ShiftStrategy
    targetwalkers::Int
    ζ::Float64 = 0.3 # damping parameter, best left at value of 0.3
    a::Int = 10 # delay for updating shift
end

"""
    LogUpdate(ζ = 0.3) <: ShiftStrategy
Strategy for updating the shift according to the log formula with damping
parameter `ζ`.

```math
S^{n+1} = S^n -\\frac{ζ}{dτ}\\ln\\left(\\frac{\\|Ψ\\|_1^{n+1}}{\\|Ψ\\|_1^n}\\right)
```
"""
@with_kw struct LogUpdate <: ShiftStrategy
    ζ::Float64 = 0.3 # damping parameter, best left at value of 0.3
end

"""
    DelayedLogUpdate(ζ = 0.3, a = 10) <: ShiftStrategy
Strategy for updating the shift according to the log formula with damping
parameter `ζ` and delay of `a` steps.

```math
S^{n+a} = S^n -\\frac{ζ}{a dτ}\\ln\\left(\\frac{\\|Ψ\\|_1^{n+a}}{\\|Ψ\\|_1^n}\\right)
```
"""
@with_kw struct DelayedLogUpdate <: ShiftStrategy
    ζ::Float64 = 0.3 # damping parameter, best left at value of 0.3
    a::Int = 10 # delay for updating shift
end

"`DontUpdate() <: ShiftStrategy` Don't update the `shift`."
struct DontUpdate <: ShiftStrategy end

"""
    update_shift(s <: ShiftStrategy, shift, shiftMode, tnorm, pnorm, dτ, step, df)
Update the shift according to strategy `s`. See [`ShiftStrategy`](@ref).
"""
@inline function update_shift(s::LogUpdate,
                        shift, shiftMode,
                        tnorm, pnorm, dτ, step, df)
    # return new shift and new shiftMode
    return shift - s.ζ/dτ * log(tnorm/pnorm), true
end

@inline function update_shift(s::DelayedLogUpdate,
                        shift, shiftMode,
                        tnorm, pnorm, dτ, step, df)
    # return new shift and new shiftMode
    if shift % s.a == 0 && size(df,1) > s.a
        prevnorm = df[end-s.a+1,:norm]
        return shift - s.ζ/(dτ * s.a) * log(tnorm/prevnorm), true
    else
        return shift, true
    end
end

@inline function update_shift(s::LogUpdateAfterTargetWalkers,
                        shift, shiftMode, tnorm, args...)
    if shiftMode || tnorm > s.targetwalkers
        return update_shift(LogUpdate(s.ζ), shift, true, tnorm, args...)
    end
    return shift, false
end

@inline function update_shift(s::DelayedLogUpdateAfterTargetWalkers,
                        shift, shiftMode, tnorm, args...)
    if shiftMode || tnorm > s.targetwalkers
        return update_shift(DelayedLogUpdate(s.ζ,s.a), shift, true, tnorm, args...)
    end
    return shift, false
end

@inline update_shift(::DontUpdate, shift, args...) = (shift, false)
