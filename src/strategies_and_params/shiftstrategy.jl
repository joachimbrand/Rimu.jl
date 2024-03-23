"""
Abstract type for defining the strategy for controlling the norm, potentially by updating
the `shift`. The concrete types are typically stateful and store the necessary information.
Passed as a parameter to [`lomc!`](@ref).

## Implemented strategies:

* [`DontUpdate`](@ref)
* [`DoubleLogUpdate`](@ref) - default in [`lomc!()`](@ref)
* [`LogUpdate`](@ref)
* [`LogUpdateAfterTargetWalkers`](@ref) - FCIQMC standard
* [`DoubleLogUpdateAfterTargetWalkers`](@ref)
"""
abstract type ShiftStrategy end

"""
    DefaultShiftParameters
Default mutable struct for storing the shift parameters.

See [`shift_parameters`](@ref).
"""
mutable struct DefaultShiftParameters{S, N}
    shift::S # for current time step
    pnorm::N # norm from previous time step
    time_step::Float64
    counter::Int
    shift_mode::Bool
end

"""
    initialise_shift_parameters(s::ShiftStrategy, shift, norm, time_step, counter=0, shift_mode=false)
Initiatlise a struct to store the shift parameters.
"""
function initialise_shift_parameters(
    ::ShiftStrategy, shift, norm, time_step,
    counter=0, shift_mode=false
)
    return DefaultShiftParameters(shift, norm, time_step, counter, shift_mode)
end

"""
    update_shift!(
        shift_parameters,
        s <: ShiftStrategy,
        v_new,
        v_old,
        step,
        report
    ) -> shift_stats, continue
Update the `shift_parameters` according to strategy `s`. See [`ShiftStrategy`](@ref).
Returns a named tuple of the shift statistics and a boolean `continue` indicating whether
the simulation should continue.
"""
update_shift!

"""
    update_shift(s <: ShiftStrategy, shift, tnorm, pnorm, dτ, step, df, v_new, v_old)
Update the shift according to strategy `s`. See [`ShiftStrategy`](@ref).
"""
update_shift

"""
    DontUpdate(; targetwalkers = 1_000_000) <: ShiftStrategy
Don't update the `shift`.  Return when `targetwalkers` is reached.

See [`ShiftStrategy`](@ref), [`lomc!`](@ref).
"""
Base.@kwdef struct DontUpdate <: ShiftStrategy
    targetwalkers::Int = 1_000_000
end

@inline function update_shift(s::DontUpdate, shift, tnorm, args...)
    return shift, tnorm, tnorm < s.targetwalkers
end

function update_shift!(sp, s::DontUpdate, v_new, _...)
    norm = walkernumber(v_new)
    return (; shift=sp.shift, norm), norm < s.targetwalkers
end

"""
    LogUpdateAfterTargetWalkers(targetwalkers, ζ = 0.08) <: ShiftStrategy
Strategy for updating the shift: After `targetwalkers` is reached, update the
shift according to the log formula with damping parameter `ζ`.

See [`LogUpdate`](@ref), [`ShiftStrategy`](@ref), [`lomc!`](@ref).
"""
Base.@kwdef struct LogUpdateAfterTargetWalkers <: ShiftStrategy
    targetwalkers::Int
    ζ::Float64 = 0.08 # damping parameter, best left at value of 0.3
    shift_mode::Ref{Bool} = Ref(false)
end
@inline function update_shift(s::LogUpdateAfterTargetWalkers,
                        shift, tnorm, args...)
    if s.shift_mode[] || real(tnorm) > s.targetwalkers
        s.shift_mode[] = true
        return update_shift(LogUpdate(s.ζ), shift, tnorm, args...)
    end
    return shift, tnorm, true
end

function update_shift!(sp, s::LogUpdateAfterTargetWalkers, v_new, _...)
    @unpack shift, pnorm, time_step, shift_mode = sp
    tnorm = walkernumber(v_new)
    if shift_mode || real(tnorm) > s.targetwalkers
        shift_mode = true
        dτ = time_step
        shift -= s.ζ / dτ * log(tnorm / pnorm)
    end
    pnorm = tnorm
    @pack! sp = shift, pnorm, shift_mode
    return (; shift, norm=tnorm, shift_mode), true
end

"""
    LogUpdate(ζ = 0.08) <: ShiftStrategy
Strategy for updating the shift according to the log formula with damping
parameter `ζ`.

```math
S^{n+1} = S^n -\\frac{ζ}{dτ}\\ln\\left(\\frac{\\|Ψ\\|_1^{n+1}}{\\|Ψ\\|_1^n}\\right)
```

See [`ShiftStrategy`](@ref), [`lomc!`](@ref).
"""
Base.@kwdef struct LogUpdate <: ShiftStrategy
    ζ::Float64 = 0.08 # damping parameter, best left at value of 0.3
end

@inline function update_shift(s::LogUpdate,
                        shift,
                        tnorm, pnorm, dτ, args...)
    # return new shift
    return shift - s.ζ/dτ * log(tnorm/pnorm), tnorm, true
end

function update_shift!(sp, s::LogUpdate, v_new, _...)
    @unpack shift, pnorm, time_step = sp
    tnorm = walkernumber(v_new)
    dτ = time_step
    shift -= s.ζ / dτ * log(tnorm / pnorm)
    pnorm = tnorm
    @pack! sp = shift, pnorm
    return (; shift, norm=tnorm), true
end

"""
    DoubleLogUpdate(; targetwalkers = 1000, ζ = 0.08, ξ = ζ^2/4) <: ShiftStrategy
Strategy for updating the shift according to the log formula with damping
parameter `ζ` and `ξ`.

```math
S^{n+1} = S^n -\\frac{ζ}{dτ}\\ln\\left(\\frac{\\|Ψ\\|_1^{n+1}}{\\|Ψ\\|_1^n}\\right)-\\frac{ξ}{dτ}\\ln\\left(\\frac{\\|Ψ\\|_1^{n+1}}{\\|Ψ\\|_1^\\text{target}}\\right)
```
When ξ = ζ^2/4 this corresponds to critical damping with a damping time scale
T = 2/ζ.

See [`ShiftStrategy`](@ref), [`lomc!`](@ref).
"""
struct DoubleLogUpdate{T} <: ShiftStrategy
    targetwalkers::T
    ζ::Float64 # damping parameter, best left at value of 0.08
    ξ::Float64  # restoring force to bring walker number to the target
end
function DoubleLogUpdate(;targetwalkers = 1000,  ζ = 0.08, ξ = ζ^2/4)
    return DoubleLogUpdate(targetwalkers, ζ, ξ)
end

@inline function update_shift(s::DoubleLogUpdate,
                        shift,
                        tnorm, pnorm, dτ, args...)
    new_shift = shift - s.ξ/dτ * log(tnorm/s.targetwalkers) - s.ζ/dτ * log(tnorm/pnorm)
    return new_shift, tnorm, true
end

function update_shift!(sp, s::DoubleLogUpdate, v_new, _...)
    @unpack shift, pnorm, time_step = sp
    tnorm = walkernumber(v_new)
    dτ = time_step
    shift -= s.ξ / dτ * log(tnorm / s.targetwalkers) + s.ζ / dτ * log(tnorm / pnorm)
    pnorm = tnorm
    @pack! sp = shift, pnorm
    return (; shift, norm=tnorm), true
end

"""
    DoubleLogUpdateAfterTargetWalkers(targetwalkers, ζ = 0.08, ξ = 0.0016) <: ShiftStrategy
Strategy for updating the shift: After `targetwalkers` is reached, update the
shift according to the log formula with damping parameter `ζ` and `ξ`.

See [`DoubleLogUpdate`](@ref), [`ShiftStrategy`](@ref), [`lomc!`](@ref).
"""
Base.@kwdef struct DoubleLogUpdateAfterTargetWalkers <: ShiftStrategy
    targetwalkers::Int
    ζ::Float64 = 0.08 # damping parameter, best left at value of 0.3
    ξ::Float64 = 0.0016 # restoring force to bring walker number to the target
    shift_mode::Ref{Bool} = Ref(false)
end

@inline function update_shift(s::DoubleLogUpdateAfterTargetWalkers,
                        shift,
                        tnorm, pnorm, dτ, step, df, args...)
    if s.shift_mode[] || real(tnorm) > s.targetwalkers
        s.shift_mode[] = true
        return update_shift(
            DoubleLogUpdate(s.targetwalkers,s.ζ,s.ξ),
            shift,
            tnorm,
            pnorm,
            dτ,
            step,
            df
        )
    end
    return shift, tnorm, true
end

function update_shift!(sp, s::DoubleLogUpdateAfterTargetWalkers, v_new, _...)
    @unpack shift, pnorm, time_step, shift_mode = sp
    tnorm = walkernumber(v_new)
    if shift_mode || real(tnorm) > s.targetwalkers
        shift_mode = true
        dτ = time_step
        shift -= s.ξ / dτ * log(tnorm / s.targetwalkers) + s.ζ / dτ * log(tnorm / pnorm)
    end
    pnorm = tnorm
    @pack! sp = shift, pnorm, shift_mode
    return (; shift, norm=tnorm, shift_mode), true
end

# more experimental strategies from here on:

"""
    DoubleLogSumUpdate(; targetwalkers = 1000, ζ = 0.08, ξ = ζ^2/4, α = 1/2) <: ShiftStrategy
Strategy for updating the shift according to the log formula with damping
parameters `ζ` and `ξ`.

```math
S^{n+1} = S^n -\\frac{ζ}{dτ}\\ln\\left(\\frac{N_\\mathrm{w}^{n+1}}{N_\\mathrm{w}^n}\\right)
- \\frac{ξ}{dτ}\\ln\\left(\\frac{N_\\mathrm{w}^{n+1}}{N_\\mathrm{w}^\\text{target}}\\right),
```
where ``N_\\mathrm{w} =`` `(1-α)*walkernumber() + α*UniformProjector()⋅ψ` computed with
[`walkernumber()`](@ref) and [`UniformProjector()`](@ref).
When ξ = ζ^2/4 this corresponds to critical damping with a damping time scale
T = 2/ζ.


See [`ShiftStrategy`](@ref), [`lomc!`](@ref).
"""
struct DoubleLogSumUpdate{T} <: ShiftStrategy
    targetwalkers::T
    ζ::Float64 # damping parameter, best left at value of 0.08
    ξ::Float64  # restoring force to bring walker number to the target
    α::Float64  # mixing angle for (1-α)*walkernumber + α*UniformProjector()⋅ψ
end
function DoubleLogSumUpdate(;targetwalkers = 1000,  ζ = 0.08, ξ = ζ^2/4, α = 1/2)
    DoubleLogSumUpdate(targetwalkers,  ζ, ξ, α)
end

@inline function update_shift(s::DoubleLogSumUpdate,
                        shift,
                        tnorm, pnorm, dτ, step, df, v_new, v_old
)
    tp = DictVectors.UniformProjector() ⋅ v_new
    pp = DictVectors.UniformProjector() ⋅ v_old
    twn = (1 - s.α) * tnorm + s.α * tp
    pwn = (1 - s.α) * pnorm + s.α * pp
    # return new shift
    new_shift = shift - s.ξ/dτ * log(twn/s.targetwalkers) - s.ζ/dτ * log(twn/pwn)
    return new_shift, tnorm, true
end
function update_shift!(sp, s::DoubleLogSumUpdate, v_new, v_old, _...)
    @unpack shift, pnorm, time_step = sp
    tnorm = walkernumber(v_new)
    dτ = time_step
    tp = DictVectors.UniformProjector() ⋅ v_new
    pp = DictVectors.UniformProjector() ⋅ v_old # could be cached
    twn = (1 - s.α) * tnorm + s.α * tp
    pwn = (1 - s.α) * pnorm + s.α * pp
    # return new shift
    shift -= s.ξ / dτ * log(twn / s.targetwalkers) + s.ζ / dτ * log(twn / pwn)
    pnorm = tnorm
    @pack! sp = shift, pnorm
    return (; shift, norm=tnorm, up=tp), true
end


"""
    DoubleLogProjected(; target, projector, ζ = 0.08, ξ = ζ^2/4) <: ShiftStrategy
Strategy for updating the shift according to the log formula with damping
parameter `ζ` and `ξ` after projecting onto `projector`.

```math
S^{n+1} = S^n -\\frac{ζ}{dτ}\\ln\\left(\\frac{P⋅Ψ^{(n+1)}}{P⋅Ψ^{(n)}}\\right)-\\frac{ξ}{dτ}\\ln\\left(\\frac{P⋅Ψ^{(n+1)}}{\\text{target}}\\right)
```

Note that adjusting the keyword `maxlength` in [`lomc!`](@ref) is advised as the
default may not be appropriate.

See [`ShiftStrategy`](@ref), [`lomc!`](@ref).
"""
struct DoubleLogProjected{T,P} <: ShiftStrategy
    target::T
    projector::P
    ζ::Float64 # damping parameter, best left at value of 0.08
    ξ::Float64 # restoring force to bring walker number to the target
end
function DoubleLogProjected(; target, projector, ζ = 0.08, ξ = ζ^2/4)
    return DoubleLogProjected(target, freeze(projector), ζ, ξ)
end

@inline function update_shift(s::DoubleLogProjected,
                        shift,
                        tnorm, pnorm, dτ, step, df, v_new, v_old)
    # return new shift
    tp = s.projector⋅v_new
    pp = s.projector⋅v_old
    new_shift = shift - s.ζ/dτ * log(tp/pp) - s.ξ/dτ * log(tp/s.target)
    return new_shift, tnorm, true
end
function update_shift!(sp, s::DoubleLogProjected, v_new, v_old, _...)
    @unpack shift, pnorm, time_step = sp
    tnorm = walkernumber(v_new)
    dτ = time_step
    tp = s.projector ⋅ v_new
    pp = s.projector ⋅ v_old
    # return new shift
    shift -= s.ξ / dτ * log(tp / s.target) + s.ζ / dτ * log(tp / pp)
    pnorm = tnorm
    @pack! sp = shift, pnorm
    return (; shift, norm=tnorm, tp, pp), true
end
