"""
Abstract type for defining the strategy for updating the `shift` with
[`update_shift()`](@ref). Implemented strategies:

* [`DontUpdate`](@ref)
* [`DoubleLogUpdate`](@ref) - default in [`lomc!()`](@ref)
* [`LogUpdate`](@ref)
* [`LogUpdateAfterTargetWalkers`](@ref) - FCIQMC standard
* [`DoubleLogUpdateAfterTargetWalkers`](@ref)
"""
abstract type ShiftStrategy end

"""
    update_shift(s <: ShiftStrategy, shift, shiftMode, tnorm, pnorm, dτ, step, df, v_new, v_old)
Update the shift according to strategy `s`. See [`ShiftStrategy`](@ref).
"""
update_shift

"""
    DontUpdate(; targetwalkers = 1_000_000) <: ShiftStrategy
Don't update the `shift`.  Return when `targetwalkers` is reached.
"""
@with_kw struct DontUpdate <: ShiftStrategy
    targetwalkers::Int = 1_000_000
end

@inline function update_shift(s::DontUpdate, shift, _, tnorm, args...)
    return shift, false, tnorm, tnorm < s.targetwalkers
end

"""
    LogUpdateAfterTargetWalkers(targetwalkers, ζ = 0.08) <: ShiftStrategy
Strategy for updating the shift: After `targetwalkers` is reached, update the
shift according to the log formula with damping parameter `ζ`.
See [`LogUpdate`](@ref).
"""
@with_kw struct LogUpdateAfterTargetWalkers <: ShiftStrategy
    targetwalkers::Int
    ζ::Float64 = 0.08 # damping parameter, best left at value of 0.3
end
@inline function update_shift(s::LogUpdateAfterTargetWalkers,
                        shift, shiftMode, tnorm, args...)
    if shiftMode || real(tnorm) > s.targetwalkers
        return update_shift(LogUpdate(s.ζ), shift, true, tnorm, args...)
    end
    return shift, false, tnorm, true
end

"""
    LogUpdate(ζ = 0.08) <: ShiftStrategy
Strategy for updating the shift according to the log formula with damping
parameter `ζ`.

```math
S^{n+1} = S^n -\\frac{ζ}{dτ}\\ln\\left(\\frac{\\|Ψ\\|_1^{n+1}}{\\|Ψ\\|_1^n}\\right)
```
"""
@with_kw struct LogUpdate <: ShiftStrategy
    ζ::Float64 = 0.08 # damping parameter, best left at value of 0.3
end

@inline function update_shift(s::LogUpdate,
                        shift, shiftMode,
                        tnorm, pnorm, dτ, args...)
    # return new shift and new shiftMode
    return shift - s.ζ/dτ * log(tnorm/pnorm), true, tnorm, true
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
                        shift, shiftMode,
                        tnorm, pnorm, dτ, args...)
    new_shift = shift - s.ξ/dτ * log(tnorm/s.targetwalkers) - s.ζ/dτ * log(tnorm/pnorm)
    return new_shift, true, tnorm, true
end

"""
    DoubleLogUpdateAfterTargetWalkers(targetwalkers, ζ = 0.08, ξ = 0.0016) <: ShiftStrategy
Strategy for updating the shift: After `targetwalkers` is reached, update the
shift according to the log formula with damping parameter `ζ` and `ξ`.
See [`DoubleLogUpdate`](@ref).
"""
@with_kw mutable struct DoubleLogUpdateAfterTargetWalkers <: ShiftStrategy
    targetwalkers::Int
    ζ::Float64 = 0.08 # damping parameter, best left at value of 0.3
    ξ::Float64 = 0.0016 # restoring force to bring walker number to the target
end

@inline function update_shift(s::DoubleLogUpdateAfterTargetWalkers,
                        shift, shiftMode,
                        tnorm, pnorm, dτ, step, df, args...)
    if shiftMode || real(tnorm) > s.targetwalkers
            return update_shift(
                DoubleLogUpdate(s.targetwalkers,s.ζ,s.ξ),
                shift,
                true,
                tnorm,
                pnorm,
                dτ,
                step,
                df
            )
    end
    return shift, false, tnorm, true
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
                        shift, shiftMode,
                        tnorm, pnorm, dτ, step, df, v_new, v_old
)
    tp = DictVectors.UniformProjector() ⋅ v_new
    pp = DictVectors.UniformProjector() ⋅ v_old
    twn = (1 - s.α) * tnorm + s.α * tp
    pwn = (1 - s.α) * pnorm + s.α * pp
    # return new shift and new shiftMode
    new_shift = shift - s.ξ/dτ * log(twn/s.targetwalkers) - s.ζ/dτ * log(twn/pwn)
    return new_shift, true, tnorm, true
end

"""
    TripleLogUpdate(; targetwalkers = 1000, ζ = 0.08, ξ = ζ^2/4, η = 0.01) <: ShiftStrategy
Strategy for updating the shift according to the extended log formula with damping
parameters `ζ`, `ξ`, and `η`.

```math
S^{n+1} = S^n -\\frac{ζ}{dτ}\\ln\\left(\\frac{N_\\mathrm{w}^{n+1}}{N_\\mathrm{w}^n}\\right)
- \\frac{ξ}{dτ}\\ln\\left(\\frac{N_\\mathrm{w}^{n+1}}{N_\\mathrm{w}^\\text{target}}\\right)
- \\frac{η}{dτ}\\ln\\left(\\frac{\\|ℜ(Ψ^{n+1})\\|_1^2 + \\|ℑ(Ψ^{n+1})\\|_1^2}
{\\|ℜ(Ψ^{n})\\|_1^2 + \\|ℑ(Ψ^{n})\\|_1^2}\\right),
```
where ``N_\\mathrm{w}`` is the [`walkernumber()`](@ref).
When ξ = ζ^2/4 this corresponds to critical damping with a damping time scale
T = 2/ζ.
"""
struct TripleLogUpdate{T} <: ShiftStrategy
    targetwalkers::T
    ζ::Float64 # damping parameter
    ξ::Float64  # restoring force to bring walker number to the target
    η::Float64
end
function TripleLogUpdate(;targetwalkers = 1000,  ζ = 0.08, ξ = ζ^2/4, η = 0.01)
    return TripleLogUpdate(targetwalkers, ζ, ξ, η)
end

@inline function update_shift(s::TripleLogUpdate,
                        shift, shiftMode,
                        tnorm, pnorm, dτ, step, df, v_new, v_old
)
    tp = abs2(DictVectors.Norm1ProjectorPPop() ⋅ v_new)
    pp = abs2(DictVectors.Norm1ProjectorPPop() ⋅ v_old)
    # return new shift and new shiftMode
    new_shift = shift - s.ξ/dτ * log(tnorm/s.targetwalkers) - s.ζ/dτ * log(tnorm/pnorm)
    # new_shift -= s.η/dτ * log(tp/pp)
    new_shift -= s.η/dτ * log(tp/s.targetwalkers)
    return new_shift, true, tnorm, true
end

"""
    DoubleLogProjected(; target, projector, ζ = 0.08, ξ = ζ^2/4) <: ShiftStrategy
Strategy for updating the shift according to the log formula with damping
parameter `ζ` and `ξ` after projecting onto `projector`.

```math
S^{n+1} = S^n -\\frac{ζ}{dτ}\\ln\\left(\\frac{P⋅Ψ^{(n+1)}}{P⋅Ψ^{(n)}}\\right)-\\frac{ξ}{dτ}\\ln\\left(\\frac{P⋅Ψ^{(n+1)}}{\\text{target}}\\right)
```
"""
struct DoubleLogProjected{P} <: ShiftStrategy
    target::Float64
    projector::P
    ζ::Float64 # damping parameter, best left at value of 0.08
    ξ::Float64 # restoring force to bring walker number to the target
end
DoubleLogProjected(; target, projector, ζ = 0.08, ξ = ζ^2/4) = DoubleLogProjected(target, projector, ζ, ξ)

@inline function update_shift(s::DoubleLogProjected,
                        shift, shiftMode,
                        tnorm, pnorm, dτ, step, df, v_new, v_old)
    # return new shift and new shiftMode
    tp = s.projector⋅v_new
    pp = s.projector⋅v_old
    new_shift = shift - s.ζ/dτ * log(tp/pp) - s.ξ/dτ * log(tp/s.target)
    return new_shift, true, tnorm, true
end
