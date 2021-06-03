"""
Abstract type for defining the strategy for updating the `shift` with
[`update_shift()`](@ref). Implemented strategies:

* [`DoubleLogUpdate`](@ref) - default in [`lomc!()`](@ref)
* [`DontUpdate`](@ref)
* [`LogUpdate`](@ref)
* [`DelayedLogUpdate`](@ref)
* [`LogUpdateAfterTargetWalkers`](@ref) - FCIQMC standard
* [`DelayedLogUpdateAfterTargetWalkers`](@ref)
* [`DoubleLogUpdateAfterTargetWalkers`](@ref)
* [`DoubleLogUpdateAfterTargetWalkersSwitch`](@ref)
* [`HistoryLogUpdate`](@ref)
* [`DoubleLogProjected`](@ref)
"""
abstract type ShiftStrategy end

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

"""
    DelayedLogUpdateAfterTargetWalkers(targetwalkers, ζ = 0.08, a = 10) <: ShiftStrategy
Strategy for updating the shift: After `targetwalkers` is reached, update the
shift according to the log formula with damping parameter `ζ` and delay of
`a` steps. See [`DelayedLogUpdate`](@ref).
"""
@with_kw struct DelayedLogUpdateAfterTargetWalkers <: ShiftStrategy
    targetwalkers::Int
    ζ::Float64 = 0.08 # damping parameter, best left at value of 0.3
    a::Int = 10 # delay for updating shift
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
DoubleLogUpdate(;targetwalkers = 1000,  ζ = 0.08, ξ = ζ^2/4) = DoubleLogUpdate(targetwalkers, ζ, ξ)

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
# """
#     DoubleLogUpdateMemoryNoise(;targetwalkers = 100,
#                                ζ = 0.3,
#                                ξ = 0.0225,
#                                Δ = 1) <: ShiftStrategy
# Strategy for updating the shift according to the log formula with damping
# parameter `ζ` and `ξ`.
# ```math
# S^{n+1} = S^n -\\frac{ζ}{dτ}\\ln\\left(\\frac{\\|Ψ\\|_1^{n+1}}{\\|Ψ\\|_1^n}\\right)\\frac{ζ}{dτ}\\ln\\left(\\frac{\\|Ψ\\|_1^{n+1}}{\\|Ψ\\|_1^\\text{target}}\\right)
# ```
# Before updating the shift memory noise with a memory length of `Δ` is applied,
# where `Δ = 1` means no memory noise.
# """
# struct DoubleLogUpdateMemoryNoise <: ShiftStrategy
#     targetwalkers::Int
#     ζ::Float64 # damping parameter, best left at value of 0.3
#     ξ::Float64 # restoring force to bring walker number to the target
#     Δ::Int # length of memory noise buffer
#     noiseBuffer::DataStructures.CircularBuffer{Float64} # buffer for memory noise
# end
# # construct with the following constructor:
# function DoubleLogUpdateMemoryNoise(;targetwalkers = 100, ζ = 0.3, ξ = 0.0225, Δ = 1)
#     cb = DataStructures.CircularBuffer{Float64, Δ}
#     DoubleLogUpdateMemoryNoise(targetwalkers, ζ, ξ, Δ, cb)
# end

"""
    LogUpdateAfterTargetWalkers(targetwalkers, ζ = 0.08, ξ = 0.0016) <: ShiftStrategy
Strategy for updating the shift: After `targetwalkers` is reached, update the
shift according to the log formula with damping parameter `ζ` and `ξ`.
See [`DoubleLogUpdate`](@ref).
"""
@with_kw mutable struct DoubleLogUpdateAfterTargetWalkers <: ShiftStrategy
    targetwalkers::Int
    ζ::Float64 = 0.08 # damping parameter, best left at value of 0.3
    ξ::Float64 = 0.0016 # restoring force to bring walker number to the target
end

"""
    LogUpdateAfterTargetWalkersSwitch(targetwalkers, ζ = 0.08, ξ = 0.0016) <: ShiftStrategy
Strategy for updating the shift: After `targetwalkers` is reached, update the
shift according to the log formula with damping parameter `ζ` and `ξ`. After `a` steps
the strategy switches to [`LogUpdate`](@ref).
See [`DoubleLogUpdate`](@ref).
"""
@with_kw mutable struct DoubleLogUpdateAfterTargetWalkersSwitch <: ShiftStrategy
    targetwalkers::Int
    ζ::Float64 = 0.08 # damping parameter, best left at value of 0.3
    ξ::Float64 = 0.0016 # restoring force to bring walker number to the target
    a::Int = 100 # time period that allows double damping
end

"""
    DelayedDoubleLogUpdate(; targetwalkers = 1000, ζ = 0.08, ξ = ζ^2/4, A=10) <: ShiftStrategy
Strategy for updating the shift according to the log formula with damping
parameter `ζ` and `ξ` and delay of `A` steps.
See [`DoubleLogUpdate`](@ref).

```math
S^{n+A} = S^n -\\frac{ζ}{A dτ}\\ln\\left(\\frac{\\|Ψ\\|_1^{n+A}}{\\|Ψ\\|_1^n}\\right)-\\frac{ξ}{A dτ}\\ln\\left(\\frac{\\|Ψ\\|_1^{n+A}}{\\|Ψ\\|_1^\\text{target}}\\right)
```
"""
@with_kw struct DelayedDoubleLogUpdate <: ShiftStrategy
    targetwalkers::Int
    ζ::Float64 = 0.08 # damping parameter, best left at value of 0.3
    ξ::Float64 = 0.0016 # restoring force to bring walker number to the target
    A::Int = 10 # delay for updating shift
end

"""
    DelayedDoubleLogUpdateAfterTW(; targetwalkers = 1000, ζ = 0.08, ξ = ζ^2/4, A=10) <: ShiftStrategy
Strategy for updating the shift according to the log formula with damping
parameter `ζ` and `ξ` and delay of `A` steps after the number of target walkers is reached.
See [`DoubleLogUpdate`](@ref).

```math
S^{n+A} = S^n -\\frac{ζ}{A dτ}\\ln\\left(\\frac{\\|Ψ\\|_1^{n+A}}{\\|Ψ\\|_1^n}\\right)-\\frac{ξ}{A dτ}\\ln\\left(\\frac{\\|Ψ\\|_1^{n+A}}{\\|Ψ\\|_1^\\text{target}}\\right)
```
"""
@with_kw struct DelayedDoubleLogUpdateAfterTW <: ShiftStrategy
    targetwalkers::Int
    ζ::Float64 = 0.08 # damping parameter, best left at value of 0.3
    ξ::Float64 = 0.0016 # restoring force to bring walker number to the target
    A::Int = 10 # delay for updating shift
end

"""
    DelayedLogUpdate(ζ = 0.08, a = 10) <: ShiftStrategy
Strategy for updating the shift according to the log formula with damping
parameter `ζ` and delay of `a` steps.

```math
S^{n+a} = S^n -\\frac{ζ}{a dτ}\\ln\\left(\\frac{\\|Ψ\\|_1^{n+a}}{\\|Ψ\\|_1^n}\\right)
```
"""
@with_kw struct DelayedLogUpdate <: ShiftStrategy
    ζ::Float64 = 0.08 # damping parameter, best left at value of 0.3
    a::Int = 10 # delay for updating shift
end

"`DontUpdate() <: ShiftStrategy` Don't update the `shift`."
struct DontUpdate <: ShiftStrategy end

"""
    HistoryLogUpdate(df::DataFrame; d = 100, k=1, ζ= 0.08)
Strategy for updating the shift according to log formula but with walker
numbers accumulated from `k` samples of the history with delay `d`. A
recent history has to be passed with the data frame `df` for initialisation.
```math
N_w^{n} = \\sum_{i=0}^{k-1} \\|Ψ\\|_1^{n-i} \\\\
S^{n+a} = S^n -\\frac{ζ}{a dτ}\\ln\\left(\\frac{N_w^{n+1}}{N_w^n}\\right)
```
"""
mutable struct HistoryLogUpdate{T} <: ShiftStrategy
    ζ::Float64 # damping parameter, best left at value of 0.3
    d::Int # delay for time window
    k::Int # number of samples to take from history
    n_w::T # for remembering last time step's sum of walker numbers
end
function HistoryLogUpdate(df::DataFrame; d = 100, k = 1, ζ= 0.08)
    size(df,1) ≤ d*k && @error "insufficient history for `HistoryLogUpdate`"
    n_w = sum(df[end-i*d, :norm] for i in 0:(k-1))
    return HistoryLogUpdate(ζ, d, k, n_w)
end

"""
    update_shift(s <: ShiftStrategy, shift, shiftMode, tnorm, pnorm, dτ, step, df, v_new, v_old)
Update the shift according to strategy `s`. See [`ShiftStrategy`](@ref).
"""
@inline function update_shift(s::HistoryLogUpdate,
                        shift, shiftMode,
                        tnorm, pnorm, dτ, step, df, args...)
    prev_n_w = s.n_w # previous sum of walker numbers
    # compute sum of walker numbers from history
    s.n_w = sum([df[end-i*s.d, :norm] for i in 0:(s.k-1)])
    # note that this will fail with a BoundsError if the history is not long enough
    # return new shift and new shiftMode
    return shift - s.ζ/dτ * log(s.n_w/prev_n_w), true, tnorm
end

@inline function update_shift(s::LogUpdate,
                        shift, shiftMode,
                        tnorm, pnorm, dτ, args...)
    # return new shift and new shiftMode
    return shift - s.ζ/dτ * log(tnorm/pnorm), true, tnorm
end

@inline function update_shift(s::DoubleLogUpdate,
                        shift, shiftMode,
                        tnorm, pnorm, dτ, args...)
    # return new shift and new shiftMode
    return shift - s.ξ/dτ * log(tnorm/s.targetwalkers) - s.ζ/dτ * log(tnorm/pnorm), true, tnorm
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
    return new_shift, true, tnorm
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
    return new_shift, true, tnorm
end

@inline function update_shift(s::DoubleLogProjected,
                        shift, shiftMode,
                        tnorm, pnorm, dτ, step, df, v_new, v_old)
    # return new shift and new shiftMode
    tp = s.projector⋅v_new
    pp = s.projector⋅v_old
    return shift - s.ζ/dτ * log(tp/pp) - s.ξ/dτ * log(tp/s.target) , true, tnorm
end

@inline function update_shift(s::DelayedLogUpdate,
                        shift, shiftMode,
                        tnorm, pnorm, dτ, step, df, args...)
    # return new shift and new shiftMode
    if step % s.a == 0 && size(df,1) > s.a
        prevnorm = df[end-s.a+1,:norm]
        return shift - s.ζ/(dτ * s.a) * log(tnorm/prevnorm), true, tnorm
        # return shift - s.ζ/(dτ * s.a) * log(tnorm/pnorm), true, tnorm
    else
        return shift, true, pnorm # important: return the old norm - not updated
    end
end

@inline function update_shift(s::LogUpdateAfterTargetWalkers,
                        shift, shiftMode, tnorm, args...)
    if shiftMode || real(tnorm) > s.targetwalkers
        return update_shift(LogUpdate(s.ζ), shift, true, tnorm, args...)
    end
    return shift, false, tnorm
end

@inline function update_shift(s::DoubleLogUpdateAfterTargetWalkers,
                        shift, shiftMode,
                        tnorm, pnorm, dτ, step, df, args...)
    if shiftMode || real(tnorm) > s.targetwalkers
            return update_shift(DoubleLogUpdate(s.targetwalkers,s.ζ,s.ξ), shift, true, tnorm, pnorm, dτ, step, df)
    end
    return shift, false, tnorm
end

@inline function update_shift(s::DoubleLogUpdateAfterTargetWalkersSwitch,
                        shift, shiftMode,
                        tnorm, pnorm, dτ, step, df, args...)
    if shiftMode || real(tnorm) > s.targetwalkers
        if s.a > 0
            s.a -= 1
            return update_shift(DoubleLogUpdate(s.targetwalkers,s.ζ,s.ξ), shift, true, tnorm, pnorm, dτ, step, df)
        else
            return update_shift(LogUpdate(s.ζ), shift, true, tnorm, pnorm, dτ, step, df)
        end
    end
    return shift, false, tnorm
end

@inline function update_shift(s::DelayedDoubleLogUpdate,
                        shift, shiftMode,
                        tnorm, pnorm, dτ, step, df, args...)
    # return new shift and new shiftMode
    if step % s.A == 0 && size(df,1) >= s.A
        prevnorm = df[end-s.A+1,:norm]
        return shift - s.ζ/(dτ * s.A) * log(tnorm/prevnorm) - s.ξ/(dτ * s.A) * log(tnorm/s.targetwalkers), true, tnorm
    else
        return shift, true, pnorm # important: return the old norm - not updated
    end
end

@inline function update_shift(s::DelayedDoubleLogUpdateAfterTW,
                        shift, shiftMode,
                        tnorm, pnorm, dτ, step, df, args...)
    # return new shift and new shiftMode
    if real(tnorm) > s.targetwalkers
        return update_shift(DelayedDoubleLogUpdate(s.targetwalkers,s.ζ,s.ξ,s.A), shift, shiftMode, tnorm, pnorm, dτ, step, df, args...)
    else
        return update_shift(DoubleLogUpdate(s.targetwalkers,s.ζ,s.ξ), shift, shiftMode, tnorm, pnorm, dτ, args...)
    end
end

@inline function update_shift(s::DelayedLogUpdateAfterTargetWalkers,
                        shift, shiftMode, tnorm, pnorm, args...)
    if shiftMode || real(tnorm) > s.targetwalkers
        return update_shift(DelayedLogUpdate(s.ζ,s.a), shift, true, tnorm, pnorm, args...)
    end
    return shift, false, pnorm
end

@inline update_shift(::DontUpdate, shift, tnorm, args...) = (shift, false, tnorm)
