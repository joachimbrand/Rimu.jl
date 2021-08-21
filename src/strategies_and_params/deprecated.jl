# This file should be deleted at some point and is only here to ease the transition to the new
# version.
export DelayedLogUpdateAfterTargetWalkers, HistoryLogUpdate, DelayedLogUpdate
export DelayedDoubleLogUpdateAfterTW, DoubleLogUpdateAfterTargetWalkersSwitch
export DelayedDoubleLogUpdate

function DelayedLogUpdateAfterTargetWalkers(args...; kwargs...)
    error("`DelayedLogUpdateAfterTargetWalkers` is deprecated. See `ShiftStrategy`!")
end
# @with_kw struct DelayedLogUpdateAfterTargetWalkers <: ShiftStrategy
#     targetwalkers::Int
#     ζ::Float64 = 0.08 # damping parameter, best left at value of 0.3
#     a::Int = 10 # delay for updating shift
# end
#
# @inline function update_shift(s::DelayedLogUpdateAfterTargetWalkers,
#                         shift, shiftMode, tnorm, pnorm, args...)
#     if shiftMode || real(tnorm) > s.targetwalkers
#         return update_shift(DelayedLogUpdate(s.ζ,s.a), shift, true, tnorm, pnorm, args...)
#     end
#     return shift, false, pnorm
# end

function HistoryLogUpdate(args...; kwargs...)
    error("`HistoryLogUpdate` is deprecated. See `ShiftStrategy`!")
end

# """
#     HistoryLogUpdate(df::DataFrame; d = 100, k=1, ζ= 0.08)
# Strategy for updating the shift according to log formula but with walker
# numbers accumulated from `k` samples of the history with delay `d`. A
# recent history has to be passed with the data frame `df` for initialisation.
# ```math
# N_w^{n} = \\sum_{i=0}^{k-1} \\|Ψ\\|_1^{n-i} \\\\
# S^{n+a} = S^n -\\frac{ζ}{a dτ}\\ln\\left(\\frac{N_w^{n+1}}{N_w^n}\\right)
# ```
# """
# mutable struct HistoryLogUpdate{T} <: ShiftStrategy
#     ζ::Float64 # damping parameter, best left at value of 0.3
#     d::Int # delay for time window
#     k::Int # number of samples to take from history
#     n_w::T # for remembering last time step's sum of walker numbers
# end
# function HistoryLogUpdate(df::DataFrame; d = 100, k = 1, ζ= 0.08)
#     size(df,1) ≤ d*k && @error "insufficient history for `HistoryLogUpdate`"
#     n_w = sum(df[end-i*d, :norm] for i in 0:(k-1))
#     return HistoryLogUpdate(ζ, d, k, n_w)
# end
#
# @inline function update_shift(s::HistoryLogUpdate,
#                         shift, shiftMode,
#                         tnorm, pnorm, dτ, step, df, args...)
#     prev_n_w = s.n_w # previous sum of walker numbers
#     # compute sum of walker numbers from history
#     s.n_w = sum([df[end-i*s.d, :norm] for i in 0:(s.k-1)])
#     # note that this will fail with a BoundsError if the history is not long enough
#     # return new shift and new shiftMode
#     return shift - s.ζ/dτ * log(s.n_w/prev_n_w), true, tnorm
# end


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

# function copytight(args...; kwargs...)
#     @warn "`copytight` is deprecated. Use `copy`" maxlog=1
#     return copy(args...; kwargs...)
# end

function DelayedLogUpdate(args...; kwargs...)
    error("`DelayedLogUpdate` is deprecated. Use `DontUpdate` for `a` steps and a continuation run with `LogUpdate`. See `ShiftStrategy`!")
end

# """
#     DelayedLogUpdate(ζ = 0.08, a = 10) <: ShiftStrategy
# Strategy for updating the shift according to the log formula with damping
# parameter `ζ` and delay of `a` steps.
#
# ```math
# S^{n+a} = S^n -\\frac{ζ}{a dτ}\\ln\\left(\\frac{\\|Ψ\\|_1^{n+a}}{\\|Ψ\\|_1^n}\\right)
# ```
# """
# @with_kw struct DelayedLogUpdate <: ShiftStrategy
#     ζ::Float64 = 0.08 # damping parameter, best left at value of 0.3
#     a::Int = 10 # delay for updating shift
# end
#
# @inline function update_shift(s::DelayedLogUpdate,
#                         shift, shiftMode,
#                         tnorm, pnorm, dτ, step, df, args...)
#     # return new shift and new shiftMode
#     if step % s.a == 0 && size(df,1) > s.a
#         prevnorm = df[end-s.a+1,:norm]
#         return shift - s.ζ/(dτ * s.a) * log(tnorm/prevnorm), true, tnorm
#         # return shift - s.ζ/(dτ * s.a) * log(tnorm/pnorm), true, tnorm
#     else
#         return shift, true, pnorm # important: return the old norm - not updated
#     end
# end

function DelayedDoubleLogUpdateAfterTW(args...; kwargs...)
    error("`DelayedDoubleLogUpdateAfterTW` is deprecated. See `ShiftStrategy`!")
end

# """
#     DelayedDoubleLogUpdateAfterTW(; targetwalkers = 1000, ζ = 0.08, ξ = ζ^2/4, A=10) <: ShiftStrategy
# Strategy for updating the shift according to the log formula with damping
# parameter `ζ` and `ξ` and delay of `A` steps after the number of target walkers is reached.
# See [`DoubleLogUpdate`](@ref).
#
# ```math
# S^{n+A} = S^n -\\frac{ζ}{A dτ}\\ln\\left(\\frac{\\|Ψ\\|_1^{n+A}}{\\|Ψ\\|_1^n}\\right)-\\frac{ξ}{A dτ}\\ln\\left(\\frac{\\|Ψ\\|_1^{n+A}}{\\|Ψ\\|_1^\\text{target}}\\right)
# ```
# """
# @with_kw struct DelayedDoubleLogUpdateAfterTW <: ShiftStrategy
#     targetwalkers::Int
#     ζ::Float64 = 0.08 # damping parameter, best left at value of 0.3
#     ξ::Float64 = 0.0016 # restoring force to bring walker number to the target
#     A::Int = 10 # delay for updating shift
# end
#
# @inline function update_shift(s::DelayedDoubleLogUpdateAfterTW,
#                         shift, shiftMode,
#                         tnorm, pnorm, dτ, step, df, args...)
#     # return new shift and new shiftMode
#     if real(tnorm) > s.targetwalkers
#         return update_shift(DelayedDoubleLogUpdate(s.targetwalkers,s.ζ,s.ξ,s.A), shift, shiftMode, tnorm, pnorm, dτ, step, df, args...)
#     else
#         return update_shift(DoubleLogUpdate(s.targetwalkers,s.ζ,s.ξ), shift, shiftMode, tnorm, pnorm, dτ, args...)
#     end
# end

function DoubleLogUpdateAfterTargetWalkersSwitch(args...; kwargs...)
    error("`DoubleLogUpdateAfterTargetWalkersSwitch` is deprecated. See `ShiftStrategy`!")
end

# """
#     DoubleLogUpdateAfterTargetWalkersSwitch(targetwalkers, ζ = 0.08, ξ = 0.0016) <: ShiftStrategy
# Strategy for updating the shift: After `targetwalkers` is reached, update the
# shift according to the log formula with damping parameter `ζ` and `ξ`. After `a` steps
# the strategy switches to [`LogUpdate`](@ref).
# See [`DoubleLogUpdate`](@ref).
# """
# @with_kw mutable struct DoubleLogUpdateAfterTargetWalkersSwitch <: ShiftStrategy
#     targetwalkers::Int
#     ζ::Float64 = 0.08 # damping parameter, best left at value of 0.3
#     ξ::Float64 = 0.0016 # restoring force to bring walker number to the target
#     a::Int = 100 # time period that allows double damping
# end
#
# @inline function update_shift(s::DoubleLogUpdateAfterTargetWalkersSwitch,
#                         shift, shiftMode,
#                         tnorm, pnorm, dτ, step, df, args...)
#     if shiftMode || real(tnorm) > s.targetwalkers
#         if s.a > 0
#             s.a -= 1
#             return update_shift(DoubleLogUpdate(s.targetwalkers,s.ζ,s.ξ), shift, true, tnorm, pnorm, dτ, step, df)
#         else
#             return update_shift(LogUpdate(s.ζ), shift, true, tnorm, pnorm, dτ, step, df)
#         end
#     end
#     return shift, false, tnorm
# end

function DelayedDoubleLogUpdate(args...; kwargs...)
    error("`DelayedDoubleLogUpdate` is deprecated. See `ShiftStrategy`!")
end

# """
#     DelayedDoubleLogUpdate(; targetwalkers = 1000, ζ = 0.08, ξ = ζ^2/4, A=10) <: ShiftStrategy
# Strategy for updating the shift according to the log formula with damping
# parameter `ζ` and `ξ` and delay of `A` steps.
# See [`DoubleLogUpdate`](@ref).
#
# ```math
# S^{n+A} = S^n -\\frac{ζ}{A dτ}\\ln\\left(\\frac{\\|Ψ\\|_1^{n+A}}{\\|Ψ\\|_1^n}\\right)-\\frac{ξ}{A dτ}\\ln\\left(\\frac{\\|Ψ\\|_1^{n+A}}{\\|Ψ\\|_1^\\text{target}}\\right)
# ```
# """
# @with_kw struct DelayedDoubleLogUpdate <: ShiftStrategy
#     targetwalkers::Int
#     ζ::Float64 = 0.08 # damping parameter, best left at value of 0.3
#     ξ::Float64 = 0.0016 # restoring force to bring walker number to the target
#     A::Int = 10 # delay for updating shift
# end
#
# @inline function update_shift(s::DelayedDoubleLogUpdate,
#                         shift, shiftMode,
#                         tnorm, pnorm, dτ, step, df, args...)
#     # return new shift and new shiftMode
#     if step % s.A == 0 && size(df,1) >= s.A
#         prevnorm = df[end-s.A+1,:norm]
#         return shift - s.ζ/(dτ * s.A) * log(tnorm/prevnorm) - s.ξ/(dτ * s.A) * log(tnorm/s.targetwalkers), true, tnorm
#     else
#         return shift, true, pnorm # important: return the old norm - not updated
#     end
# end
