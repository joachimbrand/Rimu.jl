# # define parameters and strategies for fciqmc as well as methods that use them
#
# FciqmcRunStrategy, RunTillLastStep
#
# TimeStepStrategy, update_dτ()
#
# ShiftStrategy, update_shift()
#
# # We also include traits for state vectors:
#
# StochasticStyle(),
# IsStochastic(), IsStochasticNonlinear(), IsDeterministic(),
# IsStochasticWithThreshold(), IsSemistochastic()
#

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
For alternative strategies, see [`FciqmcRunStrategy`](@ref).
"""
@with_kw mutable struct RunTillLastStep <: FciqmcRunStrategy
    step::Int = 0 # number of current/starting timestep
    laststep::Int = 50 # number of final timestep
    shiftMode::Bool = false # whether to adjust shift
    shift::Float64 = 0.0 # starting/current value of shift
    dτ::Float64 = 0.01 # time step
end

"""
Abstract type for strategies for reporting data in a DataFrame with
[`report!()`](@ref). Implemented strategies:
   * [`EveryTimeStep`](@ref)
   * [`EveryKthStep`](@ref)
   * [`ReportDFAndInfo`](@ref)
"""
abstract type ReportingStrategy end

"Report every time step."
struct EveryTimeStep <: ReportingStrategy end

"""
    EveryKthStep(;k = 10)
Report every `k`th step.
"""
@with_kw struct EveryKthStep <: ReportingStrategy
    k::Int = 10
end

"""
    ReportDFAndInfo(; k=10, i=100, io=stdout, writeinfo=true)
Report every `k`th step in DataFrame and write info message to `io` every `i`th
step (unless `writeinfo == false`). The flag `writeinfo` is useful for
controlling info messages in MPI codes.
"""
@with_kw struct ReportDFAndInfo <: ReportingStrategy
    k::Int = 10 # how often to write to DataFrame
    i::Int = 100 # how often to write info message
    io::IO = stdout # IO stream for info messages
    writeinfo::Bool = true # write info only if true - useful for MPI codes
end

"""
    report!(df::DataFrame, t::Tuple, s<:ReportingStrategy)
Record results in `df` and write informational messages according to strategy
`s`. See [`ReportingStrategy`](@ref).
"""
report!(df::DataFrame,t::Tuple,s::EveryTimeStep) = push!(df,t)

function report!(df::DataFrame,t::Tuple,s::EveryKthStep)
    step = t[1]
    step % s.k == 0 && push!(df,t) # only push to df if step is multiple of s.k
    return df
end

function report!(df::DataFrame,t::Tuple,s::ReportDFAndInfo)
    step = t[1]
    step % s.k == 0 && push!(df,t) # only push to df if step is multiple of s.k
    if s.writeinfo && step % s.i == 0
        println(s.io, "Step ", step)
        flush(s.io)
    end
    return df
end

"""
Abstract type for strategies for updating the time step with
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

"Slow down/Speed up `dτ` to control the psips overshoot."
@with_kw mutable struct OvershootControl <: TimeStepStrategy
    targetwalkers::Int
    dτinit::Float64
    speedup::Bool
end

@inline function update_dτ(s::OvershootControl, dτ, tnorm, args...)
    if tnorm >= s.targetwalkers
        s.speedup = true
    end
    if s.speedup && dτ < s.dτinit
        dτ += 0.1*(1-dτ/s.dτinit)*dτ
    else
        dτ = (1-tnorm/(s.targetwalkers*1.1))*s.dτinit
    end
    return dτ
end
# here we implement the trivial strategy: don't change dτ

"""
Abstract type for defining the strategy for injectimg memory noise.
Implemented strategies:

  * [`NoMemory`](@ref)
  * [`DeltaMemory`](@ref)
"""
abstract type MemoryStrategy end

"""
    NoMemory <: MemoryStrategy
Default strategy for [`MemoryStrategy`](@ref) indicating that
no memory noise will be used.
"""
struct NoMemory <: MemoryStrategy end

"""
    DeltaMemory(Δ::Int) <: MemoryStrategy
Before updating the shift memory noise with a memory length of `Δ` is applied,
where `Δ = 1` means no memory noise.
"""
mutable struct DeltaMemory <: MemoryStrategy
    Δ::Int # length of memory noise buffer
    pnorm::Float64 # previous norm
    noiseBuffer::DataStructures.CircularBuffer{Float64} # buffer for memory noise
end
DeltaMemory(Δ::Int) = DeltaMemory(Δ, NaN, DataStructures.CircularBuffer{Float64}(Δ))


"""
Abstract type for defining the strategy for updating the `shift` with
[`update_shift()`](@ref). Implemented strategies:

   * [`DontUpdate`](@ref)
   * [`LogUpdate`](@ref)
   * [`DelayedLogUpdate`](@ref)
   * [`LogUpdateAfterTargetWalkers`](@ref)
   * [`DelayedLogUpdateAfterTargetWalkers`](@ref)
   * [`HistoryLogUpdate`](@ref)
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
    DoubleLogUpdate(targetwalkers, ζ = 0.3, ξ = 0.0225) <: ShiftStrategy
Strategy for updating the shift according to the log formula with damping
parameter `ζ` and `ξ`.

```math
S^{n+1} = S^n -\\frac{ζ}{dτ}\\ln\\left(\\frac{\\|Ψ\\|_1^{n+1}}{\\|Ψ\\|_1^n}\\right)\\frac{ζ}{dτ}\\ln\\left(\\frac{\\|Ψ\\|_1^{n+1}}{\\|Ψ\\|_1^\\text{target}}\\right)
```
"""
@with_kw struct DoubleLogUpdate <: ShiftStrategy
    targetwalkers::Int
    ζ::Float64 = 0.3 # damping parameter, best left at value of 0.3
    ξ::Float64 = 0.0225 # restoring force to bring walker number to the target
end

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
    LogUpdateAfterTargetWalkers(targetwalkers, ζ = 0.3, ξ = 0.0225) <: ShiftStrategy
Strategy for updating the shift: After `targetwalkers` is reached, update the
shift according to the log formula with damping parameter `ζ` and `ξ`.
See [`DoubleLogUpdate`](@ref).
"""
@with_kw mutable struct DoubleLogUpdateAfterTargetWalkers <: ShiftStrategy
    targetwalkers::Int
    ζ::Float64 = 0.3 # damping parameter, best left at value of 0.3
    ξ::Float64 = 0.0225 # restoring force to bring walker number to the target
end

"""
    LogUpdateAfterTargetWalkersSwitch(targetwalkers, ζ = 0.3, ξ = 0.0225) <: ShiftStrategy
Strategy for updating the shift: After `targetwalkers` is reached, update the
shift according to the log formula with damping parameter `ζ` and `ξ`. After `a` steps
the strategy swiches to [`LogUpdate`](@ref).
See [`DoubleLogUpdate`](@ref).
"""
@with_kw mutable struct DoubleLogUpdateAfterTargetWalkersSwitch <: ShiftStrategy
    targetwalkers::Int
    ζ::Float64 = 0.3 # damping parameter, best left at value of 0.3
    ξ::Float64 = 0.0225 # restoring force to bring walker number to the target
    a::Int = 100 # time period that allows double damping
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
    HistoryLogUpdate(df::DataFrame; d = 100, k=1, ζ= 0.3)
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
function HistoryLogUpdate(df::DataFrame; d = 100, k = 1, ζ= 0.3)
    size(df,1) ≤ d*k && @error "insufficient history for `HistoryLogUpdate`"
    n_w = sum(df[end-i*d, :norm] for i in 0:(k-1))
    return HistoryLogUpdate(ζ, d, k, n_w)
end

"""
    update_shift(s <: ShiftStrategy, shift, shiftMode, tnorm, pnorm, dτ, step, df)
Update the shift according to strategy `s`. See [`ShiftStrategy`](@ref).
"""
@inline function update_shift(s::HistoryLogUpdate,
                        shift, shiftMode,
                        tnorm, pnorm, dτ, step, df)
    prev_n_w = s.n_w # previous sum of walker numbers
    # compute sum of walker numbers from history
    s.n_w = sum([df[end-i*s.d, :norm] for i in 0:(s.k-1)])
    # note that this will fail with a BoundsError if the history is not long enough
    # return new shift and new shiftMode
    return shift - s.ζ/dτ * log(s.n_w/prev_n_w), true, tnorm
end

@inline function update_shift(s::LogUpdate,
                        shift, shiftMode,
                        tnorm, pnorm, dτ, step, df)
    # return new shift and new shiftMode
    return shift - s.ζ/dτ * log(tnorm/pnorm), true, tnorm
end

@inline function update_shift(s::DoubleLogUpdate,
                        shift, shiftMode,
                        tnorm, pnorm, dτ, step, df)
    # return new shift and new shiftMode
    return shift - s.ξ/dτ * log(tnorm/s.targetwalkers) - s.ζ/dτ * log(tnorm/pnorm), true, tnorm
end

@inline function update_shift(s::DelayedLogUpdate,
                        shift, shiftMode,
                        tnorm, pnorm, dτ, step, df)
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
    if shiftMode || tnorm > s.targetwalkers
        return update_shift(LogUpdate(s.ζ), shift, true, tnorm, args...)
    end
    return shift, false, tnorm
end

@inline function update_shift(s::DoubleLogUpdateAfterTargetWalkers,
                        shift, shiftMode,
                        tnorm, pnorm, dτ, step, df, args...)
    if shiftMode || tnorm > s.targetwalkers
            return update_shift(DoubleLogUpdate(s.targetwalkers,s.ζ,s.ξ), shift, true, tnorm, pnorm, dτ, step, df)
    end
    return shift, false, tnorm
end

@inline function update_shift(s::DoubleLogUpdateAfterTargetWalkersSwitch,
                        shift, shiftMode,
                        tnorm, pnorm, dτ, step, df, args...)
    if shiftMode || tnorm > s.targetwalkers
        if s.a > 0
            s.a -= 1
            return update_shift(DoubleLogUpdate(s.targetwalkers,s.ζ,s.ξ), shift, true, tnorm, pnorm, dτ, step, df)
        else
            return update_shift(LogUpdate(s.ζ), shift, true, tnorm, pnorm, dτ, step, df)
        end
    end
    return shift, false, tnorm
end

@inline function update_shift(s::DelayedLogUpdateAfterTargetWalkers,
                        shift, shiftMode, tnorm, pnorm, args...)
    if shiftMode || tnorm > s.targetwalkers
        return update_shift(DelayedLogUpdate(s.ζ,s.a), shift, true, tnorm, pnorm, args...)
    end
    return shift, false, pnorm
end

@inline update_shift(::DontUpdate, shift, tnorm, args...) = (shift, false, tnorm)


# let's decide whether a simulation is deterministic, stochastic, or
# semistochastic upon a trait on the vector type

"""
    StochasticStyle(V)
    StochasticStyle(typeof(V))
`StochasticStyle` specifies the native style of the generalised vector `V` that
determines how simulations are to proceed. This can be fully stochastic (with
`IsStochastic`), fully deterministic (with `IsDeterministic`), or semistochastic
(with [`IsSemistochastic`](@ref)).
"""
abstract type StochasticStyle end

struct IsStochastic <: StochasticStyle end

struct IsStochasticNonlinear <: StochasticStyle
    c::Float64 # parameter of nonlinear correction applied to local shift
end

struct IsDeterministic <: StochasticStyle end

"""
    IsStochasticWithThreshold(threshold::Float16)
Trait for generalised vector of configurations indicating stochastic
propagation with real walker numbers and cutoff `threshold`.
```
> StochasticStyle(V) = IsStochasticWithThreshold(threshold)
```
During stochastic propoagation, walker numbers small than `threshold` will be
stochastically projected to either zero or `threshold`.
"""
struct IsStochasticWithThreshold <: StochasticStyle
    threshold::Float32
end

"""
    IsSemistochastic(threshold::Float16, d_space)
Trait for generalised vector of configurations indicating semistochastic
propagation. Set with [`setSemistochastic!`](@ref).
```
> StochasticStyle(V) = IsSemistochastic(threshold, d_space)
```
where `d_space` is a vector of addresses defining the the stochastic subspace.
"""
struct IsSemistochastic{T} <: StochasticStyle
    threshold::Float16
    d_space::Vector{T} # list of addresses in deterministic space
end

"""
    setSemistochastic!(dv, threshold::Float16, d_space)
Set the deterministic space for `dv` with threshold `threshold`, where
`d_space` is a vector of addresses defining the the stochastic subspace.
"""
function setSemistochastic!(dv, threshold::Float16, d_space)
    clearDSpace!(dv)
    for add in d_space
        (val, flag) = dv[add]
        dv[add] = (val, flag | one(typeof(flag)))
    end
    StochasticStyle(dv) = IsSemistochastic(threshold, d_space)
    dv
end

"""
    clearDFlags!(dv)
Clear all flags in `dv` of the deterministic bit (rightmost bit).
"""
function clearDFlags!(dv)
    for (add, (val, flag)) in pairs(dv)
        # delete determimistic bit (rightmost) in `flag`
        dv[add] = (val, flag ⊻ one(typeof(flag)))
    end
    dv
end

# some sensible defaults
StochasticStyle(A::Union{AbstractArray,AbstractDVec}) = StochasticStyle(typeof(A))
StochasticStyle(::Type{<:Array}) = IsDeterministic()
StochasticStyle(::Type{Vector{Int}}) = IsStochastic()
# the following works for dispatch, i.e. the function is evaluated at compile time
function StochasticStyle(T::Type{<:AbstractDVec})
    ifelse(eltype(T) <: Integer, IsStochastic(), IsDeterministic())
end

# """
#     tnorm = apply_memory_noise!(v, w, s_strat, pnorm, tnorm, shift, dτ)
# Apply memory noise to `v` according to the shift update strategy `s_strat`
# returning the updated norm.
# If the strategy does not require updating the norm, it does nothing and returns
# `tnorm`.
# """
# apply_memory_noise!(v, w, s_strat, pnorm, tnorm, shift, dτ) = tnorm
#
# # maybe it would be better to apply this before projecting to threshold
# function apply_memory_noise!(v, w, s_strat::DoubleLogUpdateMemoryNoise,
#                              pnorm, tnorm, shift, dτ)
#     if StochasticStyle(v) ∉ [IsStochasticWithThreshold, IsSemistochastic]
#         @error "Memory noise not defined for $(StochasticStyle(v)). Use `IsStochasticWithThreshold` or `IsSemistochastic` instead."
#     end
#     @unpack noisebuffer = s_strat # extract noisebuffer from shift strategy
#     r̃ = (pnorm - tnorm)/(dτ*pnorm) + shift # instantaneous noisy correction
#     push!(noisebuffer, r̃) # add to noisebuffer
#     r_noise = r̃ - sum(noisebuffer)/length(noisebuffer) # subtract Δ average
#     lv = localpart(v)
#     for (add, val) in kvpairs(w)
#         c̃ = lv[add]
#         c = c̃ + dτ*r_noise*val
#         if sign(c̃) == sign(c)
#             lv[add] = c
#             # !!! Careful! This could lead to a sign change!!!!
#         end
#     end
#     return norm(v, 1) # MPI sycncronising: total number of psips
# end

"""
Abstract type for defining the stategy of projection for fciqmc with
floating point walker number with [`norm_project`](@ref).
Implemented stategies:

   * [`NoProjection`](@ref)
   * [`ThresholdProject`](@ref)
   * [`ScaledThresholdProject`](@ref)
"""
abstract type ProjectStrategy end

"Do not project the walker amplitudes. See [`norm_project`](@ref)."
struct NoProjection <: ProjectStrategy end

"""
    ThresholdProject(threshold = 1.0) <: ProjectStrategy
Project stochastically for walker amplitudes below `threshold`.
See [`norm_project`](@ref).
"""
@with_kw struct ThresholdProject <: ProjectStrategy
    threshold::Float32 = 1.0f0
end

"""
    ScaledThresholdProject(threshold = 1.0) <: ProjectStrategy
Project stochastically for walker amplitudes below `threshold` and scale
configuration array as to keep the norm constant. As a consequence, the
final configuration amplitudes may be smaller than `threshold`.
See [`norm_project`](@ref).
"""
@with_kw struct ScaledThresholdProject <: ProjectStrategy
    threshold::Float32 = 1.0f0
end
