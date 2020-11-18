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
     FciqmcRunStrategy{T}
Abstract type representing the strategy for running and terminating
[`fciqmc!()`](@ref). The type parameter `T` is relevant for reporting the shift
and the norm.

Implemented strategies:

   * [`RunTillLastStep`](@ref)
"""
abstract type FciqmcRunStrategy{T} end


@with_kw mutable struct RunTillLastStep{T} <: FciqmcRunStrategy{T}
    step::Int = 0 # number of current/starting timestep
    laststep::Int = 100 # number of final timestep
    shiftMode::Bool = false # whether to adjust shift
    shift::T = 0.0 # starting/current value of shift
    dτ::Float64 = 0.01 # time step
end
@doc """
    RunTillLastStep(step::Int = 0 # number of current/starting timestep
                 laststep::Int = 100 # number of final timestep
                 shiftMode::Bool = false # whether to adjust shift
                 shift = 0.0 # starting/current value of shift
                 dτ::Float64 = 0.01 # current value of time step
    ) <: FciqmcRunStrategy
Parameters for running [`fciqmc!()`](@ref) for a fixed number of time steps.
For alternative strategies, see [`FciqmcRunStrategy`](@ref).
""" RunTillLastStep

"""
    ReportingStrategy
Abstract type for strategies for reporting data in a DataFrame with
[`report!()`](@ref). It also affects the calculation and reporting of
projected quantities in the DataFrame.

# Implemented strategies:
   * [`EveryTimeStep`](@ref)
   * [`EveryKthStep`](@ref)
   * [`ReportDFAndInfo`](@ref)

Every strategy accepts the keyword arguments `projector` and `hproj`
according to which
a projection of the instantaneous coefficient vector `projector⋅v` and
`hproj⋅v` are
reported to the DataFrame  in the fields `df.vproj` and `df.hproj`,
respectively. Possible values for `projector` are
* `nothing` - no projections are computed (default)
* `dv::AbstractDVec` - compute projection onto coefficient vector `dv` (set up with [`copytight`](@ref) to conserve memory)
* [`UniformProjector()`](@ref) - projection onto vector of all ones
* [`NormProjector()`](@ref) - compute 1-norm instead of projection
* [`Norm2Projector()`](@ref) - compute 2-norm instead of projection

In order to help set up the calculation of the projected energy,
where `df.hproj` should report `dot(projector, ham, v)`, the keyword `hproj`
accepts the following values (for `ReportingStrategy`s passed to `lomc!()`):
* `:auto` - choose method depending on `projector` and `ham` (default)
* `:lazy` - compute `dot(projector, ham, v)` every time (slow)
* `:eager` -  precompute `hproj` as `ham'*v` (fast, requires `adjoint(ham)`)
* `:not` - don't compute second projector (equivalent to `nothing`)

# Examples
```julia
r_strat = EveryTimeStep(projector = copytight(svec))
```
Record the projected energy components `df.vproj = svec⋅v` and
`df.hproj = dot(svec,ham,v)` with respect to
the starting vector (performs fast eager calculation if
`Hamiltonians.LOStructure(ham) == Hamiltonians.HermitianLO()`),
and report every time step.

```julia
r_strat = EveryKthStep(k=10, projector = UniformProjector(), hproj = :lazy)
```
Record the projection of the instananeous coefficient vector `v` onto
the uniform vector of all 1s into `df.vproj` and of `ham⋅v` into `df.hproj`,
and report every `k`th time step.
"""
abstract type ReportingStrategy{P1,P2} end

@with_kw struct EveryTimeStep{P1,P2} <: ReportingStrategy{P1,P2}
    projector::P1 = nothing # no projection by default
    hproj::P2 = :auto # choose automatically by default
end
@doc """
    EveryTimeStep(;projector = nothing, hproj = :auto)
Report every time step. Include projection onto `projector`. See
[`ReportingStrategy`](@ref) for details.
""" EveryTimeStep

# function EveryTimeStep(; projector = missing, ham = missing)
#     EveryTimeStep(projector, ham'*projector)
#     # we need the adjoint of the Hamiltonian here because eventually we want to
#     # compute df.hproj = dot(projector, ham, v) [== (ham'*projector)⋅v]
# end

@with_kw struct EveryKthStep{P1,P2} <: ReportingStrategy{P1,P2}
    k::Int = 10
    projector::P1 = nothing # no projection by default
    hproj::P2 = :auto # choose automatically by default
end
@doc """
    EveryKthStep(;k = 10, projector = nothing, hproj = :auto)
Report every `k`th step. Include projection onto `projector`. See
[`ReportingStrategy`](@ref) for details.
""" EveryKthStep

@with_kw struct ReportDFAndInfo{P1,P2} <: ReportingStrategy{P1,P2}
    k::Int = 10 # how often to write to DataFrame
    i::Int = 100 # how often to write info message
    io::IO = stdout # IO stream for info messages
    writeinfo::Bool = true # write info only if true - useful for MPI codes
    projector::P1 = nothing # no projection by default
    hproj::P2 = :auto # choose automatically by default
end
@doc """
    ReportDFAndInfo(; k=10, i=100, io=stdout, writeinfo=true, projector = nothing, hproj = :auto)
Report every `k`th step in DataFrame and write info message to `io` every `i`th
step (unless `writeinfo == false`). The flag `writeinfo` is useful for
controlling info messages in MPI codes. Include projection onto `projector`.
See [`ReportingStrategy`](@ref) for details.
""" ReportDFAndInfo

"""
    compute_proj_observables(v, ham, r::ReportingStrategy)
Compute the projection of `r.projector⋅v` and `r.hproj⋅v` or
`r.projector⋅ham*v` according to
the [`ReportingStrategy`](@ref) `r`.
"""
function compute_proj_observables(v, ham, ::RS) where
                        {P1 <: Nothing, P2 <: Nothing,
                         RS<:ReportingStrategy{P1,P2}}
    return missing, missing # nothing to do
end

# catch an error
function compute_proj_observables(v, ham, ::RS) where
                        {P1, P2 <: Symbol,
                         RS<:ReportingStrategy{P1,P2}}
    throw(ErrorException("`Symbol` is not a valid type for `hproj`. Use `refine_r_strat()`!"))
    return missing, missing
end

#  single projector, e.g. for norm calculation
function compute_proj_observables(v, ham, r::RS) where
                        {P1, P2 <: Nothing,
                         RS<:ReportingStrategy{P1,P2}}
    return r.projector⋅v, missing
end
# The dot products work across MPI when `v::MPIData`; MPI sync

# (slow) generic version with single projector, e.g. for computing projected energy
function compute_proj_observables(v, ham, r::RS) where
                        {P1, P2 <: Missing,
                         RS<:ReportingStrategy{P1,P2}}
    return r.projector⋅v, dot(r.projector, ham, v)
end
# The dot products work across MPI when `v::MPIData`; MPI sync

# fast version with 2 projectors, e.g. for computing projected energy
function compute_proj_observables(v, ham, r::ReportingStrategy)
    return r.projector⋅v, r.hproj⋅v
end
# The dot products work across MPI when `v::MPIData`; MPI sync

# # version for `Norm?Projector`s
# # Only norm of vector is computed to save time
# function compute_proj_observables(v, ham, r::RS) where
#                         {DV<:Union{NormProjector,Norm2Projector}, RS<:ReportingStrategy{DV}}
#     return r.projector⋅v, missing
# end
# # The dot products work across MPI when `v::MPIData`; MPI sync

"""
    report!(df::DataFrame, t::Tuple, s<:ReportingStrategy)
Record results in `df` and write informational messages according to strategy
`s`. See [`ReportingStrategy`](@ref).
"""
report!(df::DataFrame,t::Tuple,s::EveryTimeStep) = push!(df,t)
# report!(df::DataFrame,t::Tuple,s::Union{EveryTimeStep, ReportPEnergy}) = push!(df,t)

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

# @with_kw mutable struct OvershootControl <: TimeStepStrategy
#     targetwalkers::Int
#     dτinit::Float64
#     speedup::Bool
# end
# @doc "Slow down/Speed up `dτ` to control the psips overshoot." OvershootControl

# @inline function update_dτ(s::OvershootControl, dτ, tnorm, args...)
#     if tnorm >= s.targetwalkers
#         s.speedup = true
#     end
#     if s.speedup && dτ < s.dτinit
#         dτ += 0.1*(1-dτ/s.dτinit)*dτ
#     else
#         dτ = (1-tnorm/(s.targetwalkers*1.1))*s.dτinit
#     end
#     return dτ
# end
# here we implement the trivial strategy: don't change dτ

"""
Abstract type for defining the strategy for injectimg memory noise.
Implemented strategies:

  * [`NoMemory`](@ref)
  * [`DeltaMemory`](@ref)
  * [`ShiftMemory`](@ref)
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
Before updating the shift, memory noise with a memory length of `Δ` is applied,
where `Δ = 1` means no memory noise.

```
r̃ = (pnorm - tnorm)/(dτ*pnorm) + shift
r = r̃ - <r̃>
```
"""
mutable struct DeltaMemory <: MemoryStrategy
    Δ::Int # length of memory noise buffer
    pnorm::Float64 # previous norm
    noiseBuffer::DataStructures.CircularBuffer{Float64} # buffer for memory noise
end
DeltaMemory(Δ::Int) = DeltaMemory(Δ, NaN, DataStructures.CircularBuffer{Float64}(Δ))

"""
    DeltaMemory2(Δ::Int) <: MemoryStrategy
Before updating the shift, memory noise with a memory length of `Δ` is applied,
where `Δ = 1` means no memory noise.

```
r̃ = pnorm - tnorm + shift*dτ*pnorm
r = (r̃ - <r̃>)/(dτ*pnorm)
```
The long-term average of `r` is not guaranteed to be zero.
"""
mutable struct DeltaMemory2 <: MemoryStrategy
    Δ::Int # length of memory noise buffer
    pnorm::Float64 # previous norm
    noiseBuffer::DataStructures.CircularBuffer{Float64} # buffer for memory noise
end
DeltaMemory2(Δ::Int) = DeltaMemory2(Δ, NaN, DataStructures.CircularBuffer{Float64}(Δ))

"""
    DeltaMemory3(Δ::Int, level::Float64) <: MemoryStrategy
Before updating the shift, apply multiplicative memory noise with a
memory length of `Δ` at level `level`,
where `Δ = 1` means no memory noise.

```
r̃ = (pnorm - tnorm)/pnorm + dτ*shift
r = r̃ - <r̃>
w .*= 1 + level*r
```
"""
mutable struct DeltaMemory3 <: MemoryStrategy
    Δ::Int # length of memory noise buffer
    level::Float64 # previous norm
    noiseBuffer::DataStructures.CircularBuffer{Float64} # buffer for memory noise
end
DeltaMemory3(Δ::Int,level::Float64) = DeltaMemory3(Δ, level, DataStructures.CircularBuffer{Float64}(Δ))


"""
    ShiftMemory(Δ::Int) <: MemoryStrategy
Effectively replaces the fluctuating `shift` update procedure for the
coefficient vector by an averaged `shift` over `Δ` timesteps,
where `Δ = 1` means no averaging.
"""
struct ShiftMemory <: MemoryStrategy
    Δ::Int # length of memory noise buffer
    noiseBuffer::DataStructures.CircularBuffer{Float64} # buffer for memory noise
end
ShiftMemory(Δ::Int) = ShiftMemory(Δ, DataStructures.CircularBuffer{Float64}(Δ))

"""
    ProjectedMemory(Δ::Int, projector, pp::Number) <: MemoryStrategy
    ProjectedMemory(Δ::Int, projector, v::AbstractDVec)
Before updating the shift, apply memory noise to minimize the fluctuations
of the overlap of the coefficient vector with `projector`.
Averaging over `Δ` time steps is applied, where `Δ = 1` means no memory noise
is applied. Use `pp` to initialise the value of the projection or pass `v` in
order to initialise the projection with `pp = projector.v`.
```
r̃ = (projector⋅v - projector⋅w)/projector⋅v + dτ*shift
r = r̃ - <r̃>
```
where `v` is the coefficient vector before and `w` after applying a regular
FCIQMC step.
"""
mutable struct ProjectedMemory{D} <: MemoryStrategy
    Δ::Int # length of memory noise buffer
    pp::Float64 # previous projection
    projector::D # projector
    noiseBuffer::DataStructures.CircularBuffer{Float64} # buffer for memory noise
end
ProjectedMemory(Δ::Int, projector, pp::Number) = ProjectedMemory(Δ, pp, projector, DataStructures.CircularBuffer{Float64}(Δ))
function ProjectedMemory(Δ::Int, projector, v::AbstractDVec)
    pp = projector⋅v
    ProjectedMemory(Δ, pp, projector, DataStructures.CircularBuffer{Float64}(Δ))
end

mutable struct ProjectedMemory2{D} <: MemoryStrategy
    Δ::Int # length of memory noise buffer
    pp::Float64 # previous projection
    projector::D # projector
    noiseBuffer::DataStructures.CircularBuffer{Float64} # buffer for memory noise
end
ProjectedMemory2(Δ::Int, projector, pp::Number) = ProjectedMemory2(Δ, pp, projector, DataStructures.CircularBuffer{Float64}(Δ))
function ProjectedMemory2(Δ::Int, projector, v::AbstractDVec)
    pp = projector⋅v
    ProjectedMemory2(Δ, pp, projector, DataStructures.CircularBuffer{Float64}(Δ))
end

mutable struct ProjectedMemory3{D} <: MemoryStrategy
    Δ::Int # length of memory noise buffer
    pp::Float64 # previous projection
    projector::D # projector
    noiseBuffer::DataStructures.CircularBuffer{Float64} # buffer for memory noise
end
ProjectedMemory3(Δ::Int, projector, pp::Number) = ProjectedMemory3(Δ, pp, projector, DataStructures.CircularBuffer{Float64}(Δ))
function ProjectedMemory3(Δ::Int, projector, v::AbstractDVec)
    pp = projector⋅v
    ProjectedMemory3(Δ, pp, projector, DataStructures.CircularBuffer{Float64}(Δ))
end
mutable struct ProjectedMemory4{D} <: MemoryStrategy
    Δ::Int # length of memory noise buffer
    pp::Float64 # previous projection
    projector::D # projector
    noiseBuffer::DataStructures.CircularBuffer{Float64} # buffer for memory noise
end
ProjectedMemory4(Δ::Int, projector, pp::Number) = ProjectedMemory4(Δ, pp, projector, DataStructures.CircularBuffer{Float64}(Δ))
function ProjectedMemory4(Δ::Int, projector, v::AbstractDVec)
    pp = projector⋅v
    ProjectedMemory4(Δ, pp, projector, DataStructures.CircularBuffer{Float64}(Δ))
end


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

@with_kw struct LogUpdateAfterTargetWalkers <: ShiftStrategy
    targetwalkers::Int
    ζ::Float64 = 0.08 # damping parameter, best left at value of 0.3
end
@doc """
    LogUpdateAfterTargetWalkers(targetwalkers, ζ = 0.08) <: ShiftStrategy
Strategy for updating the shift: After `targetwalkers` is reached, update the
shift according to the log formula with damping parameter `ζ`.
See [`LogUpdate`](@ref).
""" LogUpdateAfterTargetWalkers


@with_kw struct DelayedLogUpdateAfterTargetWalkers <: ShiftStrategy
    targetwalkers::Int
    ζ::Float64 = 0.08 # damping parameter, best left at value of 0.3
    a::Int = 10 # delay for updating shift
end
@doc """
    DelayedLogUpdateAfterTargetWalkers(targetwalkers, ζ = 0.08, a = 10) <: ShiftStrategy
Strategy for updating the shift: After `targetwalkers` is reached, update the
shift according to the log formula with damping parameter `ζ` and delay of
`a` steps. See [`DelayedLogUpdate`](@ref).
""" DelayedLogUpdateAfterTargetWalkers


@with_kw struct LogUpdate <: ShiftStrategy
    ζ::Float64 = 0.08 # damping parameter, best left at value of 0.3
end
@doc """
    LogUpdate(ζ = 0.08) <: ShiftStrategy
Strategy for updating the shift according to the log formula with damping
parameter `ζ`.

```math
S^{n+1} = S^n -\\frac{ζ}{dτ}\\ln\\left(\\frac{\\|Ψ\\|_1^{n+1}}{\\|Ψ\\|_1^n}\\right)
```
""" LogUpdate


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
struct DoubleLogUpdate <: ShiftStrategy
    targetwalkers::Int
    ζ::Float64 # damping parameter, best left at value of 0.08
    ξ::Float64  # restoring force to bring walker number to the target
end
DoubleLogUpdate(;targetwalkers = 1000,  ζ = 0.08, ξ = ζ^2/4) = DoubleLogUpdate(targetwalkers, ζ, ξ)

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


@with_kw mutable struct DoubleLogUpdateAfterTargetWalkers <: ShiftStrategy
    targetwalkers::Int
    ζ::Float64 = 0.08 # damping parameter, best left at value of 0.3
    ξ::Float64 = 0.0016 # restoring force to bring walker number to the target
end
@doc """
    LogUpdateAfterTargetWalkers(targetwalkers, ζ = 0.08, ξ = 0.0016) <: ShiftStrategy
Strategy for updating the shift: After `targetwalkers` is reached, update the
shift according to the log formula with damping parameter `ζ` and `ξ`.
See [`DoubleLogUpdate`](@ref).
""" DoubleLogUpdateAfterTargetWalkers


@with_kw mutable struct DoubleLogUpdateAfterTargetWalkersSwitch <: ShiftStrategy
    targetwalkers::Int
    ζ::Float64 = 0.08 # damping parameter, best left at value of 0.3
    ξ::Float64 = 0.0016 # restoring force to bring walker number to the target
    a::Int = 100 # time period that allows double damping
end
@doc """
    LogUpdateAfterTargetWalkersSwitch(targetwalkers, ζ = 0.08, ξ = 0.0016) <: ShiftStrategy
Strategy for updating the shift: After `targetwalkers` is reached, update the
shift according to the log formula with damping parameter `ζ` and `ξ`. After `a` steps
the strategy switches to [`LogUpdate`](@ref).
See [`DoubleLogUpdate`](@ref).
""" DoubleLogUpdateAfterTargetWalkersSwitch

@with_kw struct DelayedDoubleLogUpdate <: ShiftStrategy
    targetwalkers::Int
    ζ::Float64 = 0.08 # damping parameter, best left at value of 0.3
    ξ::Float64 = 0.0016 # restoring force to bring walker number to the target
    A::Int = 10 # delay for updating shift
end
@doc """
    DelayedDoubleLogUpdate(; targetwalkers = 1000, ζ = 0.08, ξ = ζ^2/4, A=10) <: ShiftStrategy
Strategy for updating the shift according to the log formula with damping
parameter `ζ` and `ξ` and delay of `A` steps.
See [`DoubleLogUpdate`](@ref).

```math
S^{n+A} = S^n -\\frac{ζ}{A dτ}\\ln\\left(\\frac{\\|Ψ\\|_1^{n+A}}{\\|Ψ\\|_1^n}\\right)-\\frac{ξ}{A dτ}\\ln\\left(\\frac{\\|Ψ\\|_1^{n+A}}{\\|Ψ\\|_1^\\text{target}}\\right)
```
""" DelayedDoubleLogUpdate


@with_kw struct DelayedDoubleLogUpdateAfterTW <: ShiftStrategy
    targetwalkers::Int
    ζ::Float64 = 0.08 # damping parameter, best left at value of 0.3
    ξ::Float64 = 0.0016 # restoring force to bring walker number to the target
    A::Int = 10 # delay for updating shift
end
@doc """
    DelayedDoubleLogUpdateAfterTW(; targetwalkers = 1000, ζ = 0.08, ξ = ζ^2/4, A=10) <: ShiftStrategy
Strategy for updating the shift according to the log formula with damping
parameter `ζ` and `ξ` and delay of `A` steps after the number of target walkers is reached.
See [`DoubleLogUpdate`](@ref).

```math
S^{n+A} = S^n -\\frac{ζ}{A dτ}\\ln\\left(\\frac{\\|Ψ\\|_1^{n+A}}{\\|Ψ\\|_1^n}\\right)-\\frac{ξ}{A dτ}\\ln\\left(\\frac{\\|Ψ\\|_1^{n+A}}{\\|Ψ\\|_1^\\text{target}}\\right)
```
""" DelayedDoubleLogUpdateAfterTW


@with_kw struct DelayedLogUpdate <: ShiftStrategy
    ζ::Float64 = 0.08 # damping parameter, best left at value of 0.3
    a::Int = 10 # delay for updating shift
end
@doc """
    DelayedLogUpdate(ζ = 0.08, a = 10) <: ShiftStrategy
Strategy for updating the shift according to the log formula with damping
parameter `ζ` and delay of `a` steps.

```math
S^{n+a} = S^n -\\frac{ζ}{a dτ}\\ln\\left(\\frac{\\|Ψ\\|_1^{n+a}}{\\|Ψ\\|_1^n}\\right)
```
""" DelayedLogUpdate

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


# let's decide whether a simulation is deterministic, stochastic, or
# semistochastic upon a trait on the vector type

"""
    StochasticStyle(v)
    StochasticStyle(typeof(v))
`StochasticStyle` specifies the native style of the generalised vector `v` that
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
During stochastic propagation, walker numbers small than `threshold` will be
stochastically projected to either zero or `threshold`.

The trait can be conveniently defined on an instance of a generalised vector with the macro
[`@setThreshold`](@ref). Example:
```julia-repl
julia> dv = DVec(Dict(nearUniform(BoseFS{3,3})=>3.0))
julia> @setThreshold dv 0.6
julia> StochasticStyle(dv)
IsStochasticWithThreshold(0.6f0)
```
"""
struct IsStochasticWithThreshold <: StochasticStyle
    threshold::Float32
end

"""
    @setThreshold dv threshold
A macro to set a threshold for non-integer walker number FCIQMC. Technically, the macro sets the
trait [`StochasticStyle`](@ref) of the generalised vector `dv` to
[`IsStochasticWithThreshold(threshold)`](@ref), where `dv` must be a type that supports floating
point walker numbers. Also available as function, see [`setThreshold`](@ref).

Example usage:
```julia-repl
julia> dv = DVec(Dict(nearUniform(BoseFS{3,3})=>3.0))
julia> @setThreshold dv 0.6
IsStochasticWithThreshold(0.6f0)
```
"""
macro setThreshold(dv, threshold)
    return esc(quote
        @assert !(valtype($dv) <:Integer) "`valtype(dv)` must not be integer."
        Rimu.StochasticStyle(::Type{typeof($dv)}) = IsStochasticWithThreshold($threshold)
        Rimu.StochasticStyle($dv)
    end)
end

"""
    setThreshold(dv, threshold)
Set a threshold for non-integer walker number FCIQMC. Technically, the function sets the
trait [`StochasticStyle`](@ref) of the generalised vector `dv` to
[`IsStochasticWithThreshold(threshold)`](@ref), where `dv` must be a type that supports floating
point walker numbers. Also available as macro, see [`@setThreshold`](@ref).

Example usage:
```julia-repl
julia> dv = DVec(Dict(nearUniform(BoseFS{3,3})=>3.0))
julia> setThreshold(dv, 0.6)
IsStochasticWithThreshold(0.6f0)
```
"""
function setThreshold(dv, threshold)
    @assert !(valtype(dv) <:Integer) "`valtype(dv)` must not be integer."
    @eval Rimu.StochasticStyle(::Type{typeof($dv)}) = IsStochasticWithThreshold($threshold)
    return Rimu.StochasticStyle(dv)
end

"""
    @setDeterministic dv
A macro to undo the effect of [`@setThreshold`] and set the
trait [`StochasticStyle`](@ref) of the generalised vector `dv` to
[`IsDeterministic()`](@ref).
"""
macro setDeterministic(dv)
    return esc(quote
        @assert !(valtype($dv) <:Integer) "`valtype(dv)` must not be integer."
        Rimu.StochasticStyle(::Type{typeof($dv)}) = IsDeterministic()
        Rimu.StochasticStyle($dv)
    end)
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
        # delete deterministic bit (rightmost) in `flag`
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
Abstract type for defining the strategy of projection for fciqmc with
floating point walker number with [`norm_project`](@ref).
Implemented strategies:

   * [`NoProjection`](@ref)
   * [`NoProjectionTwoNorm`](@ref)
   * [`ThresholdProject`](@ref)
   * [`ScaledThresholdProject`](@ref)
"""
abstract type ProjectStrategy end

"Do not project the walker amplitudes. See [`norm_project`](@ref)."
struct NoProjection <: ProjectStrategy end

"""
Do not project the walker amplitudes. Use two-norm to
calculate walker numbers. This affects reported "norm" but also the shift update procedures.
See [`norm_project`](@ref).
"""
struct NoProjectionTwoNorm <: ProjectStrategy end


@with_kw struct ThresholdProject <: ProjectStrategy
    threshold::Float32 = 1.0f0
end
@doc """
    ThresholdProject(threshold = 1.0) <: ProjectStrategy
Project stochastically for walker amplitudes below `threshold`.
See [`norm_project`](@ref).
""" ThresholdProject


@with_kw struct ScaledThresholdProject <: ProjectStrategy
    threshold::Float32 = 1.0f0
end
@doc """
    ScaledThresholdProject(threshold = 1.0) <: ProjectStrategy
Project stochastically for walker amplitudes below `threshold` and scale
configuration array as to keep the norm constant. As a consequence, the
final configuration amplitudes may be smaller than `threshold`.
See [`norm_project`](@ref).
""" ScaledThresholdProject

@with_kw struct ComplexNoiseCancellation <: ProjectStrategy
    κ::Float32 = 1.0f0
end
@doc """
    ComplexNoiseCancellation(κ = 1.0) <: ProjectStrategy
Use complex noise cancellation strategy with imaginary noise injected at
level `κ`.
See [`norm_project`](@ref).
""" ComplexNoiseCancellation
