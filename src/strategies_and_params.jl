# # define parameters and strategies for fciqmc as well as methods that use them
#
# FciqmcRunStrategy, RunTillLastStep
#
# ReportingStrategy
#
# MemoryStrategy
#
# TimeStepStrategy, update_dτ()
#
# ShiftStrategy, update_shift()

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
Abstract type for strategies for reporting data in a DataFrame with [`report!()`](@ref). It
also affects the calculation and reporting of projected quantities in the DataFrame.

# Implemented strategies:

* [`EveryTimeStep`](@ref)
* [`EveryKthStep`](@ref)
* [`ReportDFAndInfo`](@ref)
* [`ReportToFile`](@ref)

Every strategy accepts the keyword arguments `projector` and `hproj` according to which a
projection of the instantaneous coefficient vector `projector⋅v` and `hproj⋅v` are reported
to the DataFrame in the fields `df.vproj` and `df.hproj`, respectively. Possible values for
`projector` are

* `nothing` - no projections are computed (default)
* `dv::AbstractDVec` - compute projection onto coefficient vector `dv` (set up with [`copy`](@ref) to conserve memory)
* [`UniformProjector()`](@ref) - projection onto vector of all ones (i.e. sum of elements)
* [`NormProjector()`](@ref) - compute 1-norm (instead of projection)
* [`Norm1ProjectorPPop()`](@ref) - compute 1-norm per population
* [`Norm2Projector()`](@ref) - compute 2-norm

In order to help set up the calculation of the projected energy, where `df.hproj` should
report `dot(projector, ham, v)`, the keyword `hproj` accepts the following values (for
`ReportingStrategy`s passed to `lomc!()`):

* `:auto` - choose method depending on `projector` and `ham` (default)
* `:lazy` - compute `dot(projector, ham, v)` every time (slow)
* `:eager` -  precompute `hproj` as `ham'*v` (fast, requires `adjoint(ham)`)
* `:not` - don't compute second projector (equivalent to `nothing`)

# Interface

A `ReportingStrategy` must define the following:

* [`report!`](@ref)
* [`report_after_step`](@ref) (optional)
* [`finalize_report!`](@ref) (optional)

# Examples

```julia
r_strat = EveryTimeStep(projector = copy(svec))
```
Record the projected energy components `df.vproj = svec⋅v` and
`df.hproj = dot(svec,ham,v)` with respect to
the starting vector (performs fast eager calculation if
`Hamiltonians.LOStructure(ham) ≠ Hamiltonians.AdjointUnknown()`),
and report every time step.

```julia
r_strat = EveryKthStep(k=10, projector = UniformProjector(), hproj = :lazy)
```
Record the projection of the instananeous coefficient vector `v` onto
the uniform vector of all 1s into `df.vproj` and of `ham⋅v` into `df.hproj`,
and report every `k`th time step.
"""
abstract type ReportingStrategy{P1,P2} end

"""
     report!(::ReportingStrategy, step, report::Report, keys, values, id="")
     report!(::ReportingStrategy, step, report::Report, nt, id="")

Report `keys` and `values` to `report`, which will be converted to a `DataFrame` before
[`lomc!`](@ref) exits. Alternatively, a `nt::NamedTuple` can be passed in place of `keys`
and `values`. If `id` is specified, it is appended to all `keys`. This is used to
differentiate between values reported by different replicas.

To overload this function for a new `ReportingStrategy`, overload
`report!(::ReportingStrategy, step, args...)` and apply the report by calling
`report!(args...)`.
"""
function report!(::ReportingStrategy, _, args...)
    report!(args...)
    return nothing
end

"""
    report_after_step(::ReportingStrategy, step, report, state)

This function is called at the very end of a step. It can let the `ReportingStrategy`
print some information to output.
"""
function report_after_step(::ReportingStrategy, args...)
    return nothing
end

"""
    finalize_report!(::ReportingStrategy, report)

Finalize the report. This function is called after all steps in [`lomc!`](@ref) have finished.
"""
function finalize_report!(::ReportingStrategy, report)
    DataFrame(report)
end

"""
    EveryTimeStep(;projector = nothing, hproj = :auto)
Report every time step. Include projection onto `projector`. See
[`ReportingStrategy`](@ref) for details.
"""
@with_kw struct EveryTimeStep{P1,P2} <: ReportingStrategy{P1,P2}
    projector::P1 = nothing # no projection by default
    hproj::P2 = :auto # choose automatically by default
end

# function EveryTimeStep(; projector = missing, ham = missing)
#     EveryTimeStep(projector, ham'*projector)
#     # we need the adjoint of the Hamiltonian here because eventually we want to
#     # compute df.hproj = dot(projector, ham, v) [== (ham'*projector)⋅v]
# end

"""
    EveryKthStep(;k = 10, projector = nothing, hproj = :auto)
Report every `k`th step. Include projection onto `projector`. See
[`ReportingStrategy`](@ref) for details.
"""
@with_kw struct EveryKthStep{P1,P2} <: ReportingStrategy{P1,P2}
    k::Int = 10
    projector::P1 = nothing # no projection by default
    hproj::P2 = :auto # choose automatically by default
end

"""
    ReportDFAndInfo(; k=10, i=100, io=stdout, writeinfo=true, projector = nothing, hproj = :auto)
Report every `k`th step in DataFrame and write info message to `io` every `i`th
step (unless `writeinfo == false`). The flag `writeinfo` is useful for
controlling info messages in MPI codes. Include projection onto `projector`.
See [`ReportingStrategy`](@ref) for details.
"""
@with_kw struct ReportDFAndInfo{P1,P2} <: ReportingStrategy{P1,P2}
    k::Int = 10 # how often to write to DataFrame
    i::Int = 100 # how often to write info message
    io::IO = stdout # IO stream for info messages
    writeinfo::Bool = true # write info only if true - useful for MPI codes
    projector::P1 = nothing # no projection by default
    hproj::P2 = :auto # choose automatically by default
end

"""
    ReportToFile(; kwargs...) <: ReportingStrategy

Reporting strategy that writes the report directly to a file. Useful when dealing with long
jobs or large numbers of replicas, when the report can incur a significant memory cost.

# Keyword arguments

* `filename`: the file to report to. If the file already exists, a new file is created.
* `chunk_size = 1000`: the size of each chunk that is written to the file.
* `save_if = true`: if this value is true, save the report, otherwise ignore it. Use
  `save_if=is_mpi_root()` when running MPI jobs.
* `return_df`: if this value is true, read the file and return the data frame at the end of
  computation. Otherwise, an empty `DataFrame` is returned.
* `io=stdout`: The `IO` to print messages to. Set to `devnull` if you don't want to see
  messages printed out.
* `projector = nothing`: include projection onto `projector`
* `hproj = :auto`: secondary projector
See `ReportingStrategy` for details regarding the use of projectors.
"""
@with_kw struct ReportToFile{P1,P2} <: ReportingStrategy{P1,P2}
    filename::String
    chunk_size::Int = 1000
    save_if::Bool = true
    return_df::Bool = false
    io::IO = stdout
    projector::P1 = nothing
    hproj::P2 = :auto
end
function report!(s::ReportToFile, _, args...)
    if s.save_if
        report!(args...)
    end
    return nothing
end
function refine_r_strat(s::ReportToFile{P1,P2}, ham::H) where {P1,P2,H}
    if s.save_if
        # If filename exists, add -1 to the end of it. If that exists as well,
        # increment the number after the dash
        new_filename = s.filename
        if isfile(new_filename)
            base, ext = splitext(new_filename)
            new_filename = string(base, "-", 1, ext)
        end
        while isfile(new_filename)
            base, ext = splitext(new_filename)
            m = match(r"(.*)-([0-9]+)$", base)
            if !isnothing(m)
                new_filename = string(m[1], "-", parse(Int, m[2]) + 1, ext)
            end
        end
        if s.filename ≠ new_filename
            println(s.io, "File `$(s.filename)` exists. Using `$(new_filename)`.")
            s = @set s.filename = new_filename
        else
            println(s.io, "Saving report to `$(s.filename)`.")
        end
    end
    # Do the standard refine_r_strat to take care of projectors.
    return invoke(refine_r_strat, Tuple{ReportingStrategy{P1,P2}, H}, s, ham)
end
function report_after_step(s::ReportToFile, step, report, state)
    if s.save_if && step % s.chunk_size == 0
        # Report some stats:
        print(s.io, "[ ", lpad(step, 11), " | ")
        shift = lpad(round(state.replicas[1].params.shift, digits=4), 10)
        norm = lpad(round(state.replicas[1].pnorm, digits=4), 10)
        println(s.io, "shift: ", shift, " | norm: ", norm)

        if isfile(s.filename)
            Arrow.append(s.filename, report.data)
        else
            Arrow.write(s.filename, report.data; file=false)
        end
        empty!(report)
    end
end
function finalize_report!(s::ReportToFile, report)
    if s.save_if
        println(s.io, "Finalizing")
        if isfile(s.filename)
            Arrow.append(s.filename, report.data)
        else
            Arrow.write(s.filename, report.data; file=false)
        end
        if s.return_df
            return DataFrame(Arrow.Table(s.filename))
        end
    end
    return DataFrame()
end

"""
    compute_proj_observables(v, ham, r::ReportingStrategy)
Compute the projection of `r.projector⋅v` and `r.hproj⋅v` or
`r.projector⋅ham*v` according to
the [`ReportingStrategy`](@ref) `r`.
"""
function compute_proj_observables(v, ham, ::ReportingStrategy{Nothing,Nothing})
    return (;)
end

# catch an error
function compute_proj_observables(v, ham, ::ReportingStrategy{<:Any,Symbol})
    error("`Symbol` is not a valid type for `hproj`. Use `refine_r_strat`!")
end

#  single projector, e.g. for norm calculation
function compute_proj_observables(v, ham, r::ReportingStrategy{<:Any,Nothing})
    return (; vproj=r.projector⋅v)
end
# The dot products work across MPI when `v::MPIData`; MPI sync

# (slow) generic version with single projector, e.g. for computing projected energy
function compute_proj_observables(v, ham, r::ReportingStrategy{<:Any,Missing})
    return (; vproj=r.projector⋅v, hproj=dot(r.projector, ham, v))
end
# The dot products work across MPI when `v::MPIData`; MPI sync

# fast version with 2 projectors, e.g. for computing projected energy
function compute_proj_observables(v, ham, r::ReportingStrategy)
    return (; vproj=r.projector⋅v, hproj=r.hproj⋅v)
end
# The dot products work across MPI when `v::MPIData`; MPI sync

# # version for `Norm?Projector`s
# # Only norm of vector is computed to save time
# function compute_proj_observables(v, ham, r::RS) where
#                         {DV<:Union{NormProjector,Norm2Projector}, RS<:ReportingStrategy{DV}}
#     return r.projector⋅v, missing
# end
# # The dot products work across MPI when `v::MPIData`; MPI sync
function report!(s::EveryKthStep, step, args...)
    step % s.k == 0 && report!(args...)
    return nothing
end
function report!(s::ReportDFAndInfo, step, args...)
    step % s.k == 0 && report!(args...)
    return nothing
end

function report_after_step(s::ReportDFAndInfo, step, args...)
    if s.writeinfo && step % s.i == 0
        println(s.io, "Step ", step)
        flush(s.io)
    end
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
    PurgeNegatives <: MemoryStrategy
Purge all negative sign walkers.
"""
struct PurgeNegatives <: MemoryStrategy end

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

"""
    ReplicaStrategy{N}

An abstract type that controles how [`lomc!`](@ref) uses replicas. A subtype of
`ReplicaStrategy{N}` operates on `N` replicas and must implement the following function:

* [`replica_stats(::ReplicaStrategy{N}, ::NTuple{N,ReplicaState})`](@ref) - return a tuple
  of `String`s or `Symbols` of replica statistic names and a tuple of the values.  These
  will be reported to the `DataFrame` returned by [`lomc!`](@ref)

Concrete implementations:

* [`NoStats`](@ref): run (possibly one) replica(s), but don't report any additional info.
* [`AllOverlaps`](@ref): report overlaps between all pairs of replica vectors.

"""
abstract type ReplicaStrategy{N} end

num_replicas(::ReplicaStrategy{N}) where {N} = N

"""
    replica_stats(::ReplicaStrategy{N}, replicas::NTuple{N,ReplicaState}) -> (names, values)

Return the names and values of statistics reported by `ReplicaStrategy`. `names` should be
a tuple of `Symbol`s or `String`s and `values` should be a tuple of the same length.
"""
replica_stats

"""
    NoStats(N=1) <: ReplicaStrategy{N}

The default [`ReplicaStrategy`](@ref). `N` replicas are run, but no statistics are collected.
"""
struct NoStats{N} <: ReplicaStrategy{N} end
NoStats(N=1) = NoStats{N}()

replica_stats(::NoStats, _) = (), ()

"""
    AllOverlaps(n=2, operator=nothing) <: ReplicaStrategy{n}

Run `n` replicas and report overlaps between all pairs of replica vectors. If operator is
not `nothing`, the overlap `dot(c1, operator, c2)` is reported as well. If operator is a tuple
of operators, the overlaps are computed for all operators.

Column names in the report are of the form c{i}_dot_c{j} for vector-vector overlaps, and
c{i}_Op{k}_c{j} for operator overlaps.

See [`ReplicaStrategy`](@ref) and [`AbstractHamiltonian`](@ref) (for an interface for
implementing operators).
"""
struct AllOverlaps{N,O} <: ReplicaStrategy{N}
    operators::O
end

function AllOverlaps(; num_replicas=2, operators=nothing)
    return AllOverlaps(num_replicas, operators)
end
function AllOverlaps(num_replicas=2, operator=nothing)
    if isnothing(operator)
        operators = ()
    elseif operator isa Tuple
        operators = operator
    else
        operators = (operator,)
    end
    return AllOverlaps{num_replicas, typeof(operators)}(operators)
end

function replica_stats(rs::AllOverlaps{N}, replicas) where {N}
    T = promote_type((valtype(r.v) for r in replicas)..., eltype.(rs.operators)...)
    names = String[]
    values = T[]
    for i in 1:N, j in i+1:N
        push!(names, "c$(i)_dot_c$(j)")
        push!(values, dot(localpart(replicas[i].v), localpart(replicas[j].v)))
        for (k, op) in enumerate(rs.operators)
            push!(names, "c$(i)_Op$(k)_c$(j)")
            push!(values, dot(replicas[i].v, op, replicas[j].v))
        end
    end

    num_reports = (N * (N - 1) ÷ 2) * (length(rs.operators) + 1)
    return SVector{num_reports,String}(names).data, SVector{num_reports,T}(values).data
end
