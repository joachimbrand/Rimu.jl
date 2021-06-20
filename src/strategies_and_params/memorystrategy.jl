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
    noiseBuffer::CircularBuffer{Float64} # buffer for memory noise
end
DeltaMemory(Δ::Int) = DeltaMemory(Δ, NaN, CircularBuffer{Float64}(Δ))

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
    noiseBuffer::CircularBuffer{Float64} # buffer for memory noise
end
DeltaMemory2(Δ::Int) = DeltaMemory2(Δ, NaN, CircularBuffer{Float64}(Δ))

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
    noiseBuffer::CircularBuffer{Float64} # buffer for memory noise
end
DeltaMemory3(Δ::Int,level::Float64) = DeltaMemory3(Δ, level, CircularBuffer{Float64}(Δ))


"""
    ShiftMemory(Δ::Int) <: MemoryStrategy
Effectively replaces the fluctuating `shift` update procedure for the
coefficient vector by an averaged `shift` over `Δ` timesteps,
where `Δ = 1` means no averaging.
"""
struct ShiftMemory <: MemoryStrategy
    Δ::Int # length of memory noise buffer
    noiseBuffer::CircularBuffer{Float64} # buffer for memory noise
end
ShiftMemory(Δ::Int) = ShiftMemory(Δ, CircularBuffer{Float64}(Δ))

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
    noiseBuffer::CircularBuffer{Float64} # buffer for memory noise
end
ProjectedMemory(Δ::Int, projector, pp::Number) = ProjectedMemory(Δ, pp, projector, CircularBuffer{Float64}(Δ))
function ProjectedMemory(Δ::Int, projector, v::AbstractDVec)
    pp = projector⋅v
    ProjectedMemory(Δ, pp, projector, CircularBuffer{Float64}(Δ))
end

mutable struct ProjectedMemory2{D} <: MemoryStrategy
    Δ::Int # length of memory noise buffer
    pp::Float64 # previous projection
    projector::D # projector
    noiseBuffer::CircularBuffer{Float64} # buffer for memory noise
end
ProjectedMemory2(Δ::Int, projector, pp::Number) = ProjectedMemory2(Δ, pp, projector, CircularBuffer{Float64}(Δ))
function ProjectedMemory2(Δ::Int, projector, v::AbstractDVec)
    pp = projector⋅v
    ProjectedMemory2(Δ, pp, projector, CircularBuffer{Float64}(Δ))
end

mutable struct ProjectedMemory3{D} <: MemoryStrategy
    Δ::Int # length of memory noise buffer
    pp::Float64 # previous projection
    projector::D # projector
    noiseBuffer::CircularBuffer{Float64} # buffer for memory noise
end
ProjectedMemory3(Δ::Int, projector, pp::Number) = ProjectedMemory3(Δ, pp, projector, CircularBuffer{Float64}(Δ))
function ProjectedMemory3(Δ::Int, projector, v::AbstractDVec)
    pp = projector⋅v
    ProjectedMemory3(Δ, pp, projector, CircularBuffer{Float64}(Δ))
end
mutable struct ProjectedMemory4{D} <: MemoryStrategy
    Δ::Int # length of memory noise buffer
    pp::Float64 # previous projection
    projector::D # projector
    noiseBuffer::CircularBuffer{Float64} # buffer for memory noise
end
ProjectedMemory4(Δ::Int, projector, pp::Number) = ProjectedMemory4(Δ, pp, projector, CircularBuffer{Float64}(Δ))
function ProjectedMemory4(Δ::Int, projector, v::AbstractDVec)
    pp = projector⋅v
    ProjectedMemory4(Δ, pp, projector, CircularBuffer{Float64}(Δ))
end
