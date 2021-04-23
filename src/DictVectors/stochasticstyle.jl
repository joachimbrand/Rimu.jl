"""
    StochasticStyle(v)
    StochasticStyle(typeof(v))
`StochasticStyle` specifies the native style of the generalised vector `v` that
determines how simulations are to proceed. This can be fully stochastic (with
`IsStochastic`), fully deterministic (with `IsDeterministic`), or stochastic with
floating point walker numbers and threshold (with [`IsStochasticWithThreshold`](@ref)).
"""
abstract type StochasticStyle end

# some sensible defaults
StochasticStyle(A::Union{AbstractArray,AbstractDVec}) = StochasticStyle(typeof(A))
StochasticStyle(::Type{<:Array}) = IsDeterministic()
StochasticStyle(::Type{Vector{Int}}) = IsStochastic()
function StochasticStyle(T::Type{<:AbstractDVec{K,V}}) where {K,V<:AbstractFloat}
    IsDeterministic()
end
function StochasticStyle(T::Type{<:AbstractDVec{K,V}}) where {I<:Integer,K,V<:Complex{I}}
    IsStochastic2Pop()
end
function StochasticStyle(T::Type{<:AbstractDVec})
    IsStochastic()
end

"""
    IsStochastic()
Trait for generalised vector of configurations indicating stochastic
propagation as seen in the original FCIQMC algorithm.
"""
struct IsStochastic <: StochasticStyle end

"""
    IsStochastic2Pop()
Trait for generalised vector of configurations indicating stochastic
propagation with complex walker numbers representing two populations of integer
walkers.
"""
struct IsStochastic2Pop <: StochasticStyle end

"""
    IsStochastic2PopInitiator()
Trait for generalised vector of configurations indicating stochastic
propagation with complex walker numbers representing two populations of integer
walkers. Initiator algorithm will be used.
"""
struct IsStochastic2PopInitiator <: StochasticStyle end

"""
    IsStochastic2PopWithThreshold(threshold::Float32)
Trait for generalised vector of configurations indicating stochastic
propagation with complex walker numbers representing two populations of real
walkers and cutoff `threshold`.
```
> StochasticStyle(V) = IsStochastic2PopWithThreshold(threshold)
```
During stochastic propagation, walker numbers small than `threshold` will be
stochastically projected to either zero or `threshold`.

The trait can be conveniently defined on an instance of a generalised vector
with the function [`setThreshold`](@ref). Example:
```julia-repl
julia> dv = DVec(nearUniform(BoseFS{3,3}) => 2.0+3.0im; capacity = 10)
julia> setThreshold(dv, 0.6)
julia> StochasticStyle(dv)
IsStochastic2PopWithThreshold(0.6f0)
```
"""
struct IsStochastic2PopWithThreshold <: StochasticStyle
    threshold::Float32
end

struct IsStochasticNonlinear <: StochasticStyle
    c::Float64 # parameter of nonlinear correction applied to local shift
end

"""
    IsDeterministic()
Trait for generalised vector of configuration indicating deterministic propagation of walkers.
"""
struct IsDeterministic <: StochasticStyle end

"""
    IsStochasticWithThreshold(threshold::Float32)
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


###
### Style setting Macros
###
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
    function DictVectors.StochasticStyle(::Type{typeof($dv)})
        IsStochasticWithThreshold($threshold)
    end
    DictVectors.StochasticStyle($dv)
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
    @eval DictVectors.StochasticStyle(::Type{typeof($dv)}) = IsStochasticWithThreshold($threshold)
    return DictVectors.StochasticStyle(dv)
end

function setThreshold(dv::AbstractDVec{K,V}, threshold) where {K,V<:Complex}
    @assert !(real(valtype(dv)) <:Integer) "`valtype(dv)` must not be integer."
    @eval DictVectors.StochasticStyle(::Type{typeof($dv)}) = IsStochastic2PopWithThreshold($threshold)
    return DictVectors.StochasticStyle(dv)
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
        DictVectors.StochasticStyle(::Type{typeof($dv)}) = IsDeterministic()
        DictVectors.StochasticStyle($dv)
    end)
end

"""
    IsDynamicSemistochastic

Parameters:

* `rel_threshold = 1.0`: If the walker number on a configuration times this threshold
  is greater than the number of offdiagonals, spawning is done deterministically. Should be
  set to at most 1 for best performance.

* `abs_threshold = Inf`: If the walker number on a configuration is greater than this value,
  spawning is done deterministically. Can be set to e.g
  `abs_threshold = 0.1 * target_walkers`.

* `proj_threshold = 1.0`: Values below this number are stochastically projected to this
  value or zero. See also [`IsStochasticWithThreshold`](@ref).
"""
Base.@kwdef struct IsDynamicSemistochastic<:StochasticStyle
    rel_threshold::Float64 = 1.0
    abs_threshold::Float64 = Inf
    proj_threshold::Float64 = 1.0
end

# TODO this is here for testing purposes. Should be deleted.
Base.@kwdef struct IsStochasticWithThresholdAndInitiator<:StochasticStyle
    alpha::Float64 = 1.0
    beta::Float64 = 0.0
    threshold::Float64 = 1.0
end
