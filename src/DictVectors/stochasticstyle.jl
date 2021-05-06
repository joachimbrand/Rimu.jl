"""
    StochasticStyle(v)
    StochasticStyle(typeof(v))
`StochasticStyle` specifies the native style of the generalised vector `v` that
determines how simulations are to proceed. This can be fully stochastic (with
`IsStochastic`), fully deterministic (with `IsDeterministic`), or stochastic with
floating point walker numbers and threshold (with [`IsStochasticWithThreshold`](@ref)).

When defining a new `StochasticStyle`, subtype it as `MyStyle<:StochasticStyle{T}` where `T`
is the concrete value type the style is designed to work with.
"""
abstract type StochasticStyle{T} end

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
struct IsStochastic <: StochasticStyle{Int} end

"""
    IsStochastic2Pop()
Trait for generalised vector of configurations indicating stochastic
propagation with complex walker numbers representing two populations of integer
walkers.
"""
struct IsStochastic2Pop <: StochasticStyle{Complex{Int}} end


"""
    IsDeterministic()
Trait for generalised vector of configuration indicating deterministic propagation of walkers.
"""
struct IsDeterministic <: StochasticStyle{Float32} end

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
struct IsStochasticWithThreshold <: StochasticStyle{Float32}
    threshold::Float32
end


"""
    IsDynamicSemistochastic

Similar to [`IsStochasticWithThreshold`](@ref), but does exact spawning when the number of
walkers in a configuration is high.

Parameters:

* `rel_threshold = 1.0`: If the walker number on a configuration times this threshold
  is greater than the number of offdiagonals, spawning is done deterministically. Should be
  set to 1 or more for best performance.

* `abs_threshold = Inf`: If the walker number on a configuration is greater than this value,
  spawning is done deterministically. Can be set to e.g
  `abs_threshold = 0.1 * target_walkers`.

* `proj_threshold = 1.0`: Values below this number are stochastically projected to this
  value or zero. See also [`IsStochasticWithThreshold`](@ref).
"""
Base.@kwdef struct IsDynamicSemistochastic <: StochasticStyle{Float32}
    rel_threshold::Float64 = 1.0
    abs_threshold::Float64 = Inf
    proj_threshold::Float64 = 1.0
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
