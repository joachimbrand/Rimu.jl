"""
    StochasticStyle(v)
`StochasticStyle` specifies the native style of the generalised vector `v` that determines
how simulations are to proceed. This can be fully stochastic (with `IsStochasticInteger`),
fully deterministic (with `IsDeterministic`), or stochastic with floating point walker
numbers and threshold (with [`IsStochasticWithThreshold`](@ref)).

When defining a new `StochasticStyle`, subtype it as `MyStyle<:StochasticStyle{T}` where `T`
is the concrete value type the style is designed to work with.
"""
abstract type StochasticStyle{T} end

Base.eltype(::Type{<:StochasticStyle{T}}) where {T} = T

"""
    StyleUnknown()
Trait for value types not (currently) compatible with FCIQMC. This style makes it possible to
construct dict vectors with unsupported `valtype`s.
"""
struct StyleUnknown{T} <: StochasticStyle{T} end

"""
    IsStochasticInteger()
Trait for generalised vector of configurations indicating stochastic
propagation as seen in the original FCIQMC algorithm.
"""
struct IsStochasticInteger <: StochasticStyle{Int} end

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
struct IsDeterministic <: StochasticStyle{Float64} end

"""
    IsStochasticWithThreshold(threshold::Float64)
Trait for generalised vector of configurations indicating stochastic
propagation with real walker numbers and cutoff `threshold`.

During stochastic propagation, walker numbers small than `threshold` will be
stochastically projected to either zero or `threshold`.
"""
struct IsStochasticWithThreshold <: StochasticStyle{Float64}
    threshold::Float64
end

"""
    IsDynamicSemistochastic

Similar to [`IsStochasticWithThreshold`](@ref), but does exact spawning when the number of
walkers in a configuration is high.

Parameters:

* `late_projection = true`: If set to true, threshold projection is done after all spawns are
  collected, otherwise, values are projected as they are being spawned.

* `rel_threshold = 1.0`: If the walker number on a configuration times this threshold
  is greater than the number of offdiagonals, spawning is done deterministically. Should be
  set to 1 or more for best performance.

* `abs_threshold = Inf`: If the walker number on a configuration is greater than this value,
  spawning is done deterministically. Can be set to e.g
  `abs_threshold = 0.1 * target_walkers`.

* `proj_threshold = 1.0`: Values below this number are stochastically projected to this
  value or zero. See also [`IsStochasticWithThreshold`](@ref).
"""
struct IsDynamicSemistochastic{P}<:StochasticStyle{Float64}
    rel_threshold::Float64
    abs_threshold::Float64
    proj_threshold::Float64
end
function IsDynamicSemistochastic(
    ; late_projection::Bool=true, rel_threshold=1.0, abs_threshold=Inf, proj_threshold=1.0
)
    return IsDynamicSemistochastic{late_projection}(
        Float64(rel_threshold), Float64(abs_threshold), Float64(proj_threshold)
    )
end

# Defaults for arrays.
StochasticStyle(::AbstractArray{AbstractFloat}) = IsDeterministic()
StochasticStyle(::AbstractArray{T}) where {T} = default_style(T)

"""
    default_style(::Type)

Pick a [`StochasticStyle`](@ref) based on the value type. Throws an error if no known default
style is known.
"""
default_style(::Type{<:Integer}) = IsStochasticInteger()
default_style(::Type{<:AbstractFloat}) = IsDeterministic()
default_style(::Type{<:Complex{<:Integer}}) = IsStochastic2Pop()
default_style(::Type{T}) where T = StyleUnknown{T}()
