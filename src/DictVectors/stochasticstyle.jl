"""
    StochasticStyle(v)
`StochasticStyle` specifies the native style of the generalised vector `v` that determines
how simulations are to proceed. This can be fully stochastic (with `IsStochasticInteger`),
fully deterministic (with `IsDeterministic`), or stochastic with floating point walker
numbers and threshold (with [`IsStochasticWithThreshold`](@ref)).

When defining a new `StochasticStyle`, subtype it as `MyStyle<:StochasticStyle{T}` where `T`
is the concrete value type the style is designed to work with.

For it to work with FCIQMC, a `StochasticStyle` must define the following:

* [`fciqmc_col!(::StochasticStyle, w, H, address, value, shift, dτ)`](@ref)
* [`step_stats(::StochasticStyle)`](@ref)

Optionally, it can also define [`update_dvec!`](@ref), which can be used to perform arbitrary
transformations on the `dvec` after the spawning step is complete.
"""
abstract type StochasticStyle{T} end

Base.eltype(::Type{<:StochasticStyle{T}}) where {T} = T

"""
    StyleUnknown{T}() <: StochasticStyle
Trait for value types not (currently) compatible with FCIQMC. This style makes it possible to
construct dict vectors with unsupported `valtype`s.

See also [`StochasticStyle`](@ref).
"""
struct StyleUnknown{T} <: StochasticStyle{T} end

"""
    IsStochasticInteger() <: StochasticStyle
Trait for generalised vector of configurations indicating stochastic propagation as seen in
the original FCIQMC algorithm.

See also [`StochasticStyle`](@ref).
"""
struct IsStochasticInteger{T<:Integer} <: StochasticStyle{T} end
IsStochasticInteger() = IsStochasticInteger{Int}()

"""
    IsStochastic2Pop() <: StochasticStyle
Trait for generalised vector of configurations indicating stochastic propagation with
complex walker numbers representing two populations of integer walkers.

See also [`StochasticStyle`](@ref).
"""
struct IsStochastic2Pop{T<:Complex{<:Integer}} <: StochasticStyle{T} end
IsStochastic2Pop() = IsStochastic2Pop{Complex{Int}}()

"""
    IsDeterministic() <: StochasticStyle
Trait for generalised vector of configuration indicating deterministic propagation of walkers.

See also [`StochasticStyle`](@ref).
"""
struct IsDeterministic{T<:AbstractFloat} <: StochasticStyle{T} end
IsDeterministic() = IsDeterministic{Float64}()

"""
    IsStochasticWithThreshold(threshold=1.0) <: StochasticStyle
Trait for generalised vector of configurations indicating stochastic
propagation with real walker numbers and cutoff `threshold`.

During stochastic propagation, walker numbers small than `threshold` will be
stochastically projected to either zero or `threshold`.

See also [`StochasticStyle`](@ref).
"""
struct IsStochasticWithThreshold{T<:AbstractFloat} <: StochasticStyle{T}
    threshold::T
end
IsStochasticWithThreshold(t=1.0) = IsStochasticWithThreshold{Float64}(Float64(t))

"""
    IsDynamicSemistochastic(rel_threshold=1, abs_threshold=Inf, proj_threshold=1) <: StochasticStyle

Similar to [`IsStochasticWithThreshold`](@ref), but does exact spawning when the number of
walkers in a configuration is high.

Unlike with [`IsStochasticWithThreshold`](@ref), when `late_projection` is set to `true`,
walker annihilation is done before the stochastic projection.

## Parameters:

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

See also [`StochasticStyle`](@ref).
"""
struct IsDynamicSemistochastic{T<:AbstractFloat,P}<:StochasticStyle{T}
    rel_threshold::T
    abs_threshold::T
    proj_threshold::T
end
function IsDynamicSemistochastic{T}(
    ; late_projection::Bool=true, rel_threshold=1.0, abs_threshold=Inf, proj_threshold=1.0
) where {T}
    return IsDynamicSemistochastic{T,late_projection}(
        T(rel_threshold), T(abs_threshold), T(proj_threshold)
    )
end
IsDynamicSemistochastic(; kwargs...) = IsDynamicSemistochastic{Float64}(kwargs...)

# Defaults for arrays.
StochasticStyle(::AbstractArray{AbstractFloat}) = IsDeterministic()
StochasticStyle(::AbstractArray{T}) where {T} = default_style(T)

"""
    default_style(::Type)

Pick a [`StochasticStyle`](@ref) based on the value type. Throws an error if no known default
style is known.
"""
default_style(::Type{T}) where {T<:Integer} = IsStochasticInteger{T}()
default_style(::Type{T}) where {T<:AbstractFloat} = IsDeterministic{T}()
default_style(::Type{T}) where {T<:Complex{<:Integer}} = IsStochastic2Pop{T}()
default_style(::Type{T}) where T = StyleUnknown{T}()
