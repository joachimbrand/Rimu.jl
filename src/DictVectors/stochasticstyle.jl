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
    IsStochasticInteger{T=Int}() <: StochasticStyle{T}
Trait for generalised vector of configurations indicating stochastic propagation as seen in
the original FCIQMC algorithm.

See also [`StochasticStyle`](@ref).
"""
struct IsStochasticInteger{T<:Integer} <: StochasticStyle{T} end
IsStochasticInteger() = IsStochasticInteger{Int}()

"""
    IsStochastic2Pop{T=Complex{Int}}() <: StochasticStyle{T}
Trait for generalised vector of configurations indicating stochastic propagation with
complex walker numbers representing two populations of integer walkers.

See also [`StochasticStyle`](@ref).
"""
struct IsStochastic2Pop{T<:Complex{<:Integer}} <: StochasticStyle{T} end
IsStochastic2Pop() = IsStochastic2Pop{Complex{Int}}()

"""
    IsDeterministic{T=Float64}() <: StochasticStyle{T}
Trait for generalised vector of configuration indicating deterministic propagation of walkers.
The optional [`compression`](@ref) argument can set a [`CompressionStrategy`](@ref).

See also [`StochasticStyle`](@ref).
"""
struct IsDeterministic{T<:AbstractFloat,C<:CompressionStrategy} <: StochasticStyle{T}
    compression::C
end
IsDeterministic{T}(; compression::C=NoCompression()) where {T,C} = IsDeterministic{T,C}(compression)
IsDeterministic(; kwargs...) = IsDeterministic{Float64}(; kwargs...)

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
IsStochasticWithThreshold(t=1.0) = IsStochasticWithThreshold{typeof(float(t))}(float(t))

"""
    IsDynamicSemistochastic{T=Float64}(rel_threshold=1, abs_threshold=Inf, proj_threshold=1) <: StochasticStyle{T}

TODO: rework this text.

QMC propagation with non-integer walker numbers and reduced noise. All possible spawns are performed
deterministically when number of walkers in a configuration is high. Stochastic vector compression with
threshold `proj_threshold` is applied after spawning and diagonal death steps.

Unlike with [`IsStochasticWithThreshold`](@ref), when `late_projection` is set to `true`,
walker annihilation is done before the stochastic vector compression.

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
struct IsDynamicSemistochastic{T<:AbstractFloat,C<:CompressionStrategy} <: StochasticStyle{T}
    strength::T
    compression::C
end
function IsDynamicSemistochastic{T}(
    ; proj_threshold=1.0, strength=1.0, compression::C=ThresholdCompression(proj_threshold),
) where {T,C}
    return IsDynamicSemistochastic{T,C}(strength, compression)
end
IsDynamicSemistochastic(; kwargs...) = IsDynamicSemistochastic{Float64}(; kwargs...)

struct IsDynamicSemistochasticProjectedSpawns{T<:AbstractFloat} <: StochasticStyle{T}
    strength::T
    threshold::T
end
function IsDynamicSemistochasticProjectedSpawns{T}(; strength=1, threshold=1) where {T}
    return IsDynamicSemistochasticProjectedSpawns(strength, threshold)
end
function IsDynamicSemistochasticProjectedSpawns(; kwargs...)
    return IsDynamicSemistochasticProjectedSpawns{Float64}(kwargs...)
end

struct IsExplosive{T<:AbstractFloat,C<:CompressionStrategy} <: StochasticStyle{T}
    splatter_factor::T
    explosion_threshold::T
    compression::C
    delay_factor::T
end
function IsExplosive{T}(
    ;
    splatter_factor=one(T),
    explosion_threshold=one(T),
    proj_threshold=one(T),
    compression=ThresholdCompression(proj_threshold),
    delay_factor=one(T),
) where {T}
    return IsExplosive(
        T(splatter_factor), T(explosion_threshold), T(proj_threshold), T(delay_factor),
    )
end
IsExplosive(; kwargs...) = IsExplosive{Float64}(; kwargs...)

struct IsDynamicSemistochasticPlus{T<:AbstractFloat} <: StochasticStyle{T}
    target_len_before::Int
    target_len_after::Int
    threshold::T
    strength::T
end
function IsDynamicSemistochasticPlus{T}(target_len_before, target_len_after) where T
    return IsDynamicSemistochasticPlus{T}(target_len_before, target_len_after, one(T), one(T))
end
function IsDynamicSemistochasticPlus(target_len_before, target_len_after)
    return IsDynamicSemistochasticPlus{Float64}(target_len_before, target_len_after)
end

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
