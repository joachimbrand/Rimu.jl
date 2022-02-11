"""
    StochasticStyle(v)

Abstract type. When called as a function it returns the native style of the
generalised vector `v` that determines how simulations are to proceed.

# Implemented styles

* [`IsStochasticInteger`](@ref) - integer walker FCIQMC
* [`IsDeterministic`](@ref) - perform deterministic variant of power method
* [`IsStochasticWithThreshold`](@ref) - floating point walker FCIQMC
* [`IsDynamicSemistochastic`](@ref) - floating point walker FCIQMC with smarter spawning and
  vector compression
* [`IsStochastic2Pop`](@ref)
* [`IsExplosive`](@ref)

# Usage

Concrete `StochasticStyle`s can be used for the `style` keyword argument of
[`lomc!`](@ref) and [`DVec`](@ref).

# Interface

When defining a new `StochasticStyle`, subtype it as `MyStyle<:StochasticStyle{T}` where `T`
is the concrete value type the style is designed to work with.

For it to work with [`lomc!`](@ref), a `StochasticStyle` must define the following:

* [`fciqmc_col!(::StochasticStyle, w, H, address, value, shift, dτ)`](@ref)
* [`step_stats(::StochasticStyle)`](@ref)

and optionally
* [`CompressionStrategy(::StochasticStyle)`](@ref) for vector compression after annihilations,
* [`update_dvec!`](@ref) for arbitrary transformations after the spawning step.

See also [`StochasticStyles`](@ref), [`Interfaces`](@ref).
"""
abstract type StochasticStyle{T} end

StochasticStyle(::AbstractVector{T}) where T = default_style(T)

Base.eltype(::Type{<:StochasticStyle{T}}) where {T} = T

"""
    StyleUnknown{T}() <: StochasticStyle

Trait for value types not (currently) compatible with FCIQMC. This style makes it possible to
construct dict vectors with unsupported `valtype`s.

See also [`StochasticStyle`](@ref).
"""
struct StyleUnknown{T} <: StochasticStyle{T} end

"""
    default_style(::Type)

Pick a [`StochasticStyle`](@ref) based on the value type. Throws an error if no known default
style is known.
"""
default_style(::Type{T}) where T = StyleUnknown{T}()

"""
    CompressionStrategy

The `CompressionStrategy` controls how a vector is compressed after a step.

## Standard implementations (subtypes):
* [`NoCompression`](@ref): no vector compression
* [`ThresholdCompression`](@ref): compress elements below a threshold

## Usage
A subtype of `CompressionStrategy` can be passed as a keyword argument to the
constructors for some [`StochasticStyle`](@ref)s. Calling
`CompressionStrategy(s::StochasticStyle)` returns a relevant subtype. The
default is [`NoCompression`](@ref).

## Interface
When defining a new `CompressionStrategy`, subtype it as
`MyCompressionStrategy <: CompressionStrategy` and define
a method for
* [`compress!(s::MyCompressionStrategy, v, hamiltonian, shift, dτ)`](@ref)
"""
abstract type CompressionStrategy end

"""
    NoCompression <: CompressionStrategy end

Default [`CompressionStrategy`](@ref). Leaves the vector intact.
"""
struct NoCompression <: CompressionStrategy end

CompressionStrategy(::StochasticStyle) = NoCompression()

"""
    compress!(::CompressionStrategy, v, hamiltonian, shift, dτ)

Compress the vector `v` and return it. The additional arguments `hamiltonian` and `shift`
are passed through to be used by some of the [`CompressionStrategy`](@ref)s.
"""
compress!(::NoCompression, v, _, _, _) = v

"""
    update_dvec!([::StochasticStyle,] dvec, hamiltonian, shift, dτ) -> dvec, nt

Perform an arbitrary transformation on `dvec` after the spawning step is completed and
report statistics to the `DataFrame`.

Returns the new `dvec` and a `NamedTuple` `nt` of statistics to be reported.

When extending this function for a custom [`StochasticStyle`](@ref), define a method
for the five-argument call signature!

The default implementation uses [`CompressionStrategy`](@ref) to compress the vector.

Note: `update_dvec!` may return a new vector.
"""
update_dvec!(v, args...) = update_dvec!(StochasticStyle(v), v, args...)
function update_dvec!(s::StochasticStyle, v, args...)
    return update_dvec!(CompressionStrategy(s), v, args...)
end
update_dvec!(::NoCompression, v, args...) = v, NamedTuple()
function update_dvec!(c::CompressionStrategy, v, args...)
    len_before = length(v)
    return compress!(c, v, args...), (; len_before)
end

"""
    step_stats(::StochasticStyle)

Return a tuple of names (`Symbol` or `String`) and a tuple of zeros of values of the same
length. These will be reported as columns in the `DataFrame` returned by [`lomc!`](@ref).
"""
step_stats(v, n) = step_stats(StochasticStyle(v), n)
function step_stats(s::StochasticStyle, ::Val{N}) where N
    if N == 1
        return step_stats(s)
    else
        names, stats = step_stats(s)
        return names, MVector(ntuple(_ -> stats, Val(N)))
    end
end

"""
    fciqmc_col!(w, ham, add, num, shift, dτ)
    fciqmc_col!(::StochasticStyle, args...)

Spawning and diagonal step of FCIQMC for single column of `ham`. In essence it computes

`w .+= (1 .+ dτ.*(shift .- ham[:,add])).*num`.

The [`StochasticStyle(w)`](@ref), picks the algorithm used.
"""
function fciqmc_col!(w, ham, add, num, shift, dτ)
    return fciqmc_col!(StochasticStyle(w), w, ham, add, num, shift, dτ)
end
