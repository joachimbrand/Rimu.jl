"""
    StochasticStyle(v)
Abstract type. When called as a function it returns the native style of the
generalised vector `v` that determines how simulations are to proceed.

# Implemented styles
* [`IsStochasticInteger`](@ref) - integer walker FCIQMC
* [`IsDeterministic`](@ref) - perform deterministic variant of power method
* [`IsStochasticWithThreshold`](@ref) - floating point walker FCIQMC
* [`IsDynamicSemistochastic`](@ref)

# Usage
Concrete `StochasticStyle`s can be used for the `style` keyword argument of
[`lomc!`](@ref) and [`DVec`](@ref).

# Interface

When defining a new `StochasticStyle`, subtype it as `MyStyle<:StochasticStyle{T}` where `T`
is the concrete value type the style is designed to work with.

For it to work with [`lomc!`](@ref), a `StochasticStyle` must define the following:

* [`fciqmc_col!(::StochasticStyle, w, H, address, value, shift, dτ)`](@ref)
* [`step_stats(::StochasticStyle)`](@ref)

Optionally, it can also define [`update_dvec!`](@ref), which can be used to perform arbitrary
transformations on the generalised vector after the spawning step is complete.
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
    default_style(::Type)

Pick a [`StochasticStyle`](@ref) based on the value type. Throws an error if no known default
style is known.
"""
default_style(::Type{T}) where T = StyleUnknown{T}()

StochasticStyle(::AbstractVector{T}) where T = default_style(T)
