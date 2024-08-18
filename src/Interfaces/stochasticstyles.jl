"""
    StochasticStyle(v)

Abstract type. When called as a function it returns the native style of the
generalised vector `v` that determines how simulations are to proceed.

# Usage

Concrete `StochasticStyle`s can be used for the `style` keyword argument of
[`ProjectorMonteCarloProblem`](@ref Main.ProjectorMonteCarloProblem),
[`DVec`](@ref Main.DictVectors.DVec) and
[`PDVec`](@ref Main.DictVectors.PDVec). The following styles are available:

* [`IsStochasticInteger`](@ref Main.StochasticStyles.IsStochasticInteger)
* [`IsDeterministic`](@ref Main.StochasticStyles.IsDeterministic)
* [`IsStochasticWithThreshold`](@ref Main.StochasticStyles.IsStochasticWithThreshold)
* [`IsDynamicSemistochastic`](@ref Main.StochasticStyles.IsDynamicSemistochastic)
* [`StyleUnknown`](@ref Main.StochasticStyles.StyleUnknown)

# Extended Help
## Interface

When defining a new `StochasticStyle`, subtype it as `MyStyle<:StochasticStyle{T}` where `T`
is the concrete value type the style is designed to work with.

For it to work with [`ProjectorMonteCarloProblem`](@ref Main.ProjectorMonteCarloProblem), a `StochasticStyle` must define the
following:

* [`apply_column!(::StochasticStyle, w, H, address, value)`](@ref)
* [`step_stats(::StochasticStyle)`](@ref)

and optionally

* [`CompressionStrategy(::StochasticStyle)`](@ref) for vector compression after
  annihilations,

See also [`StochasticStyles`](@ref Main.StochasticStyles), [`Interfaces`](@ref).
"""
abstract type StochasticStyle{T} end

StochasticStyle(::AbstractVector{T}) where T = default_style(T)

Base.eltype(::Type{<:StochasticStyle{T}}) where {T} = T
VectorInterface.scalartype(::Type{<:StochasticStyle{T}}) where {T} = T

"""
    StyleUnknown{T}() <: StochasticStyle

Trait for value types not (currently) compatible with FCIQMC. This style makes it possible
to construct dict vectors with unsupported `valtype`s.

See also [`StochasticStyle`](@ref).
"""
struct StyleUnknown{T} <: StochasticStyle{T} end

"""
    default_style(::Type)

Pick a [`StochasticStyle`](@ref) based on the value type. Returns [`StyleUnknown`](@ref) if
no known default style is set.
"""
default_style(::Type{T}) where T = StyleUnknown{T}()

"""
    CompressionStrategy

The `CompressionStrategy` controls how a vector is compressed after a step.

## Default implementation:

* [`NoCompression`](@ref): no vector compression

## Usage

A subtype of `CompressionStrategy` can be passed as a keyword argument to the constructors
for some [`StochasticStyle`](@ref)s. Calling `CompressionStrategy(s::StochasticStyle)`
returns a relevant subtype. The default is [`NoCompression`](@ref).

## Interface

When defining a new `CompressionStrategy`, subtype it as `MyCompressionStrategy <:
CompressionStrategy` and define these methods:

* [`compress!(s::CompressionStrategy, v)`](@ref compress!)
* [`compress!(s::CompressionStrategy, w, v)`](@ref compress!)
* [`step_stats(s::CompressionStrategy)`](@ref step_stats)
"""
abstract type CompressionStrategy end

"""
    NoCompression <: CompressionStrategy end

Default [`CompressionStrategy`](@ref). Leaves the vector intact.
"""
struct NoCompression <: CompressionStrategy end

CompressionStrategy(::StochasticStyle) = NoCompression()
CompressionStrategy(v) = CompressionStrategy(StochasticStyle(v))

"""
    compress!([::CompressionStrategy,] v) -> ::NTuple{N,::Symbol}, ::NTuple{N}
    compress!([::CompressionStrategy,] w, v) -> ::NTuple{N,::Symbol}, ::NTuple{N}

Compress the vector `v`. The one-argument version compresses the vector in-place. The
two-argument vector stores the result in `w`. The [`CompressionStrategy`](@ref) associated
with the [`StochasticStyle`](@ref) of `v` is used to determine the type of compression.

Returns two tuples, containing the names and values of statistics that are to be reported.
"""
compress!(v) = compress!(CompressionStrategy(StochasticStyle(v)), v)
compress!(w, v) = compress!(CompressionStrategy(StochasticStyle(v)), w, v)

step_stats(::NoCompression) = (), ()

compress!(::NoCompression, v) = ()
function compress!(::NoCompression, w, v)
    for (add, val) in pairs(v)
        w[add] = val
    end
    return ()
end

"""
    step_stats(::StochasticStyle)
    step_stats(::CompressionStrategy)

Return a tuple of stat names (`Symbol` or `String`) and a tuple of zeros of the same
length. These will be reported as columns in the `DataFrame` returned by
[`ProjectorMonteCarloProblem`](@ref Main.ProjectorMonteCarloProblem).
"""
step_stats(v) = step_stats(StochasticStyle(v))

"""
    apply_column!(v, op, addr, num, boost=1) -> stats::Tuple

Apply the product of column `addr` of the operator `op` and the scalar `num` to the
vector `v` according to the [`StochasticStyle`](@ref) of `v`. By expectation value this
should be equivalent to

```
v .+= op[:, add] .* num
```

This is used to perform the spawning step in FCIQMC and to implement operator-vector
multiplications. Mutates `v` and reports spawning statistics.

The `boost` argument multiplicatively increases the number of spawns to be performed without
affecting the expectation value of the procedure.
"""
function apply_column!(v, ham, add, val, boost=1)
    return apply_column!(StochasticStyle(v), v, ham, add, val, boost)
end
