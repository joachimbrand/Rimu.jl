"""
    StochasticStyle(v)

Abstract type. When called as a function it returns the native style of the
generalised vector `v` that determines how simulations are to proceed.

# Usage

Concrete `StochasticStyle`s can be used for the `style` keyword argument of
[`lomc!`](@ref Main.lomc!) and [`DVec`](@ref Main.DictVectors.DVec).

# Interface

When defining a new `StochasticStyle`, subtype it as `MyStyle<:StochasticStyle{T}` where `T`
is the concrete value type the style is designed to work with.

For it to work with [`lomc!`](@ref Main.lomc!), a `StochasticStyle` must define the
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

* [`compress!(s::MyCompressionStrategy, v)`](@ref compress!)
* [`compress!(s::MyCompressionStrategy, w, v)`](@ref compress!)
"""
abstract type CompressionStrategy end

"""
    NoCompression <: CompressionStrategy end

Default [`CompressionStrategy`](@ref). Leaves the vector intact.
"""
struct NoCompression <: CompressionStrategy end

CompressionStrategy(::StochasticStyle) = NoCompression()

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

compress!(::NoCompression, v) = (), ()
function compress!(::NoCompression, w, v)
    copy!(w, v)
    return (), ()
end

"""
    step_stats(::StochasticStyle)

Return a tuple of names (`Symbol` or `String`) and a tuple of zeros of values of the same
length. These will be reported as columns in the `DataFrame` returned by [`lomc!`](@ref Main.lomc!).
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
