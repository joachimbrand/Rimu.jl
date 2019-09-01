"""
    DictVectors.AbstractDVec{K,V}
Abstract type for sparse vectors with `valtype()` `V` based on dictionary-like
structures.
They behave like vectors except that indexing is performed with an arbitrary
`keytype()` `K` (no order required). `getindex()` returns `zero(V)` for any
unknown key and
zeroed elements should be deleted with `delete!()`. Thus, iteration typically
returns only non-zero entries. In variance to `AbstractVector`, `length()` will
return the number of non-zero elements, while `capacity()` returns the holding
capacity (assumed fixed) of the object.
Even though `AbstractDVec` is similar to both `AbstractDict` and
`AbstractVector`, it is explicitly not subtyped to either.
The aim is to create data
structures suitable for FCIQMC and compatible with KrylovKit.jl.

### Interface
The interface is similar to the `AbstractDict` interface.
Implement what would be needed for the `AbstractDict` interface
(`setindex!, getindex, delete!, length,
haskey, empty!, isempty`) and, in addition:
- `capacity(dv)`: holding capacity
- `similar(dv [,Type])`
- `iterate()`: should return `Pair{K,V}`
"""
abstract type AbstractDVec{K,V} end

"""
    DictVectors.capacity(dv::AbstractDVec)
    capacity(dv, s = :effective)
gives the effective holding capacity of `dv`.

Optional argument `s`:
- `:effective`  the number of elements that
can be stored savely (default)
- `:allocated` actual internal memory allocation
"""
capacity
# doc string here only; needs to be defined for each concrete type.

# Note: The following method definitions for the supertype
# also extend (by a fallback method) to instances of the type.
# However, for concrete subtypes, the methods have to be defined again as the
# supertype methods do not apply (because the supertype and subtype are distinct
# types).
Base.valtype(::AbstractDVec{K,V}) where V where K = V
Base.eltype(::AbstractDVec{K,V}) where V where K = V
# conficts with definition and expected behaviour of AbstractDict
# but is needed for KrylovKit
Base.keytype(::AbstractDVec{K,V}) where V where K = K

Base.isreal(v::AbstractDVec) = valtype(v) <: Real
Base.ndims(::AbstractDVec) = 1

"""
    zero!(v::AbstractDVec)
Replace `v` by a zero vector as an inplace operation. For `AbstractDVec` types
it means removing all non-zero elements.
"""
zero!(v::AbstractDVec) = empty!(v)

"""
    norm_sqr(x::AbstractDVec)
Fast calculation of the square of the 2-norm of `x`.
"""
function norm_sqr(x::AbstractDVec{K,V}) where K where V<:Number
    return mapreduce(p->abs2(p[2]), +, x)
end

"""
    norm(x::AbstractDVec{K,V})
Computes the 2-norm of the DVec x.
"""
LinearAlgebra.norm(x::AbstractDVec) = sqrt(norm_sqr(x))

# # fastest
# """
#     norm2(x::DVec{K,V})
# Computes the 2-norm of the DVec x.
# """
# function norm2(x::DVec{K,V}) where K where V<:Number
#     return sqrt(mapreduce(p->abs2(p[2]), +, x))
# end
#
# function norm2alt3(x::DVec{K,V}) where K where V<:Real
#     return sqrt(mapreduce(p->p[2].^2, +, x))
# end

function norm1(x::AbstractDVec{K,V}) where K where V<:Number
    return mapreduce(p->abs(p[2]), +, x)|>Float64
end

"""
    normInf(x::AbstractDVec)
Infinity norm: largest absolute value of entries.
"""
normInf(x::AbstractDVec) = mapreduce(p->abs(p[2]), max, x)|>Float64

"""
    norm(x::AbstractDVec, p)
Computes the p-norm of the DVec x. Implemented for `p ∈ {1, 2, Inf}`.
Returns zero if `x` is empty.
"""
function LinearAlgebra.norm(x::AbstractDVec, p::Real)
    if length(x) == 0
        return 0.0 # return type is Float64
    elseif p === 2
        return norm(x)
    elseif p === 1
        return norm1(x)
    elseif p === Inf
        return normInf(x)
    else
        error("$p-norm of DVec of length $(length(x)) is not implemented.")
    end
end

@inline function Base.copy!(w::AbstractDVec, v)
    @boundscheck length(v) ≤ capacity(w) || throw(BoundsError()) # prevent overflow
    empty!(w) # since the values are not ordered, just forget about old ones
    @inbounds return copyto!(w, v)
end # copy!

@inline function Base.copyto!(w::AbstractDVec, v)
    @boundscheck length(v) ≤ capacity(w) || throw(BoundsError()) # prevent overflow
    for (key, val) in v
        w[key] = val
    end
    return w
end

function Base.copy(v::AbstractDVec)
    w = empty(v) # new adv of same type and same length
    @inbounds return copyto!(w, v)
end # copy


"""
    fill!(da::AbstractDVec, x)
Empties `da` if `x==zero(valtype(da))` and throws an error otherwise.
"""
function Base.fill!(da::AbstractDVec{K,V}, x::V) where V where K
    x == zero(V) || error("Trying to fill! $(typeof(da)) object with $x instead of $(zero(V))")
    return empty!(da) # remove all elements but keep capacity
end

# multiply with scalar and copy!
function LinearAlgebra.mul!(w::AbstractDVec, v::AbstractDVec, α::Number)
    if length(v) > capacity(w)
        error("Not enough capacity to copy to `AbstractDVec` with `mul!()`.")
    end
    empty!(w) # since the values are not ordered, just forget about old ones
    for (key, val) in v
        w[key] = val*α
    end
    return w
end # mul!

"""
    add!(x::AbstractDVec,y::AbstactDVec)
Inplace add `x+y` and store result in `x`.
"""
@inline function add!(x::AbstractDVec{K,V1},y::AbstractDVec{K,V2}) where {K,V1,V2}
    for (k,v) in y
        x[k] += v
    end
    return x
end

# BLAS-like function: y += α*x
@inline function LinearAlgebra.axpy!(α::Number,x::AbstractDVec,y::AbstractDVec)
    for (k,v) in x
        y[k] += α*v
    end
    return y
end

# generic multiply with scalar inplace - this is slow (360 times slower than
# the fast version for FastDVec)
function LinearAlgebra.rmul!(w::AbstractDVec, α::Number)
    for (k,v) in w
        w[k] = v*α
    end
    return w
end # rmul!

# copying (save) multiplication with scalar
*(w::AbstractDVec, α::Number) = rmul!(copy(w), α)

# BLAS-like function: y = α*x + β*y
function LinearAlgebra.axpby!(α::Number, x::AbstractDVec, β::Number, y::AbstractDVec)
    rmul!(y,β) # multiply every non-zero element
    axpy!(α, x, y)
    return y
end

function LinearAlgebra.dot(x::AbstractDVec{K,V}, y::AbstractDVec{K,V}) where {K,V}
    result = zero(V) # identical value types
    if length(x) < length(y) # try to save time by looking for the smaller vec
        for (key, val) in x
            result += conj(val)*y[key]
        end
    else
        for (key, val) in y
            result += conj(x[key])*val
        end
    end
    return result # same type as valtype(x) - could be complex!
end

function LinearAlgebra.dot(x::AbstractDVec{K,T1}, y::AbstractDVec{K,T2}) where {K,T1, T2}
    # for mixed value types
    result = zero(promote_type(T1,T2))
    if length(x) < length(y) # try to save time by looking for the smaller vec
        for (key, val) in x
            result += conj(val)*y[key]
        end
    else
        for (key, val) in y
            result += conj(x[key])*val
        end
    end
    return result # the type is promote_type(T1,T2) - could be complex!
end

## some methods below that we could inherit from AbstracDict with subtyping

"""
    isequal(l::AbstractDVec, r::AbstractDVec)
Returns `true` if all non-zero entries have the same value. Equality of
flags is not tested.
"""
function isequal(l::AbstractDVec, r::AbstractDVec)
    l === r && return true
    if length(l) != length(r) return false end
    for (lk,lv) in l
        if !isequal(r[lk],lv)
            return false
        end
    end
    true
end

==(l::AbstractDVec, r::AbstractDVec) = isequal(l,r)

# Iterators for `keys()` and `values()`, general fallback code for AbstractDVec
# Not sure this code is needed. It will be slower than defining methods
# for concrete types that can take account of the specific structure.
struct ADVKeysIterator{DV}
    dv::DV
end
Base.keys(dv::AbstractDVec) = ADVKeysIterator(dv)
function Base.iterate(ki::ADVKeysIterator)
    it = iterate(ki.dv)
    it == nothing && return nothing
    pair, state = it
    return (pair[1],state)
end
function Base.iterate(ki::ADVKeysIterator, oldstate)
    it = iterate(ki.dv, oldstate)
    it == nothing && return nothing
    pair, state = it
    return (pair[1],state)
end
Base.length(ki::ADVKeysIterator) = length(ki.dv)
Base.eltype(::Type{ADVKeysIterator{DV}}) where DV = keytype(DV)
Base.IteratorSize(::Type{ADVKeysIterator}) = HasLength()

struct ADVValuesIterator{DV}
    dv::DV
end
Base.values(dv::AbstractDVec) = ADVValuesIterator(dv)
function Base.iterate(ki::ADVValuesIterator)
    it = iterate(ki.dv)
    it == nothing && return nothing
    pair, state = it
    return (pair[2],state)
end
function Base.iterate(ki::ADVValuesIterator, oldstate)
    it = iterate(ki.dv, oldstate)
    it == nothing && return nothing
    pair, state = it
    return (pair[2],state)
end
Base.length(ki::ADVValuesIterator) = length(ki.dv)
Base.eltype(::Type{ADVValuesIterator{DV}}) where DV = valtype(DV)
Base.IteratorSize(::Type{ADVValuesIterator}) = HasLength()
