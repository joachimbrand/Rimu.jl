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
- `iterate()`: should return values of type `V`
- `pairs()`: should return an iterator over `key::K => content` pairs. If `content ≠ value::V` then provide `values()` iterator as well!
"""
abstract type AbstractDVec{K,V} end

"""
    DictVectors.capacity(dv::AbstractDVec, [s = :effective])
    capacity(dvs::Tuple, [s = :effective])
gives the effective holding capacity of `dv`. If a tuple of `dvs` is given it
aggregates the capacities.

Optional argument `s`:
- `:effective`  the number of elements that
can be stored savely (default)
- `:allocated` actual internal memory allocation
"""
capacity
# doc string here only; needs to be defined for each concrete type.

# make capacity aggregate over tuple
capacity(tup::Tuple, args...) = sum(capacity.(tup, args...))

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

"""
    pairtype(dv)
Returns the type of stored data, as returned by the `pairs()` iterator.
"""
pairtype(dv) = pairtype(typeof(dv))
pairtype(::AbstractDVec{K,V}) where {K,V} = Pair{K,V} # need this for each concrete type

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
    return isempty(x) ? 0.0 : mapreduce(p->abs2(p), +, x)
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
    return isempty(x) ? 0.0 : mapreduce(p->abs(p), +, x)|>Float64
end

"""
    normInf(x::AbstractDVec)
Infinity norm: largest absolute value of entries.
"""
normInf(x::AbstractDVec) = isempty(x) ? 0.0 : mapreduce(p->abs(p), max, x)|>Float64

"""
    norm(x::AbstractDVec, p)
Computes the p-norm of the DVec x. Implemented for `p ∈ {1, 2, Inf}`.
Returns zero if `x` is empty.
"""
function LinearAlgebra.norm(x::AbstractDVec, p::Real)
    if p === 2
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
    for (key, val) in kvpairs(v)
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
@inline function LinearAlgebra.mul!(w::AbstractDVec, v::AbstractDVec, α::Number)
    @boundscheck length(v) ≤ capacity(w) || throw(BoundsError()) # prevent overflow
    empty!(w) # since the values are not ordered, just forget about old ones
    for (key, val) in kvpairs(v)
        w[key] = val*α
    end
    return w
end # mul!


# copying (save) multiplication with scalar
# For compatibility with KrylovKit v0.5:
function *(α::N,dv::AbstractDVec{K,V}) where {K,V,N<:Number}
    w = similar(dv, promote_type(N,V))
    @inbounds mul!(w,dv,α)
    return w
end

*(dv::AbstractDVec, α::Number) =  α*dv


"""
    add!(x::AbstractDVec,y::AbstactDVec)
Inplace add `x+y` and store result in `x`.
"""
@inline function add!(x::AbstractDVec{K,V1},y::AbstractDVec{K,V2}) where {K,V1,V2}
    for (k,v) in kvpairs(y)
        x[k] += v
    end
    return x
end

# BLAS-like function: y .+= α*x
"""
    axpy!(α::Number, X::AbstractDVec, Y::AbstractDVec)
    axpy!(α::Number, X::AbstractDVec, Ys::Tuple, batchsize)
Overwrite `Y` with `α*X + Y` where `α` is scalar for `AbstractDVec`s.
If a tuple `Ys` is passed with `Threads.nthreads()` `AbstractDVec`s, then
perform the operation in parallel over threads with `batchsize` elements at a
time.
"""
@inline function LinearAlgebra.axpy!(α::Number,x::AbstractDVec,y::AbstractDVec)
    for (k,v) in kvpairs(x)
        y[k] += α*v
    end
    return y
end
# multithreaded version
@inline function LinearAlgebra.axpy!(α::Number,x::AbstractDVec,ys::NTuple{NT,W};
        batchsize = batchsize = max(20, min(length(x)÷NT, round(Int,sqrt(length(x))*10)))
    ) where {NT, W<:AbstractDVec}
    @boundscheck @assert NT == Threads.nthreads()
    if length(x) < NT*20 # just use main thread and copy into ys[1]
        axpy!(α, x, ys[1])
        return ys
    end
    return threaded_axpy!(α, x, ys, batchsize)
end
# unsafe version: we know that multithreaded is applicable
@inline function threaded_axpy!(α, x, ys, batchsize)
    @sync for btr in Iterators.partition(pairs(x), batchsize)
        Threads.@spawn for (k,v) in btr
            y = ys[Threads.threadid()]
            y[k] += α*v
        end
    end # all threads have returned; now running on single thread again
    return ys
end
# NOTE: the multithreaded version is allocating memory of quite considerable
# size, apparently due to the `Iterators.partition()` iterator.

# generic multiply with scalar inplace - this is slow (360 times slower than
# the fast version for FastDVec)
function LinearAlgebra.rmul!(w::AbstractDVec, α::Number)
    for (k,v) in kvpairs(w)
        w[k] = v*α
    end
    return w
end # rmul!

# BLAS-like function: y = α*x + β*y
function LinearAlgebra.axpby!(α::Number, x::AbstractDVec, β::Number, y::AbstractDVec)
    rmul!(y,β) # multiply every non-zero element
    axpy!(α, x, y)
    return y
end

# function LinearAlgebra.dot(x::AbstractDVec{K,V}, y::AbstractDVec{K,V}) where {K,V}
#     result = zero(V) # identical value types
#     if length(x) < length(y) # try to save time by looking for the smaller vec
#         for (key, val) in kvpairs(x)
#             result += conj(val)*y[key]
#         end
#     else
#         for (key, val) in y
#             result += conj(x[key])*val
#         end
#     end
#     return result # same type as valtype(x) - could be complex!
# end

function LinearAlgebra.dot(x::AbstractDVec{K,T1}, y::AbstractDVec{K,T2}) where {K,T1, T2}
    # for mixed value types
    result = zero(promote_type(T1,T2))
    if length(x) < length(y) # try to save time by looking for the smaller vec
        for (key, val) in kvpairs(x)
            result += conj(val)*y[key]
        end
    else
        for (key, val) in kvpairs(y)
            result += conj(x[key])*val
        end
    end
    return result # the type is promote_type(T1,T2) - could be complex!
end
# For MPI version see mpi_helpers.jl

# threaded dot()
function LinearAlgebra.dot(x::AbstractDVec{K,T1}, ys::NTuple{N, AbstractDVec{K,T2}}) where {N, K, T1, T2}
    results = zeros(promote_type(T1,T2), N)
    Threads.@threads for i in 1:N
        results[i] = x⋅ys[i]
    end
    return sum(results)
end
# function myspawndot(x::AbstractDVec{K,T1}, ys::NTuple{N, AbstractDVec{K,T2}}) where {N, K, T1, T2}
#     results = zeros(promote_type(T1,T2), N)
#     @sync for i in 1:N
#         Threads.@spawn results[i] = x⋅ys[i] # using dynamic scheduler
#     end
#     return sum(results)
# end
# # This version with `Threads.@spawn` was slightly slower (651 μs vs 640 μs)
# and needed more memory allocations (3.14 KiB vs 2.86 KiB) in an example
# compared to the `Threads.@threads` version implemented above.
# With 4 threads we got a speedup of 3.5 compared to single threaded sum(map(...))
# for DVecs with ≈ 23_000 entries.


## some methods below that we could inherit from AbstracDict with subtyping

"""
    isequal(l::AbstractDVec, r::AbstractDVec)
Returns `true` if all non-zero entries have the same value. Equality of
flags is not tested unless both `l` and `r` support flags.
"""
function isequal(l::AbstractDVec, r::AbstractDVec)
    l === r && return true
    if length(l) != length(r) return false end
    for (lk,lv) in kvpairs(l)
        if !isequal(r[lk],lv)
            return false
        end
    end
    true
end

==(l::AbstractDVec, r::AbstractDVec) = isequal(l,r)

# Iterators for `keys()` and `pairs()`
struct ADVKeysIterator{DV}
    dv::DV
end
Base.keys(dv::AbstractDVec) = ADVKeysIterator(dv)
function Base.iterate(ki::ADVKeysIterator, oldstate...)
    it = iterate(pairs(ki.dv), oldstate...)
    it === nothing && return nothing
    pair, state = it
    return (pair[1],state)
end
Base.length(ki::ADVKeysIterator) = length(ki.dv)
Base.eltype(::Type{ADVKeysIterator{DV}}) where DV = keytype(DV)
Base.IteratorSize(::Type{ADVKeysIterator}) = HasLength()

# iterator over pairs
"""
    ADVPairsIterator
Iterator type for pairs from a [`AbstractDVec`](@ref).
"""
struct ADVPairsIterator{DV}
    dv::DV
end
Base.length(ki::ADVPairsIterator) = length(ki.dv)
Base.eltype(::Type{ADVPairsIterator{DV}}) where DV = Pair{keytype(DV),valtype(DV)}
Base.IteratorSize(::Type{ADVPairsIterator}) = HasLength()

Base.pairs(dv::AbstractDVec) = ADVPairsIterator(dv)

"""
    kvpairs(collection)
Return an iterator over `key => value` pairs ignoring any flags. If no flags
are present, eg. for generic `AbstractDVec`, this falls back to
[`Base.pairs`](@ref).
"""
kvpairs(v) = pairs(v)

# iteration over values is default
Base.values(dv::AbstractDVec) = dv

# struct ADVValuesIterator{DV}
#     dv::DV
# end
# Base.values(dv::AbstractDVec) = ADVValuesIterator(dv)
# Base.length(ki::ADVValuesIterator) = length(ki.dv)
# Base.eltype(::Type{ADVValuesIterator{DV}}) where DV = valtype(DV)
# Base.IteratorSize(::Type{ADVValuesIterator}) = HasLength()
#
# # fallback method for value iteration - from pairs
# # This will not always work or not be the fastest way
# @inline function Base.iterate(ki::ADVValuesIterator, oldstate...)
#     it = iterate(pairs(ki.dv), oldstate...)
#     it == nothing && return nothing
#     pair, state = it
#     @inbounds return (pair[2],state)
# end
#
# Base.iterate(dv::AbstractDVec) = iterate(values(dv))
# Base.iterate(dv::AbstractDVec, state) = iterate(values(dv), state)

# struct UniformProjector{K,V} <: AbstractDVec{K,V} end
# UniformProjector(::Type{AbstractDVec{K,V}}) where {K,V} = UniformProjector{K,V}()
# UniformProjector(::DV) where DV <: AbstractDVec = UniformProjector(DV)
#
# struct NormProjector{K,V} <: AbstractDVec{K,V} end
# NormProjector(::Type{AbstractDVec{K,V}}) where {K,V} = NormProjector{K,V}()
# NormProjector(::DV) where DV <: AbstractDVec = NormProjector(DV)
#
# function LinearAlgebra.dot(x::NormProjector{K,T1}, y::AbstractDVec{K,T2}) where {K,T1, T2}
#     # dot returns the promote_type of the arguments.
#     # NOTE that this can be different from the return type of norm()->Float64
#     return convert(promote_type(T1,T2),norm(y,1))
# end
#
# function LinearAlgebra.dot(x::UniformProjector{K,T1}, y::AbstractDVec{K,T2}) where {K,T1, T2}
#     # dot returns the promote_type of the arguments.
#     # NOTE that this can be different from the return type of norm()->Float64
#     return convert(promote_type(T1,T2), sum(values(y)))
# end


# Define this type union for local (non-MPI) data
DVecOrVec = Union{AbstractDVec,AbstractVector}

"""
    UniformProjector()
Represents a vector with all elements 1. To be used with [`dot()`](@ref).
Minimizes memory allocations.

```julia
UniformProjector()⋅v == sum(v)
dot(UniformProjector(), LO, v) == sum(LO*v)
```

See also [`ReportingStrategy`](@ref) for use
of projectors in FCIQMC.
"""
struct UniformProjector end

LinearAlgebra.dot(::UniformProjector, y::DVecOrVec) = sum(y)
# a specialised fast and non-allocating method for
# `dot(::UniformProjector, A::AbstractHamiltonian, y)` is defined in `Hamiltonians.jl`

"""
    NormProjector()
Results in computing the one-norm when used in `dot()`. E.g.
```julia
dot(NormProjector(),x)
-> norm(x,1) # with type valtype(x)
```
`NormProjector()` thus represents the vector `sign.(x)`.

See also [`ReportingStrategy`](@ref) for use
of projectors in FCIQMC.
"""
struct NormProjector end

LinearAlgebra.dot(::NormProjector, y::DVecOrVec) = convert(valtype(y),norm(y,1))
# dot returns the promote_type of the arguments.
# NOTE that this can be different from the return type of norm()->Float64
# NOTE: This operation should work for `MPIData` and is MPI synchronizing

"""
    Norm2Projector()
Results in computing the two-norm when used in `dot()`. E.g.
```julia
dot(NormProjector(),x)
-> norm(x,2) # with type Float64
```

See also [`ReportingStrategy`](@ref) for use
of projectors in FCIQMC.
"""
struct Norm2Projector end

LinearAlgebra.dot(::Norm2Projector, y::DVecOrVec) = norm(y,2)
# NOTE that this returns a `Float64` opposite to the convention for
# dot to return the promote_type of the arguments.
