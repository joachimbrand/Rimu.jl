"""
    DFVec{K,V,F}(capacity) <: AbstractDVec{K,V <: Number}
    DFVec(d::Dict [, capacity])
    DFVec(v::Vector{V} [, capacity])
Construct a wrapped dictionary with minimum capacity `capacity` to
represent a vector-like object with `valtype(dv) == V`. The value of the
`Dict` are of type `Tuple{V,F}`, which allows for storing a flag of type `F`
for each entry. Indexing is done with an
arbitrary (in general non-integer) `keytype(dv) == K`.
When constructed from a `Vector{V}`,
the keys will be integers `∈ [0, length(v)]` and the flag `zero(UInt16)`.
See [`AbstractDVec`](@ref). The
method [`capacity()`](@ref) is defined but not a strict upper limit as `Dict`
objects can expand.
"""
struct DFVec{K,V,F} <: AbstractDVec{K,V}
    d::Dict{K,Tuple{V,F}}
end
# default constructor from `Dict` just wraps the dict, no copying or allocation:
# dv = DFVec(Dict(k => (t,f), ...))


function DFVec(dict::Dict{K,Tuple{V,F}}, capacity::Int) where {K, V <: Number, F}
    if capacity*3 ≥ length(dict.keys)*2
        # requested capacity ≥ 2/3 of allocated memory to avoid rehashing
        # be triggered by `setindex!`
        sizehint!(dict, (capacity*3)>>1+1)
        # ensures memory allocation > 3/2 rquested capacity
    end # does nothing if dict is large enough (shrinking is not implemented)
    return DFVec(dict)
end

# by specifying keytype, eltype, flag type and capacity
function DFVec{K,V,F}(capacity::Int) where {K, V <: Number, F}
    return DFVec(Dict{K,Tuple{V,F}}(), capacity)
end

# from Vector
function DFVec(t::Vector{V}, capacity = length(t), F = UInt16) where V <: Number
    vs = map(x->tuple(x,zero(F)),t) # all flags are set to zero
    ps = map(Pair,keys(t),vs) # create iterator over pairs
    return DFVec(Dict(ps), capacity)
end

# from AbstractDVec or AbstractDict; note that a new dict is constructed and data is copied
@inline function DFVec(d::Union{AbstractDict{K,Tuple{V,F}},
                                AbstractDVec{K,Tuple{V,F}}},
              capacity = length(d)) where {K,V <: Number,F}
    @boundscheck length(d) ≤ capacity || throw(BoundsError()) # prevent overflow
    dv = DFVec{K,V,F}(capacity)
    @inbounds return copyto!(dv, d)
end

# from AbstractDVec or AbstractDict
@inline function DFVec(d::Union{AbstractDVec{K,V}, AbstractDict{K,V}},
                       capacity = length(d), F = UInt16
                      ) where {K, V <: Number}
    @boundscheck length(d) ≤ capacity || throw(BoundsError()) # prevent overflow
    dv = DFVec{K,V,F}(capacity)
    for (key, val) in d
        dv[key] = (val, zero(F))
    end
    return dv
end

# from DFVec
@inline function DFVec(adv::DFVec{K,V,F}, capacity = capacity(adv), FN = F) where {K,V,F}
    @boundscheck length(adv) ≤ capacity || throw(BoundsError()) # prevent overflow
    dv = DFVec{K,V,FN}(capacity) # allocate new DFVec
    @inbounds return copyto!(dv,adv) # need specific form
end

@inline function Base.copyto!(w::DFVec{K,V,F}, v::DFVec{K1,V1,F1}) where {K,V,F,K1,V1,F1}
    @boundscheck length(v) ≤ capacity(w) || throw(BoundsError()) # prevent overflow
    @assert K == promote_type(K,K1) && V == promote_type(V,V1) && F == promote_type(F,F1)
    for (k, (v, f)) in pairs(v)
        w[K(k)] = tuple(V(v),F(f))
    end
    return w
end

# the following also create and allocated new DFVec objects
Base.empty(dv::DFVec{K,V,F}) where {K,V,F} = DFVec{K,V,F}(capacity(dv))
Base.empty(dv::DFVec, ::Type{V}) where {V} = empty(dv,keytype(dv),V,flagtype(dv))
function Base.empty(dv::DFVec, ::Type{K}, ::Type{V}, ::Type{F}) where {K,V,F}
    return DFVec{K,V,F}(capacity(dv))
end

# thus empty() and zero() will conserve the capacity of dv
Base.zero(dv::DFVec) = empty(dv)
Base.similar(dv::DFVec, args...) = empty(dv, args...)
# Base.similar(dv::DFVec, ::Type{V}) where {V} = empty(dv, V)
# Base.similar(dv::DFVec) = empty(dv)

flagtype(::Type{DFVec{K,V,F}}) where {K,V,F} = F
flagtype(dv) = flagtype(typeof(dv))

Base.keytype(::Type{DFVec{K,V,F}}) where {K,V,F} = K
Base.valtype(::Type{DFVec{K,V,F}}) where {K,V,F} = V
Base.eltype(::Type{DFVec{K,V,F}}) where {K,V,F} = V
# for instances of AbstractDVec, eltype is already defined
# for the type we need to do it here because it has to be specific

capacity(dv::DFVec) = (2*length(dv.d.keys))÷3
# 2/3 of the allocated storage size
# if number of elements grows larger, Dict will start rehashing and allocating
# new memory
function capacity(dv::DFVec, s::Symbol)
    if s ==:allocated
        return length(dv.d.keys)
    elseif s == :effective
        return capacity(dv)
    else
        ArgumentError("Option symbol $s not recognized")
    end
end

# getindex returns the default value without adding it to dict
Base.getindex(dv::DFVec, key) = get(dv, key, zero(eltype(dv)))

"""
    get(dv::DFVec, key, deftup::Tuple)
Return tuple containing value and flag if `key` exists and otherwise `deftup`.
"""
Base.get(dv::DFVec, key, deftup::Tuple) = get(dv.d, key, deftup)

"""
    get(dv::DFVec, key, defnum::Number)
Return value only if `key` exists and otherwise `defnum`
"""
function Base.get(dv::DFVec{K,V,F}, key, defnum::Number) where {K,V,F}
    return get(dv.d, key, (zero(F),defnum))[1] # only return number
end

"""
    gettuple(dv::DFVec, key)
Retrieve the tuple `(t,f)` stored at the given `key`. If the key is not
found, return a tuple of zeros.
"""
function gettuple(dv::DFVec{K,V,F}, key) where {K,V,F}
    return get(dv, key, tuple(zero(F),zero(V)))
end

# Iterators
# iterator over pairs
Base.pairs(dv::DFVec) = dv.d # just return the contained dictionary

# """
#     TuplePairIterator
# Iterator type returned by [`tuplepairs()`](@ref).
# """
# struct TuplePairIterator{K,V,F}
#     dv::DFVec{K,V,F}
# end
# Base.length(ki::TuplePairIterator) = length(ki.dv)
# Base.eltype(::Type{TuplePairIterator{K,V,F}}) where {K,V,F} = Pair{K,Tuple{V,F}}
# Base.IteratorSize(::Type{TuplePairIterator}) = HasLength()
#
# """
#     tuplepairs(dv::DFVec)
# An iterator that yields `k => (v, f)` where `k` is a key and `(v, f)` is the
# corresponding tuple with value `v` and flag `f` stored in the [`DFVec`](@ref)
# `dv`.
# """
# function tuplepairs(dv::DFVec)
#     return TuplePairIterator(dv)
# end
# Base.iterate(ti::TuplePairIterator) = iterate(ti.dv.d)
# Base.iterate(ti::TuplePairIterator, state) = iterate(ti.dv.d, state)

# """
#     TupleIterator
# Iterator type returned by [`tuples()`](@ref).
# """
# struct TupleIterator{K,V,F}
#     dv::DFVec{K,V,F}
# end
# Base.length(ki::TupleIterator) = length(ki.dv)
# Base.eltype(::Type{TupleIterator{K,V,F}}) where {K,V,F} = Tuple{V,F}
# Base.IteratorSize(::Type{TupleIterator}) = HasLength()

"""
    tuples(dv::DFVec)
An iterator that yields tuples `(v, f)` with value `v` and flag `f`
stored in the [`DFVec`](@ref) `dv`.
"""
tuples(dv::DFVec) = values(dv.d) # just the values of the dict
# function tuples(dv::DFVec)
#     return TupleIterator(dv)
# end
# function Base.iterate(ti::TupleIterator)
#     ps = iterate(ti.dv.d)
#     ps == nothing && return nothing
#     pair, state = ps
#     return pair[2], state
# end
# function Base.iterate(ti::TupleIterator, oldstate)
#     ps = iterate(ti.dv.d, oldstate)
#     ps == nothing && return nothing
#     pair, state = ps
#     return pair[2], state
# end

"""
    FlagsIterator
Iterator type returned by [`flags()`](@ref).
"""
struct FlagsIterator{K,V,F}
    dv::DFVec{K,V,F}
end
Base.length(ki::FlagsIterator) = length(ki.dv)
Base.eltype(::Type{FlagsIterator{K,V,F}}) where {K,V,F} = F
Base.IteratorSize(::Type{FlagsIterator}) = HasLength()

"""
    flags(dv::DFVec)
An iterator that yields flags stored in the [`DFVec`](@ref) `dv`.
"""
function flags(dv::DFVec)
    return FlagsIterator(dv)
end
@inline function Base.iterate(fi::FlagsIterator)
    ps = iterate(fi.dv.d)
    ps == nothing && return nothing
    pair, state = ps
    @inbounds return pair[2][2], state
end
@inline function Base.iterate(fi::FlagsIterator, oldstate)
    ps = iterate(fi.dv.d, oldstate)
    ps == nothing && return nothing
    pair, state = ps
    @inbounds return pair[2][2], state
end

"""
    KVPairsIterator
Iterator type returned by [`kvpairs()`](@ref).
"""
struct KVPairsIterator{K,V,F}
    dv::DFVec{K,V,F}
end
Base.length(ki::KVPairsIterator) = length(ki.dv)
Base.eltype(::Type{KVPairsIterator{K,V,F}}) where {K,V,F} = Pair{K,V}
Base.IteratorSize(::Type{KVPairsIterator}) = HasLength()

"""
    kvpairs(dv::DFVec)
An iterator that yields `key => value` pairs stored in the [`DFVec`](@ref) `dv`
ignoring any `flags`. In contrast, [`pairs()`](@ref) will return pairs
`key => (value, flag)`.
"""
function kvpairs(dv::DFVec)
    return KVPairsIterator(dv)
end
@inline function Base.iterate(kvi::KVPairsIterator)
    ps = iterate(kvi.dv.d)
    ps == nothing && return nothing
    pair, state = ps
    @inbounds return Pair(pair[1], pair[2][1]), state
end
@inline function Base.iterate(kvi::KVPairsIterator, oldstate)
    ps = iterate(kvi.dv.d, oldstate)
    ps == nothing && return nothing
    pair, state = ps
    @inbounds return Pair(pair[1], pair[2][1]), state
end

# Note that standard iteration over a `DFVec` will return `key => val` pairs
# as required for the `AbstractDVec` interface without the `flag` information.
# Use `tuples()` or `tuplepairs()` in order to access the full information.
@inline function Base.iterate(dv::DFVec)
    ps = iterate(dv.d)
    ps == nothing && return nothing
    pair, state = ps
    @inbounds return Pair(pair[1], pair[2][1]), state
end
@inline function Base.iterate(dv::DFVec, oldstate)
    ps =  iterate(dv.d, oldstate)
    ps == nothing && return nothing
    pair, state = ps
    @inbounds return Pair(pair[1], pair[2][1]), state
end
Base.IteratorSize(::Type{DFVec}) = HasLength()


# most functions are simply delegated to the wrapped dictionary
@delegate DFVec.d [get!, haskey, getkey, pop!, isempty, length, keys]

# Some functions are delegated, but then need to return the main dictionary
# NOTE: push! is not included below, because the fallback version just
#       calls setindex!
@delegate_return_parent DFVec.d [delete!, empty!, sizehint!]

function Base.setindex!(dv::DFVec{K,V,F}, v::V, key::K) where {K, V<:Number, F}
    if v == zero(V)
        delete!(dv, key)
    else
        setindex!(dv.d, (v, zero(F)), key)
    end
    return dv
end

function Base.setindex!(dv::DFVec{K,V,F}, v::Tuple{V,F}, key::K) where {K,V,F}
    if v == (zero(V), zero(F)) # delete only if value and flag are both zero
        delete!(dv, key)
    else
        setindex!(dv.d, v, key)
    end
    return dv
end

# should be much faster than generic version from AbstractDVec
function LinearAlgebra.rmul!(w::DFVec, α::Number)
    rmul!(w.d.vals,α)
    return w
end # rmul!

function Base.show(io::IO, da::DFVec{K,V,F}) where {K,V,F}
    print(io, "DFVec{$K,$V,$F} with $(length(da)) entries and capacity $(capacity(da)):")
    for (i,p) in enumerate(pairs(da))
        print(io, "\n  ", p)
        if i>15
            print(io, "\n  ⋮   => ⋮")
            break
        end
    end
end

"""
    isequal(l::DFVec, r::DFVec)
Returns `true` if all non-zero entries have the same value and the same flag.
"""
function isequal(l::DFVec, r::DFVec)
    l === r && return true
    if length(l) != length(r) return false end
    for (lk,lv) in pairs(l)
        if !isequal(gettuple(r,lk),lv)
            return false
        end
    end
    true
end
