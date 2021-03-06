# create data structure that encapsulates the way juliamc is storing walkers
"""
    DictVectors.FastDVec{K,V}(capacity)
Create a dictionary-like array indexed by keys of type `K` and values of type
`V` with a maximum capacity of `capacity`.

    FastDVec(a::AbstractArray, [capacity = length(a)])
    FastDVec(d::AbstractDict{K,V}, [capacity = length(d)])
    FastDVec(d::AbstractDVec{K,V}, [capacity = length(d)])
Construct a `FastDVec` object from an existing array or dictionary.

    FastDVec(args...; capacity)
If the keyword argument `capacity` is passed then args are parsed as for `Dict`.
"""
mutable struct FastDVec{K,V} <: AbstractDVec{K,V}
    # V = value, e.g. a walker type like W3
    # K = key, e.g. an address type like BSAdd64
    vals::FastBuf{V} # WS_ to store walkers
    emptyslots::FastBuf{Int} # EMPTY_WS_ to remember empty slots
    capacity::Int # maximum number of entries
    hashrange::Int # size of hashtable; must be power of 2!
    hashtable::Array{Array{Tuple{K,Int},1},1} # HASH_T_WS_
end

# The default constructor initialises an empty FastDVec of the correct
# type with the requested capacity
function FastDVec{K,V}(capacity::Int) where V <: Number where K
    capacity =  capacity < 16 ? 16 : capacity::Int # minimum size
    hashrange = 2^ceil(Int,log2(0.7*capacity)) # power of 2 > 0.7*capacity
    hashtable = [Array{Tuple{K,Int}}(undef,0) for i = 1 : hashrange]
    vals = FastBuf{V}(capacity)
    emptyslots = FastBuf{Int}(capacity)
    FastDVec{K,V}(vals, emptyslots, capacity, hashrange, hashtable)
end

function FastDVec(p::Pair{K,V}; capacity) where {K,V}
    fdv = FastDVec{K,V}(capacity)
    fdv[p.first] = p.second
    return fdv
end

FastDVec(args...; capacity) = FastDVec(Dict(args...), capacity)

# convenience constructors
function FastDVec(a::AbstractVector, capacity = length(a))
    fdv = FastDVec{Int,eltype(a)}(capacity)
    for i = 1:length(a)
        fdv[i] = a[i]
    end
    return fdv
end

function FastDVec(d::AbstractDict{K,V},
                  capacity = length(d)) where K where V
    fdv = FastDVec{K,V}(capacity)
    for (k,v) in d
        fdv[k] = v
    end
    return fdv
end
function FastDVec(d::AbstractDVec{K,V}, # inherit capacity by default
                  capacity = capacity(d)) where K where V
    fdv = FastDVec{K,V}(capacity)
    for (k,v) in pairs(d)
        fdv[k] = v
    end
    return fdv
end

## Note: This can be used for copying:
# nv = FastDVec(ov)
## creates a new object in new memory with the same elements. Use
# nv = FastDVec(ov,ov.capacity)
## for a copy that preserves the capacity.

capacity(da::FastDVec) = da.capacity # not exported but useful here
function capacity(dv::FastDVec, s::Symbol)
    if s ==:allocated || s == :effective
        return capacity(dv)
    else
        ArgumentError("Option symbol $s not recognized")
    end
end

Base.length(da::FastDVec) = length(da.vals) - length(da.emptyslots)
Base.size(da::FastDVec) = (length(da),)
## eltype() is already provided by AbstractDVec
# Base.eltype(::Type{FastDVec{K,V}}) where {K,V} = V

function Base.similar(fdv::FastDVec{K,V}, ::Type{T}) where K where V where T
    hashtable = [Array{Tuple{K,Int}}(undef,0) for i = 1 : fdv.hashrange]
    # need to construct new hashtable
    FastDVec(similar(fdv.vals, T),similar(fdv.emptyslots),
             fdv.capacity,fdv.hashrange,hashtable)
end
Base.similar(fdv::FastDVec{K,V}) where K where V = similar(fdv, V)

Base.empty(fdv::FastDVec) = similar(fdv)
Base.empty(fdv::FastDVec, T) = similar(fdv, T)
Base.zero(fdv::FastDVec) = similar(fdv)

function Base.empty!(da::FastDVec)
    empty!(da.vals)
    empty!(da.emptyslots)
    @inbounds for i = 1: length(da.hashtable)
        empty!(da.hashtable[i])
    end
    return da
end

Base.isempty(da::FastDVec) = length(da) == 0


hashindex(key, hashrange) = (((hash(key)%Int) & (hashrange-1)) + 1)::Int
# uses built-in hash() function to produce a hash value of type Int that
# is smaller than hashrange.
# Note: this requires that hashrange is a power of 2. The function does not
# check this explicitly for speed reasons.

function Base.haskey(da::FastDVec, key)
    hind = hashindex(key, da.hashrange)
    for (add, vind) in da.hashtable[hind]
        if add == key # compare key with entry in hashtable
            return true # entry found
        end
    end # unsuccessful if we get here
    return false # key not found
end

Base.getindex(da::FastDVec{K,V}, key::K) where {K,V} = get(da, key, zero(V))

@inline function Base.get(da::FastDVec{K,V}, key::K, default::V) where K where V
    hind = hashindex(key, da.hashrange)
    @inbounds for (add, vind) in da.hashtable[hind]
        if add == key # compare key with entry in hashtable
            return da.vals[vind] # entry found, return value
        end
    end # unsuccessful if we get here
    return default # key not found
end


@inline function Base.setindex!(da::FastDVec{K,V}, v::V, key::K) where K where V
    if v == zero(V)
        delete!(da, key)
    else
        hind = hashindex(key, da.hashrange) # index into hashtable
        @inbounds for (add, vind) in da.hashtable[hind] # look through existing entries
            if add == key # compare key with entry in hashtable
                da.vals[vind] = v # reset existing value
                return  da # success, we are done
            end
        end # key does not yet exist
        if isempty(da.emptyslots) # add to end of vals
            push!(da.vals,v)
            vind = length(da.vals)
        else # fill an empty slot
            vind = pop!(da.emptyslots) # next empty slot
            da.vals[vind] = v
        end #still need to update hashtable
        push!(da.hashtable[hind], (key,vind))
    end
    return da
end # setindex!

function Base.delete!(da::FastDVec{K,V}, key::K) where K where V
    hind = hashindex(key, da.hashrange) # index into hashtable
    @inbounds for (ind, (add, vind)) in enumerate(da.hashtable[hind]) # look through existing entries
        if add == key # compare key with entry in hashtable
            push!(da.emptyslots, vind) # remember empty slot
            deleteat!(da.hashtable[hind], ind)
            return  da # success, we are done
        end
    end # key does not yet exist
    return da
    # error("No entry with key $key in $da.")
end


# # Iteration over all vals.
# # Note: this includes empty slots!
# @inline function val_iterate(da::FastDVec, i = 1)
#     if i == length(da.vals) + 1
#         nothing
#     else
#         return (da.vals[i], i+1)
#     end
# end

# Iterators
# non-exportet internal functions called by the iterators
@inline function _pair_iterate(dv::FastDVec, state = (1,1))
    hind, pos = state
    @boundscheck hind ≤ dv.hashrange || throw(BoundsError())
    list = dv.hashtable[hind]
    @inbounds while pos > length(list)
        hind += 1
        pos = 1
        if hind > dv.hashrange
            return nothing
        end
        list = dv.hashtable[hind]
    end
    @inbounds key, vind = list[pos]
    @inbounds val = dv.vals[vind]
    state = (hind,pos + 1)
    return (Pair(key,val), state)
end

# iteration over the FastDVec returns values
@inline function Base.iterate(fdv::FastDVec, state...)
    it = _pair_iterate(fdv,state...)
    it === nothing && return nothing
    pair, state = it
    return pair[2], state # value
end

function Base.show(io::IO, da::FastDVec{K,V}) where V where K
    print(io, "FastDVec{$K,$V}([")
    init = true
    for p in pairs(da)
        if init
            init = false
        else
            print(io, ", ")
        end
        print(io, p)
    end
    print(io, "])\n")
end

## the following code seems to work but is not really faster than the generic one
## generic code moved to `abstractdvec.jl`
# function Base.copyto!(w::FastDVec{K,V},v::FastDVec{K,V}) where K where V
#     if w.capacity == v.capacity
#         copyto!(w.vals,v.vals)
#         copyto!(w.emptyslots, v.emptyslots)
#         for i = 1: length(v.hashtable)
#             w.hashtable[i] = copy(v.hashtable[i])
#         end
#         return w
#     elseif length(v) > capacity(w)
#         error("Not enough capacity to copy `FastDVec` with `copyto!()`.")
#     end
#     empty!(w) # since the values are not ordered, just forget about old ones
#     for (key, val) in v
#         w[key] = val
#     end
#     return w
# end # copyto!

# multiply with scalar inplace - this is very fast
function LinearAlgebra.rmul!(w::FastDVec, α::Number)
    rmul!(w.vals,α)
    return w
end # mul!

# Iterators for `keys()` and `pairs()`
struct FDVKeysIterator{DV}
    dv::DV
end
Base.keys(dv::AbstractDVec) = FDVKeysIterator(dv)
function Base.iterate(ki::FDVKeysIterator, oldstate...)
    it = iterate(pairs(ki.dv), oldstate...)
    it === nothing && return nothing
    pair, state = it
    return (pair[1],state)
end
Base.length(ki::FDVKeysIterator) = length(ki.dv)
Base.eltype(::Type{FDVKeysIterator{DV}}) where DV = keytype(DV)
Base.IteratorSize(::Type{FDVKeysIterator}) = HasLength()

@inline function Base.iterate(ki::FDVKeysIterator{DV}, state...) where DV<:FastDVec
    it = _pair_iterate(ki.dv,state...)
    it === nothing && return nothing
    pair, state = it
    return pair[1], state # key
end

# iterator over pairs
"""
    FDVPairsIterator
Iterator type for pairs from a [`AbstractDVec`](@ref).
"""
struct FDVPairsIterator{DV}
    dv::DV
end
Base.length(ki::FDVPairsIterator) = length(ki.dv)
Base.eltype(::Type{FDVPairsIterator{DV}}) where DV = Pair{keytype(DV),valtype(DV)}
Base.IteratorSize(::Type{FDVPairsIterator}) = HasLength()

Base.pairs(dv::AbstractDVec) = FDVPairsIterator(dv)

@inline function Base.iterate(pit::FDVPairsIterator, state...)
    return _pair_iterate(pit.dv, state...)
end
