"Provides the `FastBuf` data structure."
module FastBufs
# to implement a fast first-in-last out buffer with fixed capacity
# code is based on CircularDeque from DataStructures.jl

import Base: length, isempty, iterate, show, empty!, getindex, setindex!,
             push!, pop!, eltype, similar, copyto!

export FastBuf
# export capacity

"""
    FastBuf{T}(n)

Create a first-in-last-out buffer of maximum capacity `n`. The element type
is `T`. Fill a FastBuf with `push!()`, retrieve with `pop!()`. Accessing elements
with `getindex()` and `setindex!()` as well as iteration are also supported.
Check the capacity of a `FastBuf` with `capacity()`.
"""
mutable struct FastBuf{T} <: AbstractVector{T}
    buffer::Vector{T}
    capacity::Int
    last::Int
end

# default constructor
FastBuf{T}(n::Int) where {T} = FastBuf(Vector{T}(undef, n), n, 0)

# similar() makes an empty FastBuf with the same capacity and eltype
function Base.similar(fb::FastBuf, ::Type{T}) where T
    FastBuf(Vector{T}(undef, fb.capacity), fb.capacity, 0)
end
Base.similar(fb::FastBuf{T}) where T = similar(fb,T)

Base.length(D::FastBuf) = D.last
Base.size(D::FastBuf) = (D.last,)
Base.eltype(::Type{FastBuf{T}}) where {T} = T

"Gives the capacity of a `FastBuf` object."
capacity(D::FastBuf) = D.capacity

function Base.empty!(D::FastBuf)
    D.last = 0
    D
end

Base.isempty(D::FastBuf) = D.last == 0


@inline function back(D::FastBuf)
    @boundscheck D.last > 0 || throw(BoundsError())
    D.buffer[D.last]
end

@inline function Base.push!(D::FastBuf, v)
    @boundscheck D.last < D.capacity || throw(BoundsError()) # prevent overflow
    D.last += 1
    @inbounds D.buffer[D.last] = v
    D
end

@inline function Base.setindex!(D::FastBuf, v, i)
    @boundscheck 1 <= i <= D.last || throw(BoundsError())
    @inbounds D.buffer[i] = v
end

@inline function Base.pop!(D::FastBuf)
    v = back(D)
    D.last -= 1
    v
end

# getindex sans bounds checking
@inline function _unsafe_getindex(D::FastBuf, i::Integer)
    @inbounds D.buffer[i]
end

@inline function Base.getindex(D::FastBuf, i::Integer)
    @boundscheck 1 <= i <= D.last || throw(BoundsError())
    return _unsafe_getindex(D, i)
end

# Iteration via getindex
@inline function iterate(d::FastBuf, i = 1)
    i == d.last + 1 ? nothing : (_unsafe_getindex(d, i), i+1)
end

function Base.show(io::IO, D::FastBuf{T}) where T
    print(io, "FastBuf{$T}([")
    for i = 1:length(D)
        print(io, D[i])
        i < length(D) && print(io, ',')
    end
    print(io, "])")
end

# needed because behaviour is different from AbstractVector
function Base.copyto!(dest::FastBuf,ori::AbstractVector)
    dest.capacity >= length(ori) || error("Insufficient capacity") # prevent overflow
    for (i, val) in enumerate(ori) # using iteration as above
        dest.buffer[i] = val
    end
    dest.last =length(ori)
    return dest
end


end # module FastVec
