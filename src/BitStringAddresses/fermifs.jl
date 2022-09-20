"""
    FermiFS{N,M,S} <: SingleComponentFockAddress

Address type that represents a Fock state of `N` fermions of the same spin in `M` modes
by wrapping a bitstring of type `S <: BitString`.

# Constructors

* `FermiFS{N,M}(bs::BitString)`: Unsafe constructor. Does not check whether the number of
  particles in `bs` is equal to `N`.

* `FermiFS(::BitString)`: Automatically determine `N` and `M`. This constructor is not type
  stable!

* `FermiFS{[N,M,S]}(onr)`: Create `FermiFS{N,M}` from [`onr`](@ref) representation. This is
  efficient as long as at least `N` is provided.

See also: [`SingleComponentFockAddress`](@ref), [`BoseFS`](@ref), [`BitString`](@ref).
"""
struct FermiFS{N,M,S<:BitString{M}} <: SingleComponentFockAddress{N,M}
    bs::S

    FermiFS{N,M,S}(bs::S) where {N,M,S<:BitString{M}} = new{N,M,S}(bs)
end

function FermiFS{N,M,S}(onr::Union{SVector{M},NTuple{M}}) where {N,M,C,T,S<:BitString{M,C,T}}
    @boundscheck sum(onr) == N && all(in((0, 1)), onr) || error("Invalid ONR: may only contain 0s and 1s.")
    result = zero(SVector{C,T})
    for mode in 1:M
        iszero(onr[mode]) && continue
        minus_j, offset = fldmod(mode - 1, 64)
        j = C - minus_j
        new = result[j] | T(1) << T(offset)
        result = setindex(result, new, j)
    end
    return FermiFS{N,M,S}(S(SVector(result)))
end

function FermiFS{N,M}(onr::Union{AbstractVector,Tuple}) where {N,M}
    S = typeof(BitString{M}(0))
    return FermiFS{N,M,S}(SVector{M}(onr))
end
function FermiFS{N}(onr::Union{SVector{M},Tuple{M}}) where {N,M}
    return FermiFS{N,M}(onr)
end
function FermiFS(onr::Union{AbstractVector,Tuple})
    M = length(onr)
    N = sum(onr)
    return FermiFS{N,M}(onr)
end

function print_address(io::IO, f::FermiFS{N,M}) where {N,M}
    if get(io, :compact, false)
        print(io, "|", join(map(o -> o == 0 ? '⋅' : '↑', onr(f))), "⟩")
    else
        print(io, "FermiFS{$N,$M}(", tuple(onr(f)...), ")")
    end
end

Base.bitstring(a::FermiFS) = bitstring(a.bs)
Base.isless(a::FermiFS, b::FermiFS) = isless(a.bs, b.bs)
Base.hash(a::FermiFS,  h::UInt) = hash(a.bs, h)
Base.:(==)(a::FermiFS, b::FermiFS) = a.bs == b.bs
num_occupied_modes(::FermiFS{N}) where {N} = N

function near_uniform(::Type{FermiFS{N,M}}) where {N,M}
    return FermiFS([fill(1, N); fill(0, M - N)])
end

onr(a::FermiFS) = SVector(m_onr(a))

find_mode(f::FermiFS, i) = FermiFSIndex(Int(is_occupied(f, i)), i, i-1)
find_mode(f::FermiFS, is::Tuple) = map(i -> find_mode(f, i), is)

occupied_modes(a::FermiFS{N,<:Any,S}) where {N,S} = FermiOccupiedModes{N,S}(a.bs)

Base.length(::FermiOccupiedModes{N}) where {N} = N
Base.eltype(::FermiOccupiedModes) = FermiFSIndex

function Base.iterate(o::FermiOccupiedModes)
    c = 0
    chunk = o.bs.chunks[end]
    while iszero(chunk)
        c += 1
        chunk = o.bs.chunks[end - c]
    end
    zeros = trailing_zeros(chunk % Int)
    return iterate(o, (chunk >> (zeros % UInt64), c * 64 + zeros, c))
end
function Base.iterate(o::FermiOccupiedModes, st)
    chunk, index, c = st
    while iszero(chunk)
        c += 1
        c == num_chunks(o.bs) && return nothing
        chunk = o.bs.chunks[end - c]
        index = c * 64
    end
    zeros = trailing_zeros(chunk % Int)
    index += zeros
    chunk >>= zeros
    return FermiFSIndex(1, index + 1, index), (chunk >> 1, index + 1, c)
end

function Base.iterate(o::FermiOccupiedModes{<:Any,<:BitString{<:Any,1,T}}) where {T}
    chunk = o.bs.chunks[end]
    zeros = trailing_zeros(chunk % Int)
    return iterate(o, (chunk >> (zeros % T), zeros))
end
function Base.iterate(o::FermiOccupiedModes{<:Any,<:BitString{<:Any,1,T}}, st) where {T}
    chunk, index = st
    iszero(chunk) && return nothing
    chunk >>= 0x1
    index += 1
    zeros = trailing_zeros(chunk % Int)
    return FermiFSIndex(1, index, index - 1), (chunk >> (zeros % T), index + zeros)
end

function find_occupied_mode(a::FermiFS, i::Integer)
    for k in occupied_modes(a)
        i -= 1
        i == 0 && return k
    end
    return FermiFSIndex(0, 0, 0)
end

@inline function m_onr(a::FermiFS{<:Any,M}) where {M}
    result = zero(MVector{M,Int32})
    @inbounds for (_, mode) in occupied_modes(a)
        result[mode] = 1
    end
    return result
end

function Base.reverse(f::FermiFS)
    return typeof(f)(bitreverse(f.bs))
end

function is_occupied(a::FermiFS{<:Any,M,S}, mode) where {M,T,S<:BitString{<:Any,1,T}}
    @boundscheck 1 ≤ mode ≤ M || throw(BoundsError(a, mode))
    return a.bs.chunks[1] & (T(1) << (mode - 1) % T) > 0
end
function is_occupied(a::FermiFS{<:Any,M}, mode) where {M}
    @boundscheck 1 ≤ mode ≤ M || throw(BoundsError(a, mode))
    j, i = fldmod1(mode, 64)
    return a.bs.chunks[end + 1 - j] & (UInt(1) << UInt(i - 1)) > 0
end

"""
    _flip_and_count(bs::BitString, k)

Count the number of ones before the `k`-th mode, flip the `k`th bit. Return the new
bitstring, the count, and the value of the bit after the flip.
"""
@inline function _flip_and_count(bs::BitString{<:Any,1,T}, k::Unsigned) where {T}
    chunk = bs.chunks[1]
    # highlights the k-th bit
    kmask = one(T) << k

    count = count_ones((kmask - 0x1) & chunk)
    chunk = chunk ⊻ kmask
    val = chunk & kmask > 0
    return typeof(bs)(chunk), count, val
end
@inline function _flip_and_count(bs::BitString, k::Unsigned)
    j, i = fldmod(k % Int, UInt(64))
    j = length(bs.chunks) - j
    chunk = bs.chunks[j]

    kmask = one(UInt64) << i

    count = count_ones((kmask - 0x1) & chunk)
    chunk = chunk ⊻ kmask
    val = chunk & kmask > 0

    for k in j + 1:num_chunks(bs)
        count += count_ones(bs.chunks[k])
    end
    return typeof(bs)(setindex(bs.chunks, chunk, j)), count, val
end

function excitation(a::FermiFS, creations::NTuple{N}, destructions::NTuple{N}) where {N}
    bs = a.bs
    count = 0
    for i in N:-1:1
        d = destructions[i].mode
        bs, x, val = _flip_and_count(bs, UInt(d - 0x1))
        val && return a, 0.0
        count += x
    end
    for i in N:-1:1
        c = creations[i].mode
        bs, x, val = _flip_and_count(bs, UInt(c - 0x1))
        !val && return a, 0.0
        count += x
    end
    return typeof(a)(bs), ifelse(iseven(count), 1.0, -1.0)
end
