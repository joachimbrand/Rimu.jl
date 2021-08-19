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
struct FermiFS{N,M,S<:BitString{M}} <: SingleComponentFockAddress{M}
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

function Base.show(io::IO, f::FermiFS{N,M}) where {N,M}
    print(io, "FermiFS{$N,$M}(", tuple(onr(f)...), ")")
end

Base.bitstring(a::FermiFS) = bitstring(a.bs)
Base.isless(a::FermiFS, b::FermiFS) = isless(a.bs, b.bs)
Base.hash(a::FermiFS,  h::UInt) = hash(a.bs, h)
Base.:(==)(a::FermiFS, b::FermiFS) = a.bs == b.bs
num_particles(::Type{FermiFS{N,M,S}}) where {N,M,S} = N
num_modes(::Type{FermiFS{N,M,S}}) where {N,M,S} = M
num_components(::Type{<:FermiFS}) = 1
num_occupied_modes(::FermiFS{N}) where {N} = N
find_mode(::FermiFS, i) = i

function near_uniform(::Type{FermiFS{N,M}}) where {N,M}
    return FermiFS([fill(1, N); fill(0, M - N)])
end

onr(a::FermiFS) = SVector(m_onr(a))

"""
    FermiOccupiedModes

Iterator over occupied modes in address. See [`occupied_modes`](@ref).
"""
struct FermiOccupiedModes{N,S}
    bs::S
end

occupied_modes(a::FermiFS{N,<:Any,S}) where {N,S} = FermiOccupiedModes{N,S}(a.bs)

Base.length(::FermiOccupiedModes{N}) where {N} = N
Base.eltype(::FermiOccupiedModes) = Int

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
    return index + 1, (chunk >> 1, index + 1, c)
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
    return index, (chunk >> (zeros % T), index + zeros)
end

function find_occupied_mode(a::FermiFS, i)
    for k in occupied_modes(a)
        i -= 1
        i == 0 && return k
    end
    return 0
end

@inline function m_onr(a::FermiFS{<:Any,M}) where {M}
    result = zero(MVector{M,Int32})
    j = num_chunks(a.bs)
    @inbounds for mode in occupied_modes(a)
        result[mode] = 1
    end
    return result
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

function move_particle(a::FermiFS{<:Any,<:Any,S}, from, to) where {T,S<:BitString{<:Any,1,T}}
    if is_occupied(a, from) && !is_occupied(a, to)
        from, to = minmax(from, to)
        new_chunk, value = _move_particle(a.bs.chunks[1], from % T, to % T)
        return typeof(a)(S(new_chunk)), value
    else
        return a, ifelse(from==to, Int(is_occupied(a, from), 0)
    end
end

function move_particle(a::FermiFS, from, to)
    if is_occupied(a, from) && !is_occupied(a, to)
        return _move_particle(a, from % UInt64, to % UInt64)
    else
        return a,  ifelse(from==to, Int(is_occupied(a, from), 0)
    end
end

# Note: the methods with underscores accept unsigned from/to and assume they are ordered.
# This allows us to not worry about converting types and swapping all the time. It also
# assumes from and to are valid positions in the bitstring.
@inline function _move_particle(chunk::T, from::T, to::T) where {T<:Unsigned}
    # Masks that locate positions `from` and `to`.
    from_mask = T(1) << (from - T(1))
    to_mask = T(1) << (to - T(1))

    # Mask for counting how many particles lie between them.
    between_mask = ((T(1) << (to - from - T(1))) - T(1)) << from

    chunk ⊻= from_mask | to_mask
    num_between = count_ones(chunk & between_mask)
    return chunk, ifelse(iseven(num_between), 1, -1)
end
@inline function _move_particle(a::FermiFS{<:Any,<:Any,S}, from::UInt64, to::UInt64) where {S}
    # Ensure they are ordered.
    from, to = minmax(from, to)
    result = a.bs.chunks

    # Get chunk and offset.
    i, from_offset = divrem(from - 1, 64)
    j, to_offset = divrem(to - 1, 64)
    # Indexing from right -> make it from left
    i = UInt64(lastindex(result)) - i
    j = UInt64(lastindex(result)) - j

    if i == j
        new_chunk, value = _move_particle(
            result[i], from_offset + UInt64(1), to_offset + UInt64(1)
        )
        result = setindex(result, new_chunk, i % Int)
    else
        from_mask = UInt64(1) << from_offset
        to_mask = UInt64(1) << to_offset
        result = setindex(result, result[i] ⊻ from_mask, i % Int)
        result = setindex(result, result[j] ⊻ to_mask, j % Int)

        count = (
            count_ones(result[i] & (-UInt64(1) << (from_offset + UInt(1)))) +
            count_ones(result[j] & ~(-UInt64(1) << to_offset))
        )
        for k in j+1:i-1
            count += count_ones(result[k])
        end
        value = ifelse(iseven(count), 1, -1)
    end
    return typeof(a)(S(result)), value
end
