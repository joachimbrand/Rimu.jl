struct FermiFS{N,M,S<:BitString{M}} <: AbstractFockAddress
    bs::S

    FermiFS{N,M,S}(bs::S) where {N,M,S<:BitString{M}} = new{N,M,S}(bs)
end

function FermiFS{N,M,S}(onr::Union{SVector{M},NTuple{M}}) where {N,M,C,T,S<:BitString{M,C,T}}
    @boundscheck sum(onr) == N && all(in((0, 1)), onr) || error("invalid ONR")
    result = zero(SVector{C,T})
    for orbital in 1:M
        iszero(onr[orbital]) && continue
        minus_j, offset = fldmod(orbital - 1, 64)
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

Base.isless(a::FermiFS, b::FermiFS) = isless(a.bs, b.bs)
Base.hash(a::FermiFS,  h::UInt) = hash(a.bs, h)
Base.:(==)(a::FermiFS, b::FermiFS) = a.bs == b.bs
num_particles(::Type{FermiFS{N,M,S}}) where {N,M,S} = N
num_modes(::Type{FermiFS{N,M,S}}) where {N,M,S} = M
num_components(::Type{<:FermiFS}) = 1

onr(a::FermiFS) = SVector(m_onr(a))

@inline function m_onr(a::FermiFS{<:Any,M}) where {M}
    result = zero(MVector{M,Int32})
    j = num_chunks(a.bs)
    @inbounds for orbital in occupied_orbitals(a)
        result[orbital] = 1
    end
    return result
end

function is_occupied(a::FermiFS{<:Any,M}, orbital) where {M}
    @boundscheck 1 ≤ orbital ≤ M || throw(BoundsError(a, orbital))
    j, i = fldmod1(orbital, 64)
    return a.bs.chunks[end + 1 - j] & (UInt(1) << UInt(i - 1)) > 0
end

"""
    move_particle(a::FermiFS, from, to)

Move particle from location `from` to location `to`. Note: the check that `from` and `to` are
valid can be elided with `@inbounds`.
"""
function move_particle(a::FermiFS{<:Any,<:Any,S}, from, to) where {T,S<:BitString{<:Any,1,T}}
    new_chunk, value = _move_particle(a.bs.chunks[1], from, to)
    return typeof(a)(S(new_chunk)), value
end

function move_particle(a::FermiFS{<:Any,<:Any,S}, from, to) where {S}
    T = chunk_type(a.bs)
    # Ensure they are ordered.
    from, to = minmax(from, to)
    result = a.bs.chunks

    # Get chunk and offset.
    i, from_offset = divrem(from - 1, 64)
    j, to_offset = divrem(to - 1, 64)
    # Indexing from right -> make it from left
    i = lastindex(result) - i
    j = lastindex(result) - j

    if i == j
        new_chunk, value = _move_particle(result[i], from_offset + 1, to_offset + 1)
        result = setindex(result, new_chunk, i)
    else
        result = setindex(result, result[i] ⊻ T(1) << T(from_offset), i)
        result = setindex(result, result[j] ⊻ T(1) << T(to_offset), j)

        count = 0
        count += count_ones(result[i] & (-UInt(1) << (from_offset + 1)))
        count += count_ones(result[j] & ~(-UInt(1) << to_offset))
        for k in j+1:i-1
            count += count_ones(result[k])
        end
        value = ifelse(iseven(count), 1, -1)
    end
    return typeof(a)(S(result)), value
end

function _move_particle(bs::T, from, to) where {T<:Unsigned}
    # Masks that locate positions `from` and `to`.
    from_mask = T(1) << T(from - 1)
    to_mask = T(1) << T(to - 1)

    # Mask for counting how many particles lie between them.
    between_mask = T(2^(abs(from - to) - 1) - 1) << T(min(from, to))

    bs ⊻= from_mask | to_mask
    num_between = count_ones(bs & between_mask)
    return bs, ifelse(iseven(num_between), 1, -1)
end

struct FermiOccupiedOrbitals{N,S}
    bs::S
end

occupied_orbitals(a::FermiFS{N,<:Any,S}) where {N,S} = FermiOccupiedOrbitals{N,S}(a.bs)

function Base.iterate(o::FermiOccupiedOrbitals)
    c = 0
    chunk = o.bs.chunks[end]
    while iszero(chunk)
        c += 1
        chunk = o.bs.chunks[end - c]
    end
    zeros = trailing_zeros(chunk % Int)
    return iterate(o, (chunk >> (zeros % UInt64), c * 64 + zeros, c))
end
function Base.iterate(o::FermiOccupiedOrbitals, st)
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

function Base.iterate(o::FermiOccupiedOrbitals{<:Any,<:BitString{<:Any,1,T}}) where {T}
    chunk = o.bs.chunks[end]
    zeros = trailing_zeros(chunk % Int)
    return iterate(o, (chunk >> (zeros % T), zeros))
end
function Base.iterate(o::FermiOccupiedOrbitals{<:Any,<:BitString{<:Any,1,T}}, st) where {T}
    chunk, index = st
    iszero(chunk) && return nothing
    chunk >>= 0x1
    index += 1
    zeros = trailing_zeros(chunk % Int)
    return index, (chunk >> (zeros % T), index + zeros)
end
