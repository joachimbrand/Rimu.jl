struct FermiFS{N,M,S<:BitString{M}} <: AbstractFockAddress
    bs::S

    FermiFS{N,M,S}(bs::S) where {N,M,S<:BitString{M}} = new{N,M,S}(bs)
end

function FermiFS{N,M,S}(onr::Union{SVector{M},NTuple{M}}) where {N,M,C,T,S<:BitString{M,C,T}}
    @boundscheck sum(onr) == N && all(in((0, 1)), onr) || error("Invalid ONR: may only contain 0s and 1s.")
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

function near_uniform(::Type{FermiFS{N,M}}) where {N,M}
    return FermiFS([fill(1, N); fill(0, M - N)])
end

onr(a::FermiFS) = SVector(m_onr(a))

struct FermiOccupiedOrbitals{N,S}
    bs::S
end

"""
    occupied_orbitals(f::FermiFS)

Iterate over occupied orbitals in `FermiFS` address. Iterates values of type `Int`.

# Example

```jldoctest
julia> f = FermiFS((1,1,0,1,0,0,1))
julia> for i in occupied_orbitals(f)
    @show i
end
i = 1
i = 2
i = 4
i = 7
```
"""
occupied_orbitals(a::FermiFS{N,<:Any,S}) where {N,S} = FermiOccupiedOrbitals{N,S}(a.bs)

Base.length(o::FermiOccupiedOrbitals{N}) where {N} = N
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

function find_particle(a::FermiFS, i)
    for k in occupied_orbitals(a)
        i -= 1
        i == 0 && return k
    end
    return 0
end

@inline function m_onr(a::FermiFS{<:Any,M}) where {M}
    result = zero(MVector{M,Int32})
    j = num_chunks(a.bs)
    @inbounds for orbital in occupied_orbitals(a)
        result[orbital] = 1
    end
    return result
end

function is_occupied(a::FermiFS{<:Any,M,S}, orbital) where {M,T,S<:BitString{<:Any,1,T}}
    @boundscheck 1 ≤ orbital ≤ M || throw(BoundsError(a, orbital))
    return a.bs.chunks[1] & (T(1) << (i - 1) % T) > 0
end
function is_occupied(a::FermiFS{<:Any,M}, orbital) where {M}
    @boundscheck 1 ≤ orbital ≤ M || throw(BoundsError(a, orbital))
    j, i = fldmod1(orbital, 64)
    return a.bs.chunks[end + 1 - j] & (UInt(1) << UInt(i - 1)) > 0
end

"""
    move_particle(a::FermiFS, from, to)

Move particle from location `from` to location `to`.

Returns new address and the sign. If the move is not legal, return a sign of zero.
"""
function move_particle(a::FermiFS{<:Any,<:Any,S}, from, to) where {T,S<:BitString{<:Any,1,T}}
    from, to = minmax(from, to)
    new_chunk, value = _move_particle(a.bs.chunks[1], from % T, to % T)
    return typeof(a)(S(new_chunk)), value
end

function move_particle(a::FermiFS, from, to)
    return _move_particle(a, from % UInt64, to % UInt64)
end

# Note: the methods with underscores accept unsigned from/to and assume they are ordered.
# This allows us to not worry about converting types and swapping all the time.
@inline function _move_particle(chunk::T, from::T, to::T) where {T<:Unsigned}
    # Masks that locate positions `from` and `to`.
    from_mask = T(1) << (from - T(1))
    to_mask = T(1) << (to - T(1))

    if (from_mask & chunk > 0) == (to_mask & chunk > 0)
        # Illegal move
        return chunk, 0
    else
        # Mask for counting how many particles lie between them.
        between_mask = ((T(1) << (to - from - T(1))) - T(1)) << from

        chunk ⊻= from_mask | to_mask
        num_between = count_ones(chunk & between_mask)
        return chunk, ifelse(iseven(num_between), 1, -1)
    end
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
        if (result[i] & from_mask > 0) == (result[j] & to_mask > 0)
            # Illegal move
            return a, 0
        else
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
    end
    return typeof(a)(S(result)), value
end
