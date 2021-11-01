"""
    num_chunks(::Val{B})

Determine the number and type of chunks needed to store `B` bits.
"""
function num_chunks(::Val{B}) where {B}
    if B ≤ 0
        throw(ArgumentError("`B` must be positive!"))
    elseif B ≤ 8
        return 1, UInt8
    elseif B ≤ 16
        return 1, UInt16
    elseif B ≤ 32
        return 1, UInt32
    else
        return (B - 1) ÷ 64 + 1, UInt64
    end
end

"""
    check_bitstring_typeparams(::Val{B}, ::Val{N})

Check if number of bits `B` is consistent with number of chunks `N`. Throw an error if not.
"""
function check_bitstring_typeparams(::Val{B}, ::Val{N}, ::Type{UInt64}) where {B,N}
    if B > N * 64
        error("$B bits do not fit into $N 64-bit chunks")
    elseif B ≤ (N - 1) * 64
        error("$B bits fit into $(N - 1) 64-bit chunks, but $N chunks were provided")
    end
end
function check_bitstring_typeparams(::Val{B}, ::Val{1}, ::Type{T}) where {B,T}
    if B > sizeof(T) * 8
        error("$B bits do not fit into a $(sizeof(T) * 8)-bit chunk")
    end
end
function check_bitstring_typeparams(::Val{B}, ::Val{1}, ::Type{UInt64}) where {B}
    if B > 64
        error("$B bits do not fit into a 64-bit chunk")
    end
end
function check_bitstring_typeparams(::Val{B}, ::Val{N}, ::Type{T}) where {B,N,T}
    error("Only `UInt64` is supported for multi-bit chunks")
end

"""
    BitString{B,N,T<:Unsigned}

Type for storing bitstrings of static size. Holds `B` bits in `N` chunks, where each chunk is
of type `T`.

`N` is chosen automatically to accommodate `B` bits as efficiently as possible.

# Constructors

* `BitString{B,N,T}(::SVector{N,T})`: unsafe constructor. Does not check for ghost bits.

* `BitString{B,N,T}(i::T)`: as above, but sets `i` as the rightmost chunk.

* `BitString{B}(::Integer)`: Convert integer to `BitString`. Integer is truncated to the
  correct number of bits.

"""
struct BitString{B,N,T<:Unsigned}
    chunks::SVector{N,T}

    # This constructor is only to be used internally. It doesn't check for ghost bits.
    function BitString{B,N,T}(s::SVector{N,T}) where {B,N,T}
        check_bitstring_typeparams(Val(B), Val(N), T)
        return new{B,N,T}(s)
    end
    function BitString{B,N,T}(i::T) where {B,N,T<:Unsigned}
        check_bitstring_typeparams(Val(B), Val(N), T)
        return new{B,N,T}(setindex(zero(SVector{N,UInt64}), i, N))
    end
end

###
### Basic properties.
###
"""
    num_chunks(::Type{<:BitString})
    num_chunks(s::BitString)

Number of chunks in bitstring. Equivalent to `length(chunks(s))`.
"""
num_chunks(::Type{<:BitString{<:Any,N}}) where {N} = N

"""
    chunk_type(::Type{<:BitString})
    chunk_type(s::BitString)

Type of unsigned integer used to store the chunks.
"""
chunk_type(::Type{<:BitString{<:Any,<:Any,T}}) where {T} = T

"""
    num_bits(::Type{<:BitString})
    num_bits(s::BitString)

Total number of bits stored in bitstring.
"""
num_bits(::Type{<:BitString{B}}) where {B} = B

"""
    top_chunk_bits(::Type{<:BitString})
    top_chunk_bits(s::BitString)

Number of bits stored in top chunk. Equivalent to `chunk_bits(s, 1)`.
"""
function top_chunk_bits(::Type{<:BitString{B}}) where B
    return B % 64 == 0 ? 64 : B % 64
end

for f in (:num_chunks, :chunk_type, :num_bits, :top_chunk_bits)
    @eval $f(s::BitString) = $f(typeof(s))
end

"""
    chunks(s::BitString)

`SVector` that stores the chunks of `s`.
"""
chunks(s::BitString) = s.chunks

"""
    chunks_bits(::Type{<:BitString}, i)
    chunks_bits(s, i)

Number of bits in the `i`-th chunk of `s`.
"""
chunk_bits(s, i) = chunk_bits(typeof(s), i)
chunk_bits(::Type{<:BitString{B,1}}, _) where {B} = B
function chunk_bits(::Type{S}, i) where {S<:BitString}
    return ifelse(i == 1, top_chunk_bits(S), 64)
end

function ghost_bit_mask(::Type{S}) where S<:BitString
    T = chunk_type(S)
    unused_bits = sizeof(T) * 8 - top_chunk_bits(S)
    return ~zero(T) >>> unused_bits
end

"""
    remove_ghost_bits(s::BitString)

Remove set bits outside data field if any are present.

See also: [`has_ghost_bits`](@ref).
"""
function remove_ghost_bits(s::S) where {S<:BitString}
    mask = ghost_bit_mask(S)
    return S(setindex(s.chunks, s.chunks[1] & mask, 1))
end

@inline function remove_ghost_bits(s::S) where {S<:BitString{<:Any,1}}
    mask = ghost_bit_mask(S)
    return S(chunks(s) .& mask)
end

"""
    has_ghost_bits(s::BitString)

Check for bits outside data field.

See also: [`remove_ghost_bits`](@ref).
"""
function has_ghost_bits(s::S) where {S<:BitString}
    top = first(chunks(s))
    mask = ~zero(UInt64) << top_chunk_bits(S)
    return top & mask > 0
end

###
### Alternative/useful constructors. These are not super efficient, but they are safe.
###
function BitString{B}(i::Union{Int128,Int64,Int32,Int16,Int8}) where {B}
    return remove_ghost_bits(BitString{B}(unsigned(i)))
end
function BitString{B}(i::Union{UInt64,UInt32,UInt16,UInt8}) where {B}
    N, T = num_chunks(Val(B))
    s = setindex(zero(SVector{N,T}), T(i), N)
    return remove_ghost_bits(BitString{B,N,T}(s))
end
function BitString{B}(i::UInt128) where {B}
    N, T = num_chunks(Val(B))
    left = i >>> 0x40 % T # left will only be used if T == UInt64 and N > 1
    right = i  % T
    s = ntuple(Val(N)) do i
        i == N ? right : i == N - 1 ? left : zero(T)
    end
    return remove_ghost_bits(BitString{B,N,T}(SVector{N,T}(s)))
end
function BitString{B}(i::BigInt) where {B}
    N, T = num_chunks(Val(B))
    s = zero(SVector{N,T})
    j = N
    while i ≠ 0
        chunk = i & typemax(T) % T
        i >>>= 64 # Can use 64 here, as only 1-chunk addresses can be smaller
        s = setindex(s, chunk, j)
        j -= 1
    end
    return remove_ghost_bits(BitString{B,N,T}(s))
end

function Base.zero(S::Type{<:BitString{B}}) where {B}
    N, T = num_chunks(Val(B))
    BitString{B,N,T}(zero(SVector{N,T}))
end
Base.zero(s::BitString) = zero(typeof(s))

function Base.show(io::IO, s::BitString{B,N}) where {B,N}
    str = join(map(i -> repr(i)[3:end], s.chunks), '_')

    print(io, "BitString{$B}(big\"0x", str, "\")")
end
Base.bitstring(s::BitString{B}) where {B} = join(bitstring.(s.chunks))[(end - B + 1):end]

###
### Operations on BitStrings
###
for op in (:⊻, :&, :|)
    @eval (Base.$op)(l::S, r::S) where S<:BitString = S($op.(l.chunks, r.chunks))
end
Base.:~(s::S) where S<:BitString = remove_ghost_bits(S(.~(s.chunks)))

Base.count_ones(s::BitString) = sum(count_ones, s.chunks)
Base.count_zeros(s::BitString) = num_bits(s) - count_ones(s)

function _trailing(f, s::BitString)
    result = 0
    i = 0
    # Idea: if all whole chunk is the same digit, you have to look at the next one.
    # This gets compiled away if N=1
    for i in num_chunks(s):-1:1
        r = f(s.chunks[i])
        result += r
        r == chunk_bits(s, i) || break
    end
    # If top chunk occupies the whole integer, result will always be smaller or equal to B.
    if f ≢ trailing_ones && top_chunk_bits(s) ≠ 64
        return min(num_bits(s), result)
    else
        return result
    end
end

function _leading(f, s::BitString)
    N = sizeof(chunk_type(s)) * 8
    # First chunk is a special case - we have to ignore the empty space before the string.
    result = min(f(s.chunks[1] << (N - top_chunk_bits(s))), top_chunk_bits(s))

    # This gets compiled away if N=1
    if num_chunks(s) > 1 && result == top_chunk_bits(s)
        for i in 2:num_chunks(s)
            r = f(s.chunks[i])
            result += r
            r == 64 || break
        end
    end
    return result
end

Base.trailing_ones(s::BitString) = _trailing(trailing_ones, s)
Base.trailing_zeros(s::BitString) = _trailing(trailing_zeros, s)
Base.leading_ones(s::BitString) = _leading(leading_ones, s)
Base.leading_zeros(s::BitString) = _leading(leading_zeros, s)

@generated function _right_shift(s::S, k) where {S<:BitString}
    N = num_chunks(S)
    quote
        $(Expr(:meta, :inline))
        # equivalent to d, r = divrem(k, 64)
        d = k >>> 0x6
        r = k & 63
        ri = 64 - r
        mask = ~zero(UInt64) >>> ri # 2^r-1 # 0b0...01...1 with `r` 1s
        c = chunks(s)

        @nif $(N + 1) l -> (d < l) l -> (
            S(SVector((@ntuple l - 1 k -> zero(UInt64))... ,c[1] >>> r,
                      (@ntuple $N-l q -> (c[q + 1] >>> r | ((c[q] & mask) << ri)))...
                      ))
        ) l -> (
            return zero(S)
        )
    end
end

function _left_shift(s::S, k) where {S<:BitString}
    result = zeros(MVector{num_chunks(S),UInt64})
    # d, r = divrem(k, 64)
    d = k >>> 0x6
    r = k & 63

    shift = s.chunks .<< (r % UInt64)
    carry = s.chunks .>>> ((64 - r) % UInt64)

    for i in d + 1:length(result)
        @inbounds result[i - d] = shift[i] | get(carry, i + 1, zero(UInt64))
    end
    # This bit removes ghost bits.
    result[1] &= ghost_bit_mask(S)
    return S(SVector(result))
end

Base.:>>(s::BitString, k) = k ≥ 0 ? _right_shift(s, k) : _left_shift(s, -k)
Base.:<<(s::BitString, k) = k > 0 ? _left_shift(s, k) : _right_shift(s, -k)
Base.:>>>(s::BitString, k) = s >> k

# remove ghost bits must be applied to both because k might be negative.
Base.:>>(s::S, k) where S<:BitString{<:Any,1} = remove_ghost_bits(S(s.chunks .>> k))
Base.:>>(s::S, k::Unsigned) where S<:BitString{<:Any,1} = S(s.chunks .>> k)
Base.:<<(s::S, k) where S<:BitString{<:Any,1} = remove_ghost_bits(S(s.chunks .<< k))

function Base.isless(s1::B, s2::B) where {B<:BitString}
    for i in 1:num_chunks(B)
        if chunks(s1)[i] ≠ chunks(s2)[i]
            return chunks(s1)[i] < chunks(s2)[i]
        end
    end
    return false
end
Base.isodd(s::BitString) = isodd(chunks(s)[end])
Base.iseven(s::BitString) = iseven(chunks(s)[end])

# For compatibility. Changing any of the hashes will slightly change results and make the
# tests fail.
Base.hash(b::BitString{<:Any,1}, h::UInt) = hash(b.chunks[1], h)
Base.hash(b::BitString, h::UInt) = hash(b.chunks.data, h)

"""
    partial_left_shift(bs::BitString, i, j)

Shift a part of the bitstring left by one place with boundaries `i < j`.
In a `BoseFS` bitstring, it moves a particle at offset `i` to the position at
offset `j`.

See also: [`excitation`](@ref), [`partial_right_shift`](@ref).
"""
function partial_left_shift(chunk::T, i, j) where {T<:Unsigned}
    # Mask of one spanning from i to j
    mask = (T(1) << T(j - i + 1) - T(1)) << T(i)
    # Shift the part of the string that needs to be shifted, ensure a one is added at the end
    # swap shift to move in other direction
    #println(bitstring(mask))
    shifted_part = ((chunk & mask) << 0x1) & mask
    # Leave the rest intact
    intact_part = chunk & ~mask

    return shifted_part | intact_part | T(1) << T(i)
end

"""
    partial_right_shift(bs::BitString, i, j)

Shift a part of the bitstring right by one place with boundaries `i < j`.
In a `BoseFS` bitstring, it moves a particle at offset `j` to the position at
offset `i`.

See also: [`partial_left_shift`](@ref), [`excitation`](@ref).
"""
function partial_right_shift(chunk::T, i, j) where {T<:Unsigned}
    # Mask of one spanning from i to j
    mask = (T(1) << T(j - i + 1) - T(1)) << T(i)
    # Shift the part of the string that needs to be shifted, ensure a one is added at the end
    # swap shift to move in other direction
    shifted_part = ((chunk & mask) >> 0x1) & mask
    # Leave the rest intact
    intact_part = chunk & ~mask
    #println(lpad("↑" * " "^j, length(bitstring(chunk))))

    return shifted_part | intact_part | T(1) << T(j)
end

function partial_left_shift(bs::S, i, j) where {S<:BitString{<:Any,1}}
    return S(partial_left_shift(bs.chunks[1], i, j))
end

function partial_right_shift(bs::S, i, j) where {S<:BitString{<:Any,1}}
    return S(partial_right_shift(bs.chunks[1], i, j))
end

function partial_left_shift(bs::S, i, j) where {N,S<:BitString{<:Any,N}}
    result = MVector(bs.chunks)
    lo_idx = N - (i >>> 0x6)
    hi_idx = N - (j >>> 0x6)
    lo_off = i & 63
    hi_off = j & 63
    @inbounds if hi_idx == lo_idx
        result[hi_idx] = partial_left_shift(result[hi_idx], lo_off, hi_off)
    else
        # Top part first.
        chunk = result[hi_idx]
        chunk = partial_left_shift(chunk, 0, hi_off)
        # Carry bit.
        chunk &= -UInt(1) << 0x1
        chunk |= result[hi_idx + 1] >> 63
        result[hi_idx] = chunk

        idx = hi_idx + 1
        while idx < lo_idx
            chunk = result[idx]
            chunk <<= 0x1
            chunk |= result[idx + 1] >> 63
            result[idx] = chunk
            idx += 1
        end

        # Bottom part.
        chunk = result[lo_idx]
        chunk = partial_left_shift(chunk, lo_off, 64)
        result[lo_idx] = chunk
    end
    return S(SVector(result))
end

function partial_right_shift(bs::S, i, j) where {N,S<:BitString{<:Any,N}}
    result = MVector(bs.chunks)
    lo_idx = N - (i >>> 0x6)
    hi_idx = N - (j >>> 0x6)
    lo_off = i & 63
    hi_off = j & 63
    @inbounds if hi_idx == lo_idx
        result[hi_idx] = partial_right_shift(result[hi_idx], lo_off, hi_off)
    else
        # Bottom first
        chunk = result[lo_idx]
        chunk = partial_right_shift(chunk, lo_off, 64)
        # Carry bit.
        chunk &= -UInt(1) >> 0x1
        chunk |= result[lo_idx - 1] << 63
        result[lo_idx] = chunk

        idx = lo_idx - 1
        while idx > hi_idx
            chunk = result[idx]
            chunk >>= 0x1
            chunk |= result[idx - 1] << 63
            result[idx] = chunk
            idx -= 1
        end

        # Top part.
        chunk = result[hi_idx]
        chunk = partial_right_shift(chunk, 0, hi_off)
        result[hi_idx] = chunk
    end
    return S(SVector(result))
end

function Base.bitreverse(bs::BitString{B,1,T}) where {B,T}
    return typeof(bs)(SVector(bitreverse(bs.chunks[1]) >> T(sizeof(T) * 8 - B)))
end
function Base.bitreverse(bs::BitString{B,N}) where {B,N}
    return typeof(bs)(bitreverse.(reverse(bs.chunks))) >> (64 * N - B)
end
