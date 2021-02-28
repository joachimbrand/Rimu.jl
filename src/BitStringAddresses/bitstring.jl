"""
    num_chunks(::Val{B})

Determine the number of chunks needed to store `B` bits.
"""
function num_chunks(::Val{B}) where {B}
    return if B ≤ 0
        throw(ArgumentError("`B` must be positive!"))
    else # Turns out it's never worth using anything other than UInt64 here.
        return (B - 1) ÷ 64 + 1
    end
end

"""
    check_bitstring_typeparams(::Val{B}, ::Val{N})

Check if number of bits `B` is consistent with number of chunks `N`. Throw an error if not.
"""
function check_bitstring_typeparams(::Val{B}, ::Val{N}) where {B,N}
    if B > N * 64
        s = N == 1 ? "" : "s"
        error("$B bits do not fit into $N 64-bit chunk$s")
    elseif B ≤ (N - 1) * 64
        s = N == 2 ? "" : "s"
        error("$B bits fit into $(N - 1) 64-bit chunk$s, but $N chunks were provided")
    end
end

"""
    BitString{B,N} <: AbstractBitString

Type for storing bitstrings of static size. Holds `B` bits in `N` chunks, where each chunk is
an `UInt64`

`N` is chosen automatically to accommodate `B` bits as efficiently as possible.

# Constructors

* `BitString{B,N}(::SVector{N,T})`: unsafe constructor. Does not check for ghost bits.

* `BitString{B,N}(i::UInt64)`: as above, but sets `i` as the rightmost chunk.

* `BitString{B}(::Integer)`: Convert integer to `BitString`. Integer is truncated to the
  correct number of bits.

"""
struct BitString{B,N}
    chunks::SVector{N,UInt64}

    # This constructor is only to be used internally. It doesn't check for ghost bits.
    function BitString{B,N}(s::SVector{N,UInt64}) where {B,N}
        check_bitstring_typeparams(Val(B), Val(N))
        return new{B,N}(s)
    end
    function BitString{B,N}(i::UInt64) where {B,N}
        check_bitstring_typeparams(Val(B), Val(N))
        return new{B,N}(setindex(zero(SVector{N,UInt64}), i, N))
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

for f in (:num_chunks, :num_bits, :top_chunk_bits)
    @eval $f(s::BitString) = $f(typeof(s))
end

"""
    chunks(::Type{<:BitString})
    chunks(s)

`SVector` that stores the chunks of `s`.
"""
chunks(s::BitString) = s.chunks
"""
    chunks_bits(s, i)

Number of bits in the `i`-th chunk of `s`.
"""
chunk_bits(s, i) = chunk_bits(typeof(s), i)
chunk_bits(::Type{<:BitString{B,1}}, _) where {B} = B
function chunk_bits(::Type{S}, i) where {S<:BitString}
    return ifelse(i == 1, top_chunk_bits(S), 64)
end

function ghost_bit_mask(::Type{S}) where S<:BitString
    unused_bits = 64 - top_chunk_bits(S)
    return ~zero(UInt64) >>> unused_bits
end

"""
    remove_ghost_bits(s)

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
    has_ghost_bits(s)

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
    N = num_chunks(Val(B))
    s = setindex(zero(SVector{N,UInt64}), UInt64(i), N)
    return remove_ghost_bits(BitString{B,N}(s))
end
function BitString{B}(i::UInt128) where {B}
    N = num_chunks(Val(B))
    left = i >>> 0x40 % UInt64
    right = i & typemax(UInt64) % UInt64
    s = ntuple(Val(N)) do i
        i == N ? right : i == N - 1 ? left : zero(UInt64)
    end
    return remove_ghost_bits(BitString{B,N}(SVector{N,UInt64}(s)))
end
function BitString{B}(i::BigInt) where {B}
    N = num_chunks(Val(B))
    s = zero(SVector{N,UInt64})
    j = N
    while i ≠ 0
        chunk = i & typemax(UInt64) % UInt64
        i >>>= 64
        s = setindex(s, chunk, j)
        j -= 1
    end
    return remove_ghost_bits(BitString{B,N}(s))
end

function Base.zero(S::Type{<:BitString{B}}) where {B}
    N = num_chunks(Val(B))
    BitString{B,N}(zero(SVector{N,UInt64}))
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
    if top_chunk_bits(s) ≠ 64
        return min(num_bits(s), result)
    else
        return result
    end
end

function _leading(f, s::BitString)
    # First chunk is a special case - we have to ignore the empty space before the string.
    result = min(f(s.chunks[1] << (64 - top_chunk_bits(s))), top_chunk_bits(s))

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
        if k < 0
            return s << k
        elseif k == 0
            return s
        else
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
Base.:<<(s::S, k) where S<:BitString{<:Any,1} = remove_ghost_bits(S(s.chunks .<< k))

# Is this ordering needed?
function Base.isless(s1::B, s2::B) where {B<:BitString}
    for i in 1:num_chunks(B)
        if s1[i] ≠ s2[i]
            return s1[i] < s2[i]
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
