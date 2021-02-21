"""
    bitstring_storage(::Val{B}) where {B}

This function determines how to store `B` bits.
"""
function bitstring_storage(::Val{B}) where {B}
    return if B ≤ 0
        throw(ArgumentError("`B` must be positive!"))
    #= Option: use smaller types. Probably not necessary.
    elseif B ≤ 8
        SVector{1, UInt8}
    elseif B ≤ 16
        SVector{1, UInt16}
    elseif B ≤ 32
        SVector{1, UInt32}
    =#
    else # Turns out it's never worth using UInt128 here.
        SVector{(B - 1) ÷ 64 + 1, UInt64}
    end
end

"""
    BitString{B,N,T<:Unsigned} <: AbstractBitString

Type for storing bitstrings of static size. Holds `B` bits in `N` chunks, where each chunk is
an unsigned integer of type `T`.

`N` and `T` are chosen automatically to accommodate `B` bits as efficiently as possible.

# Constructors

* `BitString{B,N,T}(::SVector{N,T})`: unsafe constructor. Does not check for ghost bits.

* `BitString{B}(::Integer)`: Convert integer to `BitString`. Integer is truncated to the
  correct number of bits.

"""
struct BitString{B,N,T<:Unsigned} <: AbstractBitString
    chunks::SVector{N,T}

    # This constructor is only to be used internally. It doesn't check for ghost bits.
    function BitString{B,N,T}(s::SVector{N,T}) where {B,N,T}
        SVector{N,T} ≡ bitstring_storage(Val(B)) || error("Invalid type parameters")
        return new{B,N,T}(s)
    end
end
###
### Basic properties. These are all compile-time constants.
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
Number of bits stored in bitstring.
"""
num_bits(::Type{<:BitString{B}}) where {B} = B

"""
    chunk_type(::Type{<:BitString})
    chunk_type(s::BitString)
Type of integer used to store bitstring.
"""
chunk_type(::Type{<:BitString{<:Any,<:Any,T}}) where T = T

"""
    chunk_size(::Type{<:BitString})
    chunk_size(s::BitString)
Number of bits stored in each chunk of bitstring. Equivalent to
`sizeof(eltype(chunks(s))) * 8`.
"""
chunk_size(::Type{<:BitString{<:Any,<:Any,T}}) where T = sizeof(T) * 8

"""
    top_chunk_bits(::Type{<:BitString})
    top_chunk_bits(s::BitString)
Equivalent to `chunk_bits(s, 1)`.
"""
function top_chunk_bits(::Type{S}) where {S<:BitString}
    B = num_bits(S)
    C = chunk_size(S)
    return B % C == 0 ? C : B % C
end

for f in (:num_chunks, :num_bits, :chunk_size, :chunk_type, :top_chunk_bits)
    @eval $f(s::BitString) = $f(typeof(s))
end

"""
    chunks(::Type{<:BitString})
    chunks(s)
Return `SVector` that stores the chunks of `s`.
"""
chunks(s::BitString) = s.chunks
"""
    chunks_bits(s, i)
Return the number of bits in the `i`-th chunk of `s`.
"""
chunk_bits(s, i) = chunk_bits(typeof(s), i)
chunk_bits(::Type{S}, _) where {S<:BitString{<:Any,1}} = num_bits(S)
function chunk_bits(::Type{S}, i) where {S<:BitString}
    return ifelse(i == 1, top_chunk_bits(S), chunk_size(S))
end

"""
    remove_ghost_bits(s)
Remove set bits outside data field if any are present.

See also: [`has_ghost_bits`](@ref).
"""
function remove_ghost_bits(s::S) where {S<:BitString}
    T = chunk_type(S)
    unused_bits = chunk_size(S) - top_chunk_bits(S)
    # This compiles away if the following is true.
    if unused_bits == 0
        return s
    else
        mask = ~zero(T) >>> unused_bits
        return S(setindex(s.chunks, s.chunks[1] & mask, 1))
    end
end

@inline function remove_ghost_bits(s::S) where {S<:BitString{<:Any,1}}
    T = chunk_type(S)
    mask = ~zero(T) >>> (chunk_size(S) - top_chunk_bits(S))
    return S(SVector(chunks(s)[1] & mask))
end

"""
    has_ghost_bits(s)
Check for bits outside data field.

See also: [`remove_ghost_bits`](@ref).
"""
function has_ghost_bits(s::S) where {S<:BitString}
    top = first(chunks(s))
    mask = ~zero(top) << top_chunk_bits(S)
    return top & mask > 0
end

###
### Alternative/useful constructors. These are not super efficient.
###
function BitString{B}(i::Union{Int128,Int64,Int32,Int16,Int8}) where {B}
    return remove_ghost_bits(BitString{B}(unsigned(i)))
end
function BitString{B}(i::Union{UInt64,UInt32,UInt16,UInt8}) where {B}
    S = bitstring_storage(Val(B))
    T = eltype(S)
    N = length(S)
    s = setindex(zero(S), T(i), N)
    return remove_ghost_bits(BitString{B,N,T}(s))
end
function BitString{B}(i::UInt128) where {B}
    S = bitstring_storage(Val(B))
    T = eltype(S)
    N = length(S)
    if T ≡ UInt128
        s = setindex(zero(S), i, N)
    else
        left = i >>> 0x40 % UInt64
        right = i & typemax(UInt64) % UInt64
        s = S(ntuple(Val(N)) do i
            i == N ? right : i == N - 1 ? left : zero(UInt64)
        end)
    end
    return remove_ghost_bits(BitString{B,N,T}(s))
end
function BitString{B}(i::BigInt) where {B}
    S = bitstring_storage(Val(B))
    T = eltype(S)
    N = length(S)
    s = zero(S)
    j = N
    while i ≠ 0
        chunk = i & typemax(T) % T
        i >>>= sizeof(T) * 8
        s = setindex(s, chunk, j)
        j -= 1
    end
    return remove_ghost_bits(BitString{B,N,T}(s))
end

function Base.zero(S::Type{<:BitString{B}}) where {B}
    s = zero(bitstring_storage(Val(B)))
    N = length(s)
    T = eltype(s)
    BitString{B,N,T}(s)
end
Base.zero(s::BitString) = zero(typeof(s))

function Base.show(io::IO, s::BitString{B,N,T}) where {B,N,T}
    print(io, "BitString{$B,$N,$T}(", join(map(i -> repr(i)[3:end], s.chunks), '|'), ')')
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
    if top_chunk_bits(s) ≠ chunk_size(s)
        return min(num_bits(s), result)
    else
        return result
    end
end

function _leading(f, s::BitString)
    # First chunk is a special case - we have to ignore the empty space before the string.
    result = min(f(s.chunks[1] << (chunk_size(s) - top_chunk_bits(s))), top_chunk_bits(s))

    # This gets compiled away if N=1
    if num_chunks(s) > 1 && result == top_chunk_bits(s)
        for i in 2:num_chunks(s)
            r = f(s.chunks[i])
            result += r
            r == chunk_size(s) || break
        end
    end
    return result
end

Base.trailing_ones(s::BitString) = _trailing(trailing_ones, s)
Base.trailing_zeros(s::BitString) = _trailing(trailing_zeros, s)
Base.leading_ones(s::BitString) = _leading(leading_ones, s)
Base.leading_zeros(s::BitString) = _leading(leading_zeros, s)

@generated function _right_shift(s::S, k) where {S<:BitString}
    z = zero(chunk_type(S))
    cs = chunk_size(S)
    N = num_chunks(S)
    quote
        $(Expr(:meta, :inline))
        if k < 0
            return s << k
        elseif k == 0
            return s
        else
            d, r = divrem(k, $cs)
            ri = $cs - r
            mask = ~$z >>> ri # 2^r-1 # 0b0...01...1 with `r` 1s
            c = chunks(s)

            @nif $(N + 1) l -> (d < l) l -> (
                S(SVector((@ntuple l - 1 k -> $z)... ,c[1] >>> r,
                          (@ntuple $N-l q -> (c[q + 1] >>> r | ((c[q] & mask) << ri)))...
                          ))
            ) l -> (
                return zero(S)
            )
        end
    end
end

function _left_shift(s::S, k) where {S<:BitString}
    T = chunk_type(S)
    result = zeros(MVector{num_chunks(S),chunk_type(S)})
    d, r = divrem(k, chunk_size(S))

    shift = SVector(s.chunks) .<< (r % UInt)
    carry = s.chunks .>>> ((chunk_size(S) - r) % UInt)

    for i in d + 1:length(result)
        @inbounds result[i - d] = shift[i] | get(carry, i + 1, zero(T))
    end
    # This bit removes ghost bits.
    mask = ~zero(T) >>> (chunk_size(S) - top_chunk_bits(S))
    result[1] &= mask
    return S(SVector(result))
end

Base.:>>(s::BitString, k) = k ≥ 0 ? _right_shift(s, k) : _left_shift(s, -k)
Base.:<<(s::BitString, k) = k > 0 ? _left_shift(s, k) : _right_shift(s, -k)
Base.:>>>(s::BitString, k) = s >> k

Base.:>>(s::S, k) where S<:BitString{<:Any,1} = remove_ghost_bits(S(SVector(s.chunks[1] >> k)))
Base.:<<(s::S, k) where S<:BitString{<:Any,1} = remove_ghost_bits(S(SVector(s.chunks[1] << k)))

"""
    one_bit_mask(::Type{T}, pos) where T<:AbstractBitString
Optional faster way to to `T(1) << pos`.
"""
one_bit_mask(::Type{<:BitString{B}}, pos) where B = BitString{B}(1) << pos

"""
    three_bit_mask(::Type{T}, pos) where T<:AbstractBitString
Optional faster way to to `T(3) << pos`.
"""
two_bit_mask(::Type{<:BitString{B}}, pos) where B = BitString{B}(3) << pos

Base.isodd(s::BitString) = isodd(chunks(s)[end])
Base.iseven(s::BitString) = iseven(chunks(s)[end])
