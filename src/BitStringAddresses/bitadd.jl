"""
    BitAdd{I,B} <: AbstractBitString
    BitAdd(address::Integer, B)
    BitAdd(chunks::T, B) where T<:Union{Tuple,SVector}
    BitAdd{B}(address)

Address type that encodes a bistring address with `B` bits. The bits are
stored efficiently as `SVector` of `I` chunks of type `UInt64`.
The two-argument constructor is preferred due to safety (consistency checks).
In hot loops there maybe gain from the (unsafe) parametric constructor.
If an integer `address` is passed, its bit representation  is used to
initialize `BitAdd`. For large bit numbers, `BigInt` is convenient.
`BitAdd[i]` will return bit `i` (as `Bool`), counting
from right to left.

- `BitAdd{B}()` creates a `BitAdd` with all ones.
- `zero(BitAdd{B})`  creates a `BitAdd` with all zeros.

Note that no checking for ghost bits occurs when constructing `BitAdd` from
`SVector` or `Tuple`. See [`bitadd()`](@ref), [`check_consistency()`](@ref), and
[`remove_ghost_bits()`](@ref) methods!
"""
struct BitAdd{I,B} <: AbstractBitString
  chunks::SVector{I,UInt64} # bitstring of `B ≤ I*64` bits, stored as SVector

  # inner contructor: only allow passing `B` to enforce consistency of `B` and `I`
  function BitAdd{B}(chunks::SVector{I,UInt64}) where {B,I}
    I == (B - 1) ÷ 64 + 1 || error("in construction of `BitAdd{$I,$B}`: (B - 1) ÷ 64 + 1 evaluates to $((B - 1) ÷ 64 + 1) but should be I == $I")
    return new{I,B}(chunks)
  end
end
# Note: The default constructor does not check for ghost bits.
# Use `check_consistency()` if unsure!
# BitAdd{I,B}(chunks::SVector{I,UInt64}) where {B,I} = BitAdd{B}(chunks)

# Least specific: try to convert `chunks` to SVector; useful for tuples
# Note: This constructor does not check for ghost bits.
# Use `check_consistency()` if unsure!
BitAdd{B}(chunks) where B = BitAdd{B}(SVector(chunks))
# BitAdd{B}(chunks) where B = check_consistency(BitAdd{B}(SVector(chunks)))

# """
#     bitadd(address, numbits)
# Safely construct a `BitAdd` with `numbits` bits.
# If `address` is integer, its bit representation (right aligned) will be used.
# `Tuple` and `SVector` can be passed as well. Ghost bits will be checked for
# and produce an error if found.
#
# This function is safe but slow and should be avoided in hot loops, where type
# constructors can yield better performance.
# """
BitAdd(chunks, numbits) = check_consistency(BitAdd{numbits}(SVector(UInt64.(chunks))))
BitAdd(chunks::Integer, numbits) = BitAdd{numbits}(chunks)

# @inline function BitAdd{B}(a::Integer) where B
#   @boundscheck B < 1 && throw(BoundsError())
#   I = (B-1) ÷ 64 + 1 # number of UInt64s needed
#   adds = zeros(MVector{I,UInt64})
#   for i in I:-1:1
#      adds[i] = UInt64(a & 0xffffffffffffffff)
#      # extract rightmost 64 bits and store in adds
#      a >>>= 64 # shift bits to right by 64 bits
#   end
#   bsa = BitAdd{B}(SVector(adds))
#   @boundscheck check_consistency(bsa)
#   return bsa
# end

@inline function BitAdd{I,B}(a::Integer) where {I,B}
  @boundscheck B < 1 && throw(BoundsError())
  @boundscheck I == (B-1) ÷ 64 + 1 || throw(BoundsError())
  adds = zeros(MVector{I,UInt64})
  for i in I:-1:1
     adds[i] = UInt64(a & 0xffffffffffffffff)
     # extract rightmost 64 bits and store in adds
     a >>>= 64 # shift bits to right by 64 bits
  end
  bsa = BitAdd{B}(SVector(adds))
  @boundscheck check_consistency(bsa)
  return bsa
end

@inline function BitAdd{B}(a::Integer) where B
  @boundscheck B < 1 && throw(BoundsError())
  I = (B-1) ÷ 64 + 1 # number of UInt64s needed
  return @inbounds BitAdd{I,B}(a)
end


@inline function BitAdd{B}(address::A) where {B, A<:Union{Int,Int64,UInt,UInt64}}
  I = (B-1) ÷ 64 + 1 # number of UInt64s needed
  bsa = BitAdd{B}((NTuple{I-1,UInt64}(0 for i in 1:(I-1))..., UInt64(address),))
  @boundscheck check_consistency(bsa)
  return bsa
end

# create a BitAdd with all ones
function BitAdd{B}() where B
  (I, r) = divrem((B-1), 64) .+ 1
  first = ~UInt64(0) >>> (64 - r)
  BitAdd{B}((first, NTuple{I-1,UInt64}(~UInt64(0) for i in 1:(I-1))...,))
end

Base.zero(::Type{BitAdd{I,B}}) where {I,B} = BitAdd{B}(0)
Base.zero(b::BitAdd) = zero(typeof(b))
Base.hash(b::BitAdd,  h::UInt) = hash(b.chunks.data, h)

function check_consistency(a::BitAdd{I,B}) where {I,B}
  (d,lp) = divrem((B-1), 64) .+ 1 # bit position of leftmost bit in first chunk
  mask = ~UInt64(0)>>(64-lp)
  iszero(a.chunks[1] & ~mask) || error("ghost bits detected in $a")
  d == I || error("inconsistency in $a: $d words needed but $I present")
  length(a.chunks) == I || error("inconsistent length $(length(a.chunks)) with I = $I in $a")
  a # nothing
end

"""
    remove_ghost_bits(bs)
Remove set bits outside data field if any are present.
"""
function remove_ghost_bits(a::BitAdd{I,B}) where {I,B}
  lp = (B-1) % 64 +1 # bit position of leftmost bit in first chunk
  mask = ~UInt64(0)>>(64-lp)
  madd = MVector(a.chunks)
  madd[1] &= mask
  return BitAdd{B}(SVector(madd))
end

num_chunks(::Type{BitAdd{I,B}}) where {I,B} = I
num_bits(::Type{BitAdd{I,B}}) where {I,B} = B
chunk_size(::Type{<:BitAdd}) = 64
chunks(b::BitAdd) = b.chunks

# comparison check number of bits and then compares the tuples
Base.isless(a::T, b::T) where T<:BitAdd = isless(a.chunks, b.chunks)
function Base.isless(a::BitAdd{I1,B1}, b::BitAdd{I2,B2}) where {I1,B1,I2,B2}
  return isless(B1,B2)
end

# bit operations
Base.:⊻(a::BitAdd{I,B}, b::BitAdd{I,B}) where {I,B} = BitAdd{B}(a.chunks .⊻ b.chunks)
Base.:&(a::BitAdd{I,B}, b::BitAdd{I,B}) where {I,B} = BitAdd{B}(a.chunks .& b.chunks)
Base.:|(a::BitAdd{I,B}, b::BitAdd{I,B}) where {I,B} = BitAdd{B}(a.chunks .| b.chunks)
Base.:~(a::BitAdd{I,B}) where {I,B} = remove_ghost_bits(BitAdd{B}(.~a.chunks))

unsafe_count_ones(a::BitAdd) = sum(count_ones, a.chunks)
Base.count_ones(a::BitAdd) = unsafe_count_ones(remove_ghost_bits(a))
Base.count_zeros(a::BitAdd{I,B}) where {I,B} = B - count_ones(a)

"""
    >>>(b::BitAdd,n::Integer)
Bitshift `b` to the right by `n` bits and fill from the left with zeros.
"""
@inline function Base.:>>>(b::BitAdd{I,B},n::Integer) where {I,B}
  return BitAdd{B}(lbshr(b.chunks,n)) # devolve to shifting SVector
end

# function >>>(b::BitAdd{I,B},n::Integer) where {I,B}
#   # we assume there are no ghost bits
#   if I == 1
#     return BitAdd{B}((b.chunks[1]>>>n,))
#   elseif I == 2 || I ==3
#     # println("say Hi!")
#     return  BitAdd{B}(lbshr(b.chunks, n))
#   end
#   # d, r = divrem(n,64) # shift by `d` chunks and `r` bits
#   r = n & 63 # same as above but saves a lot of time!!
#   d = n >>> 6
#   mask = ~0 >>> (64-r) # 2^r-1 # 0b0...01...1 with `r` 1s
#   a = zeros(MVector{I,UInt64})
#   I-d > 0 && (a[d+1] = b.chunks[1]>>>r) # no carryover for leftmost chunk
#   for i = 2:(I-d) # shift chunks and `or` carryover
#     a[d+i] = (b.chunks[i]>>>r) | ((b.chunks[i-1] & mask) << (64-r) )
#   end
#   return BitAdd{B}(SVector(a))
# end

Base.:>>(b::BitAdd,n::Integer) = b >>> n

"""
    lbshr(c,k)
Apply logical bit shift to the right by `k` bits to `c`.
"""
@generated function lbshr(c::SVector{N,I}, k) where {N,I}
  # this generated function produces code at compile time that is specific
  # to the dimension `N` of `c` and thus avoids stack allocations completely.
  # println("gen N = $N, k = $k, I = $I")
  nplus1 = N + 1
  # println("gen nplus1 = $nplus1")
  quote
    $(Expr(:meta, :inline))
    r = k & 63 # same as above but saves a lot of time!!
    d = k >>> 6
    ri = 64-r
    mask = ~0 >>> ri # 2^r-1 # 0b0...01...1 with `r` 1s
    # println("quote N = $N, d = $d")
    # @nif generates a sequence of if ... ifelse ... else statments with `N`
    # branches
    @nif $nplus1 l->(d < l) l->(
          # println("d = $d; l = ",l);
          # return  zero(SVector{l-1,UInt64})
          SVector((@ntuple l-1 k->zero($I))... ,c[1] >>>r,
            # (@ntuple $N-l q -> q)...
            (@ntuple $N-l q -> (c[q+1]>>>r | ((c[q] & mask)<<ri)))...
          )
        ) l->(
          # println("d = $d; All OK, N = $N, l = ",l);
          return zero(SVector{$N,$I})
    )
  end
end

## specific version for 2 chunks
# @inline function lbshr(c::SVector{2,UInt64}, n::Integer)
#   # d, r = divrem(n,64) # shift by `d` chunks and `r` bits
#   r = n & 63 # same as above but saves a lot of time!!
#   d = n >>> 6
#   mask = ~0 >>> (64-r) # 2^r-1 # 0b0...01...1 with `r` 1s
#   a = d>0 ? zero(UInt64) : c[1] >>> r
#   b = d>1 ? zero(UInt64) : (d>0 ? c[1] >>>r : (c[2]>>>r | ((c[1] & mask)<< (64-r))))
#   return SVector(a,b)
# end

## specific version for 3 chunks; slightly slower than the below version
# @inline function lbshr(c::SVector{3,UInt64}, k::Integer)
#   # d, r = divrem(n,64) # shift by `d` chunks and `r` bits
#   r = k & 63 # same as above but saves a lot of time!!
#   d = k >>> 6
#   mask = ~0 >>> (64-r) # 2^r-1 # 0b0...01...1 with `r` 1s
#   s1 = d>0 ? zero(UInt64) : c[1] >>> r
#   s2 = d>1 ? zero(UInt64) : (d>0 ? c[1] >>>r : (c[2]>>>r | ((c[1] & mask)<< (64-r))))
#   s3 = d>2 ? zero(UInt64) : (d>1 ? c[1] >>>r : d >0 ? (c[2]>>>r | ((c[1] & mask)<< (64-r))) :
#         (c[3]>>>r | ((c[2] & mask)<< (64-r))))
#   return SVector(s1,s2,s3)
# end

## specific version for 3 chunks
# @inline function lbshr(c::SVector{3,UInt64}, k::Integer)
#   # d, r = divrem(n,64) # shift by `d` chunks and `r` bits
#   r = k & 63 # same as above but saves a lot of time!!
#   d = k >>> 6
#   mask = ~0 >>> (64-r) # 2^r-1 # 0b0...01...1 with `r` 1s
#   if d > 2
#     return zero(SVector{3,UInt64})
#   elseif d > 1
#     return SVector(zero(UInt64), zero(UInt64), c[1] >>> r)
#   elseif d > 0
#     return SVector(zero(UInt64), c[1] >>> r, (c[2]>>>r | ((c[1] & mask)<< (64-r))))
#   end
#   return SVector(c[1] >>>r, (c[2]>>>r | ((c[1] & mask)<< (64-r))), (c[3]>>>r | ((c[2] & mask)<< (64-r))))
# end

"""
    <<(b::BitAdd,n::Integer)
Bitshift `b` to the left by `n` bits and fill from the right with zeros.
"""
Base.:<<(b::BitAdd{I,B},n::Integer) where {I,B} = remove_ghost_bits(unsafe_shift_left(b,n))

## this is still memory allocating for `I>3`.
# TODO: rewrite this as generated function
function unsafe_shift_left_old(b::BitAdd{I,B},n::Integer) where {I,B}
  if I == 1
    return BitAdd{B}((b.chunks[1]<<n,))
  elseif I == 2
    return BitAdd{B}(bshl(b.chunks,n))
  end
  d, r = divrem(n,64) # shift by `d` chunks and `r` bits
  mask = ~0 << (64-r) # (2^r-1) << (64-r) # 0b1...10...0 with `r` 1s
  a = zeros(MVector{I,UInt64})
  for i in 1:(I-d-1) # shift chunks and `or` carryover
    a[i] = (b.chunks[i+d]<<r) | ((b.chunks[i+d+1] & mask) >>> (64-r))
  end
  I-d > 0 && (a[I-d] = b.chunks[I]<<r) # no carryover for rightmost chunk
  return BitAdd{B}(SVector(a))
end
# This is faster than << for BitArray (yeah!).
# About half the time and allocations.

function unsafe_shift_left(b::BitAdd{I,B},n::Integer) where {I,B}
  if I == 1
    return BitAdd{B}((b.chunks[1]<<n,))
  elseif I == 2
    return BitAdd{B}(bshl(b.chunks,n))
  end
  d, r = divrem(n,64) # shift by `d` chunks and `r` bits
  mask = ~0 << (64-r) # (2^r-1) << (64-r) # 0b1...10...0 with `r` 1s
  a = zeros(SVector{I,UInt64})
  for i in 1:(I-d-1) # shift chunks and `or` carryover
    nchunk = (b.chunks[i+d]<<r) | ((b.chunks[i+d+1] & mask) >>> (64-r))
    a = @set a[i] =  nchunk
  end
  if I-d > 0
    lchunk = b.chunks[I]<<r # no carryover for rightmost chunk
    a = @set a[I-d] = lchunk
  end
  return BitAdd{B}(SVector(a))
end
# this version does not allocate memory at all!!

function bshl(c::SVector{2, UInt64}, n::Integer)
  d, r = divrem(n,64) # shift by `d` chunks and `r` bits
  if d > 1
    return SVector(zero(UInt64), zero(UInt64))
  elseif d > 0
    return SVector(c[2]<<r,zero(UInt64))
  else
    mask = ~0 << (64-r) # (2^r-1) << (64-r) # 0b1...10...0 with `r` 1s
    l = (c[1] << r) | ((c[2] & mask) >>> (64 -r))
    return SVector(l, c[2] << r)
  end
end
function bshl(c::SVector{3, UInt64}, n::Integer)
  d, r = divrem(n,64) # shift by `d` chunks and `r` bits
  if d > 2
    return SVector(zero(UInt64), zero(UInt64), zero(UInt64))
  elseif d > 1
    return SVector(c[3]<<r, zero(UInt64), zero(UInt64))
  elseif d > 0
    l = (c[2] << r) | ((c[3] & mask) >>> (64 -r))
    return SVector(l, c[3] << r,  zero(UInt64))
  else
    mask = ~0 << (64-r) # (2^r-1) << (64-r) # 0b1...10...0 with `r` 1s
    l = (c[1] << r) | ((c[2] & mask) >>> (64 -r))
    l2 = (c[2] << r) | ((c[3] & mask) >>> (64 -r))
    return SVector(l, l2, c[3] << r)
  end
end


function Base.trailing_ones(a::BitAdd{I,B}) where {I,B}
  t = 0
  for chunk in reverse(a.chunks)
    s = trailing_ones(chunk)
    t += s
    s < 64 && break
  end
  return t # min(t, B) # assume no ghost bits
end

function Base.trailing_zeros(a::BitAdd{I,B}) where {I,B}
  t = 0
  for chunk in reverse(a.chunks)
    s = trailing_zeros(chunk)
    t += s
    s < 64 && break
  end
  return min(t, B)
end

function Base.leading_zeros(a::BitAdd{I,B}) where {I,B}
  r = (B-1)%64 + 1 # number of bits in first chunk
  t = leading_zeros(a.chunks[1]<<(64-r))
  t < r && return t # we are done
  t = r # ignore more than r zeros
  for i in 2:I
    s = leading_zeros(a.chunks[i])
    t += s
    s < 64 && break
  end
  return t
end

function Base.leading_ones(a::BitAdd{I,B}) where {I,B}
  r = (B-1)%64 + 1 # number of bits in first chunk
  t = leading_ones(a.chunks[1]<<(64-r))
  t < r && return t # we are done
  for i in 2:I
    s = leading_ones(a.chunks[i])
    t += s
    s < 64 && break
  end
  return t
end

Base.iseven(a::BitAdd) = iseven(a.chunks[end])
Base.isodd(a::BitAdd) = isodd(a.chunks[end])
# """
#     lead_bit(a)
# Value of leftmost bit in bit address `a`.
# """
# function lead_bit(a::BitAdd{I,B}) where {I,B}
#   lp = (B-1) % 64 # bit position of leftmost bit in first chunk (count from 0)
#   mask = UInt64(1)<< lp # shift a "1" to the right place
#   return (a.chunks[1]&mask) >> lp |>Bool # return type is Bool
# end
# """
#     tail_bit(a)
# Value of rightmost bit in bit address `a`.
# """
# tail_bit(a::BitAdd) = a.chunks[end] & UInt64(1) # return type is UInt64

# getindex access to individual bits in BitAdd, counting from right to left.
@inline function Base.getindex(a::BitAdd{I,B}, i::Integer) where {I,B}
  @boundscheck 0 < i ≤ B || throw(BoundsError(a,i))
  (ci, li) = divrem(i-1, 64) # chunk index and local index, from 0
  mask = UInt64(1)<< li # shift a "1" to the right place
  return (a.chunks[I-ci] & mask) >> li |> Bool # return type is Bool
end
# make BitAdd a full blown iterator (over bits)
Base.lastindex(a::BitAdd{I,B}) where {I,B} = B
Base.length(a::BitAdd{I,B}) where {I,B} = B
Base.iterate(a::BitAdd{I,B}, i=1) where {I,B} = i == B+1 ? nothing : (a[i], i+1)
Base.eltype(a::BitAdd) = Bool

# useful for visualising the bits
# slow (μs) - do not use in hot loops
function Base.bitstring(ba::BitAdd{I,B}) where {I,B}
  return reverse(mapreduce(i->repr(Int(i)),*,ba))
end

function Base.show(io::IO, ba::BitAdd{I,B}) where {I,B}
  if I == 1
    print(io, "BitAdd{$B}(0b",bitstring(ba), ")")
  # elseif I == 2
  #   num = UInt128((UInt128(ba.chunks[1])<<64) | UInt128(ba.chunks[2]))
  #   print(io, "BitAdd{$B}(",num,")")
  else
    print(io, "BitAdd{$B}(",Tuple(ba.chunks),")")
  end
  nothing
end
