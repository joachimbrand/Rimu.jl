```
    BitStringAddresses
Module with types and methods pertaining to bitstring addresses.
```
module BitStringAddresses

using StaticArrays
using Setfield

using Base.Cartesian

import Base: isless, zero, iszero, show, ==, hash

export BitStringAddressType, BSAdd64, BSAdd128
export BitAdd, BoseFS, BoseFS2C
export onr, nearUniform, nearUniformONR
export numBits, numChunks, numParticles, numModes
export BStringAdd # deprecate
export bitaddr, maxBSLength # deprecate


"""
    BitStringAddressType

Abstract type for configuration addresses with the nature of a bitstring.
A number of methods need to be implemented, in particular
`Base.isless(a,b)`, `numBits()`, `numChunks()`.
"""
abstract type BitStringAddressType end

"""
    numChunks(a)
Number of 64-bit chunks representing `a`.
"""
numChunks(T::Type) = @error "not implemented: numChunks($T)"
numChunks(b) = numChunks(typeof(b))

"""
    numBits(a)
Number of bit chunks representing `a`.
"""
numBits(T::Type) = @error "not implemented: numBits($T)"
numBits(b) = numBits(typeof(b))

"""
    BosonicFockStateAddress <: BitStringAddressType
Supertype representing a bosonic Fock state. Implement [`numModes()`](@ref)
and [`numParticles()`](@ref).
"""
abstract type BosonicFockStateAddress <: BitStringAddressType end

"""
    numParticles(a)
Number of particles represented by `a`.
"""
numParticles(::Type{T}) where T = error("not implemented: numParticles($T)")
numParticles(b) = numParticles(typeof(b))

"""
    numModes(a)
Number of modes represented by `a`.
"""
numModes(T::Type) = error("not implemented: numModes($T)")
numModes(b) = numModes(typeof(b))

#################################################
"""
    BStringAdd <: BitStringAddressType

Address type that encodes a bistring address in a one-dim. BitArray.
"""
struct BStringAdd <: BitStringAddressType
  add::BitArray{1}
end #struct

function Base.isless(a1::BStringAdd,a2::BStringAdd)
# compare the chunks of the BitArray a1.add and a2.add
  for (index,chunk) in enumerate(a1.add.chunks)
    if chunk != a2.add.chunks[index]
      return chunk < a2.add.chunks[index]
    end
  end
  return false
end

Base.zero(::Type{BStringAdd}) = BStringAdd(BitArray([false]))
Base.zero(adr::BStringAdd) = BStringAdd(BitArray([false]))
Base.iszero(adr::BStringAdd) = iszero(adr.add)

import Base: ==, <<, >>>, >>, ⊻, &, |
==(a::BStringAdd, b::BStringAdd) = a.add == b.add
<<(a::BStringAdd, n) = BStringAdd(a.add << n)
>>>(a::BStringAdd, n) = BStringAdd(a.add >>> n)
>>(a::BStringAdd, n) = a >>> n
⊻(a::BStringAdd, b::BStringAdd) = BStringAdd(a.add .⊻ b.add)
(&)(a::BStringAdd, b::BStringAdd) = BStringAdd(a.add .& b.add)
(|)(a::BStringAdd, b::BStringAdd) = BStringAdd(a.add .| b.add)

Base.hash(a::BStringAdd, h::UInt) = hash(a.add, h)

function Base.show(io::IO, a::BStringAdd)
    print(io, "BStringAdd\"")
    for elem in a.add
      print(io, elem ? "1" : "0")
    end
    print(io, "\"")
end

function Base.leading_ones(a::BStringAdd)
  t = 0
  for chunk in a.add.chunks
    s = trailing_ones(chunk)
    t += s
    s < 64 && break
  end
  return min(t, a.add.len)
end

function Base.leading_zeros(a::BStringAdd)
  t = 0
  for chunk in a.add.chunks
    s = trailing_zeros(chunk)
    t += s
    s < 64 && break
  end
  return min(t, a.add.len)
end

function Base.trailing_zeros(a::BStringAdd)
  t = 0
  for (i, chunk) in enumerate(reverse(a.add.chunks))
    if i == 1 && (r = a.add.len%64) > 0
      t = leading_zeros(a.add.chunks[end]<<(64-r))
      t < r && return t
      t = min(t, r)
    else
      s = leading_zeros(chunk)
      t += s
      s < 64 && break
    end
  end
  return min(t, a.add.len)
end

function Base.trailing_ones(a::BStringAdd)
  t = 0
  for (i, chunk) in enumerate(reverse(a.add.chunks))
    if i == 1 && (r = a.add.len%64) > 0
      t = leading_ones(a.add.chunks[end]<<(64-r))
      t < r && return t
      t = min(t, r)
    else
      s = leading_ones(chunk)
      t += s
      s < 64 && break
    end
  end
  return min(t, a.add.len)
end

Base.iseven(a::BStringAdd) = ~a.add[end]
Base.reverse(a::BStringAdd) = BStringAdd(reverse(a.add))

# Base.leading_ones(a::BStringAdd) = trailing_ones(reverse(a))


#################################################
"""
    BSAdd64 <: BitStringAddressType

Address type that encodes a bistring address in a UInt64.
"""
struct BSAdd64 <: BitStringAddressType
  add::UInt64
end

Base.isless(a1::BSAdd64,a2::BSAdd64) = isless(a1.add, a2.add)
Base.zero(::Type{BSAdd64}) = BSAdd64(0)
Base.zero(add::BSAdd64) = BSAdd64(0)
Base.hash(a::BSAdd64, h::UInt) = hash(a.add, h)
Base.trailing_ones(a::BSAdd64) = trailing_ones(a.add)
Base.trailing_zeros(a::BSAdd64) = trailing_zeros(a.add)
Base.count_ones(a::BSAdd64) = count_ones(a.add)

import Base: <<, >>>, >>, ⊻, &, |
(>>>)(a::BSAdd64, n::Integer) = BSAdd64(a.add >>> n)

numChunks(::Type{BSAdd64}) = 1
numBits(::Type{BSAdd64}) = 64
Base.bitstring(a::BSAdd64) = bitstring(a.add)

#################################################
"""
    BSAdd128 <: BitStringAddressType

Address type that encodes a bistring address in a UInt128.
"""
struct BSAdd128 <: BitStringAddressType
  add::UInt128
end
BSAdd128(bsa::BSAdd128) = bsa # convert to same type
BSAdd128(bsa::BSAdd64) = BSAdd128(bsa.add)

Base.isless(a1::BSAdd128,a2::BSAdd128) = isless(a1.add, a2.add)
#Base.typemax(BSAdd128) = BSAdd128(typemax(UInt64))
Base.zero(::Type{BSAdd128}) = BSAdd128(0)
Base.zero(add::BSAdd128) = BSAdd128(0)
Base.hash(a::BSAdd128, h::UInt) = hash(a.add, h)
Base.trailing_ones(a::BSAdd128) = trailing_ones(a.add)
Base.trailing_zeros(a::BSAdd128) = trailing_zeros(a.add)
Base.count_ones(a::BSAdd128) = count_ones(a.add)

import Base: <<, >>>, >>, ⊻, &, |
(>>>)(a::BSAdd128, n::Integer) = BSAdd128(a.add >>> n)

numChunks(::Type{BSAdd128}) = 1
numBits(::Type{BSAdd128}) = 128
Base.bitstring(a::BSAdd128) = bitstring(a.add)

#################################################
"""
    BitAdd{I,B} <: BitStringAddressType
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
struct BitAdd{I,B} <: BitStringAddressType
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

numChunks(::Type{BitAdd{I,B}}) where {I,B} = I
numBits(::Type{BitAdd{I,B}}) where {I,B} = B

# comparison check number of bits and then compares the tuples
Base.isless(a::T, b::T) where T<:BitAdd = isless(a.chunks, b.chunks)
function Base.isless(a::BitAdd{I1,B1}, b::BitAdd{I2,B2}) where {I1,B1,I2,B2}
  return isless(B1,B2)
end

# bit operations
import Base: <<, >>>, >>, ⊻, &, |
⊻(a::BitAdd{I,B}, b::BitAdd{I,B}) where {I,B} = BitAdd{B}(a.chunks .⊻ b.chunks)
(&)(a::BitAdd{I,B}, b::BitAdd{I,B}) where {I,B} = BitAdd{B}(a.chunks .& b.chunks)
(|)(a::BitAdd{I,B}, b::BitAdd{I,B}) where {I,B} = BitAdd{B}(a.chunks .| b.chunks)

unsafe_count_ones(a::BitAdd) = mapreduce(count_ones, +, a.chunks)
Base.count_ones(a::BitAdd) = unsafe_count_ones(remove_ghost_bits(a))
Base.count_zeros(a::BitAdd{I,B}) where {I,B} = B - count_ones(a)

"""
    >>>(b::BitAdd,n::Integer)
Bitshift `b` to the right by `n` bits and fill from the left with zeros.
"""
@inline function >>>(b::BitAdd{I,B},n::Integer) where {I,B}
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

(>>)(b::BitAdd,n::Integer) = b >>> n

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
<<(b::BitAdd{I,B},n::Integer) where {I,B} = remove_ghost_bits(unsafe_shift_left(b,n))

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

#################################
"""
    BoseFS{N,M,A} <: BosonicFockStateAddress <: BitStringAddressType
    BoseFS(bs::A) where A <: BitAdd
    BoseFS(bs::A, b)

Address type that represents a Fock state of `N` spinless bosons in `M` orbitals
by wrapping a bitstring of type `A`. Orbitals are stored in reverse
order, i.e. the first orbital in a `BoseFS` is stored rightmost in the
bitstring `bs`. If the number of significant bits `b` is not encoded in `A` it
must be passed as an argument (e.g. for `BSAdd64` and `BSAdd128`).
"""
struct BoseFS{N,M,A} <: BosonicFockStateAddress
  bs::A
end

BoseFS{N,M}(bs::A) where {N,M,A} = BoseFS{N,M,A}(bs)

function BoseFS(bs::A, b::Integer) where A <: BitStringAddressType
  n = count_ones(bs)
  m = b - n + 1
  bfs = BoseFS{n,m,A}(bs)
  check_consistency(bfs)
  return bfs
end

function BoseFS(bs::BitAdd{I,B}) where {B,I}
  n = count_ones(bs)
  m = B - n + 1
  return BoseFS{n,m,BitAdd{I,B}}(bs)
end

function BoseFS(bs::BStringAdd)
  n = sum(bs.add)
  m = length(bs.add) - n + 1
  return BoseFS{n,m,BStringAdd}(bs)
end


"""
    BoseFS(onr::T) where T<:Union{AbstractVector,Tuple}
    BoseFS{BST}(onr::T)
Create `BoseFS` address from an occupation number representation, specifying
the occupation number of each orbital.
If a type `BST` is given it will define the underlying
bit string type. Otherwise, the bit string type is chosen to fit the `onr`.
"""
function BoseFS(onr::T) where T<:Union{AbstractVector,Tuple}
  m = length(onr)
  n = Int(sum(onr))
  b = n + m - 1
  if b ≤ 64
    A = BSAdd64
  elseif b ≤ 128
    A = BSAdd128
  else
    A = BitAdd
  end
  BoseFS{A}(onr,Val(n),Val(m),Val(b))
end

function BoseFS{A}(onr::T) where {A, T<:Union{AbstractVector,Tuple}}
  m = length(onr)
  n = Int(sum(onr))
  b = n + m - 1
  BoseFS{A}(onr,Val(n),Val(m),Val(b))
end

# This constructor is performant!!
@inline function BoseFS{N,M,A}(onr::T) where {N, M, A, T<:Union{AbstractVector,Tuple}}
  return BoseFS{A}(onr, Val(N), Val(M), Val(N+M-1))
end

# This constructor is performant!!
@inline function BoseFS{BSAdd64}(onr::T,::Val{N},::Val{M},::Val{B}) where {N,M,B,T<:Union{AbstractVector,Tuple}}
  @boundscheck begin
    B > 64 && throw(BoundsError(BSAdd64(0),B))
    N + M - 1 == B || @error "Inconsistency in constructor BoseFS"
  end
  bs = zero(UInt64) # empty bitstring
  for on in reverse(onr)
    bs <<= on+1
    bs |= ~zero(UInt64)>>(64-on)
  end
  return BoseFS{N,M,BSAdd64}(BSAdd64(bs))
end

@inline function BoseFS{BSAdd128}(onr::T,::Val{N},::Val{M},::Val{B}) where {N,M,B,T<:Union{AbstractVector,Tuple}}
  @boundscheck begin
    B > 128 && throw(BoundsError(BSAdd128(0),B))
    N + M - 1 == B || @error "Inconsistency in constructor BoseFS"
  end
  bs = zero(UInt128) # empty bitstring
  for on in reverse(onr)
    bs <<= on+1
    bs |= ~zero(UInt128)>>(128-on)
  end
  return BoseFS{N,M,BSAdd128}(BSAdd128(bs))
end

@inline function BoseFS{BitAdd{I,B}}(onr::T,::Val{N},::Val{M},::Val{B}) where {I,N,M,B,T<:Union{AbstractVector,Tuple}}
  @boundscheck  ((N + M - 1 == B) && (I == (B-1) ÷ 64 + 1)) || @error "Inconsistency in constructor BoseFS"
  bs = BitAdd{B}(0) # empty bitstring
  for on in reverse(onr)
    bs <<= on+1
    bs |= BitAdd{B}()>>(B-on)
  end
  return BoseFS{N,M,BitAdd{I,B}}(bs)
end

@inline function BoseFS{BitAdd}(onr::T,::Val{N},::Val{M},::Val{B}) where {N,M,B,T<:Union{AbstractVector,Tuple}}
  @boundscheck  N + M - 1 == B || @error "Inconsistency in constructor BoseFS"
  bs = BitAdd{B}(0) # empty bitstring
  for on in reverse(onr)
    bs <<= on+1
    bs |= BitAdd{B}()>>(B-on)
  end
  I = (B-1) ÷ 64 + 1 # number of UInt64s needed
  return BoseFS{N,M,BitAdd{I,B}}(bs)
end

@inline function BoseFS{BStringAdd}(onr::T,::Val{N},::Val{M},::Val{B}) where {N,M,B,T<:Union{AbstractVector,Tuple}}
  @boundscheck  N + M - 1 == B || @error "Inconsistency in constructor BoseFS{BStringAdd}"
  bs = bitaddr(onr, BStringAdd)
  @boundscheck  length(bs.add) == B || @error "Inconsistency in constructor BoseFS{BStringAdd}"
  return BoseFS{N,M,BStringAdd}(bs)
end

# comparison delegates to bs
Base.isless(a::BoseFS, b::BoseFS) = isless(a.bs, b.bs)
# hashing delegates to bs
Base.hash(bba::BoseFS,  h::UInt) = hash(bba.bs, h)
Base.bitstring(b::BoseFS) = bitstring(b.bs)
numChunks(::Type{BoseFS{N,M,A}}) where {N,M,A} = numChunks(A)
numBits(::Type{BoseFS{N,M,A}}) where {N,M,A} = N+M-1 # generally true for bosons
numParticles(::Type{BoseFS{N,M,A}}) where {N,M,A} = N
numModes(::Type{BoseFS{N,M,A}}) where {N,M,A} = M

function check_consistency(b::BoseFS{N,M,A}) where {N,M,A}
  numBits(b) ≤ numBits(A) || error("Inconsistency in $b: N+M-1 = $(N+M-1), numBits(A) = $(numBits(A)).")
  check_consistency(b.bs)
end
function check_consistency(b::BoseFS{N,M,A}) where {N,M,A<:Union{BSAdd64,BSAdd128}}
  numBits(b) ≤ numBits(A) || error("Inconsistency in $b: N+M-1 = $(N+M-1), numBits(A) = $(numBits(A)).")
  leading_zeros(b.bs.add) ≥ numBits(A) - numBits(b) ||  error("Ghost bits detected in $b.")
end




#################################
"""
    BoseFS2C{NA,NB,M,AA,AB} <: BosonicFockStateAddress <: BitStringAddressType

Address type that constructed with two [`BoseFS{N,M,A}`](@ref). It represents a
Fock state with two components, e.g. two different species of bosons with particle
number `NA` from species A and particle number `NB` from species B. The number of
orbitals `M` is expacted to be the same for both components.
"""
struct BoseFS2C{NA,NB,M,AA,AB} <: BitStringAddressType
  bsa::BoseFS{NA,M,AA}
  bsb::BoseFS{NB,M,AB}
end

BoseFS2C(onr_a::Tuple, onr_b::Tuple) = BoseFS2C(BoseFS(onr_a),BoseFS(onr_b))

function Base.show(io::IO, b::BoseFS2C{NA,NB,M,AA,AB}) where {NA,NB,M,AA,AB}
  print(io, "BoseFS2C(")
  Base.show(io,b.bsa)
  print(io, ",")
  Base.show(io,b.bsb)
  print(io, ")")
end

# performant and allocation free (if benchmarked on its own):
"""
    onr(bs)
Compute and return the occupation number representation of the bit string
address `bs` as an `SVector{M,Int32}`, where `M` is the number of orbitals.
"""
function onr(bba::BoseFS{N,M,A}) where {N,M,A}
  r = zeros(MVector{M,Int32})
  address = bba.bs
  for orbitalnumber in 1:M
    bosonnumber = Int32(trailing_ones(address))
    r[orbitalnumber] = bosonnumber
    address >>>= bosonnumber + 1
    iszero(address) && break
  end
  return SVector(r)
end

@inline function m_onr(bba::BoseFS{N,M,A}) where {N,M,A}
  r = zeros(MVector{M,Int32})
  address = bba.bs
  for orbitalnumber in 1:M
    bosonnumber = Int32(trailing_ones(address))
    @inbounds r[orbitalnumber] = bosonnumber
    address >>>= bosonnumber + 1
    iszero(address) && break
  end
  return r
end
# for some reason this is slower than the above onr() when benchmarked
s_onr(arg) = m_onr(arg) |> SVector


# need a special case for BStringAdd because the bit ordering is reversed
function onr(bba::BoseFS{N,M,A}) where {N,M,A<:BStringAdd}
  return SVector{M,Int32}(onr(bba.bs,M))
end


# # works but is not faster
# @generated function onr2(bba::BoseFS{N,M,A}) where {N,M,A}
#   quote
#     address = bba.bs
#     t = @ntuple $M k->(
#       bosonnumber = trailing_ones(address);
#       address >>>= bosonnumber + 1;
#       bosonnumber
#     )
#     return SVector(t)
#   end
# end

"""
  OccupationNumberIterator(address)
An iterator over the occupation numbers in `address`.
"""
struct OccupationNumberIterator{BS}
    bs::BS
    m::Int
end

OccupationNumberIterator(ad::BoseFS{N,M}) where {N,M} = OccupationNumberIterator(ad.bs, M)

Base.length(oni::OccupationNumberIterator) = oni.m
Base.eltype(oni::OccupationNumberIterator) = Int32

function Base.iterate(oni::OccupationNumberIterator, bsstate = (oni.bs, oni.m))
    bs, m = bsstate
    iszero(m) && return nothing
    bosonnumber = Int32(trailing_ones(bs))
    bs >>>= bosonnumber + 1
    return (bosonnumber, (bs, m-1))
end

# fast and works without allocations
function i_onr(bba::BoseFS{N,M,A}) where {N,M,A}
    SVector{M,Int32}(on for on in OccupationNumberIterator(bba))
end


# need a special case for BStringAdd because the bit ordering is reversed
function i_onr(bba::BoseFS{N,M,A}) where {N,M,A <:BStringAdd}
  return SVector{M,Int32}(onr(bba.bs,M))
end

"""
    nearUniformONR(N, M) -> onr::SVector{M,Int}
Create occupation number representation `onr` distributing `N` particles in `M`
modes in a close-to-uniform fashion with each orbital filled with at least
`N ÷ M` particles and at most with `N ÷ M + 1` particles.
"""
function nearUniformONR(n::Number, m::Number)
  return nearUniformONR(Val(n),Val(m))
end
function nearUniformONR(::Val{N}, ::Val{M}) where {N, M}
  fillingfactor, extras = divrem(N, M)
  # startonr = fill(fillingfactor,M)
  startonr = fillingfactor * @MVector ones(Int,M)
  startonr[1:extras] += ones(Int, extras)
  return SVector{M}(startonr)
end

"""
    nearUniform(BoseFS{N,M})
    nearUniform(BoseFS{N,M,A}) -> bfs::BoseFS{N,M,A}
Create bosonic Fock state with near uniform occupation number of `M` modes with
a total of `N` particles. Specifying the bit address type `A` is optional.

# Examples
```jldoctest
julia> nearUniform(BoseFS{7,5,BitAdd})
BoseFS{BitAdd}((2,2,1,1,1))

julia> nearUniform(BoseFS{7,5})
BoseFS{BSAdd64}((2,2,1,1,1))
```
"""
function nearUniform(::Type{BoseFS{N,M,A}}) where {N,M,A}
  return BoseFS{A}(nearUniformONR(Val(N),Val(M)),Val(N),Val(M),Val(N+M-1))
end
function nearUniform(::Type{BoseFS{N,M}}) where {N,M}
  return BoseFS(nearUniformONR(Val(N),Val(M)))
end

# function Base.show(io::IO, b::BoseFS{N,M,A}) where {N,M,A}
#   print(io, "BoseFS{$N,$M}|")
#   r = onr(b)
#   for (i,bn) in enumerate(r)
#     isodd(i) ? print(io, bn) : print(io, "\x1b[4m",bn,"\x1b[0m")
#     # using ANSI escape sequence for underline,
#     # see http://jafrog.com/2013/11/23/colors-in-terminal.html
#     i ≥ M && break
#   end
#   print(io, "⟩")
# end
function Base.show(io::IO, b::BoseFS{N,M,A}) where {N,M,A}
  print(io, "BoseFS")
  if A <: BSAdd64
    print(io, "{BSAdd64}")
  elseif A <: BSAdd128
    print(io, "{BSAdd128}")
  elseif A <: BitAdd
    print(io, "{BitAdd}")
  else
    print(io, "{$A}")
  end
  print(io, "((")
  for (i, on) in enumerate(onr(b))
    print(io, on)
    i ≥ M && break
    print(io, ",")
  end
  print(io, "))")
end

#################################
#
# some functions for operating on addresses
#

"""
    bitaddr(onr, Type)

Calculate a bitstring address from an occupation number representation
the type of the bitstring address is passed as the second argument.
"""
function bitaddr(onr, ::Type{T}) where T<: Integer
  # calculate a bitstring address from an occupation number representation
  # the type of the bitstring address is passed as the second argument
  address = zero(T)
  for ind = length(onr):-1:1
    n = onr[ind]
    address <<= n+1 # shift n+1 zeros
    address |= T(2)^n - one(T) # add block of n 1's
  end
  return address
end

bitaddr(onr, ::Type{BoseFS})  = BoseFS(onr)
bitaddr(onr, ::Type{BoseFS{N,M,A}}) where {N,M,A}  = BoseFS{A}(onr)

function bitaddr(onr, ::Type{BitArray{1}})
  address = BitArray(undef,0)
  for ind = length(onr):-1:1
    for i = 1:onr[ind]
      insert!(address,1,1)
    end
    insert!(address,1,0)
  end
  popfirst!(address)
  return address
end

bitaddr(onr, ::Type{BSAdd64}) = BSAdd64(bitaddr(onr,UInt64))

bitaddr(onr, ::Type{BSAdd128}) = BSAdd128(bitaddr(onr,UInt128))

bitaddr(onr, ::Type{BStringAdd}) = BStringAdd(bitaddr(onr,BitArray{1}))


maxBSLength(T::Type{BSAdd64}) = 64

maxBSLength(T::Type{BSAdd128}) = 128

maxBSLength(T::Type{BStringAdd}) = Inf

"""
    onr(address, m)

Compute and return the occupation number representation as an array of `Int`
corresponding to the given address.
"""
function onr(address::A,mm::Integer) where A<:Union{Integer}
  # compute and return the occupation number representation corresponding to
  # the given address
  # note: it is much faster to pass mm as argument than to access it as global
  # This is the fastest version with 11 seconds for 30,000,000 calls
  # onr = zeros(UInt16,mm) # this limits us to 2^16-1 orbitals
  onr = zeros(Int, mm)
  orbitalnumber = 0
  while !iszero(address)
    orbitalnumber += 1
    bosonnumber = trailing_ones(address)
    # surpsingly it is faster to not check whether this is nonzero and do the
    # following operations anyway
    address >>>= bosonnumber
    # bosonnumber has now the number of bosons in orbtial orbitalnumber
    onr[orbitalnumber] = bosonnumber
    address >>>= 1 # shift right and get ready to look at next orbital
  end # while address
  return onr
end #

onr(address::BSAdd64,mm::Integer) =
    onr(address.add,mm)

onr(address::BSAdd128,mm::Integer) =
    onr(address.add,mm)

# # the most general way of calling this method
# onr(address,p::BosonicHamiltonianParameters) =
#     onr(address,p.M)

function onr(bsadd::BStringAdd, mm::Int)
  #CHANGE FROM BITSTRING representation TO occupation NUMBER representation
  #careful: add[1] is last bit
  onr = zeros(Int,mm)
  address = copy(bsadd.add)
  norb = 0
  i = 1
  while length(address) > 0
    if address[1] == 1
      norb += 1
    else
      onr[i] = norb; i += 1
      norb = 0
    end
    popfirst!(address)
  end
  onr[mm] = norb

  return onr
end #function onr(bsadd::BStringAdd...)

end # module BitStringAddresses
