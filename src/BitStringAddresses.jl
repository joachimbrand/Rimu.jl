```
    BitStringAddresses
Module with types and methods pertaining to bitstring addresses.
```
module BitStringAddresses

using StaticArrays
using Base.Cartesian

import Base: isless, zero, iszero, show, ==, hash

export BitStringAddressType, BSAdd64, BSAdd128
export BitAdd, BoseBA, onr, BoseFS
# export nbits, nchunks, nparticles, nmodes # consider
export BStringAdd, BSAdd, BSA, BoseBS # deprecate
export occupationnumberrepresentation, bitaddr, maxBSLength # deprecate


"""
    BitStringAddressType

Abstract type for configuration addresses with the nature of a bitstring.
A number of methods need to be implemented, in particular
Base.isless(a,b)
"""
abstract type BitStringAddressType end

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
import Base: <<, >>>, >>, ⊻, &, |
(>>>)(a::BSAdd64, n::Integer) = BSAdd64(a.add >>> n)

nchunks(::Type{BSAdd64}) = 1
nbits(::Type{BSAdd64}) = 64
Base.bitstring(a::BSAdd64) = bitstring(a.add)

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
import Base: <<, >>>, >>, ⊻, &, |
(>>>)(a::BSAdd128, n::Integer) = BSAdd128(a.add >>> n)

nchunks(::Type{BSAdd128}) = 1
nbits(::Type{BSAdd128}) = 128
Base.bitstring(a::BSAdd128) = bitstring(a.add)

"""
    BSAdd{I,B} <: BitStringAddressType
    BSAdd(address, B)

Address type that encodes a bistring address with `B` bits. The bits are
stored as `NTuple` of `I` integers (`UInt64`) with `B ≤ I*64`. The `address`
can be an integer, e.g. `BigInt`.
"""
struct BSAdd{I,B} <: BitStringAddressType
  add::NTuple{I,UInt64} # bitstring of `B ≤ I*64` bits, stored as NTuple
end

# general default constructor - works with BigInt - but is quite slow
# avoid using in hot loops!
@inline function BSAdd(address::Integer, nbits::Integer)
  @boundscheck nbits < 1 && throw(BoundsError())
  a = copy(address)
  I = (nbits-1) ÷ 64 + 1 # number of UInt64s needed
  adds = zeros(UInt64,I)
  for i in I:-1:1
     adds[i] = UInt64(a & 0xffffffffffffffff)
     # extract rightmost 64 bits and store in adds
     a >>>= 64 # shift bits to right by 64 bits
  end
  return BSAdd{I,nbits}(Tuple(adds))
end

# This is really fast because the compiler does all the work:
BSAdd{I,B}(address::Int) where {I,B} = BSAdd{I,B}((NTuple{I-1,UInt64}(0 for i in 1:(I-1))..., UInt64(address),))

BSAdd(address::BSAdd128, nbits=128) = BSAdd(address.add, nbits)
BSAdd(address::BSAdd64, nbits=64) = BSAdd{1,nbits}((address.add,))

Base.zero(::BSAdd{I,B}) where {I,B} = BSAdd{I,B}(0)

# comparison check number of bits and then compares the tuples
Base.isless(a::BSAdd{I,B}, b::BSAdd{I1,B}) where {I,I1,B} = isless(a.add, b.add)
function Base.isless(a::BSAdd{I1,B1}, b::BSAdd{I2,B2}) where {I1,B1,I2,B2}
  return isless(B1,B2)
end

import Base: >>>, <<
"""
    >>>(b::BSAdd,n::Integer)
Bitshift `b` to the right by `n` bits and fill from the left with zeros.
"""
function >>>(b::BSAdd{I,B},n::Integer) where {I,B}
  if I == 1
    return BSAdd{I,B}((b.add[1]>>>n,))
  end
  d, r = divrem(n,64) # shift by `d` chunks and `r` bits
  mask = ~0 >>> (64-r) # 2^r-1 # 0b0...01...1 with `r` 1s
  a = zeros(UInt64,I)
  I-d > 0 && (a[d+1] = b.add[1]>>>r) # no carryover for leftmost chunk
  for i = 2:(I-d) # shift chunks and `or` carryover
    a[d+i] = (b.add[i]>>>r) | ((b.add[i-1] & mask) << (64-r) )
  end
  return BSAdd{I,B}(Tuple(a))
  ## does the same thing in one line, but is slower:
  # return BSAdd{I,B}((NTuple{d,UInt64}(0 for i in 1:d)...,b.add[1]>>r,
  #   NTuple{I-d-1,UInt64}((b.add[i]>>>r)|((b.add[i-1]&mask)<<(64-r))
  #     for i in 2:(I-d))...,
  #   )
  # )
end
"""
    <<(b::BSAdd,n::Integer)
Bitshift `b` to the left by `n` bits and fill from the right with zeros.
"""
function <<(b::BSAdd{I,B},n::Integer) where {I,B}
  if I == 1
    return BSAdd{I,B}((b.add[1]<<n,))
  end
  d, r = divrem(n,64) # shift by `d` chunks and `r` bits
  mask = ~0 << (64-r) # (2^r-1) << (64-r) # 0b1...10...0 with `r` 1s
  a = zeros(UInt64,I)
  for i in 1:(I-d-1) # shift chunks and `or` carryover
    a[i] = (b.add[i+d]<<r) | ((b.add[i+d+1] & mask) >>> (64-r))
  end
  I-d > 0 && (a[I-d] = b.add[I]<<r) # no carryover for rightmost chunk
  return BSAdd{I,B}(Tuple(a))
end

#######################################
"""
    BSA{I,B} <: BitStringAddressType
    BSA(address, B)

Address type that encodes a bistring address with `B` bits. The bits are
stored as `SVector` of `I` integers (`UInt64`) with `B ≤ I*64`. The `address`
can be an integer, e.g. `BigInt`.

- `BSA{I,B}()` creates a BSA with all ones.
- `BSA{I,B}(add::Int)` creates BSA quickly and efficiently, where all non-zero bits are represented in `add`.
"""
struct BSA{I,B} <: BitStringAddressType
  add::SVector{I,UInt64} # bitstring of `B ≤ I*64` bits, stored as NTuple
  # end
  # inner contructor: only allow passing `B` to enforce consistency
  function BSA{B}(chunks::SVector{I,UInt64}) where {B,I}
    I == (B - 1) ÷ 64 + 1 || error("in construction of `BSA{$I,$B}`: (B - 1) ÷ 64 + 1 evaluates to $((B - 1) ÷ 64 + 1) but should be I == $I")
    return new{I,B}(chunks)
  end
end
BSA{I,B}(chunks::SVector{I,UInt64}) where {B,I} = BSA{B}(chunks)
BSA{I,B}(chunks) where {B,I} = BSA{B}(SVector(chunks))
BSA{B}(chunks) where B = BSA{B}(SVector(chunks))

# general default constructor - works with BigInt - but is quite slow
# avoid using in hot loops!
@inline function BSA(address::Integer, nbits::Integer)
  @boundscheck nbits < 1 && throw(BoundsError())
  a = copy(address)
  I = (nbits-1) ÷ 64 + 1 # number of UInt64s needed
  adds = zeros(MVector{I,UInt64})
  for i in I:-1:1
     adds[i] = UInt64(a & 0xffffffffffffffff)
     # extract rightmost 64 bits and store in adds
     a >>>= 64 # shift bits to right by 64 bits
  end
  bsa = BSA{I,nbits}(SVector(adds))
  @boundscheck check_consistency(bsa)
  return bsa
end
@inline function BSA{B}(a::Integer) where B
  @boundscheck B < 1 && throw(BoundsError())
  I = (B-1) ÷ 64 + 1 # number of UInt64s needed
  adds = zeros(MVector{I,UInt64})
  for i in I:-1:1
     adds[i] = UInt64(a & 0xffffffffffffffff)
     # extract rightmost 64 bits and store in adds
     a >>>= 64 # shift bits to right by 64 bits
  end
  bsa = BSA{B}(SVector(adds))
  @boundscheck check_consistency(bsa)
  return bsa
end

# This is really fast because the compiler does all the work:
# BSA{I,B}(address::Int) where {I,B} = BSA{I,B}((NTuple{I-1,UInt64}(0 for i in 1:(I-1))..., UInt64(address),))

@inline function BSA{I,B}(address::Int) where {I,B}
  bsa = BSA{I,B}((NTuple{I-1,UInt64}(0 for i in 1:(I-1))..., UInt64(address),))
  @boundscheck check_consistency(bsa)
  return bsa
end
@inline function BSA{B}(address::Int) where B
  I = (B-1) ÷ 64 + 1 # number of UInt64s needed
  bsa = BSA{I,B}((NTuple{I-1,UInt64}(0 for i in 1:(I-1))..., UInt64(address),))
  @boundscheck check_consistency(bsa)
  return bsa
end

# create a BSA with all ones
function BSA{I,B}() where {I,B}
  (B-1) ÷ 64 + 1 == I || error("Inconsistent I = $I with B = $B")
  r = (B-1) % 64 + 1
  first = ~UInt64(0) >>> (64 - r)
  BSA{I,B}((first, NTuple{I-1,UInt64}(~UInt64(0) for i in 1:(I-1))...,))
end

BSA(address::BSAdd128, nbits=128) = BSA(address.add, nbits)
BSA(address::BSAdd64, nbits=64) = BSA{1,nbits}((address.add,))

Base.zero(::BSA{I,B}) where {I,B} = BSA{I,B}(0)

# comparison check number of bits and then compares the tuples
Base.isless(a::BSA{I,B}, b::BSA{I,B}) where {I,B} = isless(a.add, b.add)
function Base.isless(a::BSA{I1,B1}, b::BSA{I2,B2}) where {I1,B1,I2,B2}
  return isless(B1,B2)
end
import Base: <<, >>>, >>, ⊻, &, |
⊻(a::BSA{I,B}, b::BSA{I,B}) where {I,B} = BSA{I,B}(a.add .⊻ b.add)
(&)(a::BSA{I,B}, b::BSA{I,B}) where {I,B} = BSA{I,B}(a.add .& b.add)
(|)(a::BSA{I,B}, b::BSA{I,B}) where {I,B} = BSA{I,B}(a.add .| b.add)

unsafe_count_ones(a::BSA) = mapreduce(count_ones, +, a.add)
Base.count_ones(a::BSA) = unsafe_count_ones(remove_ghost_bits(a))
# takes about the same time:
# function co2(a::BSA)
#   s = 0
#   for n in a.add
#     s += count_ones(n)
#   end
#   return s
# end

Base.count_zeros(a::BSA{I,B}) where {I,B} = B - count_ones(a)

"""
    >>>(b::BSA,n::Integer)
Bitshift `b` to the right by `n` bits and fill from the left with zeros.
"""
function >>>(b::BSA{I,B},n::Integer) where {I,B}
  if I == 1
    return BSA{I,B}((b.add[1]>>>n,))
  end
  d, r = divrem(n,64) # shift by `d` chunks and `r` bits
  mask = ~0 >>> (64-r) # 2^r-1 # 0b0...01...1 with `r` 1s
  a = zeros(MVector{I,UInt64})
  I-d > 0 && (a[d+1] = b.add[1]>>>r) # no carryover for leftmost chunk
  for i = 2:(I-d) # shift chunks and `or` carryover
    a[d+i] = (b.add[i]>>>r) | ((b.add[i-1] & mask) << (64-r) )
  end
  return BSA{I,B}(SVector(a))
end

(>>)(b::BSA,n::Integer) = b >>> n

"""
    <<(b::BSA,n::Integer)
Bitshift `b` to the left by `n` bits and fill from the right with zeros.
"""
<<(b::BSA{I,B},n::Integer) where {I,B} = remove_ghost_bits(unsafe_shift_left(b,n))

function unsafe_shift_left(b::BSA{I,B},n::Integer) where {I,B}
  if I == 1
    return BSA{I,B}((b.add[1]<<n,))
  end
  d, r = divrem(n,64) # shift by `d` chunks and `r` bits
  mask = ~0 << (64-r) # (2^r-1) << (64-r) # 0b1...10...0 with `r` 1s
  a = zeros(MVector{I,UInt64})
  for i in 1:(I-d-1) # shift chunks and `or` carryover
    a[i] = (b.add[i+d]<<r) | ((b.add[i+d+1] & mask) >>> (64-r))
  end
  I-d > 0 && (a[I-d] = b.add[I]<<r) # no carryover for rightmost chunk
  return BSA{I,B}(SVector(a))
end
# This is faster than << for BitArray (yeah!).
# About half the time and allocations.

function Base.trailing_ones(a::BSA{I,B}) where {I,B}
  t = 0
  for chunk in reverse(a.add)
    s = trailing_ones(chunk)
    t += s
    s < 64 && break
  end
  return min(t, B)
end

function Base.trailing_zeros(a::BSA{I,B}) where {I,B}
  t = 0
  for chunk in reverse(a.add)
    s = trailing_zeros(chunk)
    t += s
    s < 64 && break
  end
  return min(t, B)
end

function lead_bit(a::BSA{I,B}) where {I,B}
  lp = (B-1) % 64 # bit position of leftmost bit in first chunk (count from 0)
  mask = UInt64(1)<< lp # shift a "1" to the right place
  return (a.add[1]&mask) >> lp # return type is UInt64
end

"""
    remove_ghost_bits(bsa)
Remove set bits outside data field if any are present.
"""
function remove_ghost_bits(a::BSA{I,B}) where {I,B}
  lp = (B-1) % 64 +1 # bit position of leftmost bit in first chunk
  mask = ~UInt64(0)>>(64-lp)
  madd = MVector(a.add)
  madd[1] &= mask
  return BSA{I,B}(SVector(madd))
end

function check_consistency(a::BSA{I,B}) where {I,B}
  (d,lp) = divrem((B-1), 64) .+ 1 # bit position of leftmost bit in first chunk
  mask = ~UInt64(0)>>(64-lp)
  iszero(a.add[1] & ~mask) || error("ghost bits detected in $a")
  d == I || error("inconsistency in $a: $d words needed but $I present")
  length(a.add) == I || error("inconsistent length $(length(a.add)) with I = $I in $a")
  nothing
end


# # iterate through bits from right to left
# @inline function Base.iterate(a::BSA{I,B}, i = 1) where {I,B}
#   d,r = divrem(i-1, 64)
#   i == B+1 ? nothing : ((a.add[I-d]&UInt64(1)<<r)>>r, i+1)
# end
# iterate through bits from left to right
@inline function Base.iterate(a::BSA{I,B}, i = 1) where {I,B}
  i == B+1 ? nothing : (UInt(lead_bit(a<<(i-1))), i+1)
end

# show prints bits from left to right, i.e. in reverse order
function Base.show(io::IO, a::BSA{I,B}) where {I,B}
    print(io, "BSA{$B}\"")
    for elem ∈ a
      print(io, elem)
    end
    print(io, "\"")
end

"""
    BoseBS{N,M,I,B} <: BitStringAddressType
    BoseBS(bs::BSA)

Address type that represents `N` spinless bosons in `M` orbitals by wrapping
a `BSA{I,B}` bitstring. In the bitstring `N` ones represent `N` particles and
`M-1` zeros separate `M` orbitals. Hence the total number of bits is
`B == N+M-1` (and `I` is the number of `UInt64` words used internally to store
the bitstring.). Orbitals are stored in reverse
order, i.e. the first orbital in a `BoseBS` is stored rightmost in the `BSA`
bitstring.
"""
struct BoseBS{N,M,I,B} <: BitStringAddressType
  bs::BSA{I,B}

  # inner constructor does type checking at compile time
  function BoseBS{N,M,I,B}(bs::BSA{I,B}) where {N,M,I,B}
    I == (B-1) ÷ 64 + 1 || error("Inconsistency in `BoseBS{$N,$M,$I,$B}` detected.")
    M + N == B + 1 || error("Inconsistency in `BoseBS{$N,$M,$I,$B}` detected.")
    return new{N,M,I,B}(bs)
  end
end

# type unstable and slow - it is faster to use the native constructor -:
function BoseBS(bs::BSA{I,B}) where {I,B}
  n = count_ones(bs) # number of particles
  m = B - n + 1 # number of orbitals
  I == (B-1) ÷ 64 + 1 || @error "Inconsistency in `BSA{$I,$B}` detected."
  return BoseBS{n,m,I,B}(bs)
end

# slow due to type instability
function BoseBS(onr::AbstractVector{T}) where T<:Integer
  m = length(onr)
  n = Int(sum(onr))
  b = n + m - 1
  i = (b-1) ÷ 64 +1
  bs = BSA{i,b}(0) # empty bitstring
  for on in reverse(onr)
    bs <<= on+1
    bs |= BSA{i,b}()>>(b-on)
  end
  return BoseBS{n,m,i,b}(bs)
end

# typestable and quite fast (with SVector faster than with Vector)
function BoseBS{N,M,I,B}(onr::AbstractVector{T}) where {N,M,I,B,T<:Integer}
  M ≥ length(onr) || error("M inconsistency")
  N == Int(sum(onr)) || error("N inconsistency")
  B == N + M - 1 ||  error("B inconsistency")
  I == (B-1) ÷ 64 +1 ||  error("I inconsistency")
  bs = BSA{I,B}(0) # empty bitstring
  for on in reverse(onr)
    bs <<= on+1
    bs |= BSA{I,B}()>>(B-on)
  end
  return BoseBS{N,M,I,B}(bs)
end

function check_consistency(b::BoseBS{N,M,I,B}) where {N,M,I,B}
  N+M-1 == B || error("Inconsistency in $b: N+M-1 = $(N+M-1), B = $B")
  check_consistency(b.bs)
end

function Base.show(io::IO, b::BoseBS{N,M,I,B}) where {N,M,I,B}
  print(io, "BoseBS{$N,$M}|")
  onr = occupationnumberrepresentation(b.bs,M)
  for (i,bn) in enumerate(onr)
    isodd(i) ? print(io, bn) : print(io, "\x1b[4m",bn,"\x1b[0m")
    # using ANSI escape sequence for underline,
    # see http://jafrog.com/2013/11/23/colors-in-terminal.html
    i ≥ M && break
  end
  ## or separate occupation numbers with commas
  # for (i,bn) in enumerate(onr)
  #   print(io, bn)
  #   i ≥ M && break
  #   print(io, ",")
  # end
  print(io, "⟩")
end

#################################
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

# """
#     remove_ghost_bits(bsa)
# Remove set bits outside data field if any are present.
# """
function remove_ghost_bits(a::BitAdd{I,B}) where {I,B}
  lp = (B-1) % 64 +1 # bit position of leftmost bit in first chunk
  mask = ~UInt64(0)>>(64-lp)
  madd = MVector(a.chunks)
  madd[1] &= mask
  return BitAdd{B}(SVector(madd))
end

nchunks(T::Type) = @error "not implemented: nchunks($T)"
nchunks(::Type{BitAdd{I,B}}) where {I,B} = I
nchunks(b) = nchunks(typeof(b))
nbits(T::Type) = @error "not implemented: nbits($T)"
nbits(::Type{BitAdd{I,B}}) where {I,B} = B
nbits(b) = nbits(typeof(b))

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
function unsafe_shift_left(b::BitAdd{I,B},n::Integer) where {I,B}
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
  if B < 30
    print(io, "BitAdd{$B}|",bitstring(ba), "⟨")
  else
    bs = bitstring(ba)
    print(io, "BitAdd{$B}|",bs[1:10]," … ",bs[end-10:end], "⟨")
  end
  nothing
end

"""
    BoseBA{N,M} <: BitStringAddressType
    BoseBA(bs::BitAdd)

Address type that represents `N` spinless bosons in `M` orbitals by wrapping
a `BitAdd{I,B}` bitstring. In the bitstring `N` ones represent `N` particles and
`M-1` zeros separate `M` orbitals. Hence the total number of bits is
`B == N+M-1` (and `I` is the number of `UInt64` words used internally to store
the bitstring.). Orbitals are stored in reverse
order, i.e. the first orbital in a `BoseBA` is stored rightmost in the `BitAdd`
bitstring.
"""
struct BoseBA{N,M,I,B} <: BitStringAddressType
  bs::BitAdd{I,B}

  # inner constructor does type checking at compile time
  function BoseBA{N,M,I,B}(bs::BitAdd{I,B}) where {N,M,I,B}
    I == (B-1) ÷ 64 + 1 || error("Inconsistency in `BoseBA{$N,$M,$I,$B}` detected.")
    M + N == B + 1 || error("Inconsistency in `BoseBA{$N,$M,$I,$B}` detected.")
    return new{N,M,I,B}(bs)
  end
end

# type unstable and slow - it is faster to use the native constructor -:
function BoseBA(bs::BitAdd{I,B}) where {I,B}
  n = count_ones(bs) # number of particles
  m = B - n + 1 # number of orbitals
  I == (B-1) ÷ 64 + 1 || @error "Inconsistency in `BitAdd{$I,$B}` detected."
  return BoseBA{n,m,I,B}(bs)
end

# slow due to type instability
function BoseBA(onr::AbstractVector{T}) where T<:Integer
  m = length(onr)
  n = Int(sum(onr))
  b = n + m - 1
  i = (b-1) ÷ 64 +1
  return BoseBA{n,m,i,b}(onr)
end

# typestable and quite fast (with SVector faster than with Vector)
function BoseBA{N,M,I,B}(onr::AbstractVector{T}) where {N,M,I,B,T<:Integer}
  M ≥ length(onr) || error("M inconsistency")
  N == Int(sum(onr)) || error("N inconsistency")
  B == N + M - 1 ||  error("B inconsistency")
  I == (B-1) ÷ 64 +1 ||  error("I inconsistency")
  bs = BitAdd{B}(0) # empty bitstring
  for on in reverse(onr)
    bs <<= on+1
    bs |= BitAdd{B}()>>(B-on)
  end
  return BoseBA{N,M,I,B}(bs)
end

# create a BoseBA address with near uniform dist
# slow due to type instability
"""
    BoseBA(n::Integer, m::Integer)
Create `BoseBA` address with near uniform distribution of `n` particles
across `m` modes.
"""
function BoseBA(n::Integer, m::Integer)
  fillingfactor, extras = divrem(n, m)
  startonr = fill(fillingfactor,m)
  startonr[1:extras] += ones(Int, extras)
  return BoseBA(startonr)
end

function check_consistency(b::BoseBA{N,M,I,B}) where {N,M,I,B}
  N+M-1 == B || error("Inconsistency in $b: N+M-1 = $(N+M-1), B = $B")
  check_consistency(b.bs)
end

"""
    onr(bs)
Compute and return the occupation number representation of the bit string
address `bs` as an `SVector{M,Int}`, where `M` is the number of orbitals.
"""
function onr(bba::BoseBA{N,M,I,B}) where {N,M,I,B} # fast
  r = zeros(MVector{M,Int})
  address = bba.bs
  for orbitalnumber in 1:M
    bosonnumber = trailing_ones(address)
    r[orbitalnumber] = bosonnumber
    address >>>= bosonnumber + 1
    iszero(address) && break
  end
  return SVector(r)
end

function Base.show(io::IO, b::BoseBA{N,M,I,B}) where {N,M,I,B}
  print(io, "BoseBA{$N,$M}|")
  r = onr(b)
  for (i,bn) in enumerate(r)
    isodd(i) ? print(io, bn) : print(io, "\x1b[4m",bn,"\x1b[0m")
    # using ANSI escape sequence for underline,
    # see http://jafrog.com/2013/11/23/colors-in-terminal.html
    i ≥ M && break
  end
  ## or separate occupation numbers with commas
  # for (i,bn) in enumerate(onr)
  #   print(io, bn)
  #   i ≥ M && break
  #   print(io, ",")
  # end
  print(io, "⟩")
end

Base.bitstring(b::BoseBA) = bitstring(b.bs)

nchunks(::Type{BoseBA{N,M,I,B}}) where {N,M,I,B} = I
nbits(::Type{BoseBA{N,M,I,B}}) where {N,M,I,B} = B
nparticles(::Type{BoseBA{N,M,I,B}}) where {N,M,I,B} = N
nmodes(::Type{BoseBA{N,M,I,B}}) where {N,M,I,B} = M

nparticles(T::Type) = @error "not implemented: nparticles($T)"
nparticles(b) = nparticles(typeof(b))
nmodes(T::Type) = @error "not implemented: nmodes($T)"
nmodes(b) = nmodes(typeof(b))

# comparison delegates to BitAdd
Base.isless(a::BoseBA, b::BoseBA) = isless(a.bs, b.bs)
# hashing delegates to BitAdd
Base.hash(bba::BoseBA,  h::UInt) = hash(bba.bs, h)

#################################
"""
    BoseFS{N,M,A} <: BitStringAddressType
    BoseFS(bs::A, N = nparticles(A), M = nmodes(A))

Address type that represents a Fock state of `N` spinless bosons in `M` orbitals
by wrapping a bitstring of type `A`. Orbitals are stored in reverse
order, i.e. the first orbital in a `BoseBA` is stored rightmost in the
bitstring `bs`.
"""
struct BoseFS{N,M,A} <: BitStringAddressType
  bs::A
end

BoseFS{N,M}(bs::A) where {N,M,A} = BoseFS{N,M,A}(bs) # slow - not sure why

function BoseFS(bs::A, n=nparticles(bs), m=nmodes(bs)) where A <: BitStringAddressType
  bfs = BoseFS{n,m,A}(bs)
  check_consistency(bfs)
  return bfs
end

function BoseFS(bs::BitAdd{I,B}) where {B,I}
  n = count_ones(bs)
  m = B - n + 1
  return BoseFS{n,m,BitAdd{I,B}}(bs)
end
# create a BoseBA address with near uniform dist
# slow due to type instability
"""
    BoseFS(n::Integer, m::Integer[, BST::Type])
Create `BoseFS` address with near uniform distribution of `n` particles
across `m` modes. If a type `BST` is given it will define the underlying
bit string type. Otherwise, the bit string type is chosen automatically.
"""
function BoseFS(n::Integer, m::Integer, ::Type{T}) where T
  fillingfactor, extras = divrem(n, m)
  startonr = fill(fillingfactor,m)
  startonr[1:extras] += ones(Int, extras)
  return BoseFS{T}(startonr)
end

function BoseFS(n::Integer, m::Integer)
  fillingfactor, extras = divrem(n, m)
  startonr = fill(fillingfactor,m)
  startonr[1:extras] += ones(Int, extras)
  return BoseFS(startonr)
end

# function BoseFS(n::Integer, m::Integer)
#   bits = n+m-1
#   if bits ≤ 64
#     return BoseFS(n, m, BSAdd64)
#   elseif bits ≤ 128
#     return BoseFS(n, m, BSAdd128)
#   else
#     return BoseFS(n, m, BitAdd)
#   end
# end
# slow due to type instability
# function BoseFS(onr::AbstractVector{T}, ::Type{BitAdd}) where {T<:Integer}
#   m = length(onr)
#   n = Int(sum(onr))
#   b = n + m - 1
#   bs = BitAdd{b}(0) # empty bitstring
#   for on in reverse(onr)
#     bs <<= on+1
#     bs |= BitAdd{b}()>>(b-on)
#   end
#   return BoseFS(bs)
#   # i = (b-1) ÷ 64 +1
#   # return BoseFS{n,m,i,b}(onr)
# end
# function BoseFS(onr::AbstractVector{T}, ::Type{BSAdd128}) where {T<:Integer}
#   m = length(onr)
#   n = Int(sum(onr))
#   b = n + m - 1
#   b > 128 && throw(BoundsError(BSAdd128(0),b))
#   bs = zero(UInt128) # empty bitstring
#   for on in reverse(onr)
#     bs <<= on+1
#     bs |= ~zero(UInt128)>>(128-on)
#   end
#   return BoseFS{n,m,BSAdd128}(BSAdd128(bs))
# end
# function BoseFS(onr::AbstractVector{T}, ::Type{BSAdd64}) where {T<:Integer}
#   m = length(onr)
#   n = Int(sum(onr))
#   b = n + m - 1
#   b > 64 && throw(BoundsError(BSAdd64(0),b))
#   bs = zero(UInt64) # empty bitstring
#   for on in reverse(onr)
#     bs <<= on+1
#     bs |= ~zero(UInt64)>>(64-on)
#   end
#   return BoseFS{n,m,BSAdd64}(BSAdd64(bs))
# end

# BoseFS{A}(onr::AbstractVector) where A = BoseFS(onr,A)

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
@inline function BoseFS{BSAdd64}(onr::T,::Val{N},::Val{M},::Val{B}) where {N,M,B,T<:Union{AbstractVector,Tuple}}
  @boundscheck begin
    B > 64 && throw(BoundsError(BSAdd64(0),B))
    N + M - 1 == B || @error "Inconsistency in constructor BoseBS"
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
    N + M - 1 == B || @error "Inconsistency in constructor BoseBS"
  end
  bs = zero(UInt128) # empty bitstring
  for on in reverse(onr)
    bs <<= on+1
    bs |= ~zero(UInt128)>>(128-on)
  end
  return BoseFS{N,M,BSAdd128}(BSAdd128(bs))
end

@inline function BoseFS{BitAdd}(onr::T,::Val{N},::Val{M},::Val{B}) where {N,M,B,T<:Union{AbstractVector,Tuple}}
  @boundscheck  N + M - 1 == B || @error "Inconsistency in constructor BoseBS"
  bs = BitAdd{B}(0) # empty bitstring
  for on in reverse(onr)
    bs <<= on+1
    bs |= BitAdd{B}()>>(B-on)
  end
  I = (B-1) ÷ 64 + 1 # number of UInt64s needed
  return BoseFS{N,M,BitAdd{I,B}}(bs)
end

# # typestable and quite fast (with SVector faster than with Vector)
# function BoseFS{N,M,I,B}(onr::AbstractVector{T}) where {N,M,I,B,T<:Integer}
#   M ≥ length(onr) || error("M inconsistency")
#   N == Int(sum(onr)) || error("N inconsistency")
#   B == N + M - 1 ||  error("B inconsistency")
#   I == (B-1) ÷ 64 +1 ||  error("I inconsistency")
#   bs = BitAdd{B}(0) # empty bitstring
#   for on in reverse(onr)
#     bs <<= on+1
#     bs |= BitAdd{B}()>>(B-on)
#   end
#   return BoseFS{N,M,BitAdd{I,B}}(bs)
# end



# comparison delegates to bs
Base.isless(a::BoseFS, b::BoseFS) = isless(a.bs, b.bs)
# hashing delegates to bs
Base.hash(bba::BoseFS,  h::UInt) = hash(bba.bs, h)
Base.bitstring(b::BoseFS) = bitstring(b.bs)
nchunks(::Type{BoseFS{N,M,A}}) where {N,M,A} = nchunks(A)
nbits(::Type{BoseFS{N,M,A}}) where {N,M,A} = N+M-1 # generally true for bosons
nparticles(::Type{BoseFS{N,M,A}}) where {N,M,A} = N
nmodes(::Type{BoseFS{N,M,A}}) where {N,M,A} = M

function check_consistency(b::BoseFS{N,M,A}) where {N,M,A}
  nbits(b) ≤ nbits(A) || error("Inconsistency in $b: N+M-1 = $(N+M-1), nbits(A) = $(nbits(A))")
  check_consistency(b.bs)
end

# performant and allocation free (if benchmarked on its own):
function onr(bba::BoseFS{N,M,A}) where {N,M,A}
  r = zeros(MVector{M,Int})
  address = bba.bs
  for orbitalnumber in 1:M
    bosonnumber = trailing_ones(address)
    r[orbitalnumber] = bosonnumber
    address >>>= bosonnumber + 1
    iszero(address) && break
  end
  return SVector(r)
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

bitaddr(onr, ::Type{BoseBA{N,M,I,B}}) where {N,M,I,B} = BoseBA{N,M,I,B}(onr)
bitaddr(onr, ::Type{BoseBA})  = BoseBA(onr)

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
    occupationnumberrepresentation(address, m)

Compute and return the occupation number representation as an array of `Int`
corresponding to the given address.
"""
function occupationnumberrepresentation(address::A,mm::Integer) where A<:Union{Integer, BSA}
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

occupationnumberrepresentation(address::BSAdd64,mm::Integer) =
    occupationnumberrepresentation(address.add,mm)

occupationnumberrepresentation(address::BSAdd128,mm::Integer) =
    occupationnumberrepresentation(address.add,mm)

# # the most general way of calling this method
# occupationnumberrepresentation(address,p::BosonicHamiltonianParameters) =
#     occupationnumberrepresentation(address,p.M)

function occupationnumberrepresentation(bsadd::BStringAdd, mm::Int)
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
end #function occupationnumberrepresentation(bsadd::BStringAdd...)

end # module BitStringAddresses
