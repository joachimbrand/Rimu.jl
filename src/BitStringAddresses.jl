```
    BitStringAddresses
Module with types and methods pertaining to bitstring addresses.
```
module BitStringAddresses

export BitStringAddressType, BSAdd64, BSAdd128, BStringAdd, BSAdd
export BSA
export occupationnumberrepresentation, bitaddr, maxBSLength

using StaticArrays
import Base: isless, zero, iszero, show, ==, hash

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
"""
struct BSA{I,B} <: BitStringAddressType
  add::SVector{I,UInt64} # bitstring of `B ≤ I*64` bits, stored as NTuple
end

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
  return BSA{I,nbits}(SVector(adds))
end
# This is really fast because the compiler does all the work:
BSA{I,B}(address::Int) where {I,B} = BSA{I,B}((NTuple{I-1,UInt64}(0 for i in 1:(I-1))..., UInt64(address),))

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
Base.count_ones(a::BSA) = unsafe_count_ones(make_consistent(a))
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
function <<(b::BSA{I,B},n::Integer) where {I,B}
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

function make_consistent(a::BSA{I,B}) where {I,B}
  lp = (B-1) % 64 +1 # bit position of leftmost bit in first chunk
  mask = ~UInt64(0)>>(64-lp)
  madd = MVector(a.add)
  madd[1] &= mask
  return BSA{I,B}(SVector(madd))
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
end

function BoseBS(bs::BSA{I,B}) where {I,B}
  n = count_ones(bs) # number of particles
  m = B - n + 1 # number of orbitals
  I == (B-1) ÷ 64 + 1 || @error "Inconsistency in `BSA{$I,$B}` detected."
  return BoseBS{n,m,I,B}(bs)
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
  println(io, "⟩")
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
