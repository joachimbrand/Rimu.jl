```
    BitStringAddresses
Module with types and methods pertaining to bitstring addresses.
```
module BitStringAddresses

export BitStringAddressType, BSAdd64, BSAdd128, BStringAdd, BSAdd
export occupationnumberrepresentation, bitaddr, maxBSLength

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

import Base: ==
==(a::BStringAdd,b::BStringAdd) = a.add == b.add

Base.hash(a::BStringAdd, h::UInt) = hash(a.add, h)

function Base.show(io::IO, a::BStringAdd)
    print(io, "BStringAdd\"")
    for elem in a.add
      print(io, elem ? "1" : "0")
    end
    print(io, "\"")
end


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

function BSAdd{I,B}(address::Int) where {I,B}
  if I==1 # This is efficient because the compiler can check the condition
    return BSAdd{I,B}((UInt64(address),))
  end

  z = zeros(UInt64, I)
  z[end] = UInt64(address)
  return BSAdd{I,B}(Tuple(z))
end
# BSAdd{I,B}(address::Int) where {I==1,B} = BSAdd{I,B}((UInt64(address),))

BSAdd(address::BSAdd128, nbits=128) = BSAdd(address.add, nbits)
BSAdd(address::BSAdd64, nbits=64) = BSAdd{1,nbits}((address.add,))

Base.zero(::BSAdd{I,B}) where {I,B} = BSAdd{I,B}(Tuple(zeros(UInt64,I)))

# comparison check number of bits and then compares the tuples
Base.isless(a::BSAdd{I,B}, b::BSAdd{I1,B}) where {I,I1,B} = isless(a.add, b.add)
function Base.isless(a::BSAdd{I1,B1}, b::BSAdd{I2,B2}) where {I1,B1,I2,B2}
  return isless(B1,B2)
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
function occupationnumberrepresentation(address::Integer,mm::Integer)
  # compute and return the occupation number representation corresponding to
  # the given address
  # note: it is much faster to pass mm as argument than to access it as global
  # This is the fastest version with 11 seconds for 30,000,000 calls
  # onr = zeros(UInt16,mm) # this limits us to 2^16-1 orbitals
  onr = zeros(Int, mm)
  orbitalnumber = 0
  while address > 0
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
