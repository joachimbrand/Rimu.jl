# Julia module to contain all type and method definitions
# pertaining to walkers

module Walkers

export WalkerType, wcreate, wisdead, wloc, wnum, wabs, wsign, wsubtract!,
  wadd!, wclone!, wdie!, wkill!, wnumpsips, wnumpsipssquare

export RealWalker
export W3, W4, W5

export deadwalker

export BitStringAddressType, BSAdd64, BSAdd128, BStringAdd
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

Base.zero(BStringAdd) = BStringAdd(BitArray([false]))
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
# Base.zero(BSAdd64) = BSAdd64(0) # This does not work
Base.zero(add::BSAdd64) = BSAdd64(0)
Base.hash(a::BSAdd64, h::UInt) = hash(a.add, h)

"""
    BSAdd128 <: BitStringAddressType

Address type that encodes a bistring address in a UInt64.
"""
struct BSAdd128 <: BitStringAddressType
  add::UInt128
end

Base.isless(a1::BSAdd128,a2::BSAdd128) = isless(a1.add, a2.add)
#Base.typemax(BSAdd128) = BSAdd128(typemax(UInt64))
# Base.zero(BSAdd128) = BSAdd128(0) # This does not work
Base.zero(add::BSAdd128) = BSAdd128(0)
Base.hash(a::BSAdd128, h::UInt) = hash(a.add, h)

"""
    WalkerType

An abstract type for walkers.
"""
abstract type WalkerType end # my own abstract type for walkers


# some methods that should work for all types of walkers
function Base.isless(w1::WalkerType,w2::WalkerType) # comparison for walkers
  #this implementation is fairly generic and should work for all walkers
    wloc(w1) < wloc(w2) ? true : wloc(w1) > wloc(w2) ? false :
        wnum(w1) < wnum(w2) ? true : false
end

function wnumpsips(walkers::AbstractArray{T,1}) where T<:WalkerType
  # return sum(convert(Vector{Int},map(wabs,walkers)))
  wn = 0
  for w in walkers
    wn += wabs(w)
  end
  return wn # devectorised loop for speedup
end

function wnumpsipssquare(walkers::AbstractArray{T,1}) where T<:WalkerType
  wn2 = 0
  for w in walkers
    wn2 += wnum(w)^2
  end
  return wn2
end

function deadwalker(w::T) where T<:WalkerType
  wcreate(zero(wloc(w)),0,T)
end
# this should work for all walker types if zero(address) is defined


"""
    RealWalker <: WalkerType

Abstract type of walker that needs to implement at least the methods
- `wloc(w) <:BitStringAddressType`
- `wnum(w) <:Real`
- `wcreate(l,n,typeof(w)) <: walker` create a walker of this type
"""
abstract type RealWalker <: WalkerType end

# some generic functions that should work for all walkers belonging to this
# abstract type
#

# simplified constructors: make walker of same type with add l and num n
wcreate(w::RealWalker,l::BitStringAddressType,n::Real) = wcreate(l,n,typeof(w))
wcreate(w::RealWalker,n::Integer) = wcreate(w,wloc(w),n)

# check whether a walker is dead;
wisdead(w::RealWalker) = iszero(wnum(w))

wabs(w::RealWalker) = abs(wnum(w)) # extract number without sign
wsign(w::RealWalker) = sign(wnum(w)) # extract sign of walker
# note that the result is zero for a dead walker (num == zero)

function Base.isless(w1::T,w2::T) where T<:RealWalker # comparison for walkers, needed for sorting
    wloc(w1) < wloc(w2) ? true : wloc(w1) > wloc(w2) ? false :
        wnum(w1) < wnum(w2) ? true : false
end

function wclone!(walkers::AbstractArray{T,1},ind) where T<:RealWalker# add a psip same sign
    walkers[ind] = wcreate(walkers[ind],
                      wnum(walkers[ind]) + wsign(walkers[ind]))
end

function wdie!(walkers::AbstractArray{T,1},ind) where T<:RealWalker # remove a psip conserving the sign
    walkers[ind] = wcreate(walkers[ind],
                      wnum(walkers[ind]) - wsign(walkers[ind]))
end

"""
    W3 <: RealWalker <: WalkerType

My own type for walker with separate address and psip number.
The address is now of type `BSAdd64 <: BitStringAddressType`
in order to allow the programming of generic access to bitstring-type
addresses.
"""
struct W3 <: RealWalker  # my own type for walkers with 2 64 bit integers
  loc::BSAdd64 # location in linear space
  num::Int64 # sign and number of walkers
end

wcreate(l::BSAdd64,n::Integer,::Type{W3}) = W3(l,n)
# For convenience, also add a method where you can give the location as an
# integer
#wcreate(l::Integer,n::Integer,::Type{W3}) = W3(BSAdd64(l),n)

wloc(w::W3) = w.loc # method for extracting location
wnum(w::W3) = w.num # method for extracting number (with sign)

"""
    W4 <: RealWalker <: WalkerType

My own type for walker with separate address and psip number.
The address is now of type `BSAdd128 <: BitStringAddressType`
in order to allow the programming of generic access to bitstring-type
addresses.
"""
struct W4 <: RealWalker  # my own type for walkers with 2 64 bit integers
  loc::BSAdd128 # location in linear space
  num::Int64 # sign and number of walkers
end

wcreate(l::BSAdd128,n::Integer,::Type{W4}) = W4(l,n)
# For convenience, also add a method where you can give the location as an
# integer
#wcreate(l::Integer,n::Integer,::Type{W3}) = W3(BSAdd64(l),n)

wloc(w::W4) = w.loc # method for extracting location
wnum(w::W4) = w.num # method for extracting number (with sign)

"""
    W5 <: RealWalker <: WalkerType

My own type for walker with separate address and psip number.
The address is now of type `BStringAdd <: BitStringAddressType`
in order to allow the programming of generic access to bitstring-type
addresses of arbitrary length.
"""
struct W5 <: RealWalker
  loc::BStringAdd
  num::Int64
end

wcreate(l::BStringAdd,n::Integer,::Type{W5}) = W5(l,n)

wloc(w::W5) = w.loc # method for extracting location
wnum(w::W5) = w.num # method for extracting number (with sign)

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

end # module Walkers
