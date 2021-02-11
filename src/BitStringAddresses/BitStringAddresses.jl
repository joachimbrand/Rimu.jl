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
export bitaddr, maxBSLength # deprecate

include("abstract.jl")
include("bsadd.jl")
include("bitadd.jl")
include("bosefs.jl")

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

maxBSLength(T::Type{BSAdd64}) = 64

maxBSLength(T::Type{BSAdd128}) = 128

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

end
