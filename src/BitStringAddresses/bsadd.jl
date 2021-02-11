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
