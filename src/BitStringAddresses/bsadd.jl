"""
    BSAdd64 <: BitStringAddressType

Address type that encodes a bistring address in a UInt64.
"""
struct BSAdd64 <: BitStringAddressType
  add::UInt64
end
BSAdd64(bsa::BSAdd64) = bsa
numBits(::Type{BSAdd64}) = 64

"""
    BSAdd128 <: BitStringAddressType

Address type that encodes a bistring address in a UInt128.
"""
struct BSAdd128 <: BitStringAddressType
  add::UInt128
end
BSAdd128(bsa::BSAdd128) = bsa
BSAdd128(bsa::BSAdd64) = BSAdd128(bsa.add)
numBits(::Type{BSAdd128}) = 128

for T in (BSAdd64, BSAdd128)
    @eval begin
        Base.isless(a1::$T,a2::$T) = isless(a1.add, a2.add)
        Base.zero(::Type{$T}) = $T(0)
        Base.zero(add::$T) = $T(0)
        Base.hash(a::$T, h::UInt) = hash(a.add, h)

        Base.trailing_ones(a::$T) = trailing_ones(a.add)
        Base.trailing_zeros(a::$T) = trailing_zeros(a.add)
        Base.leading_ones(a::$T) = leading_ones(a.add)
        Base.leading_zeros(a::$T) = leading_zeros(a.add)
        Base.count_ones(a::$T) = count_ones(a.add)
        Base.count_zeros(a::$T) = count_zeros(a.add)

        Base.:&(a::$T, b::$T) = $T(a.add & b.add)
        Base.:|(a::$T, b::$T) = $T(a.add | b.add)
        Base.:⊻(a::$T, b::$T) = $T(a.add ⊻ b.add)
        Base.:~(a::$T) = $T(~a.add)
        Base.:>>>(a::$T, n) = $T(a.add >>> n)
        Base.:>>(a::$T, n) = $T(a.add >> n)
        Base.:<<(a::$T, n) = $T(a.add << n)

        Base.iseven(a::$T) = iseven(a.add)
        Base.isodd(a::$T) = isodd(a.add)

        numChunks(::Type{$T}) = 1
        Base.bitstring(a::$T) = bitstring(a.add)
    end
end
