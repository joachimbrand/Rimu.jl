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
