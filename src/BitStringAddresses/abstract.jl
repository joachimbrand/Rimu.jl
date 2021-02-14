"""
    BitStringAddressType

Abstract type for configuration addresses with the nature of a bitstring.
A number of methods need to be implemented:

* Bitwise operations: `&`, `|`, `~`, `⊻`, `<<`, `>>`, `>>>`.

* `iseven` and `isodd`.

* `numChunks`: return the number of chunks the bitstring is split into.

* `numBits`: total number of bits stored in the bitstring.

* `chunks`: return a static array containing all the chunks of the bitstring.

"""
abstract type BitStringAddressType end

"""
    numChunks(a)
Number of 64-bit chunks representing `a`.
"""
numChunks(b) = numChunks(typeof(b))

"""
    numBits(a)
Number of bit chunks representing `a`.
"""
numBits(b) = numBits(typeof(b))

"""
    chunks(a)
Return the chunks in `a`, wrapped in a static array.
"""
chunks

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
numParticles(b) = numParticles(typeof(b))

"""
    numModes(a)
Number of modes represented by `a`.
"""
numModes(b) = numModes(typeof(b))
