"""
    AbstractBitString

Abstract type for bitstring storage. Not aware of the physics involved.

* `isless`, `&`, `|`, `~`, `⊻`, `<<`, `>>`, `>>>`, `trailing_zeros`, `trailing_ones`,
 `leading_zeros`, `trailing_zeros`, `count_ones`, `count_zeros`, `bitstring`, `iseven`, and
 `isodd`.

* [`num_chunks`](@ref): return the number of chunks the bitstring is split into.

* [`num_bits`](@ref): total number of bits stored in the bitstring.

* [`chunk_size`](@ref): size of a single chunk

* [`chunks`](@ref): return a static array containing all the chunks of the bitstring.

"""
abstract type AbstractBitString end

"""
    num_chunks(::Type{<:AbstractBitString})
Number of 64-bit chunks representing `a`.
"""
num_chunks(a) = num_chunks(typeof(a))

"""
    num_bits(::Type{<:AbstractBitString})
Number of bit chunks representing `a`.
"""
num_bits(a) = num_bits(typeof(a))

"""
chunk_size(::Type{<:AbstractBitString})
Size of a single chunk in bits.
"""
chunk_size(a) = chunk_size(typeof(a))

"""
    chunks(::AbstractBitString)
Return the chunks of bitstring wrapped in a static array.
"""
chunks

"""
    AbstractFockAddress
Supertype representing a Fock state.

TODO: document interface
"""
abstract type AbstractFockAddress end

"""
    num_particles(::Type{<:AbstractFockAddress})
Number of particles represented by address.
"""
num_particles(b) = num_particles(typeof(b))

"""
    num_modes(::Type{<:AbstractFockAddress})
Number of modes represented by address.
"""
num_modes(b) = num_modes(typeof(b))
