"""
    AbstractFockAddress

Supertype representing a Fock state.
"""
abstract type AbstractFockAddress end

"""
    num_particles(::Type{<:AbstractFockAddress})
    num_particles(::AbstractFockAddress)

Number of particles represented by address.
"""
num_particles(b::AbstractFockAddress) = num_particles(typeof(b))

"""
    num_modes(::Type{<:AbstractFockAddress})
    num_modes(::AbstractFockAddress)

Number of modes represented by address.
"""
num_modes(b::AbstractFockAddress) = num_modes(typeof(b))

"""
    num_components(::Type{<:AbstractFockAddress})
    num_components(::AbstractFockAddress)

Number of components in address.
"""
num_components(b::AbstractFockAddress) = num_components(typeof(b))

"""
    FockAddressIndex

Convenience struct for indexing into a fock state.

Fields:

* `occnum`: the occupation number.
* `site`: the index of the site.
* `offset`: the bit offset of the site.
"""
struct FockAddressIndex <: FieldVector{3,Int}
    occnum::Int
    site::Int
    offset::Int
end
