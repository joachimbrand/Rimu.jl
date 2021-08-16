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
num_components(::Type{<:AbstractFockAddress}) = 1

"""
    find_mode(address, i)

Find the `i`-th mode in `address`. Returns [`BoseFSIndex`](@ref) for [`BoseFS`](@ref), and
an integer for [`FermiFS`](@ref).
"""
find_mode

"""
    find_occupied_mode(address, k)

Find the `k`-th particle in `address`. Returns [`BoseFSIndex`](@ref) for [`BoseFS`](@ref),
and an integer for [`FermiFS`](@ref).
"""
find_occupied_mode
