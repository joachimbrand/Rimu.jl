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
    SingleComponentFockAddress{M}

A type representing a single component Fock state with `M` modes.

Implemented subtypes: [`BoseFS`](@ref), [`FermiFS`](@ref).

# Supported functionality

* [`find_mode`](@ref)
* [`find_occupied_mode`](@ref)
* [`num_occupied_modes`](@ref)
* [`occupied_modes`](@ref)
* [`move_particle`](@ref)

"""
abstract type SingleComponentFockAddress{M} <: AbstractFockAddress end

"""
    find_mode(::SingleComponentFockAddress, i)

Find the `i`-th mode in address. Returns [`BoseFSIndex`](@ref) for [`BoseFS`](@ref), and
an integer for [`FermiFS`](@ref).

```jldoctest
julia> find_mode(BoseFS((1, 0, 2)), 2)
3-element Rimu.BitStringAddresses.BoseFSIndex with indices SOneTo(3):
 0
 2
 2

julia> find_mode(FermiFS((1, 1, 1, 0)), 2)
2
```
"""
find_mode

"""
    find_occupied_mode(::SingleComponentFockAddress, k)

Find the `k`-th occupied mode in address. Returns [`BoseFSIndex`](@ref) for
[`BoseFS`](@ref), and an integer for [`FermiFS`](@ref).

# Example

```jldoctest
julia> find_occupied_mode(BoseFS((1, 0, 2)), 2)
3-element Rimu.BitStringAddresses.BoseFSIndex with indices SOneTo(3):
 2
 3
 3

julia> find_occupied_mode(FermiFS((1, 1, 1, 0)), 2)
2
```
"""
find_occupied_mode

"""
    num_occupied_modes(::SingleComponentFockAddress)

Get the number of occupied modes in address. Equivalent to
`length(`[`occupied_modes`](@ref)`(address))`, or the number of non-zeros in its ONR
representation.

# Example

```jldoctest
julia> num_occupied_modes(BoseFS((1, 0, 2)))
2
julia> num_occupied_modes(FermiFS((1, 1, 1, 0)))
3
```
"""
num_occupied_modes

"""
    occupied_modes(::SingleComponentFockAddress)

Iterate over all occupied modes in an address. Iterates [`BoseFSIndex`](@ref) for
[`BoseFS`](@ref), and an integers for [`FermiFS`](@ref).

# Example

```jldoctest
julia> b = BoseFS((1,5,0,4))
julia> for (n, i, m) in occupied_modes(b)
    @show n, i, m
end
(n, i, m) = (1, 1, 0)
(n, i, m) = (5, 2, 2)
(n, i, m) = (4, 4, 9)
```

```jldoctest
julia> f = FermiFS((1,1,0,1,0,0,1))
julia> for i in occupied_modes(f)
    @show i
end
i = 1
i = 2
i = 4
i = 7
```
"""
occupied_modes

"""
    move_particle(::SingleComponentFockAddress, i, j)

Move particle from mode `i` to mode `j`. Equivalent to ``a^{\\dagger}_i a_j``.
Returns the square of the normalization factor.

Note that the modes in [`BoseFS`](@ref) are indexed by [`BoseFSIndex`](@ref), while the ones
in [`FermiFS`](@ref) are indexed by integers (see example below).

# Example

```jldoctest
julia> b = BoseFS((1, 1, 3, 0))
BoseFS{5,4}((1, 1, 3, 0))

julia> i = find_occupied_mode(b, 2)
3-element Rimu.BitStringAddresses.BoseFSIndex with indices SOneTo(3):
 1
 2
 2

julia> j = find_mode(b, i.mode + 1)
3-element Rimu.BitStringAddresses.BoseFSIndex with indices SOneTo(3):
 3
 3
 4

julia> move_particle(b, i, j)
(BoseFS{5,4}((1, 0, 4, 0)), 4)
```

```jldoctest
julia> f = FermiFS((1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0))
FermiFS{7,12}((1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0))

julia> i = find_occupied_mode(f, 2)
6

julia> move_particle(f, i, i + 1)
(FermiFS{7,12}((1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0)), 0)

julia> move_particle(f, i, i - 1)
(FermiFS{7,12}((1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0)), 1)

julia> move_particle(f, i, 12)
(FermiFS{7,12}((1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1)), -1)
```
"""
move_particle
