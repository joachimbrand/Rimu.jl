"""
    AbstractFockAddress

Supertype representing a Fock state.
"""
abstract type AbstractFockAddress{N,M} end

"""
    num_particles(::Type{<:AbstractFockAddress})
    num_particles(::AbstractFockAddress)

Number of particles represented by address.
"""
num_particles(a::AbstractFockAddress) = num_particles(typeof(a))
num_particles(::Type{<:AbstractFockAddress{N}}) where {N} = N

"""
    num_modes(::Type{<:AbstractFockAddress})
    num_modes(::AbstractFockAddress)

Number of modes represented by address.
"""
num_modes(a::AbstractFockAddress) = num_modes(typeof(a))
num_modes(::Type{<:AbstractFockAddress{<:Any,M}}) where {M} = M

"""
    num_components(::Type{<:AbstractFockAddress})
    num_components(::AbstractFockAddress)

Number of components in address.
"""
num_components(b::AbstractFockAddress) = num_components(typeof(b))

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
abstract type SingleComponentFockAddress{N,M} <: AbstractFockAddress{N,M} end

num_components(::Type{<:SingleComponentFockAddress}) = 1

"""
    find_mode(::SingleComponentFockAddress, i)

Find the `i`-th mode in address. Returns [`BoseFSIndex`](@ref) for [`BoseFS`](@ref), and
an integer index for [`FermiFS`](@ref). Does not check bounds.

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
[`BoseFS`](@ref), and an integer for [`FermiFS`](@ref). When unsuccessful it
returns zero.

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
    is_occupied(::SingleComponentFockAddress, i)

Return `true` if index `i` points to an occupied mode.
"""
is_occupied

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
julia> b = BoseFS((1,5,0,4));

julia> foreach(println, occupied_modes(b))
[1, 1, 0]
[5, 2, 2]
[4, 4, 9]
```

```jldoctest
julia> f = FermiFS((1,1,0,1,0,0,1));

julia> foreach(println, occupied_modes(f))
1
2
4
7
```
"""
occupied_modes

"""
    move_particle(add::SingleComponentFockAddress, i, j) -> nadd, α

Move particle from mode `i` to mode `j`. Returns the new Fock state address `nadd` and
amplitude `α`. Equivalent to
```math
a^{\\dagger}_i a_j |\\mathrm{add}\\rangle \\to α|\\mathrm{nadd}\\rangle
```

Note that the modes in [`BoseFS`](@ref) are indexed by [`BoseFSIndex`](@ref), while the ones
in [`FermiFS`](@ref) are indexed by integers (see example below).  For illegal moves where `α
== 0` the value of `nadd` is undefined.

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
(BoseFS{5,4}((1, 0, 4, 0)), 2.0)

julia> move_particle(b, j, j)
(BoseFS{5,4}((1, 1, 3, 0)), 3.0)
```

```jldoctest
julia> f = FermiFS((1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0))
FermiFS{7,12}((1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0))

julia> i = find_occupied_mode(f, 2)
6

julia> move_particle(f, i, i + 1)
(FermiFS{7,12}((1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0)), 0.0)

julia> move_particle(f, i, i - 1)
(FermiFS{7,12}((1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0)), 1.0)

julia> move_particle(f, i, 12)
(FermiFS{7,12}((1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1)), -1.0)
```
"""
move_particle
