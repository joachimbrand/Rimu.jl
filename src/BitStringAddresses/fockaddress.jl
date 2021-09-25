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

"""
    excitation(a::SingleComponentFockAddress, creations::NTuple{N}, destructions::NTuple{N})

Generate an excitation on address `a` by applying `creations` and `destructions`, which are
tuples of the appropriate address indices (i.e. integers for fermions and `BoseFSIndex` for
bosons).

Returns the new address and the value. If the excitations is illegal, returns an arbitrary
address and 0.0.

# Example

```jldoctest
julia> f = FermiFS((1,1,0,0,1,1,1,1))
FermiFS{6,8}((1, 1, 0, 0, 1, 1, 1, 1))

julia> excitation(f, (3,4), (2,5))
(FermiFS{6,8}((1, 0, 1, 1, 0, 1, 1, 1)), -1.0)

julia> excitation(f, (3,4), (2,2))
(FermiFS{6,8}((1, 1, 0, 0, 1, 1, 1, 1)), 0.0)
```
"""
excitation

"""
    offset_excitation(a::SingleComponentFockAddress, indices, offsets; periodic=true)

Excitation of the form

```math
a^†_{p + k}a^†_{q + l}...a_{q}a_{p},
```

where ``p,q,... ∈`` `indices` and ``k,q... ∈`` `offsets`. If `periodic=true`, boundary
conditions are periodic, otherwise zeros are returned for moves that would have moved a
particle out of bounds.

The `periodic` should be replaced by a `Geometry`.

```jldoctest
julia> b = BoseFS((1,2,3,4,5))

BoseFS{15,5}((1, 2, 3, 4, 5))

julia> offset_excitation(b, (2, 3), (-1, 1))
(BoseFS{15,5}((2, 1, 2, 5, 5)), 7.745966692414834)

julia> offset_excitation(b, (2, 3), (-2, 2))
(BoseFS{15,5}((1, 1, 2, 4, 7)), 6.48074069840786)

julia> offset_excitation(b, (2, 3), (-2, 2); periodic=false)
(BoseFS{15,5}((1, 2, 3, 4, 5)), 0.0)

```
"""
function offset_excitation(
    a::SingleComponentFockAddress{<:Any,M}, indices, offsets; periodic=true
) where {M}
    src_indices = find_occupied_mode(a, indices)
    dst_modes = mode.(src_indices) .+ offsets
    if periodic
        dst_modes = mod1.(dst_modes, M)
    elseif any(i -> i > M || i ≤ 0, dst_modes)
        return a, 0.0
    end
    dst_indices = find_mode(a, dst_modes)
    return excitation(a, dst_indices, src_indices)
end
export offset_excitation, mode

mode(i::BoseFSIndex) = i.mode
mode(i::Integer) = i
