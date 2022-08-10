"""
    abstract type LatticeGeometry{D}

A `LatticeGeometry` controls which sites in an [`AbstractFockAddress`](@ref) are considered
to be neighbours.

Currently only supported by [`HubbardRealSpace`](@ref).

# Available implementations

* [`PeriodicBoundaries`](@ref)
* [`HardwallBoundaries`](@ref)
* [`LadderBoundaries`](@ref)

# Interface to implement

* `Base.size`: return the lattice size.
* [`neighbour_site(::LatticeGeometry, ::Int, ::Int)`](@ref)
* [`num_dimensions(::LatticeGeometry)`](@ref)
* [`num_neighbours(::LatticeGeometry)`](@ref)
"""
abstract type LatticeGeometry{D} end

function Base.show(io::IO, geom::LatticeGeometry)
    print(io, nameof(typeof(geom)), size(geom))
end

"""
    onr(add::AbstractFockAddress, geom::LatticeGeometry)
Returns the occupation number representation of a Fock state address as an `SArray`
with the shape of the lattice geometry `geom`. For composite addresses, a tuple
of `onr`s is returned.
"""
function BitStringAddresses.onr(add, geom::LatticeGeometry)
    return reshape(onr(add), size(geom))
end
function BitStringAddresses.onr(add::CompositeFS, geom::LatticeGeometry)
    return map(fs -> onr(fs, geom), add.components)
end

"""
    neighbour_site(geom::LatticeGeometry, site, i)

Find the `i`-th neighbour of `site` in the geometry. If the move is illegal, return 0.
"""
neighbour_site

"""
    num_dimensions(geom::LatticeGeometry)

Return the number of dimensions of the lattice in this geometry.
"""
num_dimensions(::LatticeGeometry{D}) where D = D

"""
    num_neighbours(geom::LatticeGeometry)

Return the number of neighbours each lattice site has in this geometry.

Note that for efficiency reasons, all sites are expected to have the same number of
neighbours. If some of the neighbours are invalid, this is handled by having
[`neighbour_site`](@ref) return 0.
"""
num_neighbours


"""
    PeriodicBoundaries(size...) <: LatticeGeometry

Rectangular lattice with periodic boundary conditions of size `size`.

The dimension of the lattice is controlled by the number of arguments given to its
constructor.

This is the default geometry used by [`HubbardRealSpace`](@ref).

## Example

```
julia> lattice = PeriodicBoundaries(5, 4) # 2D lattice of size 5 × 4
PeriodicBoundaries(5, 4)

julia> num_neighbours(lattice)
4

julia> neighbour_site(lattice, 1, 1)
2

julia> neighbour_site(lattice, 1, 2)
5

julia> neighbour_site(lattice, 1, 3)
6

julia> neighbour_site(lattice, 1, 4)
16
```

## See also

* [`LatticeGeometry`](@ref)
* [`HardwallBoundaries`](@ref)
* [`LadderBoundaries`](@ref)
* [`num_neighbours`](@ref)
* [`neighbour_site`](@ref)
"""
struct PeriodicBoundaries{D} <: LatticeGeometry{D}
    size::NTuple{D,Int}
end

PeriodicBoundaries(args::Vararg{Int}) = PeriodicBoundaries(args)

Base.size(geom::PeriodicBoundaries) = geom.size

num_neighbours(::PeriodicBoundaries{D}) where {D} = 2D

function neighbour_site(geom::PeriodicBoundaries{D}, site, i) where {D}
    # Neighbour indexing
    # i |  x  y  z …
    # 0 | +1  0  0
    # 1 | -1  0  0
    # 2 |  0 +1  0
    # 3 |  0 -1  0
    # 4 |  0  0 +1
    #         ⋮    ⋱
    i -= 1
    cart_indices = CartesianIndices(size(geom))
    cart_index = Tuple(cart_indices[site])
    offset = ntuple(Val(D)) do k
        ifelse(2(k - 1) ≤ i < 2k, ifelse(iseven(i), 1, -1), 0)
    end
    new_index = CartesianIndex(mod1.(cart_index .+ offset, size(geom) .% UInt))
    return LinearIndices(cart_indices)[new_index]
end

"""
    HardwallBoundaries

Rectangular lattice with hard wall boundary conditions of size `size`.
[`neighbour_site()`](@ref) will return 0 for some neighbours of boundary sites.

The dimension of the lattice is controlled by the number of arguments given to its
constructor.

## Example

```
julia> lattice = HardwallBoundaries(5) # 1D lattice of size 5
HardwallBoundaries(5)

julia> neighbour_site(lattice, 1, 1)
2

julia> neighbour_site(lattice, 1, 2)
0

```

## See also

* [`LatticeGeometry`](@ref)
* [`PeriodicBoundaries`](@ref)
* [`LadderBoundaries`](@ref)
"""
struct HardwallBoundaries{D} <: LatticeGeometry{D}
    size::NTuple{D,Int}
end

HardwallBoundaries(args::Vararg{Int}) = HardwallBoundaries(args)

Base.size(geom::HardwallBoundaries) = geom.size

num_neighbours(::HardwallBoundaries{D}) where {D} = 2D

function neighbour_site(geom::HardwallBoundaries{D}, site, i) where {D}
    i -= 1
    cart_indices = CartesianIndices(size(geom))
    cart_index = Tuple(cart_indices[site])
    offset = ntuple(Val(D)) do k
        ifelse(2(k - 1) ≤ i < 2k, ifelse(iseven(i), 1, -1), 0)
    end
    new_index = CartesianIndex(cart_index .+ offset)
    if new_index in cart_indices
        return LinearIndices(cart_indices)[new_index]
    else
        return 0
    end
end

"""
    LadderBoundaries(size...; subgeometry=PeriodicBoundaries) <: LatticeGeometry

Lattice geometry where the first dimension is of size 2 and has hardwall boundary conditions.
Using this geometry is more efficient than using [`HardwallBoundaries`](@ref) with a size of
2, as it does not generate rejected neighbours.

In other dimensions, it behaves like its subgeometry, which can be any
[`LatticeGeometry`](@ref).

## Example

```
julia> lattice = LadderBoundaries(2, 3, 4) # 3D lattice of size 2 × 3 × 4
LadderBoundaries(2, 3, 4)

julia> num_neighbours(lattice)
5

julia> neighbour_site(lattice, 1, 1)
2

julia> neighbour_site(lattice, 1, 2)
3

julia> neighbour_site(lattice, 1, 3)
5

julia> neighbour_site(lattice, 1, 4)
7

julia> neighbour_site(lattice, 1, 5)
19
```

## See also

* [`LatticeGeometry`](@ref)
* [`PeriodicBoundaries`](@ref)
* [`HardwallBoundaries`](@ref)
"""
struct LadderBoundaries{D,G<:LatticeGeometry} <: LatticeGeometry{D}
    subgeometry::G
end

function LadderBoundaries(arg::Int, args::Vararg{Int}; subgeometry=PeriodicBoundaries)
    if arg ≠ 2
        throw(ArgumentError("First dimension must be of size 2"))
    end
    D = num_dimensions(subgeometry) + 1
    return LadderBoundaries{D}(subgeometry(args...))
end

Base.size(geom::LadderBoundaries) = (2, size(geom.subgeometry)...)

num_neighbours(geom::LadderBoundaries) = num_neighbours(geom.subgeometry) + 1

function neighbour_site(geom::LadderBoundaries, site, i)
    if i == 1
        # Make odd site even, or make even site odd.
        return site + site % 2 - (site + 1) % 2
    else
        i -= 1
        subneighbour = neighbour_site(geom.subgeometry, cld(site, 2), i)
        return ifelse(iszero(subneighbour), 0, 2 * subneighbour - site % 2)
    end
end
