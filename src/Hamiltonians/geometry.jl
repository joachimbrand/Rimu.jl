"""
    abstract type LatticeGeometry

A `LatticeGeometry` controls which sites in an [`AbstractFockAddress`](@ref) are considered
to be neighbours.

Currently only supported by [`HubbardRealSpace`](@ref).

## Interface to implement

* `Base.size`: return the lattice size.
* [`neighbour_site(::LatticeGeometry, ::Int, ::Int)`](@ref)
* [`num_neighbours(::LatticeGeometry)`](@ref)

## Available implementations

* [`PeriodicBoundaries`](@ref)
* [`HardwallBoundaries`](@ref)
* [`LadderBoundaries`](@ref)

"""
abstract type LatticeGeometry end

function Base.show(io::IO, geom::LatticeGeometry)
    print(io, nameof(typeof(geom)), size(geom))
end

"""
    neighbour_site(geom::LatticeGeometry, site, i)

Find the `i`-th neighbour of `site` in the geometry. If the move is illegal, return 0.
"""
neighbour_site

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

Rectangular lattice of any dimension with size `size` and periodic boundary conditions.

The dimension of the lattice is controlled by the number of arguments given to its
constructor.

This is the default geometry used by [`HubbardRealSpace`](@ref).

## Example

```
julia> lattice = PeriodicBoundaries(5, 4) # 2D lattice of size 5 × 4
PeriodicBoundaries(5, 4)

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
"""
struct PeriodicBoundaries{D} <: LatticeGeometry
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

Rectangular lattice of any dimension with size `size` and hardwall boundary
conditions. Sites next to the boundaries will return 0 for some of their neighbours.

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
struct HardwallBoundaries{D} <: LatticeGeometry
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
struct LadderBoundaries{G<:LatticeGeometry} <: LatticeGeometry
    subgeometry::G
end

function LadderBoundaries(arg::Int, args::Vararg{Int}; subgeometry=PeriodicBoundaries)
    if arg ≠ 2
        error("First dimension must be of size 2")
    end
    return LadderBoundaries(subgeometry(args...))
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
