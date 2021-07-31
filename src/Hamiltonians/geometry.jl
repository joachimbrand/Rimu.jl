"""
    abstract type LatticeGeometry

A `LatticeGeometry` controls which sites in an [`AbstractFockAddress`](@ref) are considered
to be neighbours.

Currently only supported by [`HubbardRealSpace`](@ref).

## Interface to implement

* `Base.size`: return the lattice size.
* [`neighbour_site(::LatticeGeometry, ::Int, ::Int)`](@ref): find the neighbour of given site.
* [`num_neighbours`](::LatticeGeometry): return the number of neighbours a site in the
  lattice has.

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
"""
num_neighbours


"""
    PeriodicBoundaries(size...) <: LatticeGeometry

Lattice of size `size` with periodic boundary conditions.

This is the default geometry used by [`HubbardRealSpace`](@ref).

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
        ifelse(2(k - 1) ≤ i < 2k, (-1)^(i & 1), 0)
    end
    new_index = CartesianIndex(mod1.(cart_index .+ offset, size(geom)))
    return LinearIndices(cart_indices)[new_index]
end

"""
    HardwallBoundaries

Lattice of size `size` with hardwall boundary conditions. Sites next to the boundaries will
return 0 for some neighbours.

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
        ifelse(2(k - 1) ≤ i < 2k, (-1)^(i & 1), 0)
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
        return 2 * neighbour_site(geom.subgeometry, cld(site, 2), i) - site % 2
    end
end
