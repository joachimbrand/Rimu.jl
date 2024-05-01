"""
    Geometry(dims::NTuple{D,Int}, fold::NTuple{D,Bool})

Represents a D-dimensional grid. Used to convert between cartesian vector indices (tuples)
and linear indices (integers).

* `dims` controls the size of the grid in each dimension
* `fold` controls whether the boundaries in each dimension are periodic (or folded in the
  case of momentum space)

`Base.getindex` can be used to convert between linear indices and vectors.

```julia
julia> geo = Geometry((2,3), (true,false))
Geometry{2}((2, 3), (true, false))

julia> geo[1]
(1, 1)

julia> geo[2]
(2, 1)

julia> geo[3]
(1, 2)

julia> geo[(1,2)]
3

julia> geo[(3,2)] # 3 is folded back into 1
3

julia> geo[(3,3)]
5

julia> geo[(3,4)] # returns nothing if out of bounds

```
"""
struct Geometry{D}
    dims::NTuple{D,Int}
    fold::NTuple{D,Bool}

    function Geometry(
        dims::NTuple{D,Int}, fold::NTuple{D,Bool}=ntuple(Returns(true), Val(D))
    ) where {D}
        return new{D}(dims, fold)
    end
end

function PeriodicBoundaries(dims::Vararg{Int,D}) where {D}
    return Geometry(dims, ntuple(Returns(true), Val(D)))
end
function HardwallBoundaries(dims::Vararg{Int,D}) where {D}
    return Geometry(dims, ntuple(Returns(false), Val(D)))
end
function LadderBoundaries(dims::Vararg{Int,D}) where {D}
    return Geometry(dims, ntuple(>(1), Val(D)))
end

Base.length(g::Geometry) = prod(g.dims)
Base.size(g::Geometry) = g.dims

"""
    num_dimensions(geom::LatticeGeometry)

Return the number of dimensions of the lattice in this geometry.
"""
num_dimensions(::Geometry{D}) where {D} = D

"""
    fold_vec(g::Geometry{D}, vec::NTuple{D,Int}) -> NTuple{D,Int}

Use the Geometry to fold the `vec` in each dimension. If folding is disabled in a
dimension, and the vector is allowed to go out of bounds.

```julia
julia> geo = Geometry((2,3), (true,false))
Geometry{2}((2, 3), (true, false))

julia> fold_vec(geo, (3,1))
(1, 1)

julia> fold_vec(geo, (3,4))
(1, 4)
```
"""
fold_vec(g::Geometry{D}, vec::NTuple{D,Int}) where {D} = _fold_vec(vec, g.fold, g.dims)
@inline _fold_vec(::Tuple{}, ::Tuple{}, ::Tuple{}) = ()
@inline function _fold_vec((x, xs...), (f, fs...), (d, ds...))
    x = f ? mod1(x, d) : x
    return (x, _fold_vec(xs, fs, ds)...)
end

function Base.getindex(g::Geometry{D}, vec::NTuple{D,Int}) where {D}
    return get(LinearIndices(g.dims), fold_vec(g, vec), 0)
end
Base.getindex(g::Geometry, i::Int) = Tuple(CartesianIndices(g.dims)[i])

"""
    UnitVectors{D} <: AbstractVector{NTuple{D,Int}}

Iterate over unit vectors in `D` dimensions.

```jldoctest
julia> UnitVectors(3)
6-element UnitVectors{3}:
 (1, 0, 0)
 (0, 1, 0)
 (0, 0, 1)
 (-1, 0, 0)
 (0, -1, 0)
 (0, 0, -1)
```
"""
struct UnitVectors{D} <: AbstractVector{NTuple{D,Int}} end

UnitVectors(D) = UnitVectors{D}()

Base.size(::UnitVectors{D}) where {D} = (2D,)
function Base.getindex(uv::UnitVectors{D}, i) where {D}
    @boundscheck 0 < i ≤ 2D || throw(BoundsError(uv, i))
    if i ≤ D
        return _unit_vec(Val(D), i, 1)
    else
        return _unit_vec(Val(D), i - D, -1)
    end
end

@inline _unit_vec(::Val{0}, _, _) = ()
@inline function _unit_vec(::Val{I}, i, x) where {I}
    val = ifelse(i == I, x, 0)
    return (_unit_vec(Val(I-1), i, x)..., val)
end

struct Offsets{D} <: AbstractVector{NTuple{D,Int}}
    geometry::Geometry{D}
end
Base.size(off::Offsets) = (length(off.geometry),)
@inline function Base.getindex(off::Offsets{D}, i) where {D}
    geo = off.geometry
    vec = geo[i]
    return add(vec, ntuple(i -> -cld(geo.dims[i], 2), Val(D)))
end

"""
    neighbour_site(geom::Geometry, site, i)

Find the `i`-th neighbour of `site` in the geometry. If the move is illegal, return 0.
"""
function neighbor_site(g::Geometry{D}, mode, chosen) where {D}
    return g[add(g[mode], UnitVectors(D)[chosen])]
end

function BitStringAddresses.onr(add, geom::Geometry)
    return reshape(onr(add), size(geom))
end
function BitStringAddresses.onr(add::CompositeFS, geom::Geometry)
    return map(fs -> onr(fs, geom), add.components)
end
