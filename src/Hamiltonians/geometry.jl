"""
    Geometry(dims::NTuple{D,Int}, fold::NTuple{D,Bool})

Represents a `D`-dimensional grid. Used to convert between cartesian vector indices (tuples
or `SVector`s) and linear indices (integers).

* `dims` controls the size of the grid in each dimension.
* `fold` controls whether the boundaries in each dimension are periodic (or folded in the
  case of momentum space).

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

julia> geo[(3,4)] # returns 0 if out of bounds
0

```
"""
struct Geometry{D,Dims,Fold}
    function Geometry(
        dims::NTuple{D,Int}, fold::NTuple{D,Bool}=ntuple(Returns(true), Val(D))
    ) where {D}
        return new{D,dims,fold}()
    end
end
Geometry(args::Vararg{Int}) = Geometry(args)

function PeriodicBoundaries(dims::NTuple{D,Int}) where {D}
    return Geometry(dims, ntuple(Returns(true), Val(D)))
end
PeriodicBoundaries(dims::Vararg{Int}) = PeriodicBoundaries(dims)

function HardwallBoundaries(dims::NTuple{D,Int}) where {D}
    return Geometry(dims, ntuple(Returns(false), Val(D)))
end
HardwallBoundaries(dims::Vararg{Int}) = HardwallBoundaries(dims)

function LadderBoundaries(dims::NTuple{D,Int}) where {D}
    return Geometry(dims, ntuple(>(1), Val(D)))
end
LadderBoundaries(dims::Vararg{Int}) = LadderBoundaries(dims)

function Base.show(io::IO, g::Geometry{<:Any,Dims,Fold}) where {Dims,Fold}
    print(io, "Geometry($Dims, $Fold)")
end

Base.size(g::Geometry{<:Any,Dims}) where {Dims} = Dims
Base.size(g::Geometry{<:Any,Dims}, i) where {Dims} = Dims[i]
Base.length(g::Geometry) = prod(size(g))
fold(g::Geometry{<:Any,<:Any,Fold}) where {Fold} = Fold

"""
    num_dimensions(geom::LatticeGeometry)

Return the number of dimensions of the lattice in this geometry.
"""
num_dimensions(::Geometry{D}) where {D} = D

"""
    fold_vec(g::Geometry{D}, vec::SVector{D,Int}) -> SVector{D,Int}

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
function fold_vec(g::Geometry{D}, vec::SVector{D,Int}) where {D}
    (_fold_vec(Tuple(vec), fold(g), size(g)))
end
@inline _fold_vec(::Tuple{}, ::Tuple{}, ::Tuple{}) = ()
@inline function _fold_vec((x, xs...), (f, fs...), (d, ds...))
    x = f ? mod1(x, d) : x
    return (x, _fold_vec(xs, fs, ds)...)
end

function Base.getindex(g::Geometry{D}, vec::Union{NTuple{D,Int},SVector{D,Int}}) where {D}
    return get(LinearIndices(size(g)), fold_vec(g, SVector(vec)), 0)
end
Base.getindex(g::Geometry, i::Int) = SVector(Tuple(CartesianIndices(size(g))[i]))

"""
    UnitVectors(D) <: AbstractVector{SVector{D,Int}}
    UnitVectors(geometry::Geometry) <: AbstractVector{SVector{D,Int}}

Iterate over unit vectors in `D` dimensions.

```jldoctest; setup=:(using Rimu.Hamiltonians: UnitVectors)
julia> UnitVectors(3)
6-element UnitVectors{3}:
 [1, 0, 0]
 [0, 1, 0]
 [0, 0, 1]
 [-1, 0, 0]
 [0, -1, 0]
 [0, 0, -1]

```
"""
struct UnitVectors{D} <: AbstractVector{SVector{D,Int}} end

UnitVectors(D) = UnitVectors{D}()
UnitVectors(::Geometry{D}) where {D} = UnitVectors{D}()

Base.size(::UnitVectors{D}) where {D} = (2D,)

function Base.getindex(uv::UnitVectors{D}, i) where {D}
    @boundscheck 0 < i ≤ length(uv) || throw(BoundsError(uv, i))
    if i ≤ D
        return SVector(_unit_vec(Val(D), i, 1))
    else
        return SVector(_unit_vec(Val(D), i - D, -1))
    end
end

@inline _unit_vec(::Val{0}, _, _) = ()
@inline function _unit_vec(::Val{I}, i, x) where {I}
    val = ifelse(i == I, x, 0)
    return (_unit_vec(Val(I-1), i, x)..., val)
end

"""
    Offsets(geometry::Geometry) <: AbstractVector{SVector{D,Int}}

```jldoctest; setup=:(using Rimu.Hamiltonians: Offsets)
julia> geometry = Geometry((3,4));

julia> reshape(Offsets(geometry), (3,4))
3×4 reshape(::Offsets{2}, 3, 4) with eltype StaticArraysCore.SVector{2, Int64}:
 [-1, -1]  [-1, 0]  [-1, 1]  [-1, 2]
 [0, -1]   [0, 0]   [0, 1]   [0, 2]
 [1, -1]   [1, 0]   [1, 1]   [1, 2]

```
"""
struct Offsets{D} <: AbstractVector{SVector{D,Int}}
    geometry::Geometry{D}
end

Base.size(off::Offsets) = (length(off.geometry),)

@inline function Base.getindex(off::Offsets{D}, i) where {D}
    @boundscheck 0 < i ≤ length(off) || throw(BoundsError(off, i))
    geo = off.geometry
    vec = geo[i]
    return vec - ones(SVector{D,Int})#+ SVector(ntuple(i -> -cld(size(geo, i), 2), Val(D)))
end

"""
    neighbor_site(geom::Geometry, site, i)

Find the `i`-th neighbor of `site` in the geometry. If the move is illegal, return 0.
"""
function neighbor_site(g::Geometry{D}, mode, chosen) where {D}
    return g[g[mode] + UnitVectors(D)[chosen]]
end

function BitStringAddresses.onr(add, geom::Geometry{<:Any,S}) where {S}
    return SArray{Tuple{S...}}(onr(add))
end
function BitStringAddresses.onr(add::CompositeFS, geom::Geometry)
    return map(fs -> onr(fs, geom), add.components)
end
