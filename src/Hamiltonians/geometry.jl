"""
    CubicGrid(dims::NTuple{D,Int}, fold::NTuple{D,Bool})

Represents a `D`-dimensional grid. Used to define a cubic lattice and boundary conditions
for some [`AbstractHamiltonian`](@ref)s. The type instance can be used to convert between
cartesian vector indices (tuples or `SVector`s) and linear indices (integers). When indexed
with vectors, it folds them back into the grid if the out-of-bounds dimension is periodic and
0 otherwise (see example below).

* `dims` controls the size of the grid in each dimension.
* `fold` controls whether the boundaries in each dimension are periodic (or folded in the
  case of momentum space).

```julia
julia> geo = CubicGrid((2,3), (true,false))
CubicGrid{2}((2, 3), (true, false))

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

See also [`PeriodicBoundaries`](@ref), [`HardwallBoundaries`](@ref) and
[`LadderBoundaries`](@ref) for special-case constructors.
"""
struct CubicGrid{D,Dims,Fold}
    function CubicGrid(
        dims::NTuple{D,Int}, fold::NTuple{D,Bool}=ntuple(Returns(true), Val(D))
    ) where {D}
        if any(≤(1), dims)
            throw(ArgumentError("All dimensions must be at least 2 in size"))
        end
        return new{D,dims,fold}()
    end
end
CubicGrid(args::Vararg{Int}) = CubicGrid(args)

"""
    PeriodicBoundaries(dims...) -> CubicGrid
    PeriodicBoundaries(dims) -> CubicGrid

Return `CubicGrid` with all dimensions periodic. Equivalent to `CubicGrid(dims)`.
"""
function PeriodicBoundaries(dims::NTuple{D,Int}) where {D}
    return CubicGrid(dims, ntuple(Returns(true), Val(D)))
end
PeriodicBoundaries(dims::Vararg{Int}) = PeriodicBoundaries(dims)

"""
    HardwallBoundaries(dims...) -> CubicGrid
    HardwallBoundaries(dims) -> CubicGrid

Return `CubicGrid` with all dimensions non-periodic. Equivalent to
`CubicGrid(dims, (false, false, ...))`.
"""
function HardwallBoundaries(dims::NTuple{D,Int}) where {D}
    return CubicGrid(dims, ntuple(Returns(false), Val(D)))
end
HardwallBoundaries(dims::Vararg{Int}) = HardwallBoundaries(dims)

"""
    LadderBoundaries(dims...) -> CubicGrid
    LadderBoundaries(dims) -> CubicGrid

Return `CubicGrid` where the first dimension is dimensions non-periodic and the rest are
periodic. Equivalent to `CubicGrid(dims, (true, false, ...))`.
"""
function LadderBoundaries(dims::NTuple{D,Int}) where {D}
    return CubicGrid(dims, ntuple(>(1), Val(D)))
end
LadderBoundaries(dims::Vararg{Int}) = LadderBoundaries(dims)

function Base.show(io::IO, g::CubicGrid{<:Any,Dims,Fold}) where {Dims,Fold}
    print(io, "CubicGrid($Dims, $Fold)")
end

Base.size(g::CubicGrid{<:Any,Dims}) where {Dims} = Dims
Base.size(g::CubicGrid{<:Any,Dims}, i) where {Dims} = Dims[i]
Base.length(g::CubicGrid) = prod(size(g))
fold(g::CubicGrid{<:Any,<:Any,Fold}) where {Fold} = Fold

"""
    num_dimensions(geom::LatticeCubicGrid)

Return the number of dimensions of the lattice in this geometry.
"""
num_dimensions(::CubicGrid{D}) where {D} = D

"""
    fold_vec(g::CubicGrid{D}, vec::SVector{D,Int}) -> SVector{D,Int}

Use the CubicGrid to fold the `vec` in each dimension. If folding is disabled in a
dimension, and the vector is allowed to go out of bounds.

```julia
julia> geo = CubicGrid((2,3), (true,false))
CubicGrid{2}((2, 3), (true, false))

julia> fold_vec(geo, (3,1))
(1, 1)

julia> fold_vec(geo, (3,4))
(1, 4)
```
"""
function fold_vec(g::CubicGrid{D}, vec::SVector{D,Int}) where {D}
    (_fold_vec(Tuple(vec), fold(g), size(g)))
end
@inline _fold_vec(::Tuple{}, ::Tuple{}, ::Tuple{}) = ()
@inline function _fold_vec((x, xs...), (f, fs...), (d, ds...))
    x = f ? mod1(x, d) : x
    return (x, _fold_vec(xs, fs, ds)...)
end

function Base.getindex(g::CubicGrid{D}, vec::Union{NTuple{D,Int},SVector{D,Int}}) where {D}
    return get(LinearIndices(size(g)), fold_vec(g, SVector(vec)), 0)
end
Base.getindex(g::CubicGrid, i::Int) = SVector(Tuple(CartesianIndices(size(g))[i]))

"""
    Directions(D) <: AbstractVector{SVector{D,Int}}
    Directions(geometry::CubicGrid) <: AbstractVector{SVector{D,Int}}

Iterate over axis-aligned direction vectors in `D` dimensions.

```jldoctest; setup=:(using Rimu.Hamiltonians: Directions)
julia> Directions(3)
6-element Directions{3}:
 [1, 0, 0]
 [0, 1, 0]
 [0, 0, 1]
 [-1, 0, 0]
 [0, -1, 0]
 [0, 0, -1]

```

See also [`CubicGrid`](@ref).
"""
struct Directions{D} <: AbstractVector{SVector{D,Int}} end

Directions(D) = Directions{D}()
Directions(::CubicGrid{D}) where {D} = Directions{D}()

Base.size(::Directions{D}) where {D} = (2D,)

function Base.getindex(uv::Directions{D}, i) where {D}
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
    Displacements(geometry::CubicGrid) <: AbstractVector{SVector{D,Int}}

Return all valid offset vectors in a [`CubicGrid`](@ref). If `center=true` the (0,0) displacement is
placed at the centre of the array.

```jldoctest; setup=:(using Rimu.Hamiltonians: Displacements)
julia> geometry = CubicGrid((3,4));

julia> reshape(Displacements(geometry), (3,4))
3×4 reshape(::Displacements{2, CubicGrid{2, (3, 4), (true, true)}}, 3, 4) with eltype StaticArraysCore.SVector{2, Int64}:
 [0, 0]  [0, 1]  [0, 2]  [0, 3]
 [1, 0]  [1, 1]  [1, 2]  [1, 3]
 [2, 0]  [2, 1]  [2, 2]  [2, 3]

julia> reshape(Displacements(geometry; center=true), (3,4))
3×4 reshape(::Displacements{2, CubicGrid{2, (3, 4), (true, true)}}, 3, 4) with eltype StaticArraysCore.SVector{2, Int64}:
 [-1, -1]  [-1, 0]  [-1, 1]  [-1, 2]
 [0, -1]   [0, 0]   [0, 1]   [0, 2]
 [1, -1]   [1, 0]   [1, 1]   [1, 2]

```
"""
struct Displacements{D,G<:CubicGrid{D}} <: AbstractVector{SVector{D,Int}}
    geometry::G
    center::Bool
end
Displacements(geometry; center=false) = Displacements(geometry, center)

Base.size(off::Displacements) = (length(off.geometry),)

@inline function Base.getindex(off::Displacements{D}, i) where {D}
    @boundscheck 0 < i ≤ length(off) || throw(BoundsError(off, i))
    geo = off.geometry
    vec = geo[i]
    if !off.center
        return vec - ones(SVector{D,Int})
    else
        return vec - SVector(ntuple(i -> cld(size(geo, i), 2), Val(D)))
    end
end

"""
    neighbor_site(geom::CubicGrid, site, i)

Find the `i`-th neighbor of `site` in the geometry. If the move is illegal, return 0.
"""
function neighbor_site(g::CubicGrid{D}, mode, chosen) where {D}
    # TODO: reintroduce LadderBoundaries small dimensions
    return g[g[mode] + Directions(D)[chosen]]
end

function BitStringAddresses.onr(address, geom::CubicGrid{<:Any,S}) where {S}
    return SArray{Tuple{S...}}(onr(address))
end
function BitStringAddresses.onr(address::CompositeFS, geom::CubicGrid)
    return map(fs -> onr(fs, geom), address.components)
end
