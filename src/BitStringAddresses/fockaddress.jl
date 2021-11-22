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
* [`is_occupied`](@ref)
* [`num_occupied_modes`](@ref)
* [`occupied_modes`](@ref): Lazy iterator.
* [`OccupiedModeMap`](@ref): `AbstractVector` with eager construction.
* [`excitation`](@ref): Create a new address.
* [`BoseFSIndex`](@ref) and [`FermiFSIndex`](@ref) for indexing.
"""
abstract type SingleComponentFockAddress{N,M} <: AbstractFockAddress{N,M} end

num_components(::Type{<:SingleComponentFockAddress}) = 1

"""
    find_mode(::SingleComponentFockAddress, i)

Find the `i`-th mode in address. Returns [`BoseFSIndex`](@ref) for [`BoseFS`](@ref), and
[`FermiFSIndex`](@ref) for [`FermiFS`](@ref). Can work on a tuple of modes. Does not check
bounds.

```jldoctest
julia> find_mode(BoseFS((1, 0, 2)), 2)
BoseFSIndex(occnum=0, mode=2, offset=2)

julia> find_mode(FermiFS((1, 1, 1, 0)), (2,3))
(FermiFSIndex(occnum=1, mode=2), FermiFSIndex(occnum=1, mode=3))
```

See [`SingleComponentFockAddress`](@ref).
"""
find_mode

"""
    find_occupied_mode(::SingleComponentFockAddress, k)
    find_occupied_mode(::BoseFS, k, [n])

Find the `k`-th occupied mode in address (with at least `n` particles).
Returns [`BoseFSIndex`](@ref) for [`BoseFS`](@ref), and [`FermiFSIndex`](@ref) for
[`FermiFS`](@ref). When unsuccessful it returns a zero index.

# Example

```jldoctest
julia> find_occupied_mode(FermiFS((1, 1, 1, 0)), 2)
FermiFSIndex(occnum=1, mode=2)

julia> find_occupied_mode(BoseFS((1, 0, 2)), 1)
BoseFSIndex(occnum=1, mode=1, offset=0)

julia> find_occupied_mode(BoseFS((1, 0, 2)), 1, 2)
BoseFSIndex(occnum=2, mode=3, offset=3)
```

See also [`occupied_modes`](@ref), [`OccupiedModeMap`](@ref),
[`SingleComponentFockAddress`](@ref).
"""
find_occupied_mode

"""
    is_occupied(::SingleComponentFockAddress, i)

Return `true` if index `i` points to an occupied mode.

See [`SingleComponentFockAddress`](@ref).
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

See [`SingleComponentFockAddress`](@ref).
"""
num_occupied_modes

"""
    occupied_modes(::SingleComponentFockAddress)

Return a lazy iterator over all occupied modes in an address. Iterates over
[`BoseFSIndex`](@ref)s for [`BoseFS`](@ref), and over [`FermiFSIndex`](@ref)s for
[`FermiFS`](@ref). See [`OccupiedModeMap`](@ref) for an eager version.

# Example

```jldoctest
julia> b = BoseFS((1,5,0,4));

julia> foreach(println, occupied_modes(b))
BoseFSIndex(occnum=1, mode=1, offset=0)
BoseFSIndex(occnum=5, mode=2, offset=2)
BoseFSIndex(occnum=4, mode=4, offset=9)
```

```jldoctest
julia> f = FermiFS((1,1,0,1,0,0,1));

julia> foreach(println, occupied_modes(f))
FermiFSIndex(occnum=1, mode=1)
FermiFSIndex(occnum=1, mode=2)
FermiFSIndex(occnum=1, mode=4)
FermiFSIndex(occnum=1, mode=7)
```
See also [`find_occupied_mode`](@ref),
[`SingleComponentFockAddress`](@ref).
"""
occupied_modes

"""
    excitation(a::SingleComponentFockAddress, creations::NTuple{N}, destructions::NTuple{N})

Generate an excitation on address `a` by applying `creations` and `destructions`, which are
tuples of the appropriate address indices (i.e. [`BoseFSIndex`](@ref) for bosons, or
[`FermiFSIndex`](@ref) for fermions).

```math
a^†_{c_1} a^†_{c_2} \\ldots a_{d_1} a_{d_2} \\ldots |\\mathrm{a}\\rangle \\to
α|\\mathrm{nadd}\\rangle
```

Returns the new address `nadd` and the value `α`. If the excitation is illegal, returns an
arbitrary address and the value `0.0`.

# Example

```jldoctest
julia> f = FermiFS((1,1,0,0,1,1,1,1))
FermiFS{6,8}((1, 1, 0, 0, 1, 1, 1, 1))

julia> i, j, k, l = find_mode(f, (3,4,2,5))
(FermiFSIndex(occnum=0, mode=3), FermiFSIndex(occnum=0, mode=4), FermiFSIndex(occnum=1, mode=2), FermiFSIndex(occnum=1, mode=5))

julia> excitation(f, (i,j), (k,l))
(FermiFS{6,8}((1, 0, 1, 1, 0, 1, 1, 1)), -1.0)
```

See [`SingleComponentFockAddress`](@ref).
"""
excitation

"""
    OccupiedModeMap(add) <: AbstractVector

Get a map of occupied modes in address as an `AbstractVector` of indices compatible with
[`excitation`](@ref) - [`BoseFSIndex`](@ref) or [`FermiFSIndex`](@ref).

`OccupiedModeMap(add)[i]` contains the index for the `i`-th occupied mode.
This is useful because repeatedly looking for occupied modes with
[`find_occupied_mode`](@ref) can be time-consuming.
`OccupiedModeMap(add)` is an eager version of the iterator returned by
[`occupied_modes`](@ref). It is similar to [`onr`](@ref) but contains more information.

# Example

```jldoctest
julia> b = BoseFS((10, 0, 0, 0, 2, 0, 1))
BoseFS{13,7}((10, 0, 0, 0, 2, 0, 1))

julia> mb = OccupiedModeMap(b)
3-element OccupiedModeMap{7, BoseFSIndex}:
 BoseFSIndex(occnum=10, mode=1, offset=0)
 BoseFSIndex(occnum=2, mode=5, offset=14)
 BoseFSIndex(occnum=1, mode=7, offset=18)

julia> f = FermiFS((1,1,1,1,0,0,1,0,0))
FermiFS{5,9}((1, 1, 1, 1, 0, 0, 1, 0, 0))

julia> mf = OccupiedModeMap(f)
5-element OccupiedModeMap{5, FermiFSIndex}:
 FermiFSIndex(occnum=1, mode=1)
 FermiFSIndex(occnum=1, mode=2)
 FermiFSIndex(occnum=1, mode=3)
 FermiFSIndex(occnum=1, mode=4)
 FermiFSIndex(occnum=1, mode=7)

julia> mf == collect(occupied_modes(f))
true

julia> dot(mf, mb)
11

julia> dot(mf, 1:20)
17
```
See also [`dot`](@ref), [`SingleComponentFockAddress`](@ref).
"""
struct OccupiedModeMap{N,T} <: AbstractVector{T}
    indices::SVector{N,T} # N = min(N, M)
    length::Int
end

function OccupiedModeMap(add::SingleComponentFockAddress{N,M}) where {N,M}
    modes = occupied_modes(add)
    T = eltype(modes)
    # There are at most N occupied modes. This could be also @generated for cases where N ≫ M
    indices = MVector{min(N,M),T}(undef)
    i = 0
    for index in modes
        i += 1
        @inbounds indices[i] = index
    end
    return OccupiedModeMap(SVector(indices), i)
end

Base.size(om::OccupiedModeMap) = (om.length,)
function Base.getindex(om::OccupiedModeMap, i)
    @boundscheck 1 ≤ i ≤ om.length || throw(BoundsError(om, i))
    return om.indices[i]
end

"""
    dot(map::OccupiedModeMap, vec::AbstractVector)
    dot(map1::OccupiedModeMap, map2::OccupiedModeMap)
Dot product extracting mode occupation numbers from an [`OccupiedModeMap`](@ref) similar
to [`onr`](@ref).

```jldoctest
julia> b = BoseFS((10, 0, 0, 0, 2, 0, 1))
BoseFS{13,7}((10, 0, 0, 0, 2, 0, 1))

julia> mb = OccupiedModeMap(b)
3-element OccupiedModeMap{7, BoseFSIndex}:
 BoseFSIndex(occnum=10, mode=1, offset=0)
 BoseFSIndex(occnum=2, mode=5, offset=14)
 BoseFSIndex(occnum=1, mode=7, offset=18)

julia> dot(mb, 1:7)
27

julia> mb⋅(1:7) == onr(b)⋅(1:7)
true
```
See also [`SingleComponentFockAddress`](@ref).
"""
function LinearAlgebra.dot(map::OccupiedModeMap, vec::AbstractVector)
    value = zero(eltype(vec))
    for index in map
        value += vec[index.mode] * index.occnum
    end
    return value
end
LinearAlgebra.dot(vec::AbstractVector, map::OccupiedModeMap) = dot(map, vec)

# Defined for consistency. Could also be used to compute cross-component interactions in
# real space.
function LinearAlgebra.dot(map1::OccupiedModeMap, map2::OccupiedModeMap)
    i = j = 1
    value = 0
    while i ≤ length(map1) && j ≤ length(map2)
        index1 = map1[i]
        index2 = map2[j]
        if index1.mode == index2.mode
            value += index1.occnum * index2.occnum
            i += 1
            j += 1
        elseif index1.mode < index2.mode
            i += 1
        else
            j += 1
        end
    end
    return value
end

"""
    parse_address(str)

Parse the compact representation of a fock state address.
"""
function parse_address(str)
    # FermiFS2C
    m = match(r"[↓⇅]", str)
    if !isnothing(m)
        m = match(r"\|([↑↓⇅⋅]+)⟩", str)
        if isnothing(m)
            throw(ArgumentError("invalid fock state format \"$str\""))
        else
            chars = Vector{Char}(m.captures[1])
            f1 = FermiFS((chars .== '↑') .| (chars .== '⇅'))
            f2 = FermiFS((chars .== '↓') .| (chars .== '⇅'))
            return CompositeFS(f1, f2)
        end
    end
    # CompositeFS
    m = match(r"⊗", str)
    if !isnothing(m)
        return CompositeFS(map(parse_address, split(str, " ⊗ "))...)
    end
    # BoseFS
    m = match(r"\|([ 0-9]+)⟩", str)
    if !isnothing(m)
        return BoseFS(parse.(Int, split(m.captures[1], ' ')))
    end
    # Single FermiFS
    m = match(r"\|([⋅↑]+)⟩", str)
    if !isnothing(m)
        return FermiFS(Vector{Char}(m.captures[1]) .== '↑')
    end
    throw(ArgumentError("invalid fock state format \"$str\""))
end

"""
    fs"\$(string)"

Parse the compact representation of a fock state.
Useful for copying the printout from a vector to the REPL.

# Example

```
julia> DVec(BoseFS{3,4}((0, 1, 2, 0)) => 1)
DVec{BoseFS{3, 4, BitString{6, 1, UInt8}},Int64} with 1 enrty, style = IsStochasticInteger{Int64}()
  fs"|0 1 2 0⟩" => 1

julia> fs"|0 1 2 0⟩" => 1 # Copied from above printout
BoseFS{3,4}((0, 1, 2, 0)) => 1
```
"""
macro fs_str(str)
    return parse_address(str)
end

function Base.show(io::IO, add::AbstractFockAddress)
    if get(io, :compact, false)
        print(io, "fs\"")
        print_address(io, add)
        print(io, "\"")
    else
        print_address(io, add)
    end
end
