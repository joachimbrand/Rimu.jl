"""
    AbstractFockAddress{N,M}

Abstract type representing a Fock state with `N` particles and `M` modes.

See also [`SingleComponentFockAddress`](@ref), [`CompositeFS`](@ref), [`BoseFS`](@ref),
[`FermiFS`](@ref).
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
    SingleComponentFockAddress{N,M} <: AbstractFockAddress{N,M}

A type representing a single component Fock state with `N` particles and `M` modes.

Implemented subtypes: [`BoseFS`](@ref), [`FermiFS`](@ref).

# Supported functionality

* [`find_mode`](@ref)
* [`find_occupied_mode`](@ref)
* [`num_occupied_modes`](@ref)
* [`occupied_modes`](@ref): Lazy iterator.
* [`OccupiedModeMap`](@ref): `AbstractVector` with eager construction.
* [`excitation`](@ref): Create a new address.
* [`BoseFSIndex`](@ref) and [`FermiFSIndex`](@ref) for indexing.

See also [`CompositeFS`](@ref), [`AbstractFockAddress`](@ref).
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
(FermiFSIndex(occnum=1, mode=2, offset=1), FermiFSIndex(occnum=1, mode=3, offset=2))
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
FermiFSIndex(occnum=1, mode=2, offset=1)

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
FermiFSIndex(occnum=1, mode=1, offset=0)
FermiFSIndex(occnum=1, mode=2, offset=1)
FermiFSIndex(occnum=1, mode=4, offset=3)
FermiFSIndex(occnum=1, mode=7, offset=6)
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
(FermiFSIndex(occnum=0, mode=3, offset=2), FermiFSIndex(occnum=0, mode=4, offset=3), FermiFSIndex(occnum=1, mode=2, offset=1), FermiFSIndex(occnum=1, mode=5, offset=4))

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
 FermiFSIndex(occnum=1, mode=1, offset=0)
 FermiFSIndex(occnum=1, mode=2, offset=1)
 FermiFSIndex(occnum=1, mode=3, offset=2)
 FermiFSIndex(occnum=1, mode=4, offset=3)
 FermiFSIndex(occnum=1, mode=7, offset=6)

julia> mf == collect(occupied_modes(f))
true

julia> dot(mf, mb)
11

julia> dot(mf, 1:20)
17
```
See also [`dot`](@ref Main.Hamiltonians.dot), [`SingleComponentFockAddress`](@ref).
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
    abstract type OccupiedModeIterator

Iterator over occupied modes with `eltype` [`BoseFSIndex`](@ref) or
[`FermiFSIndex`](@ref). A subtype of this should be returned when calling
[`occupied_modes`](@ref) on a Fock state.
"""
abstract type OccupiedModeIterator end

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

Parse the compact representation of a Fock state address.
"""
function parse_address(str)
    # CompositeFS
    m = match(r"⊗", str)
    if !isnothing(m)
        if !isnothing(match(r"[↓⇅]", str))
            throw(ArgumentError("invalid fock state format \"$str\""))
        else
            return CompositeFS(map(parse_address, split(str, r" *⊗ *"))...)
        end
    end
    # FermiFS2C
    m = match(r"[↓⇅]", str)
    if !isnothing(m)
        m = match(r"\|([↑↓⇅⋅ ]+)⟩", str)
        if isnothing(m)
            throw(ArgumentError("invalid fock state format \"$str\""))
        else
            chars = filter(!=(' '), Vector{Char}(m.captures[1]))
            f1 = FermiFS((chars .== '↑') .| (chars .== '⇅'))
            f2 = FermiFS((chars .== '↓') .| (chars .== '⇅'))
            return CompositeFS(f1, f2)
        end
    end
    # Sparse BoseFS
    m = match(r"\|b *([0-9]+): *([ 0-9]+)⟩", str)
    if !isnothing(m)
        particles = parse.(Int, filter(!isempty, split(m.captures[2], r" +")))
        return BoseFS(parse(Int, m.captures[1]), zip(particles, fill(1, length(particles))))
    end
    # Sparse FermiFS
    m = match(r"\|f *([0-9]+): *([ 0-9]+)⟩", str)
    if !isnothing(m)
        particles = parse.(Int, filter(!isempty, split(m.captures[2], r" +")))
        return FermiFS(parse(Int, m.captures[1]), zip(particles, fill(1, length(particles))))
    end
    # BoseFS
    m = match(r"\|([ 0-9]+)⟩", str)
    if !isnothing(m)
        return BoseFS(parse.(Int, split(m.captures[1], r" +")))
    end
    # Single FermiFS
    m = match(r"\|([ ⋅↑]+)⟩", str)
    if !isnothing(m)
        chars = filter(!=(' '), Vector{Char}(m.captures[1]))
        return FermiFS(chars .== '↑')
    end
    throw(ArgumentError("invalid fock state format \"$str\""))
end

"""
    fs"\$(string)"

Parse the compact representation of a Fock state.
Useful for copying the printout from a vector to the REPL.

# Example

```
julia> DVec(BoseFS{3,4}((0, 1, 2, 0)) => 1)
DVec{BoseFS{3, 4, BitString{6, 1, UInt8}},Int64} with 1 enrty, style = IsStochasticInteger{Int64}()
  fs"|0 1 2 0⟩" => 1

julia> fs"|0 1 2 0⟩" => 1 # Copied from above printout
BoseFS{3,4}((0, 1, 2, 0)) => 1
```

See also [`FermiFS`](@ref), [`BoseFS`](@ref), [`CompositeFS`](@ref), [`FermiFS2C`](@ref).
"""
macro fs_str(str)
    return parse_address(str)
end

"""
    print_address(io::IO, address)

Print the `address` to `io`. If `get(io, :compact, false) == true`, the printed form should
be parsable by [`parse_address`](@ref).

This function is used to implement `Base.show` for [`AbstractFockAddress`](@ref).
"""
print_address

function Base.show(io::IO, add::AbstractFockAddress)
    if get(io, :typeinfo, nothing) == typeof(add) || get(io, :compact, false)
        print(io, "fs\"")
        print_address(io, add; compact=true)
        print(io, "\"")
    else
        print_address(io, add; compact=false)
    end
end

function onr_sparse_string(o)
    ps = map(p -> p[1] => p[2], Iterators.filter(p -> !iszero(p[2]), enumerate(o)))
    return join(ps, ", ")
end

###
### Boson stuff
###
"""
    BoseFSIndex

Struct used for indexing and performing [`excitation`](@ref)s on a [`BoseFS`](@ref).

## Fields:

* `occnum`: the occupation number.
* `mode`: the index of the mode.
* `offset`: the position of the mode in the address. This is the bit offset of the mode when
 the address is represented by a bitstring, and the position in the list when it is
 represented by `SortedParticleList`.

"""
struct BoseFSIndex<:FieldVector{3,Int}
    occnum::Int
    mode::Int
    offset::Int
end

function Base.show(io::IO, i::BoseFSIndex)
    @unpack occnum, mode, offset = i
    print(io, "BoseFSIndex(occnum=$occnum, mode=$mode, offset=$offset)")
end
Base.show(io::IO, ::MIME"text/plain", i::BoseFSIndex) = show(io, i)

"""
    BoseOccupiedModes{C,S<:BoseFS}

Iterator for occupied modes in [`BoseFS`](@ref). The definition of `iterate` is dispatched
on the storage type.

See [`occupied_modes`](@ref).

Defining `Base.length` and `Base.iterate` for this struct is a part of the interface for an
underlying storage format used by [`BoseFS`](@ref).
"""
struct BoseOccupiedModes{N,M,S}<:OccupiedModeIterator
    storage::S
end
Base.eltype(::BoseOccupiedModes) = BoseFSIndex

# Apply destruction operator to BoseFSIndex.
@inline _destroy(d, index::BoseFSIndex) = @set index.occnum -= (d.mode == index.mode)
@inline _destroy(d) = Base.Fix1(_destroy, d)
# Apply creation operator to BoseFSIndex.
@inline _create(c, index::BoseFSIndex) = @set index.occnum += (c.mode == index.mode)
@inline _create(c) = Base.Fix1(_create, c)

"""
    bose_excitation_value(
        creations::NTuple{_,BoseFSIndex}, destructions::NTuple{_,::BoseFSIndex}
    ) -> Int

Compute the squared value of an excitation from indices. Starts by applying all destruction
operators, and then applying all creation operators. The operators must be given in reverse
order. Will return 0 if move is illegal.
"""
@inline bose_excitation_value(::Tuple{}, ::Tuple{}) = 1
@inline function bose_excitation_value((c, cs...)::NTuple{<:Any,BoseFSIndex}, ::Tuple{})
    return bose_excitation_value(map(_create(c), cs), ()) * (c.occnum + 1)
end
@inline function bose_excitation_value(
    creations::NTuple{<:Any,BoseFSIndex}, (d, ds...)::NTuple{<:Any,BoseFSIndex}
)
    return bose_excitation_value(map(_destroy(d), creations), map(_destroy(d), ds)) * d.occnum
end

"""
    from_bose_onr(::Type{B}, onr::AbstractArray) -> B

Convert array `onr` to type `B`. It is safe to assume `onr` contains a valid
occupation-number representation array. The checks are preformed in the [`BoseFS`](@ref)
constructor.

This function is a part of the interface for an underlying storage format used by
[`BoseFS`](@ref).
"""
from_bose_onr

"""
    to_bose_onr(bs::B) -> SVector

Convert `bs` to a static vector in the occupation number representation format.

This function is a part of the interface for an underlying storage format used by
[`BoseFS`](@ref).
"""
to_bose_onr

"""
    bose_excitation(
        bs::B, creations::NTuple{N,BoseFSIndex}, destructions::NTuple{N,BoseFSIndex}
    ) -> Tuple{B,Float64}

Perform excitation as if `bs` was a bosonic address. See also
[`bose_excitation_value`](@ref).

This function is a part of the interface for an underlying storage format used by
[`BoseFS`](@ref).
"""
bose_excitation

"""
    bose_num_occupied_modes(bs::B)

Return the number of occupied modes, if `bs` represents a bosonic address.

This function is a part of the interface for an underlying storage format used by
[`BoseFS`](@ref).
"""
bose_num_occupied_modes

###
### Fermion stuff
###
"""
    FermiFSIndex

Struct used for indexing and performing [`excitation`](@ref)s on a [`FermiFS`](@ref).

## Fields:

* `occnum`: the occupation number.
* `mode`: the index of the mode.
* `offset`: the position of the mode in the address. This is `mode - 1` when the address is
  represented by a bitstring, and the position in the list when using `SortedParticleList`.

"""
struct FermiFSIndex<:FieldVector{3,Int}
    occnum::Int
    mode::Int
    offset::Int
end

function Base.show(io::IO, i::FermiFSIndex)
    @unpack occnum, mode, offset = i
    print(io, "FermiFSIndex(occnum=$occnum, mode=$mode, offset=$offset)")
end
Base.show(io::IO, ::MIME"text/plain", i::FermiFSIndex) = show(io, i)

"""
    FermiOccupiedModes{N,S<:BitString}

Iterator over occupied modes in address. `N` is the number of fermions. See [`occupied_modes`](@ref).
"""
struct FermiOccupiedModes{N,S}<:OccupiedModeIterator
    storage::S
end

Base.length(::FermiOccupiedModes{N}) where {N} = N
Base.eltype(::FermiOccupiedModes) = FermiFSIndex

"""
    from_fermi_onr(::Type{B}, onr) -> B

Convert array `onr` to type `B`. It is safe to assume `onr` contains a valid
occupation-number representation array. The checks are preformed in the [`FermiFS`](@ref)
constructor.

This function is a part of the interface for an underlying storage format used by
[`FermiFS`](@ref).
"""
from_fermi_onr

"""
    fermi_find_mode(bs::B, i::Integer) -> FermiFSIndex

Find `i`-th mode in `bs` if `bs` is a fermionic address. Should return an appropriately
formatted [`FermiFSIndex`](@ref).

This function is a part of the interface for an underlying storage format used by
[`FermiFS`](@ref).
"""
fermi_find_mode

"""
    fermi_excitation(
        bs::B, creations::NTuple{N,FermiFSIndex}, destructions::NTuple{N,FermiFSIndex}
    ) -> Tuple{B,Float64}

Perform excitation as if `bs` was a fermionic address.

This function is a part of the interface for an underlying storage format used by
[`FermiFS`](@ref).
"""
fermi_excitation

###
### General
###
function LinearAlgebra.dot(occ_a::OccupiedModeIterator, occ_b::OccupiedModeIterator)
    (n_a, i_a, _), st_a = iterate(occ_a)
    (n_b, i_b, _), st_b = iterate(occ_b)

    acc = 0
    while true
        if i_a > i_b
            # b is behind and needs to do a step
            iter = iterate(occ_b, st_b)
            isnothing(iter) && return acc
            (n_b, i_b, _), st_b = iter
        elseif i_a < i_b
            # a is behind and needs to do a step
            iter = iterate(occ_a, st_a)
            isnothing(iter) && return acc
            (n_a, i_a, _), st_a = iter
        else
            # a and b are at the same position
            acc += n_a * n_b
            # now both need to do a step
            iter = iterate(occ_a, st_a)
            isnothing(iter) && return acc
            (n_a, i_a, _), st_a = iter
            iter = iterate(occ_b, st_b)
            isnothing(iter) && return acc
            (n_b, i_b, _), st_b = iter
        end
    end
end

function sparse_to_onr(M, pairs)
    onr = spzeros(Int, M)
    for (k, v) in pairs
        v ≥ 0 || throw(ArgumentError("Invalid pair `$k=>$v`: particle number negative"))
        0 < k ≤ M || throw(ArgumentError("Invalid pair `$k => $v`: key of of range `1:$M`"))
        onr[k] += v
    end
    return onr
end

"""
    OccupiedPairsMap(addr::SingleComponentFockAddress) <: AbstractVector

Get a map of all distinct pairs of indices in `addr`. Pairs involving 
multiply-occupied modes are counted once, (including self-pairing).
This is useful for cases where identifying pairs of particles for eg. 
interactions is not well-defined or efficient to do on the fly.
This is an eager iterator whose elements are a tuple of particle indices that 
can be given to `excitation`

# Example

```jldoctest
julia> addr = BoseFS((10, 0, 0, 0, 2, 0, 1))
BoseFS{13,7}((10, 0, 0, 0, 2, 0, 1))

julia> pairs = OccupiedModeMap(addr)
5-element OccupiedPairsMap{78, Tuple{BoseFSIndex, BoseFSIndex}}:
 (BoseFSIndex(occnum=10, mode=1, offset=0), BoseFSIndex(occnum=10, mode=1, offset=0))
 (BoseFSIndex(occnum=2, mode=5, offset=14), BoseFSIndex(occnum=2, mode=5, offset=14))
 (BoseFSIndex(occnum=2, mode=5, offset=14), BoseFSIndex(occnum=10, mode=1, offset=0))
 (BoseFSIndex(occnum=1, mode=7, offset=18), BoseFSIndex(occnum=10, mode=1, offset=0))
 (BoseFSIndex(occnum=1, mode=7, offset=18), BoseFSIndex(occnum=2, mode=5, offset=14))

julia> excitation(addr, pairs[2], pairs[4])
(BoseFS{13,7}((9, 0, 0, 0, 4, 0, 0)), 10.954451150103322)
```

See also [`OccupiedModeMap`](@ref).
"""
struct OccupiedPairsMap{N,T} <: AbstractVector{T}
    pairs::SVector{N,T}
    length::Int
end

function OccupiedPairsMap(addr::SingleComponentFockAddress{N}) where {N}
    omm = OccupiedModeMap(addr)
    T = eltype(omm)
    P = N * (N - 1) ÷ 2
    pairs = MVector{P,Tuple{T,T}}(undef)
    a = 0
    for i in eachindex(omm)
        p_i = omm[i]
        if p_i.occnum > 1
            a += 1
            @inbounds pairs[a] = (p_i, p_i)
        end
        for j in 1:i-1
            p_j = omm[j]
            a += 1
            @inbounds pairs[a] = (p_i, p_j)
        end
    end
    
    return OccupiedPairsMap(SVector(pairs), a)
end

Base.size(opm::OccupiedPairsMap) = (opm.length,)
function Base.getindex(opm::OccupiedPairsMap, i)
    @boundscheck 1 ≤ i ≤ opm.length || throw(BoundsError(opm, i))
    return opm.pairs[i]
end