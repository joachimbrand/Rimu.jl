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
"""
struct BoseOccupiedModes{N,M,S}
    storage::S
end

# Apply destruction operator to BoseFSIndex.
@inline _destroy(d, index::BoseFSIndex) = @set index.occnum -= (d.mode == index.mode)
@inline _destroy(d) = Base.Fix1(_destroy, d)
# Apply creation operator to BoseFSIndex.
@inline _create(c, index::BoseFSIndex) = @set index.occnum += (c.mode == index.mode)
@inline _create(c) = Base.Fix1(_create, c)

# Compute the value of an excitation. Starts by applying all destruction operators, and
# then applying all creation operators. The operators must be given in reverse order.
# Will return 0 if move is illegal.
@inline compute_excitation_value(::Tuple{}, ::Tuple{}) = 1
@inline function compute_excitation_value((c, cs...)::NTuple{<:Any,BoseFSIndex}, ::Tuple{})
    return compute_excitation_value(map(_create(c), cs), ()) * (c.occnum + 1)
end
@inline function compute_excitation_value(
    creations::NTuple{<:Any,BoseFSIndex}, (d, ds...)::NTuple{<:Any,BoseFSIndex}
)
    return compute_excitation_value(map(_destroy(d), creations), map(_destroy(d), ds)) * d.occnum
end


###
###
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
    @unpack occnum, mode = i
    print(io, "FermiFSIndex(occnum=$occnum, mode=$mode)")
end
Base.show(io::IO, ::MIME"text/plain", i::FermiFSIndex) = show(io, i)

"""
    FermiOccupiedModes{N,S<:BitString}

Iterator over occupied modes in address. `N` is the number of fermions. See [`occupied_modes`](@ref).
"""
struct FermiOccupiedModes{N,S}
    bs::S
end
