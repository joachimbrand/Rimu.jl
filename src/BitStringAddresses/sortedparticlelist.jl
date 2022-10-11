"""
    select_int_type(M)

Select unsigned integer type that can hold values up to `M`.
"""
function select_int_type(M)
    if M ≤ 0
        throw(ArgumentError("`M` must be positive!"))
    elseif M ≤ typemax(UInt8)
        return UInt8
    elseif M ≤ typemax(UInt16)
        return UInt16
    elseif M ≤ typemax(UInt32)
        return UInt32
    else
        return UInt64
    end
end

"""
    SortedParticleList{N,M,T<:Unsigned}

Type for storing sparse fock states. Stores the mode number of each particle as an entry
with only its mode stored. The entries are always kept sorted.

Iterating over `SortedParticleList`s yields occupied modes as a tuple of occupation number,
mode number, and position in list.

# Constructors

* `SortedParticleList{N,M,T}(::SVector{N,T})`: unsafe constructor. Does not sort input.

* `SortedParticleList(arr::AbstractVector)`: convert ONR to `SortedParticleList`

"""
struct SortedParticleList{N,M,T<:Unsigned}
    storage::SVector{N,T}
end
function Base.show(io::IO, ss::SortedParticleList{N,M,T}) where {N,M,T}
    print(io, "SortedParticleList{$N,$M,$T}(", Int.(ss.storage), ")")
end

function SortedParticleList{N,M}() where {N,M}
    T = select_int_type(M)
    return SortedParticleList{N,M,T}(ones(SVector{N,T}))
end
function from_onr(::Type{S}, onr) where {N,M,T,S<:SortedParticleList{N,M,T}}
    spl = zeros(MVector{N,T})
    curr = 1
    for (n, v) in enumerate(onr)
        for _ in 1:v
            spl[curr] = n
            curr += 1
        end
    end
    return SortedParticleList{N,M,T}(SVector(spl))
end

function Base.isless(ss1::SortedParticleList, ss2::SortedParticleList)
    return isless(ss1.storage, ss2.storage)
end

num_particles(::Type{<:SortedParticleList{N}}) where {N} = N
num_modes(::Type{<:SortedParticleList{<:Any,M}}) where {M} = M

###
### General functions
###
Base.eltype(::SortedParticleList) = Tuple{Int,Int,Int}

function Base.length(ss::SortedParticleList{<:Any,<:Any,T}) where {T}
    curr = zero(T)
    res = 0
    for i in ss.storage
        res += (i != curr)
        curr = i
    end
    return res
end

function Base.iterate(ss::SortedParticleList{N}, i=1) where {N}
    @inbounds if i > N
        return nothing
    else
        occnum = 1
        mode = ss.storage[i]
        offset = i
        i += 1
        while i ≤ N && ss.storage[i] == mode
            occnum += 1
            i += 1
        end
        return (occnum, Int(mode), i - occnum - 1), i
    end
end

function Base.reverse(ss::SortedParticleList{N,M,T}) where {N,M,T}
    new_storage = map(reverse(ss.storage)) do i
        T(M) - i + one(T)
    end
    return SortedParticleList{N,M,T}(new_storage)
end

# Somehow this is faster than the default method.
Base.hash(ss::SortedParticleList, u::UInt) = hash(ss.storage, u)

# In this case, getting the ONR is the same for bosons and fermions, and assumes the address
# is not malformed.
function onr(ss::SortedParticleList{<:Any,M}) where {M}
    mvec = zeros(MVector{M,Int})
    @inbounds for (occnum, mode, _) in ss
        mvec[mode] = occnum
    end
    return SVector(mvec)
end

# Same as above.
function find_mode(ss::SortedParticleList, n)
    offset = 0
    for (occnum, mode, _) in ss
        if mode == n
            return (occnum, mode, offset)
        elseif mode > n
            return (0, n, offset)
        end
        offset += occnum
    end
    return (0, n, offset)
end

"""
    move_particles(ss::SortedParticleList, dsts, srcs)

Move several particles at once. Moves `srcs[i]` to `dsts[i]`.  `dsts` and `srcs` should be
tuples of [`BoseFSIndex`](@ref) or [`FermiFSIndex`](@ref). The legality of the moves is not
checked - the result of an illegal move is undefined!
"""
function move_particles(ss::SortedParticleList{N,M,T}, dsts, srcs) where {N,M,T}
    new_storage = ss.storage
    for (dst, src) in zip(dsts, srcs)
        src_pos = 1
        @inbounds for i in 1:N
            src_pos = max(src_pos, i * (new_storage[i] == src.mode % T))
        end
        @boundscheck 0 < dst.mode ≤ M || throw(BoundsError(ss, dst.mode))
        new_storage = setindex(new_storage, dst.mode % T, src_pos)
    end
    return SortedParticleList{N,M,T}(sort(new_storage))
end

###
### Bose interface
###
function from_bose_onr(::Type{S}, onr) where{S<:SortedParticleList}
    from_onr(S, onr)
end
to_bose_onr(ss::SortedParticleList, _) = onr(ss)

bose_num_occupied_modes(ss::SortedParticleList) = length(ss)

bose_find_mode(ss::SortedParticleList, n) = find_mode(ss, n)

@inline function bose_excitation(
    ss::SortedParticleList{N,M,T}, creations, destructions
) where {N,M,T}
    creations_rev = reverse(creations)
    value = bose_excitation_value(creations_rev, destructions)
    if iszero(value)
        return ss, 0.0
    else
        return move_particles(ss, creations_rev, destructions), √value
    end
end

Base.length(bom::BoseOccupiedModes{<:Any,<:Any,<:SortedParticleList}) = length(bom.storage)
function Base.iterate(bom::BoseOccupiedModes{<:Any,<:Any,<:SortedParticleList}, i=1)
    it = iterate(bom.storage, i)
    if isnothing(it)
        return nothing
    else
        res, i = it
        return BoseFSIndex(res...), i
    end
end

###
### Fermi interface
###
function from_fermi_onr(::Type{S}, onr) where {S<:SortedParticleList}
    from_onr(S, onr)
end

# Fix offsets and occupation numbers after creation/destruction operator is applied.
# The idea behind these is to allow computing the value from the indices alone.
function _fix_pos_create(c, index)
    index = @set index.offset += (c.mode < index.mode)
    index = @set index.occnum += (c.mode == index.mode)
    return index
end
_fix_pos_create(c) = Base.Fix1(_fix_pos_create, c)
function _fix_pos_destroy(d, index)
    index = @set index.offset -= (d.mode < index.mode)
    index = @set index.occnum -= (d.mode == index.mode)
    return index
end
_fix_pos_destroy(d) = Base.Fix1(_fix_pos_destroy, d)

"""
    fermi_excitation_value_spl(
        creations::NTuple{_,FermiFSIndex}, destructions::NTuple{_,::FermiFSIndex}
    ) -> {-1,0,1}

Compute the value of an excitation from indices. Starts by applying all destruction
operators, and then applying all creation operators. The operators must be given in reverse
order. Will return 0 if move is illegal.

Note that this function only works on indices obtained from a [`SortedParticleList`](@ref).
"""
@inline fermi_excitation_value_spl(::Tuple{}, ::Tuple{}) = 1.0
@inline function fermi_excitation_value_spl((c, cs...), ::Tuple{})
    cs = map(_fix_pos_create(c), cs)
    return fermi_excitation_value_spl(cs, ()) * ifelse(isodd(c.offset), -1, 1) * (c.occnum == 0)
end
@inline function fermi_excitation_value_spl(cs, (d, ds...))
    cs = map(_fix_pos_destroy(d), cs)
    ds = map(_fix_pos_destroy(d), ds)
    return fermi_excitation_value_spl(cs, ds) * ifelse(isodd(d.offset), -1, 1) * (d.occnum == 1)
end

fermi_find_mode(ss::SortedParticleList, n) = FermiFSIndex(find_mode(ss, n))
function fermi_find_mode(ss::SortedParticleList, ns::Tuple)
    # It's OK to do that instead of the fancy method used with bosons, because the
    # assumption is that `N` is small.
    return map(n -> FermiFSIndex(find_mode(ss, n)), ns)
end

@inline function fermi_excitation(
    ss::SortedParticleList{N,M,T}, creations::NTuple{K}, destructions::NTuple{K}
) where {N,M,T,K}
    creations_rev = reverse(creations)
    destructions_rev = reverse(destructions)
    value = fermi_excitation_value_spl(creations_rev, destructions_rev)
    if iszero(value)
        return ss, 0.0
    else
        return move_particles(ss, creations_rev, destructions), float(value)
    end
end

Base.length(fom::FermiOccupiedModes{N,<:SortedParticleList}) where {N} = length(fom.storage)
function Base.iterate(fom::FermiOccupiedModes{<:Any,<:SortedParticleList}, i=1)
    itr = iterate(fom.storage, i)
    if isnothing(itr)
        return nothing
    else
        res, i = itr
        return FermiFSIndex(res...), i
    end
end
