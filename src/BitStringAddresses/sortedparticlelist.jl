export SortedParticleList

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

Type for storing sparse fock states. Stores the mode number of each particle as an entry.
The entries are always kept sorted.

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
function SortedParticleList(arr)
    M = length(arr)
    T = select_int_type(M)
    vals = T[]
    for (i, v) in enumerate(arr)
        for _ in 1:v
            push!(vals, i)
        end
    end
    N = length(vals)
    return SortedParticleList{N,M,T}(SVector{N,T}(vals))
end

###
### General functions
###
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
        return (Int(mode), occnum, i - 1), i
    end
end

function Base.reverse(ss::SortedParticleList{N,M,T}) where {N,M,T}
    new_storage = map(reverse(ss.storage)) do i
        T(M) - i + one(T)
    end
    return SortedParticleList{N,M,T}(new_storage)
end

# In this case, getting the ONR is the same for bosons and fermions, assuming the address
# is not malformed.
function onr(ss::SortedParticleList{<:Any,M}) where {M}
    mvec = zeros(MVector{M,Int})
    @inbounds for (mode, occnum) in ss
        mvec[mode] = occnum
    end
    return SVector(vec)
end

# Same as above.
function find_mode(ss::SortedParticleList, n)
    for (mode, occnum, offset) in ss
        if mode == n
            return (mode, occnum, offset)
        elseif mode > n
            return (0, 0, 0)
        end
    end
    return (0, 0, 0)
end

###
### Bose interface
###
@inline function bose_excitation(ss::SortedParticleList{N,M,T}, srcs, dsts) where {N,M,T}
    new_storage = ss.storage
    src_pos = 1
    for (src, dst) in zip(srcs, dsts)
        @inbounds for i in 1:N
            src_pos = max(src_pos, i * (ss.storage[i] == src % T))
        end
        new_storage = setindex(new_storage, dst % T, src_pos)
    end
    return SortedParticleList{N,M,T}(sort(new_storage))
end
bose_onr(ss::SortedParticleList) = onr(ss)
bose_find_mode(ss::SortedParticleList, n) = find_mode(ss, n)
bose_num_occupied_modes(ss::SortedParticleList) = length(ss)

Base.length(bom::BoseOccupiedModes{<:SortedParticleList}) = length(bom)
function Base.iterate(bom::BoseOccupiedModes{<:SortedParticleList}, i=1)
    res, i = iterate(bom, i)
    return BoseFSIndex(res...), i
end

###
### Fermi interface
###
@inline function fermi_excitation(ss::SortedParticleList{N,M,T}, srcs, dsts) where {N,M,T}
    new_storage = ss.storage
    src_pos = 1
    dst_pos = 1
    for (src, dst) in zip(srcs, dsts)
        @inbounds for i in 1:N
            src_pos = max(src_pos, i * (ss.storage[i] == src % T))
            dst_pos = max(dst_pos, i * (ss.storage[i] < dst % T))
        end
        new_storage = setindex(new_storage, dst % T, src_pos)
    end
    return SortedParticleList{N,M,T}(sort(new_storage))
end
fermi_onr(ss::SortedParticleList) = onr(ss)
fermi_find_mode(ss::SortedParticleList, n) = find_mode(ss::SortedParticleList, n)

Base.length(fom::FermiOccupiedModes{N,<:SortedParticleList}) where {N} = length(bom)
function Base.iterate(fom::FermiOccupiedModes{<:Any,<:SortedParticleList}, i=1)
    res, i = iterate(fom, i)
    return FermiFSIndex(res...), i
end
