struct FermiFS{N,M,S<:BitString{M}} <: AbstractFockAddress
    bs::S

    FermiFS{N,M,S}(bs::S) where {N,M,S<:BitString{M}} = new{N,M,S}(bs)
end

function FermiFS{N,M,S}(onr::Union{SVector{M},NTuple{M}}) where {N,M,C,T,S<:BitString{M,C,T}}
    @boundscheck sum(onr) == N && all(in((0, 1)), onr) || error("invalid ONR")
    result = zero(SVector{C,T})
    j = length(result)
    for orbital in 1:M
        i = mod1(orbital, 64)
        new = result[j]
        new |= UInt64(onr[i]) << UInt64(i - 1)
        result = setindex(result, new, j)
        j -= (i == 64)
    end
    return FermiFS{N,M,S}(S(SVector(result)))
end

function FermiFS{N,M}(onr::Union{AbstractVector,Tuple}) where {N,M}
    S = typeof(BitString{M}(0))
    return FermiFS{N,M,S}(SVector{M}(onr))
end
function FermiFS{N}(onr::Union{SVector{M},Tuple{M}}) where {N,M}
    return FermiFS{N,M}(onr)
end
function FermiFS(onr::Union{AbstractVector,Tuple})
    M = length(onr)
    N = sum(onr)
    return FermiFS{N,M}(onr)
end

function Base.show(io::IO, f::FermiFS{N,M}) where {N,M}
    print(io, "FermiFS{$N,$M}(", tuple(onr(f)...), ")")
end

Base.isless(a::FermiFS, b::FermiFS) = isless(a.bs, b.bs)
Base.hash(a::FermiFS,  h::UInt) = hash(a.bs, h)
Base.:(==)(a::FermiFS, b::FermiFS) = a.bs == b.bs
num_particles(::Type{FermiFS{N,M,S}}) where {N,M,S} = N
num_modes(::Type{FermiFS{N,M,S}}) where {N,M,S} = M
num_components(::Type{<:FermiFS}) = 1

onr(a::FermiFS) = SVector(m_onr(a))

@inline function m_onr(a::FermiFS{<:Any,M}) where {M}
    result = zero(MVector{M,Int32})
    j = num_chunks(a.bs)
    @inbounds for orbital in 1:M
        i = mod1(orbital, 64)
        result[orbital] = (chunks(a.bs)[j] & (1 << (i - 1))) > 0
        j -= (i == 64)
    end
    return result
end

function is_occupied(a::FermiFS{<:Any,M}, orbital) where {M}
    @boundscheck 1 ≤ orbital ≤ M || throw(BoundsError(a, orbital))
    j, i = fldmod1(orbital, 64)
    return a.bs.chunks[end + 1 - j] & (UInt(1) << UInt(i - 1)) > 0
end

"""
    move_particle(a::FermiFS, from, to)

Move particle from location `from` to location `to`. Note: the check that `from` and `to` are
valid can be elided with `@inbounds`.
"""
function move_particle(a::FermiFS{<:Any,<:Any,S}, from, to) where {T,S<:BitString{<:Any,1,T}}
    bs = a.bs.chunks[1]

    # Masks that locate positions `from` and `to`.
    from_mask = T(1) << T(from - 1)
    to_mask = T(1) << T(to - 1)
    # Mask for counting how many particles lie between them.
    between_mask = T(2^(abs(from - to) - 1) - 1) << T(min(from, to))

    @boundscheck begin
        bs & from_mask > 0 || error("`from` has no particle")
        bs & to_mask == 0 || error("`to` is already occupied")
    end
    bs ⊻= from_mask | to_mask
    num_between = count_ones(bs & between_mask)
    return typeof(a)(S(bs)), (-1)^num_between
end
