"""
    ONRFS{B,M,S} <: SingleComponentFockAddress{missing,M}

Address type that represents a bosonic Fock state with an indeterminate number of particles
in `M` modes. Each mode can be occupied by at most ``2^B-2`` particles. The number of
particles can be obtained by [`num_particles`](@ref) but is runtime information and not part
of the type, in contrast to [`BoseFS`](@ref). This makes this type suitable for representing
Fock states with number-nonconcerving Hamiltonians.
"""
struct ONRFS{BITS,M,S} <: SingleComponentFockAddress{missing,M}
    bs::S
end

@inline function ONRFS{BITS}(onr::Union{SVector{M},MVector{M},NTuple{M}}) where {BITS,M}
    bs = onrbs_from_onr(onr, Val(BITS))
    return ONRFS{BITS,M,typeof(bs)}(bs)
end

function ONRFS(onr, max_n)
    BITS = ceil(Int, log2(max_n+1))
    return ONRFS{BITS}(onr)
end

@inline function onrbs_from_onr(
    onr::Union{SVector{M},MVector{M},NTuple{M}}, ::Val{BITS}
) where {M,BITS}
    B = BITS * M # number of bits in the bitstring
    if B ≤ 8
        T = UInt8
    elseif B ≤ 16
        T = UInt16
    elseif B ≤ 32
        T = UInt32
    elseif B ≤ 64
        T = UInt64
    elseif B ≤ 128
        T = UInt128
    else
        T = BigInt
    end
    val = zero(T)
    for i in 1:M
        val |= T(onr[i]) << (BITS*(i-1))
    end
    return BitString{B}(val)
end

function onr(ofs::ONRFS{BITS,M,S}) where {BITS,M,B,C,S<:BitString{B,C}}
    # onr = SVector{M}(ofs.bs >> (BITS * (i - 1)) & (2^BITS - 1) for i in 1:M)
    @assert BITS * M == B
    onr = MVector{M,Int32}(undef)
    bs = ofs.bs
    for i in 1:M
        onr[i] = bs.chunks[C] & (2^BITS - 1)
        bs >>= BITS
    end
    return SVector(onr)
end

function print_address(io::IO, ofs::ONRFS{BITS,M,S}; compact=false) where {BITS,M,S}
    if compact
        print(io, BITS, "|", join(onr(ofs), ' '), "⟩")
    else
        print(io, "ONRFS{", BITS, "}(", tuple(onr(ofs)...), ")")
    end
    # print(io, "ONRFS{", BITS, "}(", tuple(onr(ofs)...), ")")
end
