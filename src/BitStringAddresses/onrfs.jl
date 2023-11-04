"""
    ONRFS{B,M,S} <: SingleComponentFockAddress{missing,M}

Address type that represents a bosonic Fock state with an indeterminate number of particles
in `M` modes. Each mode can be occupied by at most ``2^B-1`` particles. The number of
particles can be obtained by [`num_particles`](@ref) but is runtime information and not part
of the type, in contrast to [`BoseFS`](@ref). This makes this type suitable for representing
Fock states with number-nonconcerving Hamiltonians. [`excitation`](@ref)s can change the
particle number and use integer mode indices.

## Constructors

* `ONRFS{B}(onr)` constructs an `ONRFS`  with `M` modes and `B` bits per mode from an
    occupation number representation `onr` of length `M`. The occupation number
    representation is a tuple of integers, each of which is less than `2^B`.
* `ONRFS(onr, max_n)` constructs an `ONRFS` from an occupation number representation
    `onr` with `M` modes. The maximum occupation number `max_n` is used to determine the
    number of bits per mode. The number of bits per mode is the smallest integer `B` such
    that `2^B > max_n`.
* [`@fs_str`](@ref): Addresses are sometimes printed in a compact manner. This
  representation can also be used as a constructor. See the example below.

## Examples

```jldoctest
julia> ofs = ONRFS((0, 1, 5, 1, 0), 5)
ONRFS{3}((0, 1, 5, 1, 0))

julia> ONRFS{3}((0, 1, 5, 1, 0)) == ofs
true

julia> fs"|0 1 5 1 0⟩{3}" == ofs
true

julia> num_particles(ofs)
7

julia> excitation(ofs, (1,2), (3,))
(ONRFS{3}((1, 2, 4, 1, 0)), 3.1622776601683795)
```
See also: [`SingleComponentFockAddress`](@ref), [`BoseFS`](@ref).
"""
struct ONRFS{BITS,M,S} <: SingleComponentFockAddress{missing,M}
    bs::S
end

# typestable constructor from occupation number representation
@inline function ONRFS{BITS}(onr::Union{SVector{M},MVector{M},NTuple{M}}) where {BITS,M}
    bs = onrbs_from_onr(onr, Val(BITS))
    return ONRFS{BITS,M,typeof(bs)}(bs)
end

# constructor required for `onr_excitation`
@inline function ONRFS{BITS,M,B}(onr::Union{SVector{M},MVector{M},NTuple{M}}) where {BITS,M,B}
    bs = onrbs_from_onr(onr, Val(BITS))::B
    return ONRFS{BITS,M,B}(bs)
end

# type-unstable constructor from occupation number representation
function ONRFS(onr, max_n)
    BITS = ceil(Int, log2(max_n+1))
    return ONRFS{BITS}(onr)
end

# create bitstring for ONRFS from occupation number representation
# TODO: optimize
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
        print(io, "|", join(onr(ofs), ' '), "⟩{", BITS, "}")
    else
        print(io, "ONRFS{", BITS, "}(", tuple(onr(ofs)...), ")")
    end
    # print(io, "ONRFS{", BITS, "}(", tuple(onr(ofs)...), ")")
end

# Baseline version for now. TODO: optimize
function Base.getindex(ofs::ONRFS, i::Int)
    return onr(ofs)[i]
end

function Base.length(ofs::ONRFS{<:Any,M}) where {M}
    return M
end

function Base.iterate(ofs::ONRFS{<:Any,M}, state=1) where {M}
    if state > M
        return nothing
    else
        return ofs[state], state+1
    end
end

num_occupied_modes(ofs::ONRFS) = mapreduce(!iszero, +, onr(ofs))

num_particles(ofs::ONRFS) = mapreduce(identity, +, onr(ofs))

# TODO: Custom fast `excitation` for `ONRFS`
