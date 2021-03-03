"""
    AbstractFockAddress

Supertype representing a Fock state.
"""
abstract type AbstractFockAddress end

"""
    num_particles(::Type{<:AbstractFockAddress})

Number of particles represented by address.
"""
num_particles(b::AbstractFockAddress) = num_particles(typeof(b))

"""
    num_modes(::Type{<:AbstractFockAddress})

Number of modes represented by address.
"""
num_modes(b::AbstractFockAddress) = num_modes(typeof(b))

"""
    BoseFS{N,M,S} <: AbstractFockAddress
    BoseFS(bs::S) where S <: BitAdd
    BoseFS(bs::S, b)

Address type that represents a Fock state of `N` spinless bosons in `M` orbitals
by wrapping a bitstring of type `S`. Orbitals are stored in reverse
order, i.e. the first orbital in a `BoseFS` is stored rightmost in the
bitstring `bs`. If the number of significant bits `b` is not encoded in `S` it
must be passed as an argument (e.g. for `BSAdd64` and `BSAdd128`).

# Constructors

* `BoseFS{N,M}(::BitString)`: Unsafe constructor. Does not check whether the number of ones
  in a is equal to `N`.

* `BoseFS(::BitString)`: Automatically determine `N` and `M`. This constructor is not type
  stable!

* `BoseFS{[N,M,S]}(onr)`: Create `BoseFS{N,M}` from onr representation. This is efficient
  as long as at least `N` is provided.

"""
struct BoseFS{N,M,S<:BitString} <: AbstractFockAddress
    bs::S

    BoseFS{N,M,S}(bs::S) where {N,M,S} = new{N,M,S}(bs)
end

function BoseFS{N,M}(bs::BitString{B}) where {N,M,B}
    # Check for consistency between parameter, but NOT for the correct number of bits.
    N + M - 1 == B || throw(ArgumentError("type parameter mismatch"))
    return BoseFS{N,M,typeof(bs)}(bs)
end

function BoseFS(bs::BitString{B}) where B
    N = count_ones(bs)
    M = B - N + 1
    return BoseFS{N,M}(bs)
end

function BoseFS{N,M,S}(onr::Union{SVector{M},NTuple{M}}) where {N,M,S<:BitString{<:Any,1}}
    @boundscheck sum(onr) == N || error("invalid ONR")
    result = zero(UInt64)
    for i in M:-1:1
        curr_occnum = onr[i]
        result <<= curr_occnum + 1
        result |= one(UInt64) << curr_occnum - 1
    end
    return BoseFS{N,M,S}(S(SVector(result)))
end

function BoseFS{N,M,S}(onr::Union{SVector{M},NTuple{M}}) where {N,M,S<:BitString}
    @boundscheck sum(onr) == N || error("invalid ONR")
    K = num_chunks(S)
    result = zeros(MVector{K,UInt64})
    offset = 0
    bits_left = chunk_bits(S, K)
    i = 1
    j = K
    while true
        # Write number to result
        curr_occnum = onr[i]
        while curr_occnum > 0
            x = min(curr_occnum, bits_left)
            mask = (one(UInt64) << x - 1) << offset
            @inbounds result[j] |= mask
            bits_left -= x
            offset += x
            curr_occnum -= x

            if bits_left == 0
                j -= 1
                offset = 0
                bits_left = chunk_bits(S, j)
            end
        end
        offset += 1
        bits_left -= 1

        if bits_left == 0
            j -= 1
            offset = 0
            bits_left = chunk_bits(S, j)
        end
        i += 1
        i > M && break
    end
    return BoseFS{N,M}(S(SVector(result)))
end
function BoseFS{N,M,S}(onr::AbstractVector) where {N,M,S}
    return BoseFS{N,M,S}(SVector{M}(onr))
end
function BoseFS{N,M}(onr::Union{AbstractVector,Tuple}) where {N,M}
    S = typeof(BitString{N + M - 1}(0))
    return BoseFS{N,M,S}(SVector{M}(onr))
end
function BoseFS{N}(onr::Union{SVector{M},NTuple{M}}) where {N,M}
    return BoseFS{N,M}(onr)
end
function BoseFS(onr::Union{AbstractVector,Tuple})
    M = length(onr)
    N = sum(onr)
    return BoseFS{N,M}(onr)
end

function Base.show(io::IO, b::BoseFS{N,M,S}) where {N,M,S}
    print(io, "BoseFS{$N,$M}(", tuple(onr(b)...), ")")
end

Base.isless(a::BoseFS, b::BoseFS) = isless(a.bs, b.bs)
Base.hash(bba::BoseFS,  h::UInt) = hash(bba.bs, h)
num_particles(::Type{BoseFS{N,M,S}}) where {N,M,S} = N
num_modes(::Type{BoseFS{N,M,S}}) where {N,M,S} = M

"""
    nearUniformONR(N, M) -> onr::SVector{M,Int}

Create occupation number representation `onr` distributing `N` particles in `M`
modes in a close-to-uniform fashion with each orbital filled with at least
`N ÷ M` particles and at most with `N ÷ M + 1` particles.
"""
function nearUniformONR(n::Number, m::Number)
    return nearUniformONR(Val(n),Val(m))
end
function nearUniformONR(::Val{N}, ::Val{M}) where {N, M}
    fillingfactor, extras = divrem(N, M)
    # startonr = fill(fillingfactor,M)
    startonr = fillingfactor * @MVector ones(Int,M)
    startonr[1:extras] += ones(Int, extras)
    return SVector{M}(startonr)
end

"""
    nearUniform(BoseFS{N,M})
    nearUniform(BoseFS{N,M,S}) -> bfs::BoseFS{N,M,S}

Create bosonic Fock state with near uniform occupation number of `M` modes with
a total of `N` particles. Specifying the bit address type `S` is optional.

# Examples
```jldoctest
julia> nearUniform(BoseFS{7,5,BitString{14}})
BoseFS((2,2,1,1,1))

julia> nearUniform(BoseFS{7,5})
BoseFS((2,2,1,1,1))
```
"""
function nearUniform(::Type{<:BoseFS{N,M}}) where {N,M}
    return BoseFS{N,M}(nearUniformONR(Val(N),Val(M)))
end
nearUniform(b::BoseFS) = nearUniform(typeof(b))

"""
    onr(bs)

Compute and return the occupation number representation of the bit string
address `bs` as an `SVector{M,Int32}`, where `M` is the number of orbitals.
"""
onr(bba::BoseFS) = SVector(m_onr(bba))

"""
    m_onr(bs)

Compute and return the occupation number representation of the bit string
address `bs` as an `MVector{M,Int32}`, where `M` is the number of orbitals.
"""
@inline m_onr(bba::BoseFS) = m_onr(Val(num_chunks(bba.bs)), bba)

# Version specialized for single-chunk addresses.
@inline function m_onr(::Val{1}, bba::BoseFS{N,M}) where {N,M}
    result = zeros(MVector{M,Int32})
    address = bba.bs
    for orbital in 1:M
        bosons = Int32(trailing_ones(address))
        @inbounds result[orbital] = bosons
        address >>>= (bosons + 1) % UInt
        iszero(address) && break
    end
    return result
end

# Version specialized for multi-chunk addresses. This is quite a bit faster for large
# addresses.
@inline function m_onr(::Val{K}, bba::BoseFS{N,M}) where {K,N,M}
    B = num_bits(bba.bs)
    result = zeros(MVector{M,Int32})
    address = bba.bs
    orbital = 1
    i = K
    while true
        chunk = chunks(address)[i]
        bits_left = chunk_bits(address, i)
        while !iszero(chunk)
            bosons = trailing_ones(chunk)
            @inbounds result[orbital] += unsafe_trunc(Int32, bosons)
            chunk >>>= bosons % UInt
            empty_modes = trailing_zeros(chunk)
            orbital += empty_modes
            chunk >>>= empty_modes % UInt
            bits_left -= bosons + empty_modes
        end
        i == 1 && break
        i -= 1
        orbital += bits_left
    end
    return result
end

"""
    occupied_orbitals(b)

Iterate over occupied orbitals in `BoseFS` address. Returns tuples of
`(boson_number, orbital_number, bit_offset)`.

Note that the `bit_offset` is zero-based!

# Example

```jldoctest
julia> b = BoseFS((1,5,0,4))
julia> for (n, i) in occupied_orbitals(b)
    @show n, i
end
(n, i) = (1, 1)
(n, i) = (5, 2)
(n, i) = (4, 4)

"""
struct OccupiedOrbitalIterator{C,S}
    address::S
end

function occupied_orbitals(b::BoseFS{N,M,S}) where {N,M,S}
    return OccupiedOrbitalIterator{num_chunks(S),S}(b.bs)
end

# Single chunk versions are simpler.
@inline function Base.iterate(osi::OccupiedOrbitalIterator{1})
    empty_orbitals = trailing_zeros(osi.address)
    return iterate(osi, (osi.address >> empty_orbitals, empty_orbitals, 1 + empty_orbitals))
end
@inline function Base.iterate(osi::OccupiedOrbitalIterator{1}, (chunk, bit, orbital))
    if iszero(chunk)
        return nothing
    else
        bosons = trailing_ones(chunk)
        chunk >>>= (bosons % UInt)
        empty_orbitals = trailing_zeros(chunk)
        chunk >>>= (empty_orbitals % UInt)
        next_bit = bit + bosons + empty_orbitals
        return (bosons, orbital, bit), (chunk, next_bit, orbital + empty_orbitals)
    end
end

@inline function Base.iterate(osi::OccupiedOrbitalIterator)
    address = osi.address
    i = num_chunks(address)
    chunk = chunks(address)[i]
    bits_left = chunk_bits(address, i)
    orbital = 1
    return iterate(osi, (i, chunk, bits_left, orbital))
end
@inline function Base.iterate(osi::OccupiedOrbitalIterator, (i, chunk, bits_left, orbital))
    i < 1 && return nothing
    address = osi.address
    S = typeof(address)
    bit_position = 0

    # Remove and count trailing zeros.
    empty_orbitals = min(trailing_zeros(chunk), bits_left)
    chunk >>>= empty_orbitals % UInt
    bits_left -= empty_orbitals
    orbital += empty_orbitals
    while bits_left < 1
        i -= 1
        i < 1 && return nothing
        @inbounds chunk = chunks(address)[i]
        bits_left = chunk_bits(S, i)
        empty_orbitals = min(bits_left, trailing_zeros(chunk))
        orbital += empty_orbitals
        bits_left -= empty_orbitals
        chunk >>>= empty_orbitals % UInt
    end

    bit_position = chunk_bits(S, i) - bits_left + 64 * (num_chunks(address) - i)

    # Remove and count trailing ones.
    result = 0
    bosons = trailing_ones(chunk)
    bits_left -= bosons
    chunk >>>= bosons % UInt
    result += bosons
    while bits_left < 1
        i -= 1
        i < 1 && break
        @inbounds chunk = chunks(address)[i]
        bits_left = chunk_bits(S, i)

        bosons = trailing_ones(chunk)
        bits_left -= bosons
        result += bosons
        chunk >>>= bosons % UInt
    end
    return (result, orbital, bit_position), (i, chunk, bits_left, orbital)
end

"""
    BoseFS2C{NA,NB,M,AA,AB} <: AbstractFockAddress

Address type that constructed with two [`BoseFS{N,M,S}`](@ref). It represents a
Fock state with two components, e.g. two different species of bosons with particle
number `NA` from species S and particle number `NB` from species B. The number of
orbitals `M` is expected to be the same for both components.
"""
struct BoseFS2C{NA,NB,M,SA,SB} <: AbstractFockAddress
    bsa::BoseFS{NA,M,SA}
    bsb::BoseFS{NB,M,SB}
end

BoseFS2C(onr_a::Tuple, onr_b::Tuple) = BoseFS2C(BoseFS(onr_a),BoseFS(onr_b))

function Base.show(io::IO, b::BoseFS2C{NA,NB,M,AA,AB}) where {NA,NB,M,AA,AB}
    print(io, "BoseFS2C(")
    Base.show(io,b.bsa)
    print(io, ",")
    Base.show(io,b.bsb)
    print(io, ")")
end

num_particles(::Type{<:BoseFS2C{NA,NB}}) where {NA,NB} = NA + NB
num_modes(::Type{<:BoseFS2C{<:Any,<:Any,M}}) where {M} = M
