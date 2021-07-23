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

@inline function BoseFS{N,M,S}(
    onr::Union{SVector{M},NTuple{M}}
) where {N,M,S<:BitString{<:Any,1}}
    @boundscheck sum(onr) == N || error("invalid ONR")
    T = chunk_type(S)
    result = zero(T)
    for i in M:-1:1
        curr_occnum = T(onr[i])
        result <<= curr_occnum + T(1)
        result |= one(T) << curr_occnum - T(1)
    end
    return BoseFS{N,M,S}(S(SVector(result)))
end

@inline function BoseFS{N,M,S}(onr::Union{SVector{M},NTuple{M}}) where {N,M,S<:BitString}
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
Base.bitstring(b::BoseFS) = bitstring(b.bs)

Base.isless(a::BoseFS, b::BoseFS) = isless(a.bs, b.bs)
Base.hash(bba::BoseFS,  h::UInt) = hash(bba.bs, h)
Base.:(==)(a::BoseFS, b::BoseFS) = a.bs == b.bs
num_particles(::Type{BoseFS{N,M,S}}) where {N,M,S} = N
num_modes(::Type{BoseFS{N,M,S}}) where {N,M,S} = M
num_components(::Type{<:BoseFS}) = 1

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
nearUniform(b::AbstractFockAddress) = nearUniform(typeof(b))

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

function occupied_orbitals(b::BoseFS{<:Any,<:Any,S}) where {S}
    return OccupiedOrbitalIterator{num_chunks(S),S}(b.bs)
end

# Single chunk versions are simpler.
@inline function Base.iterate(osi::OccupiedOrbitalIterator{1})
    chunk = osi.address.chunks[1]
    empty_orbitals = trailing_zeros(chunk)
    return iterate(
        osi, (chunk >> (empty_orbitals % UInt), empty_orbitals, 1 + empty_orbitals)
    )
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
    find_sites(b::BoseFS, ks...)

Find the `k`-th site (occupied or not).

Return bit offset and occupation number. If multiple `k`s are given, find multiple sites in
a single pass over the address.
"""
function find_sites(b::BoseFS, ks...)
    # This is somewhat convoluted, but it allows us to find multiple sites in a single pass.
    #
    # The idea is to pass around an occupied orbital iterator and construct a tuple with
    # the results as you go along. Since we know the number of sites we're looking for, julia
    # is able to infer the length of the tuple.
    return _find_sites(b::BoseFS, ks, iterate(occupied_orbitals(b)), (0, 0, 0))
end
_find_sites(b::BoseFS, ::Tuple{}, _, _) = ()
function _find_sites(b::BoseFS, (k, ks...), state, last)
    iter = occupied_orbitals(b)
    while !isnothing(state)
        (occnum, site, offset), st = state
        there = k - site
        if there == 0
            # We are done with this site.
            # Call the next step with the next iteration.
            result = offset, occnum
            return (result, _find_sites(b, ks, iterate(iter, st), (occnum, site, offset))...)
        elseif there < 0
            # The desired site was before the one found in iteration.
            # Call the next step, but do not iterate yet.
            result = offset + there, 0
            return (result, _find_sites(b, ks, state, last)...)
        else
            # k-th site not found yet, look further.
            state = iterate(iter, st)
        end
    end
    # Remaining sites are after the last occupied site.
    last_offset, last_occnum, last_site = last
    result = (last_offset + last_occnum + k - last_site, 0)
    return (result, _find_sites(b, ks, state, last)...)
end

"""
    find_particles(b::BoseFS, i; n=1)

Find the `i`-th occupied site with an occupation number of at least `n`.

Return the bit offset, the site and the occupation number. If multiple `i`s are given, find
multiple occupied sites in a single pass over the address.
"""
function find_particles(b::BoseFS, is...; n=1)
    # This uses the same idea as find_site, explained above.
    return _find_particles(b::BoseFS, is, 0, n, iterate(occupied_orbitals(b)))
end

_find_particles(::BoseFS, ::Tuple{}, _, _, _) = ()
function _find_particles(b::BoseFS, (i, is...), counter, n, state)
    iter = occupied_orbitals(b)
    while !isnothing(state)
        (occnum, site, offset), st = state
        counter += occnum ≥ n
        if counter == i
            result = (offset, site, occnum)
            return (result, _find_particles(b, is, counter, n, iterate(iter, st))...)
        else
            state = iterate(iter, st)
        end
    end
    # This should never be returned unless we are out of bounds.
    return ((0, 0, 0), )
end

"""
    move_particle(b::BoseFS, n, by)

Move the `n`-th particle by `by` sites. Negative values of `by` move the particle to the
left and positive values move it to the right.

This is equivalent to applying a destruction operator followed by a creation operator to the
address.

Return the new fock state and the product of the occupation numbers.
"""
function move_particle(b::BoseFS{<:Any,M}, n, by) where {M}
    if by == 0
        _, _, occnum = find_particles(b, n)
        return b, occnum * (occnum - 1)
    else
        from, site, occnum1 = find_particles(b, n)
        to, occnum2 = find_sites(b, mod1(site + by, M))
        return move_particle_bitoffset(b, from, to), occnum1 * (occnum2 + 1)
    end
end

export double_move_particle
function double_move_particle(b::BoseFS{<:Any,M}, i, by) where {M}
    @assert by ≠ 0
    # Both moves from the same site.
    (from, site, occnum_i), = find_particles(b, i; n=2)
    j = mod1(site + by, M)
    k = mod1(site - by, M)

    (to_j, occnum_j), = find_sites(b, j)
    b = move_particle_bitoffset(b, from, to_j)
    # Need to find second site after first move.
    (to_k, occnum_k), = find_sites(b, k)
    b = move_particle_bitoffset(b, from, to_k)
    return b, occnum_i * (occnum_i - 1) * (occnum_j + 1) * (occnum_k + 1)
end
function double_move_particle(b::BoseFS{<:Any,M}, i, j, by) where {M}
    @assert by ≠ 0
    # Move from different sites.
    (from_i, site_i, occnum_i), (from_j, site_j, occnum_j) = find_particles(b, i, j)
    k = mod1(site_i - by, M)
    l = mod1(site_j + by, M)

    (to_k, occnum_k), = find_sites(b, k)
    b = move_particle_bitoffset(b, from_i, to_k)
    # Need to find second site after first move.
    (to_l, occnum_l), = find_sites(b, l)
    b = move_particle_bitoffset(b, from_j, to_l)
    return b, occnum_i * (occnum_k + 1) * occnum_j * (occnum_l + 1)
end

function move_particle_bitoffset(b::BoseFS, from, to)
    if to < from
        return typeof(b)(partial_left_shift(b.bs, to, from))
    else
        return typeof(b)(partial_right_shift(b.bs, from, to - 1))
    end
end
