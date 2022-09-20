"""
    BoseFS{N,M,S} <: SingleComponentFockAddress

Address type that represents a Fock state of `N` spinless bosons in `M` modes
by wrapping a bitstring of type `S <: BitString`.

# Constructors

* `BoseFS{N,M}(bs::BitString)`: Unsafe constructor. Does not check whether the number of
  particles in `bs` is equal to `N`.

* `BoseFS(::BitString)`: Automatically determine `N` and `M`. This constructor is not type
  stable!

* `BoseFS{[N,M,S]}(onr)`: Create `BoseFS{N,M}` from [`onr`](@ref) representation. This is
  efficient as long as at least `N` is provided.

See also: [`SingleComponentFockAddress`](@ref), [`FermiFS`](@ref), [`BitString`](@ref).
"""
struct BoseFS{N,M,S} <: SingleComponentFockAddress{N,M}
    bs::S
end

###
### General
###
# TODO generalize for both storage types
function BoseFS{N}(onr::Union{SVector{M},NTuple{M}}) where {N,M}
    return BoseFS{N,M}(onr)
end
function BoseFS(onr::Union{AbstractVector,Tuple})
    M = length(onr)
    N = sum(onr)
    return BoseFS{N,M}(onr)
end

function print_address(io::IO, b::BoseFS{N,M}) where {N,M}
    if get(io, :compact, false)
        print(io, "|", join(onr(b), ' '), "⟩")
    else
        print(io, "BoseFS{$N,$M}(", tuple(onr(b)...), ")")
    end
end

Base.bitstring(b::BoseFS) = bitstring(b.bs) # TODO rename?

Base.isless(a::BoseFS, b::BoseFS) = isless(a.bs, b.bs)
Base.hash(bba::BoseFS,  h::UInt) = hash(bba.bs, h)
Base.:(==)(a::BoseFS, b::BoseFS) = a.bs == b.bs


###
###
###
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

"""
    near_uniform_onr(N, M) -> onr::SVector{M,Int}

Create occupation number representation `onr` distributing `N` particles in `M`
modes in a close-to-uniform fashion with each mode filled with at least
`N ÷ M` particles and at most with `N ÷ M + 1` particles.
"""
function near_uniform_onr(n::Number, m::Number)
    return near_uniform_onr(Val(n),Val(m))
end
function near_uniform_onr(::Val{N}, ::Val{M}) where {N, M}
    fillingfactor, extras = divrem(N, M)
    # startonr = fill(fillingfactor,M)
    startonr = fillingfactor * @MVector ones(Int,M)
    startonr[1:extras] .+= 1
    return SVector{M}(startonr)
end

"""
    near_uniform(BoseFS{N,M})
    near_uniform(BoseFS{N,M,S}) -> bfs::BoseFS{N,M,S}

Create bosonic Fock state with near uniform occupation number of `M` modes with
a total of `N` particles. Specifying the bit address type `S` is optional.

# Examples
```jldoctest
julia> near_uniform(BoseFS{7,5,BitString{14}})
BoseFS{7,5}((2, 2, 1, 1, 1))

julia> near_uniform(BoseFS{7,5})
BoseFS{7,5}((2, 2, 1, 1, 1))
```
"""
function near_uniform(::Type{<:BoseFS{N,M}}) where {N,M}
    return BoseFS{N,M}(near_uniform_onr(Val(N),Val(M)))
end
near_uniform(b::AbstractFockAddress) = near_uniform(typeof(b))

@deprecate nearUniform near_uniform

"""
    onr(bs)

Compute and return the occupation number representation of the bit string
address `bs` as an `SVector{M,Int32}`, where `M` is the number of modes.
"""
onr(b::BoseFS{<:Any,M}) where {M} = bose_onr(b.bs, Val(M))

function Base.reverse(b::BoseFS)
    return typeof(b)(bitreverse(b.bs))
end

# For vacuum state
function num_occupied_modes(b::BoseFS{0})
    return 0
end
function num_occupied_modes(b::BoseFS)
    return bose_num_occupied_modes(b.bs)
end

function occupied_modes(b::BoseFS{N,M,S}) where {N,M,S}
    return BoseOccupiedModes{N,M,S}(b.bs)
end

function find_mode(b::BoseFS, index)
    last_occnum = last_mode = last_offset = 0
    for (occnum, mode, offset) in occupied_modes(b)
        dist = index - mode
        if dist == 0
            return BoseFSIndex(occnum, index, offset)
        elseif dist < 0
            return BoseFSIndex(0, index, offset + dist)
        end
        last_occnum = occnum
        last_mode = mode
        last_offset = offset
    end
    offset = last_offset + last_occnum + index - last_mode
    return BoseFSIndex(0, index, offset)
end
# Multiple in a single pass
function find_mode(b::BoseFS, indices::NTuple{N}) where {N}
    # Idea: find permutation, then use the permutation to find indices in order even though
    # they are not sorted.
    perm = sortperm(SVector(indices))
    # perm_i is the index in permutation and goes from 1:N.
    perm_i = 1
    # curr_i points to indices and result
    curr_i = perm[1]
    # index is the current index we are looking for.
    index = indices[curr_i]

    result = ntuple(_ -> BoseFSIndex(0, 0, 0), Val(N))
    last_occnum = last_mode = last_offset = 0
    @inbounds for (occnum, mode, offset) in occupied_modes(b)
        dist = index - mode
        # While loop handles duplicate entries in indices.
        while dist ≤ 0
            if dist == 0
                @set! result[curr_i] = BoseFSIndex(occnum, mode, offset)
            else
                @set! result[curr_i] = BoseFSIndex(0, index, offset + dist)
            end
            perm_i += 1
            perm_i > N && return result
            curr_i = perm[perm_i]
            index = indices[curr_i]
            dist = index - mode
        end
        last_occnum = occnum
        last_mode = mode
        last_offset = offset
    end
    # Now we have to find all indices that appear after the last occupied site.
    # While true because we break out of the loop early anyway.
    @inbounds while true
        offset = last_offset + last_occnum + index - last_mode
        @set! result[curr_i] = BoseFSIndex(0, index, offset)
        perm_i += 1
        perm_i > N && return result
        curr_i = perm[perm_i]
        index = indices[curr_i]
    end
    return result # not reached
end

function find_occupied_mode(b::BoseFS, index::Integer, n=1)
    for (occnum, mode, offset) in occupied_modes(b)
        index -= occnum ≥ n
        if index == 0
            return BoseFSIndex(occnum, mode, offset)
        end
    end
    return BoseFSIndex(0, 0, 0)
end

function excitation(b::B, creations, destructions) where {B<:BoseFS}
    new_bs, val = bose_excitation(b.bs, creations, destructions)
    return B(new_bs), val
end
