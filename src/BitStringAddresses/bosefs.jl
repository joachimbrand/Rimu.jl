"""
    BoseFS{N,M,S} <: SingleComponentFockAddress

Address type that represents a Fock state of `N` spinless bosons in `M` modes by wrapping a
[`BitString`](@ref), or a [`SortedParticleList`](@ref). Which is wrapped is chosen
automatically based on the properties of the address.

# Constructors

* `BoseFS{[N,M]}(val::Integer...)`: Create `BoseFS{N,M}` from occupation numbers. This is
  type-stable if the number of modes `M` and the number of particles `N` are provided.
  Otherwise, `M` and `N` are inferred from the arguments.

* `BoseFS{[N,M]}(onr)`: Create `BoseFS{N,M}` from occupation number representation, see
  [`onr`](@ref). This is efficient if `N` and `M` are provided, and `onr` is a
  statically-sized collection, such as a `Tuple` or `SVector`.

* `BoseFS{[N,M]}([M, ]pairs...)`: Provide the number of modes `M` and `mode =>
  occupation_number` pairs. If `M` is provided as a type parameter, it should not be
  provided as the first argument.  Useful for creating sparse addresses. `pairs` can be
  multiple arguments or an iterator of pairs.

* `BoseFS{N,M,S}(bs::S)`: Unsafe constructor. Does not check whether the number of
  particles in `bs` is equal to `N`.

* [`@fs_str`](@ref): Addresses are sometimes printed in a compact manner. This
  representation can also be used as a constructor. See the last example below.

# Examples

```jldoctest
julia> BoseFS{6,5}(0, 1, 2, 3, 0)
BoseFS{6,5}(0, 1, 2, 3, 0)

julia> BoseFS([abs(i - 3) ≤ 1 ? i - 1 : 0 for i in 1:5])
BoseFS{6,5}(0, 1, 2, 3, 0)

julia> BoseFS(5, 2 => 1, 3 => 2, 4 => 3)
BoseFS{6,5}(0, 1, 2, 3, 0)

julia> BoseFS{6,5}(i => i - 1 for i in 2:4)
BoseFS{6,5}(0, 1, 2, 3, 0)

julia> fs"|0 1 2 3 0⟩"
BoseFS{6,5}(0, 1, 2, 3, 0)

julia> fs"|b 5: 2 3 3 4 4 4⟩"
BoseFS{6,5}(0, 1, 2, 3, 0)
```

See also: [`SingleComponentFockAddress`](@ref), [`OccupationNumberFS`](@ref),
[`FermiFS`](@ref), [`CompositeFS`](@ref), [`FermiFS2C`](@ref).
"""
struct BoseFS{N,M,S} <: SingleComponentFockAddress{N,M}
    bs::S
end

@inline function BoseFS{N,M,S}(onr::Union{SVector{M},MVector{M},NTuple{M}}) where {N,M,S}
    @boundscheck begin
        sum(onr) == N || throw(ArgumentError(
            "invalid ONR: $N particles expected, $(sum(onr)) given"
        ))
        if S <: BitString
            B = num_bits(S)
            M + N - 1 == B || throw(ArgumentError(
                "invalid ONR: $B-bit BitString does not fit $N particles in $M modes"
            ))
        elseif S <: SortedParticleList
            N == num_particles(S) && M == num_modes(S) || throw(ArgumentError(
                "invalid ONR: $S does not fit $N particles in $M modes"
            ))
        end
    end
    return BoseFS{N,M,S}(from_bose_onr(S, onr))
end
function BoseFS{N,M}(onr::Union{AbstractArray{<:Integer},NTuple{M,<:Integer}}) where {N,M}
    @boundscheck begin
        sum(onr) == N || throw(ArgumentError(
            "invalid ONR: $N particles expected, $(sum(onr)) given"
        ))
        length(onr) == M || throw(ArgumentError(
            "invalid ONR: $M modes expected, $(length(onr)) given"
        ))
    end
    spl_type = select_int_type(M)

    # Pick smaller address type, but prefer sparse.
    # Alway pick dense if it fits into one chunk.

    # Compute the size of container in words
    sparse_sizeof = ceil(Int, N * sizeof(spl_type) / 8)
    dense_sizeof = ceil(Int, (N + M - 1) / 64)
    if dense_sizeof == 1 || dense_sizeof < sparse_sizeof
        S = typeof(BitString{M + N - 1}(0))
    else
        S = SortedParticleList{N,M,spl_type}
    end
    return BoseFS{N,M,S}(from_bose_onr(S, onr))
end
function BoseFS(onr::Union{AbstractArray,Tuple})
    M = length(onr)
    N = sum(onr)
    return BoseFS{N,M}(onr)
end
BoseFS(vals::Integer...) = BoseFS(vals) # specify occupation numbers
BoseFS(val::Integer) = BoseFS((val,)) # single mode address
BoseFS{N,M}(vals::Integer...) where {N,M} = BoseFS{N,M}(vals)

BoseFS(M::Integer, pairs::Pair...) = BoseFS(M, pairs)
BoseFS(M::Integer, pairs) = BoseFS(sparse_to_onr(M, pairs))
BoseFS{N,M}(pairs::Pair...) where {N,M} = BoseFS{N,M}(pairs)
BoseFS{N,M}(pairs) where {N,M} = BoseFS{N,M}(sparse_to_onr(M, pairs))

function print_address(io::IO, b::BoseFS{N,M}; compact=false) where {N,M}
    if compact && b.bs isa SortedParticleList
        print(io, "|b ", M, ": ", join(Int.(b.bs.storage), ' '), "⟩")
    elseif compact
        print(io, "|", join(onr(b), ' '), "⟩")
    elseif b.bs isa SortedParticleList
        print(io, "BoseFS{$N,$M}(", onr_sparse_string(onr(b)), ")")
    else
        print(io, "BoseFS{$N,$M}", tuple(onr(b)...))
    end
end

Base.bitstring(b::BoseFS) = bitstring(b.bs) # TODO rename?

Base.isless(a::BoseFS, b::BoseFS) = isless(a.bs, b.bs)
Base.hash(bba::BoseFS,  h::UInt) = hash(bba.bs, h)
Base.:(==)(a::BoseFS, b::BoseFS) = a.bs == b.bs

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
    near_uniform(BoseFS{N,M}) -> BoseFS{N,M}

Create bosonic Fock state with near uniform occupation number of `M` modes with
a total of `N` particles.

# Examples
```jldoctest
julia> near_uniform(BoseFS{7,5})
BoseFS{7,5}(2, 2, 1, 1, 1)

julia> near_uniform(FermiFS{3,5})
FermiFS{3,5}(1, 1, 1, 0, 0)
```
"""
function near_uniform(::Type{<:BoseFS{N,M}}) where {N,M}
    return BoseFS{N,M}(near_uniform_onr(Val(N),Val(M)))
end
near_uniform(b::AbstractFockAddress) = near_uniform(typeof(b))

onr(b::BoseFS{<:Any,M}) where {M} = to_bose_onr(b.bs, Val(M))
const occupation_number_representation = onr # resides here because `onr` has to be defined

function Base.reverse(b::BoseFS)
    return typeof(b)(reverse(b.bs))
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

# find_occupied_mode provided by generic implementation

function excitation(b::B, creations, destructions) where {B<:BoseFS}
    new_bs, val = bose_excitation(b.bs, creations, destructions)
    return B(new_bs), val
end

"""
    new_address, value = hopnextneighbour(add, chosen, boundary_condition)

Compute the new address of a hopping event for the Hubbard model. Returns the new
address and the square root of product of occupation numbers of the involved modes
multiplied by a term consistent with boundary condition as the `value`. 
The following boundary conditions are supported:

* `:periodic`: hopping over the boundary gives does not change the `value`.
* `:twisted`: hopping over the boundary flips the sign of the `value`.
* `:hard_wall`: hopping over the boundary gives a `value` of zero.
* `θ <: Number`: hopping over the boundary gives a `value` multiplied by ``\\exp(iθ)`` or ``\\exp(−iθ)`` depending on the direction of hopping.

The off-diagonals are indexed as follows:

* `(chosen + 1) ÷ 2` selects the hopping site.
* Even `chosen` indicates a hop to the left.
* Odd `chosen` indicates a hop to the right.

# Example

```jldoctest
julia> using Rimu.Hamiltonians: hopnextneighbour

julia> hopnextneighbour(BoseFS(1, 0, 1), 3)
(BoseFS{2,3}(2, 0, 0), 1.4142135623730951)

julia> hopnextneighbour(BoseFS(1, 0, 1), 4)
(BoseFS{2,3}(1, 1, 0), 1.0)

julia> hopnextneighbour(BoseFS(1, 0, 1), 3, :twisted)
(BoseFS{2,3}(2, 0, 0), -1.4142135623730951)

julia> hopnextneighbour(BoseFS(1, 0, 1), 3, :hard_wall)
(BoseFS{2,3}(2, 0, 0), 0.0)

julia> hopnextneighbour(BoseFS(1, 0, 1), 3, π/4)
(BoseFS{2,3}(2, 0, 0), 1.0000000000000002 + 1.0im)
```
"""
function hopnextneighbour(b::BoseFS{N,M,A}, chosen) where {N,M,A<:BitString}
    address = b.bs
    T = chunk_type(address)
    site = (chosen + 1) >>> 0x1
    if isodd(chosen) # Hopping to the right
        next = 0
        curr = 0
        offset = 0
        sc = 0
        reached_end = false
        for (i, (num, sn, bit)) in enumerate(occupied_modes(b))
            next = num * (sn == sc + 1) # only set next to > 0 if sites are neighbours
            reached_end = i == site + 1
            reached_end && break
            curr = num
            offset = bit + num
            sc = sn
        end
        if sc == M
            new_address = (address << 0x1) | A(T(1))
            prod = curr * (trailing_ones(address) + 1) # mul occupation num of first obital
        else
            next *= reached_end
            new_address = address ⊻ A(T(3)) << ((offset - 1) % T)
            prod = curr * (next + 1)
        end
    else # Hopping to the left
        if site == 1 && isodd(address)
            # For leftmost site, we shift the whole address circularly by one bit.
            new_address = (address >>> 0x1) | A(T(1)) << ((N + M - 2) % T)
            prod = trailing_ones(address) * leading_ones(new_address)
        else
            prev = 0
            curr = 0
            offset = 0
            sp = 0
            for (i, (num, sc, bit)) in enumerate(occupied_modes(b))
                prev = curr * (sc == sp + 1) # only set prev to > 0 if sites are neighbours
                curr = num
                offset = bit
                i == site && break
                sp = sc
            end
            new_address = address ⊻ A(T(3)) << ((offset - 1) % T)
            prod = curr * (prev + 1)
        end
    end
    return BoseFS{N,M,A}(new_address), √prod
end

function hopnextneighbour(b::SingleComponentFockAddress, i)
    src = find_occupied_mode(b, (i + 1) >>> 0x1)
    dst = find_mode(b, mod1(src.mode + ifelse(isodd(i), 1, -1), num_modes(b)))

    new_b, val = excitation(b, (dst,), (src,))
    return new_b, val
end

function hopnextneighbour(
    b::SingleComponentFockAddress, i, boundary_condition::Symbol)
    src = find_occupied_mode(b, (i + 1) >>> 0x1)
    dir = ifelse(isodd(i), 1, -1)
    dst = find_mode(b, mod1(src.mode + dir, num_modes(b)))
    new_b, val = excitation(b, (dst,), (src,))
    on_boundary = src.mode == 1 && dir == -1 || src.mode == num_modes(b) && dir == 1
    if boundary_condition == :twisted && on_boundary
        return new_b, -val
    elseif boundary_condition == :hard_wall && on_boundary
        return new_b, 0.0
    else
        return new_b, val
    end
end

function hopnextneighbour(b::SingleComponentFockAddress, i, boundary_condition::Number)
    src = find_occupied_mode(b, (i + 1) >>> 0x1)
    dir = ifelse(isodd(i), 1, -1)
    dst = find_mode(b, mod1(src.mode + dir, num_modes(b)))
    new_b, val = excitation(b, (dst,), (src,))
    if (src.mode == 1 && dir == -1)
        return new_b, val*exp(-im*boundary_condition)
    elseif (src.mode == num_modes(b) && dir == 1)
        return new_b, val*exp(im*boundary_condition)
    else
        return new_b, complex(val)
    end
end

"""
    bose_hubbard_interaction(address)

Return ``Σ_i n_i (n_i-1)`` for computing the Bose-Hubbard on-site interaction (without the
``U`` prefactor.)

# Example

```jldoctest
julia> Hamiltonians.bose_hubbard_interaction(BoseFS{4,4}((2,1,1,0)))
2
julia> Hamiltonians.bose_hubbard_interaction(BoseFS{4,4}((3,0,1,0)))
6
```
"""
function bose_hubbard_interaction(b::BoseFS{<:Any,<:Any,A}) where {A<:BitString}
    return bose_hubbard_interaction(Val(num_chunks(A)), b)
end
function bose_hubbard_interaction(b::SingleComponentFockAddress)
    return bose_hubbard_interaction(nothing, b)
end

@inline function bose_hubbard_interaction(_, b::SingleComponentFockAddress)
    result = 0
    for (n, _, _) in occupied_modes(b)
        result += n * (n - 1)
    end
    return result
end

@inline function bose_hubbard_interaction(::Val{1}, b::BoseFS{<:Any,<:Any,<:BitString})
    # currently this ammounts to counting occupation numbers of modes
    chunk = chunks(b.bs)[1]
    matrixelementint = 0
    while !iszero(chunk)
        chunk >>>= (trailing_zeros(chunk) % UInt) # proceed to next occupied mode
        bosonnumber = trailing_ones(chunk) # count how many bosons inside
        # surpsingly it is faster to not check whether this is nonzero and do the
        # following operations anyway
        chunk >>>= (bosonnumber % UInt) # remove the counted mode
        matrixelementint += bosonnumber * (bosonnumber - 1)
    end
    return matrixelementint
end
