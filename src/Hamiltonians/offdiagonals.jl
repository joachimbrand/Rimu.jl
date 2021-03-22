###
### This file contains the definition of `offdiagonals`, as well as some internal functions
### operating on Bosonic addresses.
###
"""
    AbstractOffdiagonals{A<:AbstractFockAddress,T}<:AbstractVector{Tuple{A,T}}

Iterator over new address and matrix element for reachable off-diagonal matrix elements of a
linear operator.

See [`Offdiagonals`](@ref) for a default implementation.

# Methods to define

* [`offdiagonals(h, a)::AbstractOffdiagonals`](@ref): This function is used to construct the
  correct type of offdiagonals for a given combination of hamiltonian `h` and fock address
  `a`.
* `Base.getindex(::AbstractOffdiagonals, i)`: should be equivalent to
  `get_offdiagonal(h, a, i)`.
* `Base.size(::AbstractOffdiagonals)`: should be equivalent to `num_offdiagonals(h, a)`.

"""
abstract type AbstractOffdiagonals{A<:AbstractFockAddress,T} <: AbstractVector{Tuple{A,T}} end

Base.IndexStyle(::Type{<:AbstractOffdiagonals}) = IndexLinear()

"""
    offdiagonals(h::AbstractHamiltonian, a::AbstractFockAddress)

Return an iterator over reachable off-diagonal matrix elements of type
`<:AbstractOffdiagonals`. Defaults to returning `Offdiagonals(h, a)`

# See also

* [`Offdiagonals`](@ref)
* [`AbstractOffdiagonals`](@ref)

```jldoctest
julia> addr = BoseFS((3,2,1));

julia> H = HubbardReal1D(addr);

julia> h = offdiagonals(H, addr)
6-element Rimu.Hamiltonians.Offdiagonals{Float64,BoseFS{6,3,BitString{8,1}},HubbardReal1D{Float64,BoseFS{6,3,BitString{8,1}},1.0,1.0}}:
 (BoseFS{6,3}((2, 3, 1)), -3.0)
 (BoseFS{6,3}((2, 2, 2)), -2.449489742783178)
 (BoseFS{6,3}((3, 1, 2)), -2.0)
 (BoseFS{6,3}((4, 1, 1)), -2.8284271247461903)
 (BoseFS{6,3}((4, 2, 0)), -2.0)
 (BoseFS{6,3}((3, 3, 0)), -1.7320508075688772)
```
"""
offdiagonals(h, a) = Offdiagonals(h, a)

"""
    random_offdiagonal(offdiagonals::AbstractOffdiagonals)
    random_offdiagonal(ham::AbstractHamiltonian, add)

Generate a single random excitation, i.e. choose from one of the accessible off-diagonal
elements in the column corresponding to address `add` of the Hamiltonian matrix represented
by `ham`. Alternatively, pass as argument an iterator over the accessible matrix elements.

"""
function random_offdiagonal(offdiagonals::AbstractOffdiagonals)
    nl = length(offdiagonals) # check how many sites we could get_offdiagonal to
    chosen = cRand(1:nl) # choose one of them
    naddress, melem = offdiagonals[chosen]
    return naddress, 1.0/nl, melem
end

function random_offdiagonal(ham::AbstractHamiltonian, add)
    return random_offdiagonal(offdiagonals(ham, add))
end

"""
    Offdiagonals(h, address)

Iterator over new address and matrix element for reachable off-diagonal matrix elements of
linear operator `h` from address `address`.  Represents an abstract vector containing the
possibly non-zero off-diagonal matrix elements of the column of ham indexed by add.

This is the default implementation defined in terms of [`num_offdiagonals`](@ref) and
[`get_offdiagonal`](@ref).

# See also

* [`offdiagonals`](@ref)

"""
struct Offdiagonals{T,A,H<:AbstractHamiltonian{T}} <: AbstractOffdiagonals{A,T}
    hamiltonian::H
    address::A
    length::Int
end

# default constructor
function Offdiagonals(h, a)
    return Offdiagonals(h, a, num_offdiagonals(h, a))
end

function Base.getindex(s::Offdiagonals, i)
    @boundscheck 1 ≤ i ≤ s.length || throw(BoundsError(s, i))
    new_address, matrix_element = get_offdiagonal(s.hamiltonian, s.address, i)
    return (new_address, matrix_element)
end

Base.size(s::Offdiagonals) = (s.length,)

###
### Internal functions common to several different bosonic Hamiltonians
###
"""
    numberoccupiedsites(b::BoseFS)

Return the number of occupied sites in address `b`, which is equal to the number of
non-zeros in its ONR representation.

# Example

```jldoctest
julia> using Rimu.Hamiltonians: numberoccupiedsites

julia> numberoccupiedsites(BoseFS((1, 0, 2)))
2
julia> numberoccupiedsites(BoseFS((3, 0, 0)))
1
```
"""
function numberoccupiedsites(b::BoseFS{<:Any,<:Any,S}) where S
    return numberoccupiedsites(Val(num_chunks(S)), b)
end

@inline function numberoccupiedsites(::Val{1}, b::BoseFS)
    chunk = b.bs.chunks[1]
    result = 0
    while true
        chunk >>= (trailing_zeros(chunk) % UInt)
        chunk >>= (trailing_ones(chunk) % UInt)
        result += 1
        iszero(chunk) && break
    end
    return result
end

@inline function numberoccupiedsites(_, b::BoseFS)
    # This version is faster than using the occupied_orbital iterator
    address = b.bs
    result = 0
    K = num_chunks(address)
    last_mask = UInt64(1) << 63 # = 0b100000...
    prev_top_bit = false
    # This loop compiles away for address<:BSAdd*
    for i in K:-1:1
        chunk = chunks(address)[i]
        # This part handles sites that span across chunk boundaries.
        # If the previous top bit and the current bottom bit are both 1, we have to subtract
        # 1 from the result or the mode will be counted twice.
        result -= (chunk & prev_top_bit) % Int
        prev_top_bit = (chunk & last_mask) > 0
        while !iszero(chunk)
            chunk >>>= trailing_zeros(chunk)
            chunk >>>= trailing_ones(chunk)
            result += 1
        end
    end
    return result
end

"""
    new_address, product = hopnextneighbour(add, chosen)

Compute the new address of a hopping event for the Bose-Hubbard model. Returns the new
address and the product of occupation numbers of the involved orbitals.

The off-diagonals are indexed as follows:

* `(chosen + 1) ÷ 2` selects the hopping site.
* Even `chosen` indicates a hop to the left.
* Odd `chosen` indicates a hop to the right.
* Boundary conditions are periodic.

# Example

```jldoctest
julia> using Rimu.Hamiltonians: hopnextneighbour

julia> hopnextneighbour(BoseFS((1, 0, 1)), 3)
(BoseFS{2,3}((2, 0, 0)), 2)
julia> hopnextneighbour(BoseFS((1, 0, 1)), 4)
(BoseFS{2,3}((1, 1, 0)), 1)
```
"""
function hopnextneighbour(b::BoseFS{N,M,A}, chosen) where {N,M,A}
    address = b.bs
    site = (chosen + 1) >>> 0x1
    if isodd(chosen) # Hopping to the right
        next = 0
        curr = 0
        offset = 0
        sc = 0
        reached_end = false
        for (i, (num, sn, bit)) in enumerate(occupied_orbitals(b))
            next = num * (sn == sc + 1) # only set next to > 0 if sites are neighbours
            reached_end = i == site + 1
            reached_end && break
            curr = num
            offset = bit + num
            sc = sn
        end
        if sc == M
            new_address = (address << 0x1) | A(UInt64(1))
            prod = curr * (trailing_ones(address) + 1) # mul occupation num of first obital
        else
            next *= reached_end
            new_address = address ⊻ A(UInt64(3)) << ((offset - 1) % UInt)
            prod = curr * (next + 1)
        end
    else # Hopping to the left
        if site == 1 && isodd(address)
            # For leftmost site, we shift the whole address circularly by one bit.
            new_address = (address >>> 0x1) | A(UInt64(1)) << ((N + M - 2) % UInt)
            prod = trailing_ones(address) * leading_ones(new_address)
        else
            prev = 0
            curr = 0
            offset = 0
            sp = 0
            for (i, (num, sc, bit)) in enumerate(occupied_orbitals(b))
                prev = curr * (sc == sp + 1) # only set prev to > 0 if sites are neighbours
                curr = num
                offset = bit
                i == site && break
                sp = sc
            end
            new_address = address ⊻ A(UInt64(3)) << ((offset - 1) % UInt)
            prod = curr * (prev + 1)
        end
    end
    return BoseFS{N,M,A}(new_address), prod
end

function hopnextneighbour(b::BoseFS2D, chosen)
    return hopnextneighbour(b, chosen, onr(b))
end

function hopnextneighbour(b::BoseFS2D{N,MY,MX}, chosen, onrep) where {N,MY,MX}
    directions = ((-1, 0), (1, 0), (0, -1), (0, 1))
    # Note: using divrem makes indexing zero-based, hence chosen - 1
    site, direction = divrem(chosen - 1, 4)
    m_onrep = MMatrix(onrep)

    # Find location and occupation number of the site we're hopping from.
    i, j, val = 0, 0, 0
    for k in eachindex(m_onrep)
        @inbounds v = m_onrep[k]
        site -= (v ≠ 0)
        if site == -1
            @inbounds i, j = Tuple(CartesianIndices(m_onrep)[k])
            val = v
            break
        end
    end
    # Get offset and compute neighbour position.
    o_i, o_j = directions[direction + 1]
    n_i, n_j = mod1(i + o_i, MY), mod1(j + o_j, MX)

    @inbounds m_onrep[i, j] -= 1
    newval = @inbounds m_onrep[n_i, n_j] += 1

    # Note: the follwing is the most time-consuming part.
    return @inbounds BoseFS2D{N,MY,MX}(m_onrep.data), newval * val
end
