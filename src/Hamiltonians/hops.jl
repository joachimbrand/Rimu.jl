###
### This file contains the definition of `hops`, as well as some internal functions
### operating on Bosonic addresses.
###
"""
    AbstractHops{A<:AbstractFockAddress,T}<:AbstractVector{Tuple{A,T}}

Iterator over new address and matrix element for reachable off-diagonal matrix elements of a
linear operator.

See [`Hops`](@ref) for a default implementation.

# Methods to define

* [`hops(h, a)::AbstractHops`](@ref): This function is used to construct the correct type
  of hops for a given combination of hamiltonian `h` and fock address `a`.
* `Base.getindex(::AbstractHops, i)`: should be equivalent to `hop(h, a, i)`.
* `Base.size(::AbstractHops)`: should be equivalent to `numOfHops(h, a)`.

"""
abstract type AbstractHops{A<:AbstractFockAddress,T} <: AbstractVector{Tuple{A,T}} end

Base.IndexStyle(::Type{<:AbstractHops}) = IndexLinear()

"""
    hops(h::AbstractHamiltonian, a::AbstractFockAddress)

Return an iterator over reachable off-diagonal matrix elements of type
`<:AbstractHops`. Defaults to returning `Hops(h, a)`

# See also

* [`Hops`](@ref)
* [`AbstractHops`](@ref)

```jldoctest
julia> addr = BoseFS((3,2,1));

julia> H = HubbardReal1D(addr);

julia> h = hops(H, addr)
6-element Rimu.Hamiltonians.Hops{Float64,BoseFS{6,3,BitString{8,1}},HubbardReal1D{Float64,BoseFS{6,3,BitString{8,1}},1.0,1.0}}:
 (BoseFS{6,3}((2, 3, 1)), -3.0)
 (BoseFS{6,3}((2, 2, 2)), -2.449489742783178)
 (BoseFS{6,3}((3, 1, 2)), -2.0)
 (BoseFS{6,3}((4, 1, 1)), -2.8284271247461903)
 (BoseFS{6,3}((4, 2, 0)), -2.0)
 (BoseFS{6,3}((3, 3, 0)), -1.7320508075688772)
```
"""
hops(h, a) = Hops(h, a)

"""
    generateRandHop(hops::AbstractHops)
    generateRandHop(ham::AbstractHamiltonian, add)

Generate a single random excitation, i.e. choose from one of the accessible off-diagonal
elements in the column corresponding to address `add` of the Hamiltonian matrix represented
by `ham`. Alternatively, pass as argument an iterator over the accessible matrix elements.

"""
function generateRandHop(hops::AbstractHops)
    nl = length(hops) # check how many sites we could hop to
    chosen = cRand(1:nl) # choose one of them
    naddress, melem = hops[chosen]
    return naddress, 1.0/nl, melem
end

function generateRandHop(ham::AbstractHamiltonian, add)
    return generateRandHop(hops(ham, add))
end

"""
    Hops(h, address)

Iterator over new address and matrix element for reachable off-diagonal matrix elements of
linear operator `h` from address `address`.  Represents an abstract vector containing the
possibly non-zero off-diagonal matrix elements of the column of ham indexed by add.

This is the default implementation defined in terms of [`numOfHops`](@ref) and [`hop`](@ref).

# See also

* [`hops`](@ref)

"""
struct Hops{T,A,H<:AbstractHamiltonian{T}} <: AbstractHops{A,T}
    hamiltonian::H
    address::A
    length::Int
end

# default constructor
function Hops(h, a)
    return Hops(h, a, numOfHops(h, a))
end

function Base.getindex(s::Hops, i)
    @boundscheck 1 ≤ i ≤ s.length || throw(BoundsError(s, i))
    new_address, matrix_element = hop(s.hamiltonian, s.address, i)
    return (new_address, matrix_element)
end

Base.size(s::Hops) = (s.length,)

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
        chunk >>= trailing_zeros(chunk)
        chunk >>= trailing_ones(chunk)
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

The hops are indexed as follows:

* `(chosen + 1) ÷ 2` selects the hopping site.
* Even `chosen` indicates a hop to the left.
* Odd `chosen` indicates a hop to the left.
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
    site = (chosen + 1) >>> 1
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
            new_address = ((address ⊻ (A(UInt64(1)) << (offset-1))) << 1) | A(UInt64(1))
            prod = curr * (trailing_ones(address) + 1) # mul occupation num of first obital
        else
            next *= reached_end
            new_address = address ⊻ A(UInt64(3)) << (offset - 1)
            prod = curr * (next + 1)
        end
    else # Hopping to the left
        if site == 1 && isodd(address)
            # For leftmost site, we shift the whole address circularly by one bit.
            new_address = (address >>> 1) | A(UInt64(1)) << (N + M - 2)
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
            new_address = address ⊻ A(UInt64(3)) << (offset - 1)
            prod = curr * (prev + 1)
        end
    end
    return BoseFS{N,M,A}(new_address), prod
end
