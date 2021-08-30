###
### This file contains the definition of `offdiagonals`, as well as some internal functions
### operating on Bosonic addresses.
###
"""
    AbstractOffdiagonals{A,T}<:AbstractVector{Tuple{A,T}}

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
abstract type AbstractOffdiagonals{A,T} <: AbstractVector{Tuple{A,T}} end

Base.IndexStyle(::Type{<:AbstractOffdiagonals}) = IndexLinear()

"""
    offdiagonals(h::AbstractHamiltonian, address)

Return an iterator over reachable off-diagonal matrix elements of type
`<:AbstractOffdiagonals`. Defaults to returning `Offdiagonals(h, a)`

# See also

* [`Offdiagonals`](@ref)
* [`AbstractOffdiagonals`](@ref)

```jldoctest
julia> addr = BoseFS((3,2,1));


julia> H = HubbardReal1D(addr);


julia> h = offdiagonals(H, addr)
6-element Rimu.Hamiltonians.Offdiagonals{BoseFS{6, 3, BitString{8, 1, UInt8}}, Float64, HubbardReal1D{Float64, BoseFS{6, 3, BitString{8, 1, UInt8}}, 1.0, 1.0}}:
 (BoseFS{6,3}((2, 3, 1)), -3.0)
 (BoseFS{6,3}((2, 2, 2)), -2.449489742783178)
 (BoseFS{6,3}((3, 1, 2)), -2.0)
 (BoseFS{6,3}((4, 1, 1)), -2.8284271247461903)
 (BoseFS{6,3}((4, 2, 0)), -2.0)
 (BoseFS{6,3}((3, 3, 0)), -1.7320508075688772)
```
"""
offdiagonals(h::AbstractHamiltonian, a) = Offdiagonals(h, a)

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
struct Offdiagonals{A,T,H<:AbstractHamiltonian{T}} <: AbstractOffdiagonals{A,T}
    hamiltonian::H
    address::A
    length::Int
end

# default constructor
function Offdiagonals(h, a)
    return Offdiagonals(h, a, num_offdiagonals(h, a))
end

function Base.getindex(s::Offdiagonals{A,T}, i)::Tuple{A,T} where {A,T}
    @boundscheck 1 ≤ i ≤ s.length || throw(BoundsError(s, i))
    new_address, matrix_element = get_offdiagonal(s.hamiltonian, s.address, i)
    return (new_address, matrix_element)
end

Base.size(s::Offdiagonals) = (s.length,)

###
### Internal functions common to several different bosonic Hamiltonians
###
"""
    new_address, product = hopnextneighbour(add, chosen)

Compute the new address of a hopping event for the Bose-Hubbard model. Returns the new
address and the product of occupation numbers of the involved modes.

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
    return BoseFS{N,M,A}(new_address), prod
end
