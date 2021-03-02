"""
    AbstractHops{A<:AbstractFockAddress,T}

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
julia> H = BoseHubbardMom1D(BoseFS((1,1,1,1)));

julia> h = hops(H, BoseFS((1,1,1,1)));

julia> length(h)
24

julia> h[1]
(BoseFS{4,4}((0, 0, 2, 2)), 0.25)

```
"""
hops(h, a) = Hops(h, a)

function generateRandHop(hops::AbstractHops)
    # method using the Hops-type iterator
    # generic implementation of a random excitation generator drawing from
    # a uniform distribution
    nl = length(hops) # check how many sites we could hop to
    chosen = cRand(1:nl) # choose one of them
    #chosen = _nearlydivisionless(nl) + 1 # choose one of them
    # using a faster random number algorithm
    naddress, melem = hops[chosen]
    return naddress, 1.0/nl, melem
    # return new address, generation probability, and matrix element
end

"""
    Hops(ham, add)

Iterator over new address and matrix element for reachable off-diagonal matrix elements of
linear operator `ham` from address add.  Represents an abstract vector containing the
possibly non-zero off-diagonal matrix elements of the column of ham indexed by add.

This is the default implementation defined in terms of [`numOfHops`](@ref) and [`hop`](@ref).

### Examples
```julia
new_address, matrix_element = Hops(ham, current_address)[i]
number_of_hops = length(Hops(ham, current_address))
for (add,elem) in Hops(ham, current_address)
   # do something with address and elem
end
```
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
    new_address, matrix_element = hop(s.hamiltonian, s.address, i)
    return (new_address, matrix_element)
end

Base.size(s::Hops) = (s.length,)
