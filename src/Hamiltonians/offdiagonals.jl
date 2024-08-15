###
### This file contains the definition of `offdiagonals`.
###
"""
    AbstractOffdiagonals{A,T}<:AbstractVector{Tuple{A,T}}

Iterator over new address and matrix elements for reachable off-diagonal matrix elements of
a linear operator.

See [`Offdiagonals`](@ref) for a default implementation.

# Methods to define

* [`offdiagonals(h, a)::AbstractOffdiagonals`](@ref offdiagonals): This function is used to construct the
  correct type of offdiagonals for a given combination of Hamiltonian `h` and Fock address
  `a`.
* `Base.getindex(::AbstractOffdiagonals, i)`: should be equivalent to
  `get_offdiagonal(h, a, i)`.
* `Base.size(::AbstractOffdiagonals)`: should be equivalent to `num_offdiagonals(h, a)`.

See also [`offdiagonals`](@ref), [`AbstractHamiltonian`](@ref), [`AbstractOperator`](@ref).
"""
abstract type AbstractOffdiagonals{A,T} <: AbstractVector{Tuple{A,T}} end

Base.IndexStyle(::Type{<:AbstractOffdiagonals}) = IndexLinear()

offdiagonals(h, a) = Offdiagonals(h, a)

"""
    Offdiagonals(h, address) <: AbstractOffdiagonals

Iterator over new address and matrix element for reachable off-diagonal matrix elements of
linear operator `h` from address `address`.  Represents an abstract vector containing the
non-zero off-diagonal matrix elements of the column of `h` indexed by `address`. To
construct this iterator use [`offdiagonals`](@ref).

This is the default implementation of [`AbstractOffdiagonals`](@ref) defined in terms of
[`num_offdiagonals`](@ref) and [`get_offdiagonal`](@ref).

See also [`offdiagonals`](@ref), [`AbstractHamiltonian`](@ref), [`AbstractOperator`](@ref).
"""
struct Offdiagonals{A,T,H<:AbstractOperator{T}} <: AbstractOffdiagonals{A,T}
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
