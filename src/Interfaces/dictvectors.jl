"""
    AbstractDVec{K,V}

Abstract type for data structures that behave similar to sparse vectors, but are indexed
by an arbitrary type `V` (could be non-integers) similarly to dictionaries. `AbstractDVec`s 
are  designed to work well with [`lomc!`](@ref) and
[KrylovKit](https://github.com/Jutho/KrylovKit.jl).

Concrete implementations are available as [`DVec`](@ref)
and [`InitiatorDVec`](@ref).

`AbstractDvec`s lie somewhere between `AbstractDict`s and sparse `AbstractVector`s, while
being subtyped to neither.
Generally they behave
like a dictionary, while supportting various linear algebra functionality. Indexing with a
value not stored in the dictionary returns `zero(V)`. Setting a stored value to 0 or below
`eps(V::AbstractFloat)` removes the value from the dictionary. Their `length` signals the
number of stored elements, not the size of the vector space.

They have a [`StochasticStyle`](@ref) which selects the spawning algorithm in `FCIQMC`.

To iterate over an `AbstractDVec`, use `keys`, `pairs`, or `values`.

# Interface

The interface is similar to the `AbstractDict` interface, but with the changed behaviour
as noted above.
Implement what would be needed for the `AbstractDict` interface (`pairs`, `keys`, `values`,
`setindex!, getindex, delete!, length, haskey, empty!, isempty`) and, in addition:
* [`StochasticStyle`](@ref)
* [`storage`](@ref) returns an `AbstractDict` storing the raw data with possibly
  different `valtype` than `V`.
* [`deposit`](@ref)

See also [`DictVectors`](@ref), [`Interfaces`](@ref).
"""
abstract type AbstractDVec{K,V} end

"""
    deposit!(w::AbstractDVec, add, val, parent::Pair)

Add `val` into `w` at address `add`, taking into account initiator rules if applicable.
`parent` contains the `address => value` pair from which the pair `add => val`
was created. [`InitiatorDVec`](@ref) can intercept this and add its own functionality.
"""
function deposit!(w, add, val, _)
    w[add] += convert(valtype(w), val)
end

"""
    localpart(dv) -> AbstractDVec

Get the part of `dv` that is located on this MPI rank. Returns `dv` itself for
`AbstractDVec`s.
"""
localpart(dv) = dv # default for local data

"""
    storage(dvec) -> AbstractDict

Return the raw storage associated with `dvec` as an `AbstractDict`. Used in MPI
communication.
"""
storage(v::AbstractVector) = v

StochasticStyle(::AbstractArray{T}) where {T} = default_style(T)

"""
    freeze(dv)

Create a "frozen" version of `dv` which can no longer be modified or used in the
conventional manner, but supports faster dot products.

If `dv` is an [`MPIData`](@ref), synchronize its contents among the ranks first.
"""
freeze(v::AbstractVector) = copy(v)