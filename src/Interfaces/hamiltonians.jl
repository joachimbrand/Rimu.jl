###
### This file contains abstract types, interfaces and traits.
###
"""
    AbstractHamiltonian{T}

Supertype that provides an interface for linear operators over a linear space with scalar
type `T` that are suitable for FCIQMC (with [`lomc!`](@ref Main.lomc!)). Indexing is done
with addresses (typically not integers) from an address space that may be large (and will
not need to be completely generated).

`AbstractHamiltonian` instances operate on vectors of type [`AbstractDVec`](@ref) from the
module `DictVectors` and work well with addresses of type [`AbstractFockAddress`](@ref Main.BitStringAddresses.AbstractFockAddress)
from the module `BitStringAddresses`. The type works well with the external package
[KrylovKit.jl](https://github.com/Jutho/KrylovKit.jl).

For available implementations see [`Hamiltonians`](@ref Main.Hamiltonians).

# Interface

Basic interface methods to implement:

* [`starting_address(::AbstractHamiltonian)`](@ref)
* [`diagonal_element(::AbstractHamiltonian, address)`](@ref)
* [`num_offdiagonals(::AbstractHamiltonian, address)`](@ref)
* [`get_offdiagonal(::AbstractHamiltonian, address, chosen::Integer)`](@ref) (optional, see
    below)

Optional additional methods to implement:

* [`LOStructure(::Type{typeof(lo)})`](@ref LOStructure): defaults to `AdjointUnknown`
* [`dimension(::AbstractHamiltonian, addr)`](@ref Main.Hamiltonians.dimension): defaults to
  dimension of address space
* [`allowed_address_type(h::AbstractHamiltonian)`](@ref): defaults to
  `typeof(starting_address(h))`
* [`momentum(::AbstractHamiltonian)`](@ref Main.Hamiltonians.momentum): no default

Provides the following functions and methods:

* [`offdiagonals`](@ref): iterator over reachable off-diagonal matrix elements
* [`random_offdiagonal`](@ref): function to generate random off-diagonal matrix element
* `*(H, v)`: deterministic matrix-vector multiply (allocating)
* `H(v)`: equivalent to `H * v`.
* `mul!(w, H, v)`: mutating matrix-vector multiply.
* [`dot(x, H, v)`](@ref Main.Hamiltonians.dot): compute `x⋅(H*v)` minimizing allocations.
* `H[address1, address2]`: indexing with `getindex()` - mostly for testing purposes (slow!)
* [`BasisSetRep`](@ref Main.Hamiltonians.sparse): construct a basis set repesentation
* [`sparse`](@ref Main.Hamiltonians.sparse), [`Matrix`](@ref): construct a (sparse) matrix representation

Alternatively to the above, [`offdiagonals`](@ref) can be implemented instead of
[`get_offdiagonal`](@ref). Sometimes this can be done efficiently. In this case
[`num_offdiagonals`](@ref) should provide an upper bound on the number of elements obtained
when iterating [`offdiagonals`](@ref).

See also [`Hamiltonians`](@ref Main.Hamiltonians), [`Interfaces`](@ref).
"""
abstract type AbstractHamiltonian{T} end

Base.eltype(::AbstractHamiltonian{T}) where {T} = T

"""
    allowed_address_type(h::AbstractHamiltonian)
Return the type of addresses that can be used with Hamiltonian `h`.

Part of the [`AbstractHamiltonian`](@ref) interface.

Defaults to `typeof(starting_address(h))`. Overload this function if the Hamiltonian can be
used with addresses of different types.
"""
allowed_address_type(h::AbstractHamiltonian) = typeof(starting_address(h))
allowed_address_type(::AbstractMatrix) = Integer

"""
    diagonal_element(ham, address)

Compute the diagonal matrix element of the linear operator `ham` at
address `address`.

# Example

```jldoctest
julia> address = BoseFS((3, 2, 1));


julia> H = HubbardMom1D(address);


julia> diagonal_element(H, address)
8.666666666666664
```
Part of the [`AbstractHamiltonian`](@ref) interface.
"""
diagonal_element(m::AbstractMatrix, i) = m[i, i]

"""
    num_offdiagonals(ham, address)

Compute the number of number of reachable configurations from address `address`.

# Example

```jldoctest
julia> address = BoseFS((3, 2, 1));


julia> H = HubbardMom1D(address);


julia> num_offdiagonals(H, address)
10
```
Part of the [`AbstractHamiltonian`](@ref) interface.
"""
num_offdiagonals(m::AbstractMatrix, i) = length(offdiagonals(m, i))

"""
    newadd, me = get_offdiagonal(ham, add, chosen)

Compute value `me` and new address `newadd` of a single (off-diagonal) matrix element in a
Hamiltonian `ham`. The off-diagonal element is in the same column as address `add` and is
indexed by integer index `chosen`.

# Example

```jldoctest
julia> addr = BoseFS(3, 2, 1);

julia> H = HubbardMom1D(addr);

julia> get_offdiagonal(H, addr, 3)
(BoseFS{6,3}(2, 1, 3), 1.0)
```
Part of the [`AbstractHamiltonian`](@ref) interface.
"""
get_offdiagonal(m::AbstractMatrix, i, n) = offdiagonals(m, i)[n]

"""
    starting_address(h)

Return the starting address for Hamiltonian `h`. When called on an `AbstractMatrix`,
`starting_address` returns the index of the lowest diagonal
element.

# Example

```jldoctest
julia> address = BoseFS((3, 2, 1));


julia> H = HubbardMom1D(address);


julia> address == starting_address(H)
true
```
Part of the [`AbstractHamiltonian`](@ref) interface.
"""
starting_address(m::AbstractMatrix) = findmin(real.(diag(m)))[2]

"""
    offdiagonals(h::AbstractHamiltonian, address)

Return an iterator over nonzero off-diagonal matrix elements of `h` in the same column as
`address`. Will iterate over pairs `(newaddress, matrixelement)`.

# Example

```jldoctest
julia> address = BoseFS(3,2,1);


julia> H = HubbardReal1D(address);


julia> h = offdiagonals(H, address)
6-element Rimu.Hamiltonians.Offdiagonals{BoseFS{6, 3, BitString{8, 1, UInt8}}, Float64, HubbardReal1D{Float64, BoseFS{6, 3, BitString{8, 1, UInt8}}, 1.0, 1.0}}:
 (fs"|2 3 1⟩", -3.0)
 (fs"|2 2 2⟩", -2.449489742783178)
 (fs"|3 1 2⟩", -2.0)
 (fs"|4 1 1⟩", -2.8284271247461903)
 (fs"|4 2 0⟩", -2.0)
 (fs"|3 3 0⟩", -1.7320508075688772)
```
Part of the [`AbstractHamiltonian`](@ref) interface.

# Extemded help

`offdiagonals` return and iterator of type `<:AbstractOffdiagonals`. It defaults to
returning `Offdiagonals(h, a)`

See also [`Offdiagonals`](@ref Main.Hamiltonians.Offdiagonals),
[`AbstractOffdiagonals`](@ref Main.Hamiltonians.AbstractOffdiagonals).

"""
function offdiagonals(m::AbstractMatrix, i)
    pairs = collect(zip(axes(m, 1), view(m, :, i)))
    return filter!(pairs) do ((k, v))
        k ≠ i && v ≠ 0
    end
end

"""
    random_offdiagonal(offdiagonals::AbstractOffdiagonals)
    random_offdiagonal(ham::AbstractHamiltonian, add)

Generate a single random excitation, i.e. choose from one of the accessible off-diagonal
elements in the column corresponding to address `add` of the Hamiltonian matrix represented
by `ham`. Alternatively, pass as argument an iterator over the accessible matrix elements.

Part of the [`AbstractHamiltonian`](@ref) interface.
"""
function random_offdiagonal(offdiagonals::AbstractVector)
    nl = length(offdiagonals) # check how many sites we could get_offdiagonal to
    chosen = rand(1:nl) # choose one of them
    naddress, melem = offdiagonals[chosen]
    return naddress, 1.0/nl, melem
end

function random_offdiagonal(ham, add)
    return random_offdiagonal(offdiagonals(ham, add))
end

@doc """
    LOStructure(op::AbstractHamiltonian)
    LOStructure(typeof(op))

Return information about the structure of the linear operator `op`.
`LOStructure` is used as a trait to speficy symmetries or other properties of the linear
operator `op` that may simplify or speed up calculations. Implemented instances are:

* `IsDiagonal()`: The operator is diagonal.
* `IsHermitian()`: The operator is complex and Hermitian or real and symmetric.
* `AdjointKnown()`: The operator is not Hermitian, but its
    [`adjoint`](@ref Main.Hamiltonians.adjoint) is implemented.
* `AdjointUnknown()`: [`adjoint`](@ref Main.Hamiltonians.adjoint) for this operator is not
    implemented.

Part of the [`AbstractHamiltonian`](@ref) interface.

In order to define this trait for a new linear operator type, define a method for
`LOStructure(::Type{<:MyNewLOType}) = …`.
"""
abstract type LOStructure end

struct IsDiagonal <: LOStructure end
struct IsHermitian <: LOStructure end
struct AdjointKnown <: LOStructure end
struct AdjointUnknown <: LOStructure end

# defaults
LOStructure(op) = LOStructure(typeof(op))
LOStructure(::Type) = AdjointUnknown()
LOStructure(::AbstractMatrix) = AdjointKnown()

"""
    has_adjoint(op)

Return true if `adjoint` is defined on `op`.

Part of the [`AbstractHamiltonian`](@ref) interface.

See also [`LOStructure`](@ref Main.Hamiltonians.LOStructure).
"""
has_adjoint(op) = has_adjoint(LOStructure(op))
has_adjoint(::AdjointUnknown) = false
has_adjoint(::LOStructure) = true
