"""
    check_address_type(h::AbstractOperator, addr_or_type)
Throw an `ArgumentError` if `addr_or_type` is not compatible with `h`, otherwise return
`true`. Acceptable arguments are either an address or an address type, or a tuple or
array thereof.

See also [`allows_address_type`](@ref).
"""
@inline function check_address_type(h, ::Type{A}) where {A}
    if !allows_address_type(h, A)
        throw(ArgumentError("address type $A not allowed for operator $h"))
    end
    return true
end
@inline check_address_type(h, addr) = check_address_type(h, typeof(addr))
@inline function check_address_type(h::AbstractOperator, v::Union{AbstractArray,Tuple})
    all(check_address_type(h, a) for a in v)
end

(h::AbstractOperator)(v) = h * v
(h::AbstractOperator)(w, v) = mul!(w, h, v)

BitStringAddresses.num_modes(h::AbstractHamiltonian) = num_modes(starting_address(h))

"""
    dimension(h::AbstractHamiltonian, addr=starting_address(h))
    dimension(h::AbstractOperator, addr)
    dimension(addr::AbstractFockAddress)
    dimension(::Type{<:AbstractFockAddress})

Return the estimated dimension of Hilbert space. May return a `BigInt` number.

When called on an address or address type, the dimension of the linear space spanned by the
address type is returned. When called on an `AbstractHamiltonian`, an upper bound on the
dimension of the matrix representing the Hamiltonian is returned.

# Examples

```jldoctest
julia> dimension(OccupationNumberFS(1,2,3))
16777216

julia> dimension(HubbardReal1D(OccupationNumberFS(1,2,3)))
28

julia> dimension(BoseFS{200,100})
1386083821086188248261127842108801860093488668581216236221011219101585442774669540

julia> Float64(ans)
1.3860838210861882e81
```

Part of the [`AbstractHamiltonian`](@ref) interface. See also
[`BasisSetRepresentation`](@ref).
# Extended Help

The default fallback for `dimension` called on an [`AbstractHamiltonian`](@ref) is to return
the dimension of the address space, which provides an upper bound. For new Hamiltonians a
tighter bound can be provided by defining a custom method.

When extending [`AbstractHamiltonian`](@ref), define a method for the two-argument form
`dimension(h::MyNewHamiltonian, addr)`. For number-conserving Hamiltonians, the function
[`Hamiltonians.number_conserving_dimension`](@ref) may be useful.

When extending [`AbstractFockAddress`](@ref), define a method for
`dimension(::Type{MyNewFockAddress})`.
"""
dimension(h::AbstractHamiltonian) = dimension(h, starting_address(h))
dimension(::AbstractOperator, addr) = dimension(addr)
dimension(addr::AbstractFockAddress) = dimension(typeof(addr))
dimension(::T) where {T<:Number} = typemax(T) # e.g. integer addresses

function dimension(::Type{<:BoseFS{N,M}}) where {N,M}
    return number_conserving_bose_dimension(N,M)
end
function dimension(::Type{<:OccupationNumberFS{M,T}}) where {M,T}
    n = typemax(T)
    return BigInt(n + 1)^BigInt(M)
end
function dimension(::Type{<:FermiFS{N,M}}) where {N,M}
    return number_conserving_fermi_dimension(N, M)
end
function dimension(::Type{<:BoseFS2C{NA,NB,M}}) where {NA,NB,M}
    return dimension(BoseFS{NA,M}) * dimension(BoseFS{NB,M})
end
function dimension(::Type{<:CompositeFS{<:Any,<:Any,<:Any,T}}) where {T}
    return prod(dimension, T.parameters)
    # This relies on an implementation detail of the Tuple type and may break in future
    # julia versions.
end

# for backward compatibility
function dimension(::Type{T}, h, addr=starting_address(h)) where {T}
    return T(dimension(h, addr))
end
dimension(::Type{T}, addr::AbstractFockAddress) where {T} = T(dimension(addr))

Base.isreal(h::AbstractOperator) = eltype(h) <: Real
LinearAlgebra.isdiag(h::AbstractOperator) = LOStructure(h) ≡ IsDiagonal()
LinearAlgebra.ishermitian(h::AbstractOperator) = LOStructure(h) ≡ IsHermitian()
LinearAlgebra.issymmetric(h::AbstractOperator) = ishermitian(h) && isreal(h)

BitStringAddresses.near_uniform(h::AbstractHamiltonian) = near_uniform(typeof(starting_address(h)))

"""
    number_conserving_bose_dimension(n, m)

Return the dimension of the number-conserving Fock space for `n` bosons in `m` modes:
`binomial(n + m - 1, n)`.

See also [`number_conserving_fermi_dimension`](@ref), [`number_conserving_dimension`](@ref).
"""
number_conserving_bose_dimension(n, m) = binomial(BigInt(n + m - 1), BigInt(n))
"""
    number_conserving_fermi_dimension(n, m)

Return the dimension of the number-conserving Fock space for `n` fermions in `m` modes:
`binomial(m, n)`.

See also [`number_conserving_bose_dimension`](@ref), [`number_conserving_dimension`](@ref).
"""
number_conserving_fermi_dimension(n, m) = binomial(BigInt(m), BigInt(n))

"""
    number_conserving_dimension(address <: AbstractFockAddress)

Return the dimension of the Fock space spanned by the address type assuming particle number
conservation.

See also [`number_conserving_bose_dimension`](@ref),
[`number_conserving_fermi_dimension`](@ref), [`dimension`](@ref).
"""
function number_conserving_dimension(address::Union{BoseFS,OccupationNumberFS})
    m = num_modes(address)
    n = num_particles(address)
    return number_conserving_bose_dimension(n, m)
end
function number_conserving_dimension(address::FermiFS)
    m = num_modes(address)
    n = num_particles(address)
    return number_conserving_fermi_dimension(n, m)
end
function number_conserving_dimension(addr::BoseFS2C)
    return number_conserving_dimension(addr.bsa) * number_conserving_dimension(addr.bsb)
end
function number_conserving_dimension(address::CompositeFS)
    return prod(number_conserving_dimension, address.components)
end

"""
    rayleigh_quotient(H, v)

Return the Rayleigh quotient of the linear operator `H` and the vector `v`:

```math
\\frac{⟨ v | H | v ⟩}{⟨ v|v ⟩}
```
"""
rayleigh_quotient(lo, v) = dot(v, lo, v) / norm(v)^2

"""
    TwoComponentHamiltonian{T} <: AbstractHamiltonian{T}

Abstract type for representing interacting two-component Hamiltonians in a Fock space with
two different species. At least the following fields should be present:

* `ha` Hamiltonian for species A
* `hb` Hamiltonian for species B

See [`AbstractHamiltonian`](@ref) for a list of methods that need to be defined.

Provides an implementation of [`dimension`](@ref).
"""
abstract type TwoComponentHamiltonian{T} <: AbstractHamiltonian{T} end

function dimension(h::TwoComponentHamiltonian)
    return dimension(h.ha) * dimension(h.hb)
end

"""
    momentum(ham::AbstractHamiltonian)

Momentum as a linear operator in Fock space. Pass a Hamiltonian `ham` in order to convey
information about the Fock basis. Returns an [`AbstractHamiltonian`](@ref) that represents
the momentum operator.

Note: `momentum` is currently only defined on [`HubbardMom1D`](@ref).

# Example

```jldoctest
julia> add = BoseFS((1, 0, 2, 1, 2, 1, 1, 3));


julia> ham = HubbardMom1D(add; u = 2.0, t = 1.0);


julia> mom = momentum(ham);


julia> diagonal_element(mom, add) # calculate the momentum of a single configuration
-1.5707963267948966

julia> v = DVec(add => 10; capacity=1000);


julia> rayleigh_quotient(mom, v) # momentum expectation value for state vector `v`
-1.5707963267948966
```
Part of the [`AbstractHamiltonian`](@ref) interface.
"""
momentum

function Base.getindex(ham::AbstractOperator{T}, address1, address2) where {T}
    # calculate the matrix element when only two bitstring addresses are given
    # this is NOT used for the QMC algorithm and is currently not used either
    # for building the matrix for conventional diagonalisation.
    # Only used for verifying matrix.
    # This will be slow and inefficient. Avoid using for larger Hamiltonians!
    address1 == address2 && return diagonal_element(ham, address1) # diagonal
    res = zero(T)
    for (add, val) in offdiagonals(ham, address2) # off-diag column as iterator
        if add == address1
            res += val # found address1
        end
    end
    return res
end

LinearAlgebra.adjoint(op::AbstractOperator) = adjoint(LOStructure(op), op)

"""
    adjoint(::LOStructure, op::AbstractOperator)

Represent the adjoint of an [`AbstractOperator`](@ref). Extend this method to define
custom adjoints.
"""
function LinearAlgebra.adjoint(::S, op) where {S<:LOStructure}
    throw(ArgumentError(
        "`adjoint()` is not defined for `AbstractOperator`s with `LOStructure` `$(S)`. "*
        " Is your Hamiltonian hermitian?"
    ))
end

LinearAlgebra.adjoint(::IsHermitian, op) = op # adjoint is known
function LinearAlgebra.adjoint(::IsDiagonal, op)
    if eltype(op) <: Real || eltype(eltype(op)) <: Real # for op's that return vectors
        return op
    else
        throw(ArgumentError("adjoint() is not implemented for complex diagonal Hamiltonians"))
    end
end

"""
    TransformUndoer(transform::AbstractHamiltonian, op::AbstractOperator) <: AbstractHamiltonian

Create a new operator for the purpose of calculating overlaps of transformed
vectors, which are defined by some transformation `transform`. The new operator should
represent the effect of undoing the transformation before calculating overlaps, including
with an optional operator `op`.

Not exported; transformations should define all necessary methods and properties,
see [`AbstractHamiltonian`](@ref). An `ArgumentError` is thrown if used with an
unsupported transformation.

# Example

A similarity transform ``\\hat{G} = f \\hat{H} f^{-1}`` has eigenvector
``d = f \\cdot c`` where ``c`` is an eigenvector of ``\\hat{H}``. Then the
overlap ``c' \\cdot c = d' \\cdot f^{-2} \\cdot d`` can be computed by defining all
necessary methods for `TransformUndoer(G)` to represent the operator ``f^{-2}`` and
calculating `dot(d, TransformUndoer(G), d)`.

Observables in the transformed basis can be computed by defining `TransformUndoer(G, A)`
to represent ``f^{-1} A f^{-1}``.

# Supported transformations

* [`GutzwillerSampling`](@ref)
* [`GuidingVectorSampling`](@ref)
"""
struct TransformUndoer{
    T,K<:AbstractHamiltonian,O<:Union{AbstractOperator,Nothing}
} <: AbstractHamiltonian{T}
    transform::K
    op::O
end

function TransformUndoer(k::AbstractHamiltonian, op)
    # default check
    throw(ArgumentError("Unsupported transformation: $k"))
end
TransformUndoer(k::AbstractHamiltonian) = TransformUndoer(k::AbstractHamiltonian, nothing)

# common methods
starting_address(s::TransformUndoer) = starting_address(s.transform)
dimension(s::TransformUndoer, addr) = dimension(s.transform, addr)
function Base.:(==)(a::TransformUndoer, b::TransformUndoer)
    return a.transform == b.transform && a.op == b.op
end
