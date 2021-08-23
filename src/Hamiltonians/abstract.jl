###
### This file contains abstract types, interfaces and traits.
###
"""
    AbstractHamiltonian{T}

Supertype that provides an interface for linear operators over a linear space with scalar
type `T` that are suitable for FCIQMC. Indexing is done with addresses (typically not
integers) from an address space that may be large (and will not need to be completely
generated).

`AbstractHamiltonian` instances operate on vectors of type [`AbstractDVec`](@ref) from the
module `DictVectors` and work well with addresses of type [`AbstractFockAddress`](@ref) from
the module `BitStringAddresses`. The type works well with the external package
[KrylovKit.jl](https://github.com/Jutho/KrylovKit.jl).

# Methods

Provides:

* [`offdiagonals`](@ref): iterator over reachable off-diagonal matrix elements
* [`random_offdiagonal`](@ref): function to generate random off-diagonal matrix element
* [`dimension`](@ref): get the dimension of the address space.
* `H[address1, address2]`: indexing with `getindex()` - mostly for testing purposes
* `*(H, v)`: deterministic matrix-vector multiply.
* `H(v)`: equivalent to `H * v`.
* `mul!(w, H, v)`: mutating matrix-vector multiply.
* [`dot(x, H, v)`](@ref): compute `x⋅(H*v)` minimizing allocations.

Methods that need to be implemented:

* [`num_offdiagonals(::AbstractHamiltonian, address)`](@ref)
* [`get_offdiagonal(::AbstractHamiltonian, address, chosen::Integer)`](@ref)
* [`diagonal_element(::AbstractHamiltonian, address)`](@ref)
* [`starting_address(::AbstractHamiltonian)`](@ref)

Optional methods to implement:

* [`Hamiltonians.LOStructure(::Type{<:AbstractHamiltonian})`](@ref)
* [`dimension(::Type, ::AbstractHamiltonian)`](@ref)
* [`offdiagonals(::AbstractHamiltonian, ::AbstractFockAddress)`](@ref)
* [`momentum(::AbstractHamiltonian)`](@ref)
"""
abstract type AbstractHamiltonian{T} end

Base.eltype(::AbstractHamiltonian{T}) where {T} = T

(h::AbstractHamiltonian)(v) = h * v
(h::AbstractHamiltonian)(w, v) = mul!(w, h, v)

BitStringAddresses.num_modes(h::AbstractHamiltonian) = num_modes(starting_address(h))

"""
    diagonal_element(ham, add)

Compute the diagonal matrix element of the linear operator `ham` at
address `add`.

# Example

```jldoctest
julia> addr = BoseFS((3, 2, 1));


julia> H = HubbardMom1D(addr);


julia> diagonal_element(H, addr)
8.666666666666664
```
"""
diagonal_element(m::AbstractMatrix, i) = m[i,i]

"""
    num_offdiagonals(ham, add)

Compute the number of number of reachable configurations from address `add`.

# Example

```jldoctest
julia> addr = BoseFS((3, 2, 1));


julia> H = HubbardMom1D(addr);


julia> num_offdiagonals(H, addr)
10
```
"""
num_offdiagonals

"""
    newadd, me = get_offdiagonal(ham, add, chosen)

Compute value `me` and new address `newadd` of a single (off-diagonal) matrix element in a
Hamiltonian `ham`. The off-diagonal element is in the same column as address `add` and is
indexed by integer index `chosen`.

# Example

```jldoctest
julia> addr = BoseFS((3, 2, 1));


julia> H = HubbardMom1D(addr);


julia> get_offdiagonal(H, addr, 3)
(BoseFS{6,3}((2, 1, 3)), 1.0)
```
"""
get_offdiagonal

"""
Approximate formula for log of binomial coefficient. [Source](https://en.wikipedia.org/wiki/Binomial_coefficient#Bounds_and_asymptotic_formulas)
"""
logbinomialapprox(n,k) =
    (n+0.5)*log((n+0.5)/(n-k+0.5))+k*log((n-k+0.5)/k) - 0.5*log(2π*k)

"""
    dimension(::Type{T}, h)

Return the dimension of Hilbert space as `T`. If the result does not fit into `T`, return
`nothing`. If `T<:AbstractFloat`, an approximate value computed with the improved
Stirling formula may be returned instead.

# Examples

```jldoctest
julia> dimension(HubbardMom1D(BoseFS((1,2,3))))
28
julia> dimension(HubbardMom1D(near_uniform(BoseFS{200,100})))


julia> dimension(Float64, HubbardMom1D(near_uniform(BoseFS{200,100})))
1.3862737677578234e81
julia> dimension(BigInt, HubbardMom1D(near_uniform(BoseFS{200,100})))
1386083821086188248261127842108801860093488668581216236221011219101585442774669540
```
"""
function dimension(::Type{T}, ::BoseFS{N,M}) where {N,M,T<:Integer}
    return try_binomial(T(N + M - 1), T(N))
end
function dimension(::Type{T}, ::BoseFS{N,M}) where {N,M,T<:AbstractFloat}
    return approximate_binomial(T(N + M - 1), T(N))
end
function dimension(::Type{T}, f::FermiFS{N,M}) where {N,M,T<:Integer}
    return try_binomial(T(M), T(N))
end
function dimension(::Type{T}, f::FermiFS{N,M}) where {N,M,T<:AbstractFloat}
    return approximate_binomial(T(M), T(N))
end
function dimension(::Type{T}, b::BoseFS2C) where {T}
    return dimension(T, b.bsa) * dimension(T, b.bsb)
end
function dimension(::Type{T}, c::CompositeFS) where {T}
    return prod(x -> dimension(T, x), c.components)
end

function try_binomial(n::T, k::T) where {T}
    try
        return T(binomial(n, k))
    catch
        return nothing
    end
end
function approximate_binomial(n::T, k::T) where {T}
    try
        T(binomial(Int128(n), Int128(k)))
    catch
        T(exp(logbinomialapprox(n, k)))
    end
end

dimension(h::AbstractHamiltonian) = dimension(Int, h)
dimension(::Type{T}, h::AbstractHamiltonian) where {T} = dimension(T, starting_address(h))

BitStringAddresses.near_uniform(h::AbstractHamiltonian) = near_uniform(typeof(starting_address(h)))

"""
    starting_address(h)

Return the starting address for Hamiltonian `h`. Part of the [`AbstractHamiltonian`](@ref)
interface. When called on an `AbstractMatrix` return the index of the lowest diagonal
element.

# Example

```jldoctest
julia> addr = BoseFS((3, 2, 1));


julia> H = HubbardMom1D(addr);


julia> addr == starting_address(H)
true
```
"""
starting_address(m::AbstractMatrix) = findmin(real.(diag(m)))[2]

"""
    Hamiltonians.LOStructure(op::AbstractHamiltonian)
    Hamiltonians.LOStructure(typeof(op))

`LOStructure` speficies properties of the linear operator `op`. If a special structure is
known this can speed up calculations. Implemented structures are:

* `Hamiltonians.Hermitian`: The operator is complex and Hermitian or real and symmetric.
* `Hamiltonians.AdjointKnown`: The operator is not Hermitian, but its [`adjoint`](@ref) is
  implemented.
* `Hamiltonians.AdjointUnknown`: [`adjoint`](@ref) for this operator is not implemented.

In order to define this trait for a new linear operator type, define a method for
`LOStructure(::Type{<:MyNewLOType}) = …`.

"""
abstract type LOStructure end

struct Hermitian <: LOStructure end
struct AdjointKnown <: LOStructure end
struct AdjointUnknown <: LOStructure end

# defaults
LOStructure(op::AbstractHamiltonian) = LOStructure(typeof(op))
LOStructure(::Type{<:AbstractHamiltonian}) = AdjointUnknown()

"""
    has_adjoint(op)

Return true if `adjoint` is defined on `op`.
"""
has_adjoint(op::AbstractHamiltonian) = has_adjoint(LOStructure(op))
has_adjoint(::AdjointUnknown) = false
has_adjoint(::LOStructure) = true

"""
    rayleigh_quotient(H, v)

```math
\\frac{⟨ v | H | v ⟩}{⟨ v|v ⟩}
```
"""
rayleigh_quotient(lo, v) = dot(v, lo, v)/norm(v)^2

"""
    TwoComponentHamiltonian{T} <: AbstractHamiltonian{T}

Abstract type for representing interacting two-component Hamiltonians in a Fock space with
two different species. At least the following fields should be present:

* `ha` Hamiltonian for species A
* `hb` Hamiltonian for species B

See [`AbstractHamiltonian`](@ref) for a list of methods that need to be defined.

Provides and implementation of [`dimension`](@ref).
"""
abstract type TwoComponentHamiltonian{T} <: AbstractHamiltonian{T} end

function dimension(::Type{T}, h::TwoComponentHamiltonian) where {T}
    return dimension(T, h.ha) * dimension(T, h.hb)
end

"""
    momentum(ham::AbstractHamiltonian)

Momentum as a linear operator in Fock space. Pass a Hamiltonian `ham` in order to convey
information about the Fock basis.

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
"""
momentum
