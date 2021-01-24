"""
This module defines Hamiltonian types and standard methods.
Model Hamiltonians should be subtyped to [`AbstractHamiltonian`](@ref).
Models implemented so far are:

* [`BoseHubbardReal1D`](@ref) Bose-Hubbard chain, real space
* [`ExtendedBHReal1D`](@ref) extended Bose-Hubbard model with on-site and nearest neighbour interactions, real space, one dimension
"""
module Hamiltonians

using Parameters, StaticArrays, LinearAlgebra, SparseArrays
using Setfield

import Base: *
import LinearAlgebra: mul!, dot, adjoint

using ..DictVectors
using ..BitStringAddresses
using ..ConsistentRNG

export AbstractHamiltonian, Hops, generateRandHop
export diagME, numOfHops, hop, hasIntDimension, dimensionLO, fDimensionLO
export rayleigh_quotient

export BosonicHamiltonian, bit_String_Length
export BoseHubbardReal1D, ExtendedBHReal1D, BoseHubbardReal1D2C
export BoseHubbardMom1D, Momentum, BoseHubbard2CMom1D, BoseHubbardMom1D2C
export HubbardMom1D

# First we have some generic types and methods for any linear operator
# that could be used for FCIQMC

"""
    AbstractHamiltonian{T}
Supertype that provides an interface for linear operators over a linear space with scalar
type `T` that are suitable for FCIQMC. Indexing is done with addresses (typically not integers)
from an address space that may be large (and will not need to be completely generated).

`AbstractHamiltonian` instances operate on vectors of type [`AbstractDVec`](@ref)
from the module `DictVectors` and work well with addresses of type [`BitStringAddressType`](@ref)
from the module `BitStringAddresses`. The type works well with the external package `KrylovKit.jl`.

Provides:
* [`Hops`](@ref): iterator over reachable off-diagonal matrix elements
* [`generateRandHop`](@ref): function to generate random off-diagonal matrix element
* `hamiltonian[address1, address2]`: indexing with `getindex()` - mostly for testing purposes
* `*(LO, v)` deterministic matrix-vector multiply (`== LO(v)`)
* `mul!(w, LO, v)` mutating matrix-vector multiply
* [`dot(x, LO, v)`](@ref) compute `x⋅(LO*v)` minimizing allocations

Methods that need to be implemented:
* [`numOfHops(lo::AbstractHamiltonian, address)`](@ref)
* [`hop(lo::AbstractHamiltonian, address, chosen::Integer)`](@ref)
* [`diagME(lo::AbstractHamiltonian, address)`](@ref)
* [`hasIntDimension(lo::AbstractHamiltonian)`](@ref)
* [`dimensionLO(lo::AbstractHamiltonian)`](@ref), if applicable
* [`fDimensionLO(lo::AbstractHamiltonian)`](@ref)
Optional:
* [`Hamiltonians.LOStructure(::Type{typeof(lo)})`](@ref)
"""
abstract type AbstractHamiltonian{T} end

Base.eltype(::AbstractHamiltonian{T}) where {T} = T

function *(h::AbstractHamiltonian{E}, v::AbstractDVec{K,V}) where {E, K, V}
    T = promote_type(E,V) # allow for type promotion
    w = empty(v, T) # allocate new vector; non-mutating version
    for (key,val) in pairs(v)
        w[key] += diagME(h, key)*val
        for (add,elem) in Hops(h, key)
            w[add] += elem*val
        end
    end
    return w
end

# # five argument version: not doing this now as it will be slower than 3-args
# function LinearAlgebra.mul!(w::AbstractDVec, h::AbstractHamiltonian, v::AbstractDVec, α, β)
#     rmul!(w, β)
#     for (key,val) in pairs(v)
#         w[key] += α * diagME(h, key) * val
#         for (add,elem) in Hops(h, key)
#             w[add] += α * elem * val
#         end
#     end
#     return w
# end

# three argument version
function LinearAlgebra.mul!(w::AbstractDVec, h::AbstractHamiltonian, v::AbstractDVec)
    empty!(w)
    for (key,val) in pairs(v)
        w[key] += diagME(h, key)*val
        for (add,elem) in Hops(h, key)
            w[add] += elem*val
        end
    end
    return w
end

"""
    Hamiltonians.LOStructure(op::AbstractHamiltonian)
    Hamiltonians.LOStructure(typeof(op))
`LOStructure` speficies properties of the linear operator `op`. If a special
structure is known this can speed up calculations. Implemented structures are:

* `Hamiltonians.HermitianLO` The operator is complex and hermitian or real and symmetric.
* `Hamiltonians.ComplexLO` The operator has no known specific structure.

In order to define this trait for a new linear operator type, define a method
for `LOStructure(::Type{T}) where T <: MyNewLOType = …`.
"""
abstract type LOStructure end

struct HermitianLO <: LOStructure end
struct ComplexLO <: LOStructure end

# defaults
LOStructure(op::AbstractHamiltonian) = LOStructure(typeof(op))
LOStructure(::Type{T}) where T <: AbstractHamiltonian = ComplexLO()

LinearAlgebra.adjoint(op::AbstractHamiltonian) = h_adjoint(LOStructure(op), op)

"""
    h_adjoint(los::LOStructure, op::AbstractHamiltonian)
Represent the adjoint of an `AbstractHamiltonian`. Extend this method to define
custom adjoints.
"""
function h_adjoint(los::LOStructure, op) # default
    throw(ErrorException("`adjoint()` not defined for `AbstractHamiltonian`s with `LOStructure` `$(typeof(los))`. Is your Hamiltonian hermitian?"))
    return op
end

h_adjoint(::HermitianLO, op) = op # adjoint is known

"""
    dot(x, LO::AbstractHamiltonian, v)
Evaluate `x⋅LO(v)` minimizing memory allocations.
"""
function LinearAlgebra.dot(x::AbstractDVec, LO::AbstractHamiltonian, v::AbstractDVec)
  return dot_w_trait(LOStructure(LO), x, LO, v)
end
# specialised method for UniformProjector
function LinearAlgebra.dot(::UniformProjector, LO::AbstractHamiltonian{T}, v::AbstractDVec{K,T2}) where {K, T, T2}
  result = zero(promote_type(T,T2))
  for (key,val) in pairs(v)
      result += diagME(LO, key) * val
      for (add,elem) in Hops(LO, key)
          result += elem * val
      end
  end
  return result
end

"""
    Hamiltonians.dot_w_trait(::LOStructure, x, LO::AbstractHamiltonian, v)
Internal function for making use of the `AbstractHamiltonian` trait `LOStructure`.
"""
dot_w_trait(::LOStructure, x, LO::AbstractHamiltonian, v) = dot_from_right(x,LO,v)
# default for LOs without special structure: keep order

function dot_w_trait(::HermitianLO, x, LO::AbstractHamiltonian, v)
    if length(x) < length(v)
        return conj(dot_from_right(v,LO,x)) # turn args around to execute faster
    else
        return dot_from_right(x,LO,v) # original order
    end
end

"""
    Hamiltonians.dot_from_right(x, LO, v)
Internal function evaluates the 3-argument `dot()` function in order from right
to left.
"""
function dot_from_right(x::AbstractDVec{K,T1}, LO::AbstractHamiltonian{T}, v::AbstractDVec{K,T2}) where {K, T,T1, T2}
    # function LinearAlgebra.dot(x::AbstractDVec{K,T1}, LO::AbstractHamiltonian{T}, v::AbstractDVec{K,T2}) where {K, T,T1, T2}
    result = zero(promote_type(T1,promote_type(T,T2)))
    for (key,val) in pairs(v)
        result += conj(x[key]) * diagME(LO, key) * val
        for (add,elem) in Hops(LO, key)
            result += conj(x[add]) * elem * val
        end
    end
    return result
end

"""
    Hops(ham, add)

Iterator over new address and matrix element for reachable
off-diagonal matrix elements of linear operator `ham` from address add.
Represents an abstract vector containing the possibly non-zero off-diagonal
matrix elements of the column of ham indexed by add.

### Examples
```julia
new_address, matrix_element = Hops(ham, current_address)[i]
number_of_hops = length(Hops(ham, current_address))
for (add,elem) in Hops(ham, current_address)
   # do something with address and elem
end
```
"""
struct Hops{T,A,O,I}  <: AbstractVector{T}
    h::O # AbstractHamiltonian
    add::A # address; usually a BitStringAddressType
    num::Int # number of possible hops
    info::I # reserved for additional info to be stored here
end

# default constructor
function Hops(ham::O, add::A) where {T,A,O <: AbstractHamiltonian{T}}
    return Hops{T,A,O,Nothing}(ham, add, numOfHops(ham, add), nothing)
end

Base.eltype(::Hops{T}) where {T} = T # apparently this works!

function Base.getindex(s::Hops, i::Int)
    nadd, melem = hop(s.h, s.add, i)
    return (nadd, melem)
end #  returns tuple (newaddress, matrixelement)

Base.size(s::Hops) = (s.num,)
Base.IndexStyle(::Type{<:Hops}) = IndexLinear()

"""
    generateRandHop(ham, add)
    generateRandHop(hops::Hops)

Generate a single random excitation, i.e. choose from one of the accessible
off-diagonal elements in the column corresponding to address `add` of
the Hamiltonian matrix represented by `ham`. Alternatively, pass as
argument an iterator over the accessible matrix elements.
"""
function generateRandHop(ham::AbstractHamiltonian, add)
  # generic implementation of a random excitation generator drawing from
  # a uniform distribution
  # draws a random linked site and returns the site's address and generation
  # probability
  nl = numOfHops(ham, add) # check how many sites we could hop to
  chosen = cRand(1:nl) # choose one of them
  #chosen = _nearlydivisionless(nl) + 1 # choose one of them
  # using a faster random number algorithm
  naddress, melem = hop(ham, add, chosen)
  return naddress, 1 ./nl, melem
  # return new address, generation probability, and matrix element
end

function generateRandHop(hops::Hops)
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

function Base.getindex(ham::AbstractHamiltonian{T}, address1, address2) where T
  # calculate the matrix element when only two bitstring addresses are given
  # this is NOT used for the QMC algorithm and is currenlty not used either
  # for building the matrix for conventional diagonalisation.
  # Only used for verifying matrix.
  # This will be slow and inefficient. Avoid using for larger Hamiltonians!
  address1 == address2 && return diagME(ham, address1) # diagonal
  for (add,val) in Hops(ham, address2) # off-diag column as iterator
      add == address1 && return val # found address1
  end
  return zero(T) # address1 not found
end # getindex(ham)

# A handy function that helps make use of the `AbstractHamiltonian` technology.
"""
    rayleigh_quotient(lo, v)
Compute
```math
\\frac{⟨ v | lo | v ⟩}{⟨ v|v ⟩}
```
"""
rayleigh_quotient(lo, v) = dot(v, lo, v)/norm(v)^2


"""
    sm, basis = build_sparse_matrix_from_LO(ham::AbstractHamiltonian, add; nnzs = 0)
Create a sparse matrix `sm` of all reachable matrix elements of a linear operator `ham`
starting from the address `add`. The vector `basis` contains the addresses of basis configurations.
Providing the number `nnzs` of expected calculated matrix elements may improve performance.
"""
function build_sparse_matrix_from_LO(ham::AbstractHamiltonian, fs; nnzs = 0)
    adds = [fs] # list of addresses of length linear dimension of matrix
    I = Vector{Int}(undef,0) # row indices, length nnz
    J = Vector{Int}(undef,0) # column indices, length nnz
    V = Vector{eltype(ham)}(undef,0) # values, length nnz
    if nnzs > 0
      sizehint!(I, nnzs)
      sizehint!(J, nnzs)
      sizehint!(V, nnzs)
    end

    k = 0 # 1:nnz, in principle, but possibly more as several contributions to a matrix element may occur
    i = 0 # 1:dim, column of matrix
    while true # loop over columns of the matrix
        k += 1
        i += 1 # next column
        i > length(adds) && break
        add = adds[i] # new address from list
        # compute and push diagonal matrix element
        melem = diagME(ham, add)
        push!(I, i)
        push!(J, i)
        push!(V, melem)
        for (nadd, melem) in Hops(ham, add) # loop over rows
            k += 1
            j = findnext(a->a == nadd, adds, 1) # find index of `nadd` in `adds`
            if isnothing(j)
                # new address: increase dimension of matrix by adding a row
                push!(adds, nadd)
                j = length(adds) # row index points to the new element in `adds`
            end
            # new nonzero matrix element
            push!(I, i)
            push!(J, j)
            push!(V, melem)
        end
    end
    return sparse(I,J,V), adds
    # when the index pair `(i,j)` occurs mutiple times in `I` and `J` the elements are added.
end

##########################################
#
# Specialising to bosonic model Hamiltonians
#
"""
    BosonicHamiltonian{T} <: AbstractHamiltonian{T}
Abstract type for representing Hamiltonians in a Fock space of fixed number of
scalar bosons. At least the following fields should be present:
* `n  # number of particles`
* `m  # number of modes`
* `AT # address type`

Methods that need to be implemented:
* [`numOfHops(lo::AbstractHamiltonian, address)`](@ref) - number of off-diagonal matrix elements
* [`hop(lo::AbstractHamiltonian, address, chosen::Integer)`](@ref) - access an off-diagonal m.e. by index `chosen`
* [`diagME(lo::AbstractHamiltonian, address)`](@ref) - diagonal matrix element
Optional:
* [`Hamiltonians.LOStructure(::Type{typeof(lo)})`](@ref) - can speed up deterministic calculations if `HermitianLO`

Provides:
* [`hasIntDimension(lo::AbstractHamiltonian)`](@ref)
* [`dimensionLO(lo::AbstractHamiltonian)`](@ref), might fail if linear space too large
* [`fDimensionLO(lo::AbstractHamiltonian)`](@ref)
* [`bit_String_Length`](@ref)
* [`nearUniform`](@ref), default version
"""
abstract type BosonicHamiltonian{T} <: AbstractHamiltonian{T} end

BitStringAddresses.numParticles(h::BosonicHamiltonian) = h.n
BitStringAddresses.numModes(h::BosonicHamiltonian) = h.m

"""
    bit_String_Length(ham)

Number of bits needed to represent an address for the linear operator `ham`.
"""
bit_String_Length(bh::BosonicHamiltonian) = numModes(bh) + numParticles(bh) - 1

"""
    hasIntDimension(ham)

Return `true` if dimension of the linear operator `ham` can be computed as an
integer and `false` if not.

If `true`, `dimensionLO(h)` will be successful and return an `Int`. The method
`fDimensionLO(h)` should be useful in other cases.
"""
function hasIntDimension(h)
  try
    dimensionLO(h)
    return true
  catch
    false
  end
end

"""
    dimensionLO(hamiltonian)

Compute dimension of linear operator as integer.
"""
dimensionLO(h::BosonicHamiltonian) = binomial(h.n + h.m - 1, h.n)
# formula for boson Hilbert spaces

"""
    fDimensionLO(hamiltonian)

Returns the dimension of Hilbert space as Float64. The exact result is
returned if the value is smaller than 2^53. Otherwise, an improved Stirling formula
is used.
"""
function fDimensionLO(h::BosonicHamiltonian)
  fbinomial(h.n + h.m - 1, h.n) # formula for boson Hilbert spaces
  # NB: returns a Float64
end #dimHS

"""
Compute binomial coefficient and return Float64. Stirlings formula
is used to return approximate value if integer arithmetic is insufficient.
"""
fbinomial(n,k) = try
  Float64(binomial(Int128(n), Int128(k)))
catch # if we get integer overflow
  exp(logbinomialapprox(n,k))
  # this should work unless the number is larger than 10^308
end # fbinomial

"""
Approximate formula for log of binomial coefficient. Source:
  <https://en.wikipedia.org/wiki/Binomial_coefficient#Bounds_and_asymptotic_formulas>
"""
logbinomialapprox(n,k) =
  (n+0.5)*log((n+0.5)/(n-k+0.5))+k*log((n-k+0.5)/k) - 0.5*log(2*pi*k)

"""
    nearUniform(ham)
Create bitstring address with near uniform distribution of particles
across modes for the Hamiltonian `ham`.
"""
function BitStringAddresses.nearUniform(h::BosonicHamiltonian)
    fillingfactor, extras = divrem(h.n, h.m)
    startonr = fill(fillingfactor,h.m)
    startonr[1:extras] += ones(Int, extras)
    return bitaddr(startonr, h.AT)
end

##########################################
#
# Here we start implementing model-specific code
#
###
### BoseHubbardReal1D
###


@with_kw struct BoseHubbardReal1D{T} <: BosonicHamiltonian{T}
  n::Int = 6    # number of bosons
  m::Int = 6    # number of lattice sites
  u::T = 1.0    # interaction strength
  t::T = 1.0    # hopping strength
  AT::Type = BSAdd64 # address type
end

@doc """
    ham = BoseHubbardReal1D(;[n=6, m=6, u=1.0, t=1.0, AT = BSAdd64])

Implements a one-dimensional Bose Hubbard chain in real space.

```math
\\hat{H} = -t \\sum_{\\langle i,j\\rangle} a_i^† a_j + \\frac{u}{2}\\sum_i n_i (n_i-1)
```

# Arguments
- `n::Int`: the number of bosons
- `m::Int`: the number of lattice sites
- `u::Float64`: the interaction parameter
- `t::Float64`: the hopping strength
- `AT::Type`: the address type

# Functor use:
    w = ham(v)
    ham(w, v)
Compute the matrix - vector product `w = ham * v`. The two-argument version is
mutating for `w`.

    ham(:dim)
Return the dimension of the linear space if representable as `Int`, otherwise
return `nothing`.

    ham(:fdim)
Return the approximate dimension of linear space as `Float64`.
""" BoseHubbardReal1D

# set the `LOStructure` trait
LOStructure(::Type{BoseHubbardReal1D{T}}) where T <: Real = HermitianLO()

"""
    BoseHubbardReal1D(add::BitStringAddressType; u=1.0, t=1.0)
Set up the `BoseHubbardReal1D` with the correct particle and mode number and
address type. Parameters `u` and `t` can be passed as keyword arguments.
"""
function BoseHubbardReal1D(add::BSA; u=1.0, t=1.0) where BSA <: BitStringAddressType
  n = numParticles(add)
  m = numModes(add)
  return BoseHubbardReal1D(n,m,u,t,BSA)
end

# functor definitions need to be done separately for each concrete type
function (h::BoseHubbardReal1D)(s::Symbol)
    if s == :dim # attempt to compute dimension as `Int`
        return hasIntDimension(h) ? dimensionLO(h) : nothing
    elseif s == :fdim
        return fDimensionLO(h) # return dimension as floating point
    end
    return nothing
end
# should be all that is needed to make the Hamiltonian a linear map:
(h::BoseHubbardReal1D)(v) = h*v
# (h::BoseHubbardReal1D)(w, v) = mul!(w, h, v) # mutating version
# function (h::BoseHubbardReal1D)(w, v) # mutating version
#     for (key,val) in pairs(v)
#         w[key] += diagME(h, key)*val
#         for (add,elem) in Hops(h, key)
#             w[add] += elem*val
#         end
#     end
#     return w
# end

# """
#     setupBoseHubbardReal1D(; n, m, u, t, [AT = BoseFS, genInitialONR = nearUniform])
#     -> ham::BoseHubbardReal1D, address::AT
# Set up the Hamiltonian `ham` and initial address `address` for the Bose Hubbard
# model with the given parameters as keyword arguments, see
# [`BoseHubbardReal1D`](@ref). For `AT` pass an address type (or suitable
# constructor) and for `genInitialONR` a function that takes `n` and `m` as
# arguments and returns an occupation number representation, see
# [`nearUniform()`](@ref):
#
# `onr = genInitialONR(n,m)`
# """
# function setupBoseHubbardReal1D(;n::Int, m::Int, u, t,
#                     AT = BoseFS, genInitialONR = nearUniform)
#     address = AT(genInitialONR(n,m))
#     ham = BoseHubbardReal1D(n = n,
#                             m = m,
#                             u = u,
#                             t = t,
#                             AT = typeof(address)
#     )
#     return ham, address
# end

"""
    diagME(ham, add)

Compute the diagonal matrix element of the linear operator `ham` at
address `add`.
"""
function diagME(h::BoseHubbardReal1D, address)
  h.u * bosehubbardinteraction(address) / 2
end

###
### ExtendedBoseHubbardReal1D
###


@with_kw struct ExtendedBHReal1D{T} <: BosonicHamiltonian{T}
  n::Int = 6    # number of bosons
  m::Int = 6    # number of lattice sites
  u::T = 1.0    # on-site interaction strength
  v::T = 1.0    # on-site interaction strength
  t::T = 1.0    # hopping strength
  AT::Type = BSAdd64 # address type
end
@doc """
    ham = ExtendedBHReal1D(n=6, m=6, u=1.0, v=1.0, t=1.0, AT=BSAdd64)

Implements the extended Bose Hubbard model on a one-dimensional chain
in real space.

```math
\\hat{H} = -t \\sum_{\\langle i,j\\rangle} a_i^† a_j + \\frac{u}{2}\\sum_i n_i (n_i-1) + v \\sum_{\\langle i,j\\rangle} n_i n_j
```

# Arguments
- `n::Int`: number of bosons
- `m::Int`: number of lattice sites
- `u::Float64`: on-site interaction parameter
- `v::Float64`: the next-neighbor interaction
- `t::Float64`: the hopping strength
- `AT::Type`: address type for identifying configuration
""" ExtendedBHReal1D

# set the `LOStructure` trait
LOStructure(::Type{ExtendedBHReal1D{T}}) where T <: Real = HermitianLO()

"""
    ExtendedBHReal1D(add::BitStringAddressType; u=1.0, v=1.0 t=1.0)
Set up the `BoseHubbardReal1D` with the correct particle and mode number and
address type. Parameters `u` and `t` can be passed as keyword arguments.
"""
function ExtendedBHReal1D(add::BSA; u=1.0, v=1.0, t=1.0) where BSA <: BitStringAddressType
  n = numParticles(add)
  m = numModes(add)
  return ExtendedBHReal1D(n,m,u,v,t,BSA)
end

# functor definitions need to be done separately for each concrete type
function (h::ExtendedBHReal1D)(s::Symbol)
    if s == :dim # attempt to compute dimension as `Int`
        return hasIntDimension(h) ? dimensionLO(h) : nothing
    elseif s == :fdim
        return fDimensionLO(h) # return dimension as floating point
    end
    return nothing
end
# should be all that is needed to make the Hamiltonian a linear map:
(h::ExtendedBHReal1D)(v) = h*v
# (h::ExtendedBHReal1D)(w, v) = mul!(w, h, v) # mutating version
# function (h::ExtendedBHReal1D)(w, v) # mutating version
#     for (key,val) in pairs(v)
#         w[key] += diagME(h, key)*val
#         for (add,elem) in Hops(h, key)
#             w[add] += elem*val
#         end
#     end
#     return w
# end

function diagME(h::ExtendedBHReal1D, address)
  ebhinteraction, bhinteraction= ebhm(address, h.m)
  return h.u * bhinteraction / 2 + h.v * ebhinteraction
end

# The off-diagonal matrix elements of the 1D Hubbard chain are the same for
# the extended and original Bose-Hubbard model.

BoseHubbardExtOrNot = Union{ExtendedBHReal1D, BoseHubbardReal1D}
# type alias for convenience

"""
    numOfHops(ham, add)

Compute the number of number of reachable configurations from address `add`.
"""
function numOfHops(ham::BoseHubbardExtOrNot, add)
    return numberlinkedsites(add)
end

"""
    newadd, me = hop(ham, add, chosen)

Compute matrix element of `hamiltonian` and new address of a single hop from
address `add` with integer index `chosen`.
"""
function hop(ham::BoseHubbardExtOrNot, add, chosen::Integer)
    naddress, onproduct = hopnextneighbour(add, chosen,
        ham.m, ham.n)
    return naddress, - ham.t*sqrt(onproduct)
    # return new address and matrix element
end




###
### BoseHubbardMom1D
###


@with_kw struct BoseHubbardMom1D{T, AD} <: BosonicHamiltonian{T}
  n::Int = 6    # number of bosons
  m::Int = 6    # number of lattice sites
  u::T = 1.0    # interaction strength
  t::T = 1.0    # hopping strength
  # AT::Type = BSAdd64 # address type
  add::AD       # starting address
end

@doc """
    ham = BoseHubbardMom1D(;[n=6, m=6, u=1.0, t=1.0], add = add)
    ham = BoseHubbardMom1D(add; u=1.0, t=1.0)

Implements a one-dimensional Bose Hubbard chain in momentum space.

```math
\\hat{H} = -t \\sum_{k} ϵ_k n_k + \\frac{u}{M}\\sum_{kpqr} a^†_{r} a^†_{q} a_p a_k δ_{r+q,p+k}\\\\
ϵ_k = - 2 t \\cos(k)
```

# Arguments
- `n::Int`: the number of bosons
- `m::Int`: the number of lattice sites
- `u::Float64`: the interaction parameter
- `t::Float64`: the hopping strength
- `AT::Type`: the address type

# Functor use:
    w = ham(v)
    ham(w, v)
Compute the matrix - vector product `w = ham * v`. The two-argument version is
mutating for `w`.

    ham(:dim)
Return the dimension of the linear space if representable as `Int`, otherwise
return `nothing`.

    ham(:fdim)
Return the approximate dimension of linear space as `Float64`.
""" BoseHubbardMom1D


# set the `LOStructure` trait
LOStructure(::Type{BoseHubbardMom1D{T, AD}}) where {T <: Real, AD} = HermitianLO()

"""
    BoseHubbardMom1D(add::BitStringAddressType; u=1.0, t=1.0)
Set up the `BoseHubbardMom1D` with the correct particle and mode number and
address type. Parameters `u` and `t` can be passed as keyword arguments.
"""
function BoseHubbardMom1D(add::BSA; u=1.0, t=1.0) where BSA <: BitStringAddressType
  n = numParticles(add)
  m = numModes(add)
  return BoseHubbardMom1D(n,m,u,t,add)
end

# functor definitions need to be done separately for each concrete type
function (h::BoseHubbardMom1D)(s::Symbol)
  if s == :dim # attempt to compute dimension as `Int`
      return hasIntDimension(h) ? dimensionLO(h) : nothing
  elseif s == :fdim
      return fDimensionLO(h) # return dimension as floating point
  end
  return nothing
end
# should be all that is needed to make the Hamiltonian a linear map:
(h::BoseHubbardMom1D)(v) = h*v

function numOfHops(ham::BoseHubbardMom1D, add)
  singlies, doublies = numSandDoccupiedsites(add)
  return singlies*(singlies-1)*(ham.m - 2) + doublies*(ham.m - 1)
  # number of excitations that can be made
end

function hop(ham::BoseHubbardMom1D, add::ADDRESS, chosen) where ADDRESS
  onr = BitStringAddresses.m_onr(add) # get occupation number representation as a mutable array
  singlies, doublies = numSandDoccupiedsites(add)
  onproduct = 1
  k = p = q = 0
  double = chosen - singlies*(singlies-1)*(ham.m - 2)
  # start by making holes as the action of two annihilation operators
  if double > 0 # need to choose doubly occupied site for double hole
    # c_p c_p
    double, q = fldmod1(double, ham.m-1)
    # double is location of double
    # q is momentum transfer
    for (i, occ) in enumerate(onr)
      if occ > 1
        double -= 1
        if double == 0
          onproduct *= occ*(occ-1)
          onr[i] = occ-2 # annihilate two particles in onr
          p = k = i # remember where we make the holes
          break # should break out of the for loop
        end
      end
    end
  else # need to punch two single holes
    # c_k c_p
    pair, q = fldmod1(chosen, ham.m-2) # floored integer division and modulus in ranges 1:(m-1)
    first, second = fldmod1(pair, singlies-1) # where the holes are to be made
    if second < first # put them in ascending order
      f_hole = second
      s_hole = first
    else
      f_hole = first
      s_hole = second + 1 # as we are counting through all singlies
    end
    counter = 0
    for (i, occ) in enumerate(onr)
      if occ > 0
        counter += 1
        if counter == f_hole
          onproduct *= occ
          onr[i] = occ -1 # punch first hole
          p = i # location of first hole
        elseif counter == s_hole
          onproduct *= occ
          onr[i] = occ -1 # punch second hole
          k = i # location of second hole
          break
        end
      end
    end
    # we have p<k and 1 < q < ham.m - 2
    if q ≥ k-p
      q += 1 # to avoid putting particles back into the holes
    end
  end # if double > 0 # we're done punching holes

  # now it is time to deal with two creation operators
  # c^†_k-q
  kmq = mod1(k-q, ham.m) # in 1:m # use mod1() to implement periodic boundaries
  occ = onr[kmq]
  onproduct *= occ + 1
  onr[kmq] = occ + 1
  # c^†_p+q
  ppq = mod1(p+q, ham.m) # in 1:m # use mod1() to implement periodic boundaries
  occ = onr[ppq]
  onproduct *= occ + 1
  onr[ppq] = occ + 1

  return ADDRESS(onr), ham.u/(2*ham.m)*sqrt(onproduct)
  # return new address and matrix element
end


"""
    ks(h::BoseHubbardMom1D)
Return a range for `k` values in the interval (-π, π] to be `dot()`ed to an `onr()`
occupation number representation.
"""
function ks(h::BoseHubbardMom1D)
    m = numModes(h)
    step = 2π/m
    if isodd(m)
        start = -π*(1+1/m) + step
    else
        start = -π + step
    end
    return StepRangeLen(start, step, m) # faster than range()
end # fast! - can be completely resolved by compiler


function diagME(h::BoseHubbardMom1D, add)
  onrep = BitStringAddresses.onr(add) # get occupation number representation

  # single particle part of Hubbard momentum space Hamiltonian
  # ke = -2*h.t.*cos.(ks(h))⋅onrep # works but allocates memory due to broadcasting
  # ugly but no allocations:
  ke = 0.0
  for (k,on) in zip(ks(h),onrep)
    ke += -2*h.t * cos(k) * on
  end

  # now compute diagonal interaction energy
  onproduct = 0 # Σ_kp < c^†_p c^†_k c_k c_p >
  # for p in 1:h.m
  #   for k in 1:h.m
  #     if k==p
  #       onproduct += onrep[k]*(onrep[k]-1)
  #     else
  #       onproduct += 2*onrep[k]*onrep[p] # two terms in sum over creation operators
  #     end
  #   end
  # end
  for p = 1:h.m
      # faster triangular loop; 9 μs instead of 33 μs for nearUniform(BoseFS{200,199})
      @inbounds onproduct += onrep[p] * (onrep[p] - 1)
      @inbounds @simd for k = 1:p-1
          onproduct += 4*onrep[k]*onrep[p]
      end
  end
  # @show onproduct
  pe = h.u/(2*h.m)*onproduct
  return ke + pe
end




"""
    Momentum(ham::AbstractHamiltonian) <: AbstractHamiltonian
Momentum as a linear operator in Fock space. Pass a Hamiltonian `ham` in order to convey information about the Fock basis.

Example use:
```julia
add = BoseFS((1,0,2,1,2,1,1,3)) # address for a Fock state (configuration) with 11 bosons in 8 modes
ham = BoseHubbardMom1D(add; u = 2.0, t = 1.0)
mom = Momentum(ham) # create an instance of the momentum operator
diagME(mom, add) # 10.996 - to calculate the momentum of a single configuration
v = DVec(Dict(add => 10), 1000)
rayleigh_quotient(mom, v) # 10.996 - momentum expectation value for state vector `v`
```
"""
struct Momentum{H,T} <: AbstractHamiltonian{T}
  ham::H
end
LOStructure(::Type{Momentum{H,T}}) where {H,T <: Real} = HermitianLO()
Momentum(ham::BoseHubbardMom1D{T, AD}) where {T, AD} = Momentum{typeof(ham), T}(ham)
numOfHops(ham::Momentum, add) = 0
diagME(mom::Momentum, add) = mod1(onr(add)⋅ks(mom.ham) + π, 2π) - π # fold into (-π, π]
# surpsingly this is all that is needed. We don't even have to define `hop()`, because it is never reached.
# `rayleigh_quotient(Momentum(ham),v)` is performant!

###############################################

struct HubbardMom1D{TT,U,T,N,M,AD} <: AbstractHamiltonian{T}
    add::AD # default starting address, should have N particles and M modes
    ks::SVector{M,TT} # values for k
    kes::SVector{M,TT} # values for kinetic energy
end

@doc """
    HubbardMom1D(add::BoseFS; u=1.0, t=1.0)
Implements a one-dimensional Bose Hubbard chain in momentum space.

```math
\\hat{H} = -t \\sum_{k} ϵ_k n_k + \\frac{u}{M}\\sum_{kpqr} a^†_{r} a^†_{q} a_p a_k δ_{r+q,p+k}\\\\
ϵ_k = - 2 t \\cos(k)
```

# Arguments
- `add::BoseFS`: bosonic starting address, defines number of particles and sites
- `u::Float64`: the interaction parameter
- `t::Float64`: the hopping strength

# Functor use:
    w = ham(v)
    ham(w, v)
Compute the matrix - vector product `w = ham * v`. The two-argument version is
mutating for `w`.

    ham(:dim)
Return the dimension of the linear space if representable as `Int`, otherwise
return `nothing`.

    ham(:fdim)
Return the approximate dimension of linear space as `Float64`.
""" HubbardMom1D

# constructors
function HubbardMom1D(add::BoseFS{N,M,A}; u::TT=1.0, t::TT=1.0) where {N, M, TT, A}
    step = 2π/M
    if isodd(M)
        start = -π*(1+1/M) + step
    else
        start = -π + step
    end
    kr = range(start; step = step, length = M)
    ks = SVector{M}(kr)
    kes = SVector{M}(-2*cos.(kr))
    return HubbardMom1D{TT,u,t,N,M,BoseFS{N,M,A}}(add, ks, kes)
end
# allow passing the N and M parameters for compatibility with show()
function HubbardMom1D{N,M}(add::BoseFS{N,M,A}; u::TT=1.0, t::TT=1.0) where {N, M, TT, A}
    return HubbardMom1D(add; u=u, t=t)
end

# display in a way that can be used as constructor
function Base.show(io::IO, h::HubbardMom1D{TT,U,T,N,M,AD}) where {TT,U,T,N,M,AD}
    print(io, "HubbardMom1D{$N,$M}(")
    show(io, h.add)
    print(io, "; u=$U, t=$T)")
end

Base.eltype(::HubbardMom1D{TT,U,T,N,M,AD}) where {TT,U,T,N,M,AD} = TT

# set the `LOStructure` trait
LOStructure(::Type{HubbardMom1D{TT,U,T,N,M,AD}}) where {TT<:Real,U,T,N,M,AD} = HermitianLO()

# functor definitions need to be done separately for each concrete type
function (h::HubbardMom1D)(s::Symbol)
  if s == :dim # attempt to compute dimension as `Int`
      return hasIntDimension(h) ? dimensionLO(h) : nothing
  elseif s == :fdim
      return fDimensionLO(h) # return dimension as floating point
  end
  return nothing
end
# should be all that is needed to make the Hamiltonian a linear map:
(h::HubbardMom1D)(v) = h*v

Momentum(ham::HubbardMom1D{TT,U,T,N,M,AD}) where {TT,U,T,N,M,AD} = Momentum{typeof(ham), TT}(ham)
# for Momentum
ks(h::HubbardMom1D) = h.ks

# standard interface function
function numOfHops(ham::HubbardMom1D, add)
  nSandD = numSandDoccupiedsites(add)
  return numOfHops(ham, add, nSandD)
end

# 3-argument version
@inline function numOfHops(ham::HubbardMom1D{TT,U,T,N,M,AD}, add, nSandD) where {TT,U,T,N,M,AD}
  singlies, doublies = nSandD
  return singlies*(singlies-1)*(M - 2) + doublies*(M - 1)
  # number of excitations that can be made
end

@inline function interaction_energy_diagonal(h::HubbardMom1D{TT,U,T,N,M,AD},
        onrep::StaticVector) where {TT,U,T,N,M,AD<:BoseFS}
    # now compute diagonal interaction energy
    onproduct = 0 # Σ_kp < c^†_p c^†_k c_k c_p >
    for p = 1:M
        @inbounds onproduct += onrep[p] * (onrep[p] - 1)
        @inbounds @simd for k = 1:p-1
            onproduct += 4*onrep[k]*onrep[p]
        end
    end
    # @show onproduct
    return U / 2M * onproduct
end

function kinetic_energy(h::HubbardMom1D, add::BitStringAddressType)
    onrep = BitStringAddresses.m_onr(add) # get occupation number representation
    return kinetic_energy(h, onrep)
end

@inline function kinetic_energy(h::HubbardMom1D, onrep::StaticVector)
    return h.kes⋅onrep # safe as onrep is Real
end

@inline function diagME(h::HubbardMom1D, add)
    onrep = BitStringAddresses.m_onr(add) # get occupation number representation
    return diagME(h, onrep)
end

@inline function diagME(h::HubbardMom1D, onrep::StaticVector)
    return kinetic_energy(h, onrep) + interaction_energy_diagonal(h, onrep)
end

@inline function hop(ham::HubbardMom1D{TT,U,T,N,M,AD}, add::AD, chosen::Number) where {TT,U,T,N,M,AD}
    hop(ham, add, chosen, numSandDoccupiedsites(add))
end

@inline function hop_old(ham::HubbardMom1D{TT,U,T,N,M,AD}, add::AD, chosen::Number, nSD) where {TT,U,T,N,M,AD}
  onrep =  BitStringAddresses.m_onr(add)
  # get occupation number representation as a mutable array
  singlies, doublies = nSD # precomputed `numSandDoccupiedsites(add)`
  onproduct = 1
  k = p = q = 0
  double = chosen - singlies*(singlies-1)*(M - 2)
  # start by making holes as the action of two annihilation operators
  if double > 0 # need to choose doubly occupied site for double hole
    # c_p c_p
    double, q = fldmod1(double, M-1)
    # double is location of double
    # q is momentum transfer
    for (i, occ) in enumerate(onrep)
      if occ > 1
        double -= 1
        if double == 0
          onproduct *= occ*(occ-1)
          onrep[i] = occ-2 # annihilate two particles in onrep
          p = k = i # remember where we make the holes
          break # should break out of the for loop
        end
      end
    end
  else # need to punch two single holes
    # c_k c_p
    pair, q = fldmod1(chosen, M-2) # floored integer division and modulus in ranges 1:(m-1)
    first, second = fldmod1(pair, singlies-1) # where the holes are to be made
    if second < first # put them in ascending order
      f_hole = second
      s_hole = first
    else
      f_hole = first
      s_hole = second + 1 # as we are counting through all singlies
    end
    counter = 0
    for (i, occ) in enumerate(onrep)
      if occ > 0
        counter += 1
        if counter == f_hole
          onproduct *= occ
          onrep[i] = occ -1 # punch first hole
          p = i # location of first hole
        elseif counter == s_hole
          onproduct *= occ
          onrep[i] = occ -1 # punch second hole
          k = i # location of second hole
          break
        end
      end
    end
    # we have p<k and 1 < q < ham.m - 2
    if q ≥ k-p
      q += 1 # to avoid putting particles back into the holes
    end
  end # if double > 0 # we're done punching holes

  # now it is time to deal with two creation operators
  # c^†_k-q
  kmq = mod1(k-q, M) # in 1:m # use mod1() to implement periodic boundaries
  occ = onrep[kmq]
  onproduct *= occ + 1
  onrep[kmq] = occ + 1
  # c^†_p+q
  ppq = mod1(p+q, M) # in 1:m # use mod1() to implement periodic boundaries
  occ = onrep[ppq]
  onproduct *= occ + 1
  onrep[ppq] = occ + 1

  return AD(onrep), U/(2*M)*sqrt(onproduct)
  # return new address and matrix element
end

# a non-allocating version of hop()
@inline function hop(ham::HubbardMom1D{TT,U,T,N,M,AD}, add::AD, chosen::Number, nSD) where {TT,U,T,N,M,AD}
  onrep =  BitStringAddresses.s_onr(add)
  # get occupation number representation as a static array
  singlies, doublies = nSD # precomputed `numSandDoccupiedsites(add)`
  onproduct = 1
  k = p = q = 0
  double = chosen - singlies*(singlies-1)*(M - 2)
  # start by making holes as the action of two annihilation operators
  if double > 0 # need to choose doubly occupied site for double hole
    # c_p c_p
    double, q = fldmod1(double, M-1)
    # double is location of double
    # q is momentum transfer
    for (i, occ) in enumerate(onrep)
      if occ > 1
        double -= 1
        if double == 0
          onproduct *= occ*(occ-1)
          onrep = @set onrep[i] = occ-2
          # annihilate two particles in onrep
          p = k = i # remember where we make the holes
          break # should break out of the for loop
        end
      end
    end
  else # need to punch two single holes
    # c_k c_p
    pair, q = fldmod1(chosen, M-2) # floored integer division and modulus in ranges 1:(m-1)
    first, second = fldmod1(pair, singlies-1) # where the holes are to be made
    if second < first # put them in ascending order
      f_hole = second
      s_hole = first
    else
      f_hole = first
      s_hole = second + 1 # as we are counting through all singlies
    end
    counter = 0
    for (i, occ) in enumerate(onrep)
      if occ > 0
        counter += 1
        if counter == f_hole
          onproduct *= occ
          onrep = @set onrep[i] = occ-1
          # punch first hole
          p = i # location of first hole
        elseif counter == s_hole
          onproduct *= occ
          onrep = @set onrep[i] = occ-1
          # punch second hole
          k = i # location of second hole
          break
        end
      end
    end
    # we have p<k and 1 < q < ham.m - 2
    if q ≥ k-p
      q += 1 # to avoid putting particles back into the holes
    end
  end # if double > 0 # we're done punching holes

  # now it is time to deal with two creation operators
  # c^†_k-q
  kmq = mod1(k-q, M) # in 1:m # use mod1() to implement periodic boundaries
  occ = onrep[kmq]
  onproduct *= occ + 1
  onrep = @set onrep[kmq] = occ + 1
  # c^†_p+q
  ppq = mod1(p+q, M) # in 1:m # use mod1() to implement periodic boundaries
  occ = onrep[ppq]
  onproduct *= occ + 1
  onrep = @set onrep[ppq] = occ + 1

  return AD(onrep), U/(2*M)*sqrt(onproduct)
  # return new address and matrix element
end

function hasIntDimension(h::HubbardMom1D{TT,U,T,N,M,AD}) where {TT,U,T,N,M,AD<:BoseFS}
  try
    binomial(N + M - 1, N)# formula for boson Hilbert spaces
    return true
  catch
    false
  end
end

function dimensionLO(h::HubbardMom1D{TT,U,T,N,M,AD}) where {TT,U,T,N,M,AD<:BoseFS}
    return binomial(N + M - 1, N) # formula for boson Hilbert spaces
end

function fDimensionLO(h::HubbardMom1D{TT,U,T,N,M,AD}) where {TT,U,T,N,M,AD<:BoseFS}
  fbinomial(N + M - 1, N) # formula for boson Hilbert spaces
  # NB: returns a Float64
end #dimHS

function Hops(ham::O, add::AD) where {TT,U,T,N,M,AD, O<:HubbardMom1D{TT,U,T,N,M,AD}}
    nSandD = numSandDoccupiedsites(add)::Tuple{Int64,Int64}
    # store this information for reuse
    nH = numOfHops(ham, add, nSandD)
    return Hops{TT,AD,O,Tuple{Int64,Int64}}(ham, add, nH, nSandD)
end

function Base.getindex(s::Hops{T,A,O,I}, i::Int) where {T,A,O<:HubbardMom1D,I}
    nadd, melem = hop(s.h, s.add, i, s.info)
    return (nadd, melem)
end #  returns tuple (newaddress, matrixelement)


################################################
#
# Internals of the Bose Hubbard model:
# private functions (not exported)
"""
    bosehubbardinteraction(address)

Return Σ_i *n_i* (*n_i*-1) for computing the Bose-Hubbard on-site interaction
(without the *U* prefactor.)
"""
function bosehubbardinteraction(address::T) where # T<:Integer
  T<:Union{Integer,BitAdd}
  # compute bosonnumber * (bosonnumber-1) for the Bose Hubbard Hamiltonian
  # currently this ammounts to counting occupation numbers of orbitals
  matrixelementint = 0
  while !iszero(address)
    address >>>= trailing_zeros(address) # proceed to next occupied orbital
    bosonnumber = trailing_ones(address) # count how many bosons inside
    # surpsingly it is faster to not check whether this is nonzero and do the
    # following operations anyway
    address >>>= bosonnumber # remove the countedorbital
    matrixelementint += bosonnumber * (bosonnumber-1)
  end
  return matrixelementint
end #bosehubbardinteraction


bosehubbardinteraction(b::BoseFS) = bosehubbardinteraction(b.bs)
bosehubbardinteraction(adcont::BSAdd64) = bosehubbardinteraction(adcont.add)
bosehubbardinteraction(adcont::BSAdd128) = bosehubbardinteraction(adcont.add)

function bosehubbardinteraction(bsadd::BStringAdd)
  #computes diagonal elementsDIAG_OF_K
  i = 1
  n = 0    #number particles in orbitals
  f = true #flag first zero after a 1
  matrixelementint = 0
  #address is mutable struct, so we have to copy first
  address = copy(bsadd.add)
  while length(address) > 0
    if address[1] == 1
      n += 1
      f = true
    elseif (address[1] == 0 && f == true)
      f = false
      matrixelementint += n * (n-1)
      n = 0
    end #if
    popfirst!(address) # removes first element of address
  end #while
  matrixelementint += n*(n-1)
  return matrixelementint
end # bosehubbardinteraction(bsadd::BStringAdd)

"""
    ebhm(address, m)

Compute the on-site product sum_j n_j(n_j-1) and the next neighbour term
sum_j n_j n_{j+1} with periodic boundary conditions.
"""
function ebhm(address::T, mModes) where T<:Union{Integer,BitAdd}
  # compute the diagonal matrix element of the Extended Bose Hubbard Hamiltonian
  # currently this ammounts to counting occupation numbers of orbitals
  #println("adress= ", bin(address))
  #if periodicboundericondition
  ## only periodic boundary conditions are implemented so far
  bhmmatrixelementint = 0
  ebhmmatrixelementint = 0
  bosonnumber2=0
  #address >>>= trailing_zeros(address) # proceed to next occupied orbital
  bosonnumber1 = trailing_ones(address) # count how many bosons inside
  # surpsingly it is faster to not check whether this is nonzero and do the
  # following operations anyway
  bhmmatrixelementint+= bosonnumber1 * (bosonnumber1-1)
  firstbosonnumber = bosonnumber1 #keap on memory the boson number of the first
  #to do the calculation with the last boson
  address >>>= bosonnumber1 # remove the countedorbital
  address >>>= 1
  for i=1:mModes-1
    #println("i mModes= ",i)
     # proceed to next occupied orbital
    bosonnumber2 = trailing_ones(address) # count how many bosons inside
    # surpsingly it is faster to not check whether this is nonzero and do the
    # following operations anyway
    address >>>= bosonnumber2 # remove the countedorbital
    ebhmmatrixelementint += bosonnumber2 * (bosonnumber1)
    bhmmatrixelementint+= bosonnumber2 * (bosonnumber2-1)
    bosonnumber1=bosonnumber2
    address >>>= 1
  end
  ebhmmatrixelementint+= bosonnumber2 * firstbosonnumber  #periodic bondary condition
  #end
  return ebhmmatrixelementint , bhmmatrixelementint
end #ebhm

ebhm(b::BoseFS, m) = ebhm(b.bs, m)
ebhm(b::BoseFS{N,M,A})  where {N,M,A} = ebhm(b, M)
ebhm(adcont::BSAdd64, m) = ebhm(adcont.add, m)
ebhm(adcont::BSAdd128, m) = ebhm(adcont.add, m)

function ebhm(bsadd::BStringAdd, mModes)
  #computes diagonal elementsDIAG_OF_K
  bosonnumber1 = 0    #number particles in orbitals
  bosonnumber2 = 0
  firstbosonnumber = 0    #keap on memory the boson number of the first site
  #to do the calculation with the last boson
  f = true #flag first zero
  bhmmatrixelementint = 0
  ebhmmatrixelementint = 0
  address = copy(bsadd.add)   #address is mutable struct, so we have to copy first
  while f
    if address[1] == 1  # count how many bosons inside
      firstbosonnumber += 1
    else  # remove the countedorbital and add the diagonal term
      f = false
      bhmmatrixelementint += firstbosonnumber * (firstbosonnumber-1)
    end #if
    popfirst!(address) # removes first element of address
  end
  bosonnumber1 = firstbosonnumber
  Length = length(address)
  for i=1:Length
    if address[1] == 1  # count how many bosons inside
      bosonnumber2 += 1
    else  # remove the countedorbital and add the diagonal term
      bhmmatrixelementint += bosonnumber2 * (bosonnumber2-1)
      ebhmmatrixelementint +=bosonnumber2 * bosonnumber1
      bosonnumber1 = bosonnumber2
      bosonnumber2 = 0
    end #if
    popfirst!(address) # removes first element of address
  end #while
  bhmmatrixelementint += bosonnumber2 * (bosonnumber2-1)  #add the last term (non equal to zero only if the last site is occupied)
  ebhmmatrixelementint += bosonnumber2 * bosonnumber1   #add the last term (non equal to zero only if the last site is occupied)
  ebhmmatrixelementint += bosonnumber2 * firstbosonnumber  #periodic bondary condition
  return ebhmmatrixelementint, bhmmatrixelementint
end # ebhm(bsadd::BStringAdd, ...)

"""
    singlies, doublies = numSandDoccupiedsites(address)
Returns the number of singly and doubly occupied sites for a bosonic bit string address.
"""
function numSandDoccupiedsites(address::T) where T<:Union{Integer,BitAdd}
  # returns number of singly and doubly occupied sites
  singlies = 0
  doublies = 0
  while !iszero(address)
    singlies += 1
    address >>>= trailing_zeros(address)
    occupancy = trailing_ones(address)
    if occupancy > 1
      doublies += 1
    end
    address >>>= occupancy
  end # while address
  return singlies, doublies
end

numSandDoccupiedsites(b::BoseFS) = numSandDoccupiedsites(b.bs)
numSandDoccupiedsites(a::BSAdd64) = numSandDoccupiedsites(a.add)
numSandDoccupiedsites(a::BSAdd128) = numSandDoccupiedsites(a.add)

function numSandDoccupiedsites(onrep::AbstractArray)
  # returns number of singly and doubly occupied sites
  singlies = 0
  doublies = 0
  for n in onrep
    if n > 0
      singlies += 1
      if n > 1
        doublies += 1
      end
    end
  end
  return singlies, doublies
end
# this one is faster by about a factor of 2 if you already have the onrep

function numSandDoccupiedsites(b::BoseFS2C{NA,NB,M,AA,AB}) where {NA,NB,M,AA,AB}
  # returns number of singly and doubly occupied sites
  singlies = 0
  doublies = 0
  c1 = onr(b.bsa)
  c2 = onr(b.bsb)
  for site in 1:M
    if c1[site]+c2[site] > 0
      singlies += 1
      if c1[site]+c2[site] > 1
        doublies += 1
      end
    end
  end
  return singlies, doublies
end
# this one is faster by about a factor of 2 if you already have the onrep

function numberoccupiedsites(address::T) where # T<:Integer
  T<:Union{Integer,BitAdd}
  # returns the number of occupied sites starting from bitstring address
  orbitalnumber = 0
  while !iszero(address)
    orbitalnumber += 1
    address >>>= trailing_zeros(address)
    address >>>= trailing_ones(address)
  end # while address
  return orbitalnumber
end # numberoccupiedsites

numberoccupiedsites(b::BoseFS) = numberoccupiedsites(b.bs)
numberoccupiedsites(a::BSAdd64) = numberoccupiedsites(a.add)
numberoccupiedsites(a::BSAdd128) = numberoccupiedsites(a.add)

function numberoccupiedsites(b::BoseFS2C{NA,NB,M,AA,AB}) where {NA,NB,M,AA,AB}
    c1 = onr(b.bsa)
    c2 = onr(b.bsb)
    occupiedsites = 0
    for site = 1:M
        if !iszero(c1[site]) || !iszero(c2[site])
            occupiedsites += 1
        end
    end
    return occupiedsites
end

function numberoccupiedsites(bsadd::BStringAdd)
  # counts number of occupied orbitals
  i = 1
  n = 0    #number of occupied orbitals
  f = true #flag last bit was zero
  #address is mutable struct, so we have to copy first
  address = copy(bsadd.add)
  while length(address) > 0
    if (address[1] == 1 && f == true)
      n += 1
      f = false
    elseif address[1] == 0
      f = true
    end #if
    popfirst!(address) # removes first element of address
  end #while
  return n
end #function numberoccupiedsites(bsadd::BStringAdd)

function numberlinkedsites(address)
  # return the number of other walker addresses that are linked in the
  # Hamiltonian
  # here implemented for 1D Bose Hubbard
  return 2*numberoccupiedsites(address)
end

"""
    naddress, onproduct = hopnextneighbour(add, chosen, m, n)

Compute the new address of a hopping event for the Bose-Hubbard model.
Returns the new address and the product of occupation numbers of the involved
orbitals.
"""
hopnextneighbour

function hopnextneighbour(address::T, chosen::Int,
  mmodes::Int, nparticles::Int) where T<:Union{Integer,BitAdd}
  # T<:Union{Integer,BStringAdd}
  # compute the address of a hopping event defined by chosen
  # Take care of the type of address
  site = (chosen + 1) >>> 1 # integer divide by 2 to find the orbital to hop from
  taddress = address # make copy for counting
  if isodd(chosen) #last bit; moves a boson to the right in the OCC_NUM_REP
    # move one bit to the left
    orbcount = 1 # count all orbitals; gives number of current one
    occcount = 0 # count occupied orbitals to identify the one that needs work
    bitcount = 0 # count position in bitstring
    while occcount < site
      tzeros = trailing_zeros(taddress)
      bitcount += tzeros
      orbcount += tzeros
      taddress >>>= tzeros
      occcount += 1
      tones = trailing_ones(taddress)
      bitcount += tones
      taddress >>>= tones
      # leaves taddress in a state where orbcount occupied orbitals have been
      # removed; bitcount points to the last one removed
      # 'tones' contains the occupation number of orbital 'site'
    end
    if orbcount == mmodes
      # we are on the last obital, ie. we have to remove the
      # leftmost bit (remove one from tgenerateBHHophat orbital), shift all bits to the right
      # and insert a 1 on the right to add one to the first orbital
      naddress = ( (address ⊻ (T(1) << (bitcount-1))) << 1 ) | T(1)
      tones *= (trailing_ones(address) + 1) # mul occupation num of first obital
    else
      naddress = address ⊻ (T(3) << (bitcount-1)) # shift bit by xor operation
      taddress >>>= 1 # move the next zero out to move the target obital into
      # position
      tones *= (trailing_ones(taddress) + 1) # mul occupation number +1
    end
  else # isodd(chosen): chosen is even (last bit not set); move boson left
    # move one bit to the right
    if site > 1 || iseven(address)
      bitcount = 0
      tzeros = trailing_zeros(taddress)
      bitcount += tzeros
      taddress >>>= tzeros
      orbcount = 1
      tones = 0
      while orbcount < site
        tones = trailing_ones(taddress)
        bitcount += tones
        taddress >>>= tones
        tzeros = trailing_zeros(taddress)
        bitcount += tzeros
        taddress >>>= tzeros
        orbcount += 1
      end
      if tzeros == 1
        # tones now contains occ number of orbital to hop into
        tones = (tones + 1)*trailing_ones(taddress)
      else
        # orbital to hop into is empty
        tones = trailing_ones(taddress)
      end
      naddress = address ⊻ (T(3) << (bitcount-1))
      # shifts a single one to the right
    else # first orbital: we have to right shift the whole lot
      naddress = (address >>> 1) | (T(1) << (nparticles + mmodes - 2))
        #(leading_zeros(zero(T)) - (lzeros = leading_zeros(taddress))-1))
        # adds one on the left after shifting to the right
        #
        # leading_zeros(zero(T)) is the number of bits in the address
        # it looks a bit ugly but should be fast because the compiler
        # can replace it by a number at compile time
      tones = trailing_ones(taddress) * (leading_ones(
        naddress << (leading_zeros(zero(T)) - nparticles - mmodes + 1)))
    end
  end
  return naddress, tones # return new address and product of occupation numbers
end

function hopnextneighbour(address::BoseFS{N,M,A}, chosen::Int,
                          args...) where {N,M,A}
  nbs, tones = hopnextneighbour(address.bs, chosen, M, N)
  return BoseFS{N,M,A}(nbs), tones
end

function hopnextneighbour(address::BSAdd64, chosen, mmodes, nparticles)
  naddress, tones = hopnextneighbour(address.add, chosen, mmodes, nparticles)
  return BSAdd64(naddress), tones
end

function hopnextneighbour(address::BSAdd128, chosen, mmodes, nparticles)
  naddress, tones = hopnextneighbour(address.add, chosen, mmodes, nparticles)
  return BSAdd128(naddress), tones
end

function hopnextneighbour(bsadd::BStringAdd,chosen::Int,dummy1::Int,dummy2::Int)
  # only hops to next neighbours
  site = (chosen+1) >>> 1 #find orbital to hop from
  if iseven(chosen)
    j = -1
  else
    j = 1
  end
  nadd,nchosen,ndest = m1p(bsadd,site,j)
  return nadd,nchosen*ndest #new address and product of occupation numbers
end

"""
    nadd,nchosen,ndest = m1p(bsadd,site,j)

Single particle hop in configuration given by address
from chosen orbital (has to be an occupied orbital!)
to orbital (chosen+j) (out of all M_MODES orbitals, j can be negative);
returns new address and number of particles in orbital i and orbital (i+j) +1.
WITH PERIODIC BOUNDARY CONDITIONS
"""
function m1p(bsadd::BStringAdd,chosen::Int,j::Int)
  #1particle hop in configuration given by address
  #from chosen orbital (has to be an occupied orbital!)
  #to orbital (chosen+j) (out of all M_MODES orbitals, j can be negative)
  # returns new address and number of particles in orbital i and orbital (i+j) +1
  # WITH PERIODIC BOUNDARY CONDITIONS
  nadd = copy(bsadd.add)
  nocc = 0; i = 0 #counts occupied orbitals; index on bit
  fempty = true   # flag for pointing on empty orbitals
  #find the chosen orbital
  while chosen > nocc
     i += 1
     if nadd[i] == 1 && fempty == true
       nocc += 1; fempty = false
     elseif nadd[i] == 0
       fempty = true
     end
  end #while

  # count number of particles in chosen orbital
  L = length(nadd)
  insert!(nadd,L+1,0) #insert extra 0 at end for case that chosen orbital is last orbital in rep

  nchosen = 1; i += 1
  while nadd[i] == 1
    nchosen += 1; i += 1
  end
  deleteat!(nadd,L+1) #remove extra 0
  #delete one particle from chosen orbital
  deleteat!(nadd,i-1)
  L -= 1
  i -= 1; if i == L+1 i = 0 end # i is now pointing on '0' between chosen orbital and chosen+1
                                # or at beginning of address
  if sign(j) == 1 #move forward
    ntot = 1 #counts total number of orbitals
    while ntot < j
      i += 1;
      if i == L+1
        i = 1; ntot += 1
        if j == ntot i = 0;  break end
      end
      if nadd[i] == 0  ntot += 1 end
      #i = rem((i+L), L) + 1 # periodic boundary conditions
    end #while
    i += 1#; if i == (L+1) insert end
    #i = rem((i+L), L) + 1
    #insert particle
    insert!(nadd, i, 1)
    L += 1
    #count particles in orbital
    ndest = 0
    while nadd[i] == 1
      ndest += 1
      i += 1; if i == L+1 break end
    end
  else #move back
    ntot = 0
    if i == 0 i = L+1 end
    while abs(j) > ntot
      i -= 1
      if i == 0
        ntot += 1; i = L+1
        if abs(j) == ntot
          break
        else
          i -= 1
        end
      end
      if nadd[i] == 0  ntot += 1 end
    end #while
    #insert particle
    insert!(nadd, i, 1)
    ndest = 0
    while nadd[i] == 1
      i -= 1; if i == 0 ndest += 1; break end; ndest += 1
    end
  end
  return BStringAdd(nadd), nchosen, ndest
end #m1p



##########################################
#
# Specialising to bosonic model Hamiltonians
#
"""
    TwoComponentBosonicHamiltonian{T} <: AbstractHamiltonian{T}
Abstract type for representing Hamiltonians in a Fock space of fixed number of
scalar bosons. At least the following fields should be present:
* `ha  # number of particles`
* `hb  # number of modes`
* `v  # inter-component interaction`
* `AT::BoseFS2C # address type`

Methods that need to be implemented:
* [`numOfHops(lo::AbstractHamiltonian, address)`](@ref) - number of off-diagonal matrix elements
* [`hop(lo::AbstractHamiltonian, address, chosen::Integer)`](@ref) - access an off-diagonal m.e. by index `chosen`
* [`diagME(lo::AbstractHamiltonian, address)`](@ref) - diagonal matrix element
Optional:
* [`Hamiltonians.LOStructure(::Type{typeof(lo)})`](@ref) - can speed up deterministic calculations if `HermitianLO`

Provides:
* [`hasIntDimension(lo::AbstractHamiltonian)`](@ref)
* [`dimensionLO(lo::AbstractHamiltonian)`](@ref), might fail if linear space too large
* [`fDimensionLO(lo::AbstractHamiltonian)`](@ref)

Provided by [`BosonicHamiltonian`](@ref) to be used on an individual component:
* [`bit_String_Length(ham.ha::BosonicHamiltonian)`](@ref)
* [`nearUniform(ham.ha::BosonicHamiltonian)`](@ref), default version
"""
abstract type TwoComponentBosonicHamiltonian{T} <: AbstractHamiltonian{T} end

# """
#     hasIntDimension(ham)
#
# Return `true` if dimension of the linear operator `ham` can be computed as an
# integer and `false` if not.
#
# If `true`, `dimensionLO(h)` will be successful and return an `Int`. The method
# `fDimensionLO(h)` should be useful in other cases.
# """
# function hasIntDimension(h)
#   try
#     dimensionLO(h)
#     return true
#   catch
#     false
#   end
# end

"""
    dimensionLO(hamiltonian)

Compute dimension of linear operator as integer.
"""
dimensionLO(h::TwoComponentBosonicHamiltonian) =  dimensionLO(h.ha::BosonicHamiltonian)*dimensionLO(h.hb::BosonicHamiltonian)
# formula for boson Hilbert spaces

"""
    fDimensionLO(hamiltonian)

Returns the dimension of Hilbert space as Float64. The exact result is
returned if the value is smaller than 2^53. Otherwise, an improved Stirling formula
is used.
"""
fDimensionLO(h::TwoComponentBosonicHamiltonian) = fDimensionLO(h.ha::BosonicHamiltonian)*fDimensionLO(h.hb::BosonicHamiltonian)

# """
#     nearUniform(ham)
# Create bitstring address with near uniform distribution of particles
# across modes for the Hamiltonian `ham`.
# """
# function BitStringAddresses.nearUniform(h::BosonicHamiltonian)
#     fillingfactor, extras = divrem(h.n, h.m)
#     startonr = fill(fillingfactor,h.m)
#     startonr[1:extras] += ones(Int, extras)
#     return bitaddr(startonr, h.AT)
# end

##########################################


# functor definitions need to be done separately for each concrete type
function (h::TwoComponentBosonicHamiltonian)(s::Symbol)
  if s == :dim # attempt to compute dimension as `Int`
      return hasIntDimension(h) ? dimensionLO(h) : nothing
  elseif s == :fdim
      return fDimensionLO(h) # return dimension as floating point
  end
  return nothing
end


###
### BoseHubbardReal1D2C
###


@with_kw struct BoseHubbardReal1D2C{T, BoseFS2C} <: TwoComponentBosonicHamiltonian{T}
  ha:: BoseHubbardReal1D{T}
  hb:: BoseHubbardReal1D{T}
  v::T = 1.0        # hopping strength
  add::BoseFS2C     # starting address
end

@doc """
    ham = BoseHubbardReal1D2C(ha::BoseHubbardReal1D,hb::BoseHubbardReal1D,v=1.0,add=add)
    ham = BoseHubbardReal1D2C(add::BoseFS2C; ua=1.0,ub=1.0,ta=1.0,tb=1.0,v=1.0)

Implements a two-component one-dimensional Bose Hubbard chain in real space.

```math
\\hat{H} = \\hat{H}_a + \\hat{H}_b + V\\sum_{i} n_{a_i}n_{b_i}
```

# Arguments
- `h_a::BoseHubbardReal1D` and `h_b::BoseHubbardReal1D`: standard Hamiltonian for boson A and B, see [`BoseHubbardReal1D`](@ref)
- `v::Float64`: the inter-species interaction parameter
- `add::BoseFS2C`: the two-component address type, see [`BoseFS2C`](@ref)

    ham(:dim)
Return the dimension of the linear space if representable as `Int`, otherwise
return `nothing`.

    ham(:fdim)
Return the approximate dimension of linear space as `Float64`.
""" BoseHubbardReal1D2C

# set the `LOStructure` trait
LOStructure(::Type{BoseHubbardReal1D2C{T,BoseFS2C}}) where {T <: Real,BoseFS2C} = HermitianLO()

function BoseHubbardReal1D2C(add::BoseFS2C; ua=1.0,ub=1.0,ta=1.0,tb=1.0,v=1.0)
    ha = BoseHubbardReal1D(add.bsa;u=ua,t=ta)
    hb = BoseHubbardReal1D(add.bsb;u=ub,t=tb)
    return BoseHubbardReal1D2C(ha,hb,v,add)
end

# number of excitations that can be made
function numOfHops(ham::BoseHubbardReal1D2C, add)
  return 2*(numberoccupiedsites(add.bsa)+numberoccupiedsites(add.bsb))
end

function bosehubbard2Cinteraction(add::BoseFS2C{NA,NB,M,AA,AB}) where {NA,NB,M,AA,AB}
    c1 = onr(add.bsa)
    c2 = onr(add.bsb)
    interaction = 0::Int
    for site = 1:M
        if !iszero(c2[site])
            interaction += c2[site]*c1[site]
        end
    end
    return interaction
end


"""
    diagME(ham, add)

Compute the diagonal matrix element of the linear operator `ham` at
address `add`.
"""
function diagME(ham::BoseHubbardReal1D2C, address::BoseFS2C)
  return ham.ha.u * bosehubbardinteraction(address.bsa) / 2 + ham.hb.u * bosehubbardinteraction(address.bsb) / 2 + ham.v * bosehubbard2Cinteraction(address)
end

function hop(ham::BoseHubbardReal1D2C, add, chosen::Integer)
    nhops = numOfHops(ham,add)
    nhops_a = 2*numberoccupiedsites(add.bsa)
    if chosen in 1:nhops_a
        naddress_from_bsa, onproduct = hopnextneighbour(add.bsa, chosen, ham.ha.m, ham.ha.n)
        elem = - ham.ha.t*sqrt(onproduct)
        return BoseFS2C(naddress_from_bsa,add.bsb), elem
    elseif chosen in nhops_a+1:nhops
        chosen -= nhops_a
        naddress_from_bsb, onproduct = hopnextneighbour(add.bsb, chosen, ham.hb.m, ham.hb.n)
        elem = - ham.hb.t*sqrt(onproduct)
        return BoseFS2C(add.bsa,naddress_from_bsb), elem
    end
    # return new address and matrix element
end


#
# ###
# ### BoseHubbard2CMom1D
# ###
#
# @with_kw struct BoseHubbard2CMom1D{T, BoseFS2C} <: TwoComponentBosonicHamiltonian{T}
#   na::Int = 6    # number of bosons
#   nb::Int = 6    # number of bosons
#   m::Int = 6    # number of lattice sites
#   ua::T = 1.0    # interaction strength
#   ub::T = 1.0    # interaction strength
#   ta::T = 1.0    # hopping strength
#   tb::T = 1.0    # hopping strength
#   v::T = 1.0    # hopping strength
#   add::BoseFS2C       # starting address
# end
#
# function BoseHubbard2CMom1D(add::BoseFS2C{NA,NB,M,AA,AB}; ua=1.0,ub=1.0,ta=1.0,tb=1.0,v=1.0) where {NA,NB,M,AA,AB}
#     return BoseHubbard2CMom1D(NA,NB,M,ua,ub,ta,tb,v,add)
# end
#
# function hopsWithin1C(m::Int, add)
#   singlies, doublies = numSandDoccupiedsites(add)
#   return singlies*(singlies-1)*(m - 2) + doublies*(m - 1)
#   # number of excitations that can be made
# end
#
# function numOfHops(ham::BoseHubbard2CMom1D, add)
#   sa = numberoccupiedsites(add.bsa)
#   sb = numberoccupiedsites(add.bsb)
#   return hopsWithin1C(ham.m, add.bsa) + hopsWithin1C(ham.m, add.bsb) + sa*(ham.m-1)*sb
#   # number of excitations that can be made
# end
#
# function hop(ham::BoseHubbard2CMom1D, add::BoseFS2C{NA,NB,M,AA,AB}, chosen::Integer) where {NA,NB,M,AA,AB}
#     ham_a = BoseHubbardMom1D(ham.na, ham.m, ham.ua, ham.ta, add.bsa)
#     ham_b = BoseHubbardMom1D(ham.nb, ham.m, ham.ub, ham.tb, add.bsb)
#     nhops_a = numOfHops(ham_a, add.bsa)
#     nhops_b = numOfHops(ham_b, add.bsb)
#     # println("Hops in A: $nhops_a, Hops in B: $nhops_b,")
#     # if chosen > numOfHops(ham,add)
#     #     error("Hop is out of range!")
#     if chosen ≤ nhops_a
#         naddress_from_bsa, elem = hop(ham_a, add.bsa, chosen)
#         # println("Hop in A, chosen = $chosen") # debug
#         return BoseFS2C{NA,NB,M,AA,AB}(naddress_from_bsa,add.bsb), elem
#     elseif nhops_a < chosen ≤ nhops_a+nhops_b
#         chosen -= nhops_a
#         naddress_from_bsb, elem = hop(ham_b, add.bsb, chosen)
#         # println("Hop in B, chosen = $chosen") # debug
#         return BoseFS2C{NA,NB,M,AA,AB}(add.bsa,naddress_from_bsb), elem
#     else
#         chosen -= (nhops_a+nhops_b)
#         sa = numberoccupiedsites(add.bsa)
#         sb = numberoccupiedsites(add.bsb)
#         # println("Hops across A and B: $(sa*(ham.m-1)*sb)")
#         new_bsa, new_bsb, onproduct_a, onproduct_b = hopacross2adds(add.bsa, add.bsb, chosen)
#         new_add = BoseFS2C{NA,NB,M,AA,AB}(new_bsa,new_bsb)
#         # println("Hop A to B, chosen = $chosen") # debug
#         # return new_add, elem
#         elem = ham.v/ham.m*sqrt(onproduct_a)*sqrt(onproduct_b)
#         new_add = BoseFS2C{NA,NB,M,AA,AB}(new_bsa,new_bsb)
#         return new_add, elem
#     end
#     # return new address and matrix element
# end


#
# function diagME(ham::BoseHubbard2CMom1D, add::BoseFS2C)
#     ham_a = BoseHubbardMom1D(ham.na, ham.m, ham.ua, ham.ta, add.bsa)
#     ham_b = BoseHubbardMom1D(ham.nb, ham.m, ham.ub, ham.tb, add.bsb)
#     onrep_a = BitStringAddresses.onr(add.bsa)
#     onrep_b = BitStringAddresses.onr(add.bsb)
#     interaction2c = 0
#     for p in 1:ham.m
#         for k in 1:ham.m
#           interaction2c += onrep_a[k]*onrep_b[p] # b†_p b_p a†_k a_k
#         end
#     end
#     return diagME(ham_a,add.bsa) + diagME(ham_b,add.bsb) + ham.v/ham.m*interaction2c
# end


###
### BoseHubbardMom1D2C
###

@with_kw struct BoseHubbardMom1D2C{T, BoseFS2C} <: TwoComponentBosonicHamiltonian{T}
  ha:: BoseHubbardMom1D{T}
  hb:: BoseHubbardMom1D{T}
  v::T = 1.0        # hopping strength
  add::BoseFS2C     # starting address
end

@doc """
    ham = BoseHubbardMom1D2C(ha::BoseHubbardMom1D,hb::BoseHubbardMom1D,v=1.0,add=add)
    ham = BoseHubbardMom1D2C(add::BoseFS2C; ua=1.0,ub=1.0,ta=1.0,tb=1.0,v=1.0)

Implements a two-component one-dimensional Bose Hubbard chain in momentum space.

```math
\\hat{H} = \\hat{H}_a + \\hat{H}_b + \\frac{V}{M}\\sum_{kpqr} b^†_{r} a^†_{q} b_p a_k δ_{r+q,p+k}
```

# Arguments
- `h_a::BoseHubbardMom1D` and `h_b::BoseHubbardMom1D`: standard Hamiltonian for boson A and B, see [`BoseHubbardMom1D`](@ref)
- `v::Float64`: the inter-species interaction parameter
- `add::BoseFS2C`: the two-component address type, see [`BoseFS2C`](@ref)

    ham(:dim)
Return the dimension of the linear space if representable as `Int`, otherwise
return `nothing`.

    ham(:fdim)
Return the approximate dimension of linear space as `Float64`.
""" BoseHubbardMom1D2C

# set the `LOStructure` trait
LOStructure(::Type{BoseHubbardMom1D2C{T, BoseFS2C}}) where {T <: Real, BoseFS2C} = HermitianLO()

function BoseHubbardMom1D2C(add::BoseFS2C; ua=1.0,ub=1.0,ta=1.0,tb=1.0,v=1.0)
    ha = BoseHubbardMom1D(add.bsa;u=ua,t=ta)
    hb = BoseHubbardMom1D(add.bsb;u=ub,t=tb)
    return BoseHubbardMom1D2C(ha,hb,v,add)
end

function numOfHops(ham::BoseHubbardMom1D2C, add)
  sa = numberoccupiedsites(add.bsa)
  sb = numberoccupiedsites(add.bsb)
  return numOfHops(ham.ha, add.bsa) + numOfHops(ham.hb, add.bsb) + sa*(ham.hb.m-1)*sb
  # number of excitations that can be made
end


function hop(ham::BoseHubbardMom1D2C, add::BoseFS2C{NA,NB,M,AA,AB}, chosen::Integer) where {NA,NB,M,AA,AB}
    # ham_a = BoseHubbardMom1D(ham.na, ham.m, ham.ua, ham.ta, add.bsa)
    # ham_b = BoseHubbardMom1D(ham.nb, ham.m, ham.ub, ham.tb, add.bsb)
    nhops_a = numOfHops(ham.ha, add.bsa)
    nhops_b = numOfHops(ham.hb, add.bsb)
    # println("Hops in A: $nhops_a, Hops in B: $nhops_b,")
    # if chosen > numOfHops(ham,add)
    #     error("Hop is out of range!")
    if chosen ≤ nhops_a
        naddress_from_bsa, elem = hop(ham.ha, add.bsa, chosen)
        # println("Hop in A, chosen = $chosen") # debug
        return BoseFS2C{NA,NB,M,AA,AB}(naddress_from_bsa,add.bsb), elem
    elseif nhops_a < chosen ≤ nhops_a+nhops_b
        chosen -= nhops_a
        naddress_from_bsb, elem = hop(ham.hb, add.bsb, chosen)
        # println("Hop in B, chosen = $chosen") # debug
        return BoseFS2C{NA,NB,M,AA,AB}(add.bsa,naddress_from_bsb), elem
    else
        chosen -= (nhops_a+nhops_b)
        sa = numberoccupiedsites(add.bsa)
        sb = numberoccupiedsites(add.bsb)
        # println("Hops across A and B: $(sa*(ham.m-1)*sb)")
        new_bsa, new_bsb, onproduct_a, onproduct_b = hopacross2adds(add.bsa, add.bsb, chosen)
        new_add = BoseFS2C{NA,NB,M,AA,AB}(new_bsa,new_bsb)
        # println("Hop A to B, chosen = $chosen") # debug
        # return new_add, elem
        elem = ham.v/ham.ha.m*sqrt(onproduct_a)*sqrt(onproduct_b)
        new_add = BoseFS2C{NA,NB,M,AA,AB}(new_bsa,new_bsb)
        return new_add, elem
    end
    # return new address and matrix element
end

# hopacross2adds needed for computing hops across two components
@inline function hopacross2adds(add_a::BoseFS{NA,M,AA}, add_b::BoseFS{NB,M,AB}, chosen::Integer) where {NA,NB,M,AA,AB}
    sa = numberoccupiedsites(add_a)
    sb = numberoccupiedsites(add_b)
    onrep_a = BitStringAddresses.s_onr(add_a)
    onrep_b = BitStringAddresses.s_onr(add_b)
    # b†_s b_q a†_p a_r
    s = p = q = r = 0
    onproduct_a = 1
    onproduct_b = 1
    hole_a, remainder = fldmod1(chosen, (M-1)*sb) # hole_a: position for hole_a
    p, hole_b = fldmod1(remainder, sb) # hole_b: position for hole_b
    # annihilate an A boson:
    for (i, occ) in enumerate(onrep_a)
      if occ > 0
        hole_a -= 1 # searching for the position for hole_a
        if hole_a == 0 # found the hole_a here
          onproduct_a *= occ # record the normalisation factor before annihilate
          onrep_a = @set onrep_a[i] = occ-1 # annihilate an A boson: a_r
          r = i # remember where we make the hole
          break # should break out of the for loop
        end
      end
    end
    if p ≥ r
        p += 1 # to skip the hole_a
    end
    # create an A boson:
    ΔP = p-r # change in momentun
    p = mod1(p, M) # enforce periodic boundary condition
    onrep_a = @set onrep_a[p] += 1 # create an A boson: a†_p
    onproduct_a *= onrep_a[p] # record the normalisation factor after creation
    # annihilate a B boson:
    for (i, occ) in enumerate(onrep_b)
      if occ > 0
        hole_b -= 1 # searching for the position for hole_b
        if hole_b == 0 # found the hole_b here
          onproduct_b *= occ # record the normalisation factor before annihilate
          onrep_b = @set onrep_b[i] = occ-1 # annihilate a B boson: b_q
          q = i # remember where we make the holes
          break # should break out of the for loop
        end
      end
    end
    s = mod1(q-ΔP, M) # compute s with periodic boundary condition
    # create a B boson:
    onrep_b = @set onrep_b[s] += 1 # create a B boson: b†_s
    onproduct_b *= onrep_b[s] # record the normalisation factor after creation
    if mod(q+r,M)-mod(s+p,M) != 0 # sanity check for momentum conservation
        error("Momentum is not conserved!")
    end
    return BoseFS{NA,M,AA}(onrep_a), BoseFS{NB,M,AB}(onrep_b), onproduct_a, onproduct_b
end


function diagME(ham::BoseHubbardMom1D2C, add::BoseFS2C)
    # ham_a = BoseHubbardMom1D(ham.na, ham.m, ham.ua, ham.ta, add.bsa)
    # ham_b = BoseHubbardMom1D(ham.nb, ham.m, ham.ub, ham.tb, add.bsb)
    onrep_a = BitStringAddresses.onr(add.bsa)
    onrep_b = BitStringAddresses.onr(add.bsb)
    interaction2c = 0
    for p in 1:ham.ha.m
        for k in 1:ham.ha.m
          interaction2c += onrep_a[k]*onrep_b[p] # b†_p b_p a†_k a_k
        end
    end
    return diagME(ham.ha,add.bsa) + diagME(ham.hb,add.bsb) + ham.v/ham.ha.m*interaction2c
end



end # module Hamiltonians
