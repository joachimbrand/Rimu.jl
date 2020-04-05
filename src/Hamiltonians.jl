"""
This module defines Hamiltonian types and standard methods.
Model Hamiltonians should be subtyped to [`LinearOperator`](@ref).
Models implemented so far are:

* [`BoseHubbardReal1D`](@ref) Bose-Hubbard chain, real space
* [`ExtendedBHReal1D`](@ref) extended Bose-Hubbard model with on-site and nearest neighbour interactions, real space, one dimension
"""
module Hamiltonians

using Parameters, StaticArrays, LinearAlgebra

import Base: *
import LinearAlgebra: mul!, dot

using ..DictVectors
using ..BitStringAddresses
using ..ConsistentRNG

export LinearOperator, Hops, generateRandHop
export diagME, numOfHops, hop, hasIntDimension, dimensionLO, fDimensionLO

export BosonicHamiltonian, bit_String_Length
export BoseHubbardReal1D, ExtendedBHReal1D

# First we have some generic types and methods for any linear operator
# that could be used for FCIQMC

"""
    LinearOperator{T}
Supertype that provides and interface for linear operators over `T` that are
suitable for FCIQMC. Indexing is done with addresses from a linear space that
may be large (and will not need to be completely generated).

Provides:
* [`Hops`](@ref): iterator over reachable off-diagonal matrix elements
* [`generateRandHop`](@ref): function to generate random off-diagonal matrix element
* `hamiltonian[address1, address2]`: indexing with `getindex()` - mostly for testing purposes
* `*(LO, v)` deterministic matrix-vector multiply (`== LO(v)`)
* `mul!(w, LO, v)` mutating matrix-vector multiply
* [`dot(x, LO, v)`](@ref) compute `x⋅(LO*v)` minimizing allocations

Methods that need to be implemented:
* [`numOfHops(lo::LinearOperator, address)`](@ref)
* [`hop(lo::LinearOperator, address, chosen::Integer)`](@ref)
* [`diagME(lo::LinearOperator, address)`](@ref)
* [`hasIntDimension(lo::LinearOperator)`](@ref)
* [`dimensionLO(lo::LinearOperator)`](@ref), if applicable
* [`fDimensionLO(lo::LinearOperator)`](@ref)
Optional:
* [`Hamiltonians.LOStructure(::Type{typeof(lo)})`](@ref)
"""
abstract type LinearOperator{T} end

Base.eltype(::LinearOperator{T}) where {T} = T

function *(h::LinearOperator{E}, v::AbstractDVec{K,V}) where {E, K, V}
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
# function LinearAlgebra.mul!(w::AbstractDVec, h::LinearOperator, v::AbstractDVec, α, β)
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
function LinearAlgebra.mul!(w::AbstractDVec, h::LinearOperator, v::AbstractDVec)
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
    Hamiltonians.LOStructure(op::LinearOperator)
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
LOStructure(op::LinearOperator) = LOStructure(typeof(op))
LOStructure(::Type{T}) where T <: LinearOperator = ComplexLO()


"""
    dot(x, LO::LinearOperator, v)
Evaluate `x⋅LO(v)` minimizing memory allocations.
"""
function LinearAlgebra.dot(x::AbstractDVec, LO::LinearOperator, v::AbstractDVec)
  return dot_w_trait(LOStructure(LO), x, LO, v)
end
# specialised method for UniformProjector
function LinearAlgebra.dot(::UniformProjector, LO::LinearOperator{T}, v::AbstractDVec{K,T2}) where {K, T, T2}
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
    Hamiltonians.dot_w_trait(::LOStructure, x, LO::LinearOperator, v)
Internal function for making use of the `LinearOperator` trait `LOStructure`.
"""
dot_w_trait(::LOStructure, x, LO::LinearOperator, v) = dot_from_right(x,LO,v)
# default for LOs without special structure: keep order

function dot_w_trait(::HermitianLO, x, LO::LinearOperator, v)
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
function dot_from_right(x::AbstractDVec{K,T1}, LO::LinearOperator{T}, v::AbstractDVec{K,T2}) where {K, T,T1, T2}
    # function LinearAlgebra.dot(x::AbstractDVec{K,T1}, LO::LinearOperator{T}, v::AbstractDVec{K,T2}) where {K, T,T1, T2}
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
struct Hops{T,A,O}  <: AbstractVector{T}
    h::O # Hamiltonian
    add::A # address; usually a BitStringAddressType
    num::Int # number of possible hops

    # inner constructor
    function Hops(ham::O, add::A) where {O,A}
        T = eltype(ham)
        return new{T,A,O}(ham, add, numOfHops(ham, add))
    end
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
function generateRandHop(ham::LinearOperator, add)
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

function Base.getindex(ham::LinearOperator{T}, address1, address2) where T
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

##########################################
#
# Specialising to bosonic model Hamiltonians
#
"""
    BosonicHamiltonian{T} <: LinearOperator{T}
Abstract type for representing Hamiltonians in a Fock space of fixed number of
scalar bosons. At least the following fields should be present:
* `n  # number of particles`
* `m  # number of modes`
* `AT # address type`

Methods that need to be implemented:
* [`numOfHops(lo::LinearOperator, address)`](@ref) - number of off-diagonal matrix elements
* [`hop(lo::LinearOperator, address, chosen::Integer)`](@ref) - access an off-diagonal m.e. by index `chosen`
* [`diagME(lo::LinearOperator, address)`](@ref) - diagonal matrix element
Optional:
* [`Hamiltonians.LOStructure(::Type{typeof(lo)})`](@ref) - can speed up deterministic calculations if `HermitianLO`

Provides:
* [`hasIntDimension(lo::LinearOperator)`](@ref)
* [`dimensionLO(lo::LinearOperator)`](@ref), might fail if linear space too large
* [`fDimensionLO(lo::LinearOperator)`](@ref)
* [`bit_String_Length`](@ref)
* [`nearUniform`](@ref), default version
"""
abstract type BosonicHamiltonian{T} <: LinearOperator{T} end

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
function hasIntDimension(h::BosonicHamiltonian)
  try
    binomial(h.n + h.m - 1, h.n)# formula for boson Hilbert spaces
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

"""
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
"""
@with_kw struct BoseHubbardReal1D{T} <: BosonicHamiltonian{T}
  n::Int = 6    # number of bosons
  m::Int = 6    # number of lattice sites
  u::T = 1.0    # interaction strength
  t::T = 1.0    # hopping strength
  AT::Type = BSAdd64 # address type
end

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

"""
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
"""
@with_kw struct ExtendedBHReal1D{T} <: BosonicHamiltonian{T}
  n::Int = 6    # number of bosons
  m::Int = 6    # number of lattice sites
  u::T = 1.0    # on-site interaction strength
  v::T = 1.0    # on-site interaction strength
  t::T = 1.0    # hopping strength
  AT::Type = BSAdd64 # address type
end

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

# function bosehubbardinteraction(a::BitAdd)
#   return mapreduce(bosehubbardinteraction,+,a.chunks)
# end
function bitshiftright!(v::MVector{I,UInt64}, n::Integer) where I
  if I==1
    @inbounds v[1] >>>= n
    return v
  elseif n ≥ I*64
    return fill!(v, zero(UInt64))
  end
  d, r = divrem(n,64) # shift by `d` chunks and `r` bits
  mask = ~0 >>> (64-r) # 2^r-1 # 0b0...01...1 with `r` 1s
  for i in I : -1 : d+2 #1 : I-d-1
    @inbounds v[i] = (v[i-d] >>> r) | ((v[i-d-1] & mask) << (64-r))
  end
  @inbounds v[d+1] = v[1] >>> r # no carryover for leftmost chunk
  for i in 1 : d
    @inbounds v[i] = zero(UInt64)
  end
  return v
end

# function bitshiftright(v::SVector{I,UInt64}, n::Integer) where I
#   if I==1
#     @inbounds return v[1] >>> n
#   elseif n ≥ I*64
#     return zero(SVector{I,UInt64})
#   end
#   d, r = divrem(n,64) # shift by `d` chunks and `r` bits
#   mask = ~0 >>> (64-r) # 2^r-1 # 0b0...01...1 with `r` 1s
#   p1 = (zero(UInt64) for i in 1:d)
#   @inbounds p2 = v[1] >>> r # no carryover for leftmost chunk
#   p3 = ((v[i-d] >>> r) | ((v[i-d-1] & mask) << (64-r)) for i in I : -1 : d+2)
#   return SVector(p1...,p2,p3...)
# end
@inline function bitshiftright(v::SVector{I,UInt64}, n::Integer) where I
  if I==1
    @inbounds return v[1] >>> n
  end
  d, r = divrem(n,64) # shift by `d` chunks and `r` bits
  mask = ~0 >>> (64-r) # 2^r-1 # 0b0...01...1 with `r` 1s
  return ((zero(UInt64) for i in 1:d)..., (v[1] >>> r),
    ((v[i-d] >>> r) | ((v[i-d-1] & mask) << (64-r)) for i in I : -1 : d+2)...)
end


function Base.trailing_ones(a::MVector)
  t = 0
  for chunk in reverse(a)
    s = trailing_ones(chunk)
    t += s
    s < 64 && break
  end
  return t # min(t, B) # assume no ghost bits
end

function Base.trailing_zeros(a::MVector)
  t = 0
  for chunk in reverse(a)
    s = trailing_zeros(chunk)
    t += s
    s < 64 && break
  end
  return t
end

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

# function numberoccupiedsites(address::BitAdd)
#   return mapreduce(numberoccupiedsites,+,address.chunks)
# end

numberoccupiedsites(b::BoseFS) = numberoccupiedsites(b.bs)

numberoccupiedsites(a::BSAdd64) = numberoccupiedsites(a.add)

numberoccupiedsites(a::BSAdd128) = numberoccupiedsites(a.add)

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

end # module Hamiltonians
