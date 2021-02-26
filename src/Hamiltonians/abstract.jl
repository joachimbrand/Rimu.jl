# First we have some generic types and methods for any linear operator
# that could be used for FCIQMC

"""
    AbstractHamiltonian{T}
Supertype that provides an interface for linear operators over a linear space with scalar
type `T` that are suitable for FCIQMC. Indexing is done with addresses (typically not integers)
from an address space that may be large (and will not need to be completely generated).

`AbstractHamiltonian` instances operate on vectors of type [`AbstractDVec`](@ref)
from the module `DictVectors` and work well with addresses of type [`AbstractFockAddress`](@ref)
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
    add::A # address; usually a AbstractFockAddress
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

BitStringAddresses.num_particles(h::BosonicHamiltonian) = h.n
BitStringAddresses.num_modes(h::BosonicHamiltonian) = h.m

"""
    bit_String_Length(ham)

Number of bits needed to represent an address for the linear operator `ham`.
"""
bit_String_Length(bh::BosonicHamiltonian) = num_modes(bh) + num_particles(bh) - 1

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
    startonr = fill(fillingfactor, h.m)
    startonr[1:extras] += ones(Int, extras)
    return h.AT(startonr)
end

##########################################################
#
# Specialising to two-component bosonic model Hamiltonians
#
"""
    TwoComponentBosonicHamiltonian{T} <: AbstractHamiltonian{T}
Abstract type for representing interacting two-component Hamiltonians in a Fock space of fixed number of
bosons with two different species. At least the following fields should be present:
* `ha::BosonicHamiltonian  # Hamiltonian for boson species A`
* `hb::BosonicHamiltonian  # Hamiltonian for boson species B`
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
dimensionLO(h::TwoComponentBosonicHamiltonian) =  dimensionLO(h.ha::AbstractHamiltonian)*dimensionLO(h.hb::AbstractHamiltonian)
# formula for boson Hilbert spaces

"""
    fDimensionLO(hamiltonian)

Returns the dimension of Hilbert space as Float64. The exact result is
returned if the value is smaller than 2^53. Otherwise, an improved Stirling formula
is used.
"""
fDimensionLO(h::TwoComponentBosonicHamiltonian) = fDimensionLO(h.ha::AbstractHamiltonian)*fDimensionLO(h.hb::AbstractHamiltonian)

# functor definitions need to be done separately for each concrete type
function (h::TwoComponentBosonicHamiltonian)(s::Symbol)
  if s == :dim # attempt to compute dimension as `Int`
      return hasIntDimension(h) ? dimensionLO(h) : nothing
  elseif s == :fdim
      return fDimensionLO(h) # return dimension as floating point
  end
  return nothing
end
