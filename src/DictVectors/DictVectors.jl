"""
Module that provides data structures that behave similar to sparse vectors,
but are indexed by arbitrary types (could be non-integers) similarly to
dictionaries. The idea is to do linear algebra with data structures that are
neither subtyped to `AbstractVector` nor to `AbstractDict` and are suitable
for use with `KrylovKit.jl`. For this, the
abstract type and interface [`AbstractDVec`](@ref) is provided, with the
following concrete subtypes:

 * [`DVec`](@ref)
 * [`FastDVec`](@ref)
"""
module DictVectors

using Random, LinearAlgebra
#using DataStructures

import LinearAlgebra: mul!, dot

import Base: length, iterate, getindex, setindex, setindex!, get, get!, haskey,
             getkey, pop!, isempty, empty, empty!, delete!, sizehint!,
             zero, similar, eltype, ==, isequal, copy, copyto!, *, fill!
# I don't know why we have to import == and isequal from Base. We are not
# defining any new methods, but since removing the subtyping to AbstractDict,
# suddenly we get errors like
# MethodError: no method matching isequal(::Int64, ::Int64)
# You may have intended to import Base.isequal


# include("FastBufs.jl")
using ..FastBufs

export AbstractDVec
export DVec
export FastDVec
export capacity, zero!

# The idea is to do linear algebra with data structures that are not
# subtyped to AbstractVector, much in the spirit of KrylovKit.jl.
# In particular we provide concrete data structures with the aim of being
# suitable for use with KrylovKit. From the manual:

# KrylovKit does not assume that the vectors involved in the problem are actual
# subtypes of AbstractVector. Any Julia object that behaves as a vector is
# supported, so in particular higher-dimensional arrays or any custom user type
# that supports the following functions (with v and w two instances of this type
# and α, β scalars (i.e. Number)):
#
# Base.eltype(v): the scalar type (i.e. <:Number) of the data in v
# Base.similar(v, [T::Type<:Number]): a way to construct additional similar vectors, possibly with a different scalar type T.
# Base.copyto!(w, v): copy the contents of v to a preallocated vector w
# LinearAlgebra.mul!(w, v, α): out of place scalar multiplication; multiply vector v with scalar α and store the result in w
# LinearAlgebra.rmul!(v, α): in-place scalar multiplication of v with α; in particular with α = false, v is initialized with all zeros
# LinearAlgebra.axpy!(α, v, w): store in w the result of α*v + w
# LinearAlgebra.axpby!(α, v, β, w): store in w the result of α*v + β*w
# LinearAlgebra.dot(v,w): compute the inner product of two vectors
# LinearAlgebra.norm(v): compute the 2-norm of a vector

# It turns out, KrylovKit also needs
# *(v, α::Number)
# fill!(v, α)

include("delegate.jl")

include("abstractdvec.jl")
include("dvec.jl")
include("fastdvec.jl")


end # module
