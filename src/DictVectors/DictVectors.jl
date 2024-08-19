"""
Module that provides concrete implementations of the [`AbstractDVec`](@ref) interface.

- [`DVec`](@ref): basic [`AbstractDVec`](@ref)
- [`PDVec`](@ref): parallel [`AbstractDVec`](@ref) with MPI and initiator support
- [`InitiatorDVec`](@ref): allows storing information about initiator status

See [`Interfaces`](@ref).
"""
module DictVectors

using Folds: Folds
using LinearAlgebra: LinearAlgebra, I, dot, ⋅, mul!, normalize!, rank
using Random: Random
using StaticArrays: SVector
using VectorInterface: VectorInterface, add, add!, inner, norm, scalartype,
    scale, scale!, zerovector, zerovector!, zerovector!!

import MPI

using ..Interfaces: Interfaces, AbstractDVec, AdjointUnknown,
    CompressionStrategy, IsDiagonal, LOStructure,
    apply_column!, apply_operator!, compress!,
    diagonal_element, offdiagonals, step_stats, AbstractHamiltonian, AbstractOperator,
    dot_from_right
using ..StochasticStyles: StochasticStyles, IsDeterministic

import ..Interfaces: deposit!, storage, StochasticStyle, default_style, freeze, localpart,
    working_memory

export deposit!, storage, walkernumber, walkernumber_and_length, dot_from_right
export DVec, InitiatorDVec, PDVec, PDWorkingMemory

export InitiatorRule, Initiator, SimpleInitiator, NonInitiator, CoherentInitiator

export AbstractProjector, NormProjector, Norm2Projector, UniformProjector, Norm1ProjectorPPop


# The idea is to do linear algebra with data structures that are not
# subtyped to AbstractVector, much in the spirit of VectorInterfaces.jl and KrylovKit.jl.
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
include("projectors.jl")

include("initiators.jl")
include("communicators.jl")

include("dvec.jl")
include("initiatordvec.jl")

include("pdvec.jl")
include("pdworkingmemory.jl")

end # module
