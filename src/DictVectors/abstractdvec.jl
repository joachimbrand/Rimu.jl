"""
    DictVectors.AbstractDVec{K,V}

Abstract type for sparse vectors with `valtype` `V` based on dictionary-like structures. The
vectors are designed to work well with FCIQMC and
[KrylovKit](https://github.com/Jutho/KrylovKit.jl).

They lie somewhere between `AbstractDict`s and sparse `AbstractVector`s, generally behaving
like a dictionary, while supportting various linear algebra functionality. Indexing with a
value not stored in the dictionary returns `zero(V)`. Setting a stored value to 0 or below
`eps(V::AbstractFloat)` removes the value from the dictionary. Their `length` signals the
number of stored elements, not the size of the vector space.

They have a [`StochasticStyle`](@ref) which selects the spawning algorithm in `FCIQMC`.

To iterate over an `AbstractDVec`, use `pairs` or `values`.

# Interface

The interface is similar to the `AbstractDict` interface.

Implement what would be needed for the `AbstractDict` interface (`pairs`, `keys`, `values`,
`setindex!, getindex, delete!, length, haskey, empty!, isempty`) and, in addition:
* [`StochasticStyle`](@ref)
* [`storage(dv)`](@ref) returns an `AbstractDict` storing the raw data with possibly
  different `valtype` than `V`.
"""
abstract type AbstractDVec{K,V} end

function Base.show(io::IO, dvec::AbstractDVec)
    summary(io, dvec)
    limit, _ = displaysize()
    for (i, p) in enumerate(pairs(dvec))
        if length(dvec) > i > limit - 4
            print(io, "\n  ⋮   => ⋮")
            break
        else
            print(io, "\n  ", p)
        end
    end
end

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
    storage(dvec) -> AbstractDict

Return the raw storage associated with `dvec` as an `AbstractDict`. Used in MPI
communication.
"""
storage

###
### Types
###
Base.keytype(::Type{<:AbstractDVec{K}}) where {K} = K
Base.keytype(dv::AbstractDVec) = keytype(typeof(dv))
Base.valtype(::Type{<:AbstractDVec{<:Any,V}}) where {V} = V
Base.valtype(dv::AbstractDVec) = valtype(typeof(dv))
Base.eltype(::Type{<:AbstractDVec{K,V}}) where {K,V} = Pair{K,V}
Base.eltype(dv::AbstractDVec) = eltype(typeof(dv))

Base.isreal(v::AbstractDVec) = valtype(v)<:Real
Base.ndims(::AbstractDVec) = 1

###
### copy*, zero*
###
"""
    zero!(v)

Replace `v` by a zero vector as an inplace operation. For `AbstractDVec` types it means
removing all non-zero elements. For `AbstractArrays`, it sets all of the values to zero.
"""
zero!(v::AbstractDVec) = empty!(v)
zero!(v::AbstractVector{T}) where {T} = v .= zero(T)

Base.zero(dv::AbstractDVec) = empty(dv)

function Base.similar(dvec::AbstractDVec, args...; kwargs...)
    return sizehint!(empty(dvec, args...; kwargs...), length(dvec))
end

@inline function Base.copyto!(w::AbstractDVec, v)
    sizehint!(w, length(v))
    for (key, val) in pairs(v)
        w[key] = val
    end
    return w
end
@inline function Base.copy!(w::AbstractDVec, v)
    empty!(w)
    return copyto!(w, v)
end
Base.copy(v::AbstractDVec) = copyto!(empty(v), v)

###
### Linear algebra
###
function Base.sum(f, x::AbstractDVec)
    return sum(f, values(x))
end

function LinearAlgebra.norm(x::AbstractDVec, p::Real=2)
    if p === 1
        return float(sum(abs, values(x)))
    elseif p === 2
        return sqrt(sum(abs2, values(x)))
    elseif p === Inf
        return float(mapreduce(abs, max, values(x), init=real(zero(valtype(x)))))
    else
        error("$p-norm of $(typeof(x)) is not implemented.")
    end
end

@inline function LinearAlgebra.mul!(w::AbstractDVec, v::AbstractDVec, α)
    empty!(w)
    sizehint!(w, length(v))
    for (key, val) in pairs(v)
        w[key] = val * α
    end
    return w
end

# copying multiplication with scalar
function Base.:*(α::T, x::AbstractDVec{<:Any,V}) where {T,V}
    return mul!(similar(x, promote_type(T, V)), x, α)
end
Base.:*(x::AbstractDVec, α) = α * x

"""
    add!(x::AbstractDVec,y::AbstactDVec)

Inplace add `x+y` and store result in `x`.
"""
@inline function add!(x::AbstractDVec{K}, y::AbstractDVec{K}) where {K}
    for (k, v) in pairs(y)
        x[k] += v
    end
    return x
end
add!(x::AbstractVector, y) = x .+= values(y)

@inline function LinearAlgebra.axpy!(α, x::AbstractDVec, y::AbstractDVec)
    for (k, v) in pairs(x)
        y[k] += α * v
    end
    return y
end

function LinearAlgebra.rmul!(x::AbstractDVec, α)
    for (k, v) in pairs(x)
        x[k] = v * α
    end
    return x
end

# BLAS-like function: y = α*x + β*y
function LinearAlgebra.axpby!(α, x::AbstractDVec, β, y::AbstractDVec)
    rmul!(y, β) # multiply every non-zero element
    axpy!(α, x, y)
    return y
end

function LinearAlgebra.dot(x::AbstractDVec, y::AbstractDVec)
    # For mixed value types
    result = zero(promote_type(valtype(x), valtype(y)))
    if length(x) < length(y) # try to save time by looking for the smaller vec
        for (key, val) in pairs(x)
            result += conj(val) * y[key]
        end
    else
        for (key, val) in pairs(y)
            result += conj(x[key]) * val
        end
    end
    return result # the type is promote_type(T1,T2) - could be complex!
end
# For MPI version see mpi_helpers.jl

# threaded dot()
function LinearAlgebra.dot(x::AbstractDVec{K,T1}, ys::NTuple{N, AbstractDVec{K,T2}}) where {N, K, T1, T2}
    results = zeros(promote_type(T1,T2), N)
    Threads.@threads for i in 1:N
        results[i] = x⋅ys[i]
    end
    return sum(results)
end

Base.isequal(x::AbstractDVec{K1}, y::AbstractDVec{K2}) where {K1,K2} = false
function Base.isequal(x::AbstractDVec{K}, y::AbstractDVec{K}) where {K}
    x === y && return true
    length(x) != length(y) && return false
    for (k, v) in pairs(x)
        !isequal(y[k], v) && return false
    end
    return true
end

Base.:(==)(x::AbstractDVec, y::AbstractDVec) = isequal(x, y)

# Define this type union for local (non-MPI) data
const DVecOrVec = Union{AbstractDVec,AbstractVector}

"""
Abstract supertype for projectors to be used in in lieu of DVecs or Vectors.
"""
abstract type AbstractProjector end

"""
    UniformProjector()
Represents a vector with all elements 1. To be used with [`dot()`](@ref).
Minimizes memory allocations.

```julia
UniformProjector()⋅v == sum(v)
dot(UniformProjector(), LO, v) == sum(LO*v)
```

See also [`ReportingStrategy`](@ref) for use
of projectors in FCIQMC.
"""
struct UniformProjector <: AbstractProjector end

LinearAlgebra.dot(::UniformProjector, y::DVecOrVec) = sum(values(y))
# a specialised fast and non-allocating method for
# `dot(::UniformProjector, A::AbstractHamiltonian, y)` is defined in `Hamiltonians.jl`

"""
    NormProjector()
Results in computing the one-norm when used in `dot()`. E.g.
```julia
dot(NormProjector(),x)
-> norm(x,1)
```
`NormProjector()` thus represents the vector `sign.(x)`.

See also [`ReportingStrategy`](@ref) for use
of projectors in FCIQMC.
"""
struct NormProjector <: AbstractProjector end

LinearAlgebra.dot(::NormProjector, y::DVecOrVec) = convert(valtype(y), norm(y,1))
# dot returns the promote_type of the arguments.
# NOTE that this can be different from the return type of norm()->Float64
# NOTE: This operation should work for `MPIData` and is MPI synchronizing

"""
    Norm2Projector()
Results in computing the two-norm when used in `dot()`. E.g.
```julia
dot(NormProjector(),x)
-> norm(x,2) # with type Float64
```

See also [`ReportingStrategy`](@ref) for use
of projectors in FCIQMC.
"""
struct Norm2Projector <: AbstractProjector end

LinearAlgebra.dot(::Norm2Projector, y::DVecOrVec) = norm(y, 2)
# NOTE that this returns a `Float64` opposite to the convention for
# dot to return the promote_type of the arguments.

"""
    Norm1ProjectorPPop()
Results in computing the one-norm per population when used in `dot()`. E.g.
```julia
dot(Norm1ProjectorPPop(),x)
-> norm(real.(x),1) + im*norm(imag.(x),1)
```

See also [`ReportingStrategy`](@ref) for use
of projectors in FCIQMC.
"""
struct Norm1ProjectorPPop <: AbstractProjector end

function LinearAlgebra.dot(::Norm1ProjectorPPop, y::DVecOrVec)
    T = float(valtype(y))
    if T <: Complex
        return T(sum(values(y)) do p
            abs(real(p)) + im*abs(imag(p))
        end)
    else
        return T(sum(abs, values(y)))
    end
end

# NOTE that this returns a `Float64` opposite to the convention for
# dot to return the promote_type of the arguments.
# NOTE: This operation should work for `MPIData` and is MPI synchronizing

"""
    PopsProjector()
Results in computing the projection of one population on the other
when used in `dot()`. E.g.
```julia
dot(PopsProjector(),x)
-> real(x) ⋅ imag(x)
```

See also [`ReportingStrategy`](@ref) for use
of projectors in FCIQMC.
"""
struct PopsProjector <: AbstractProjector end

function LinearAlgebra.dot(::PopsProjector, y::DVecOrVec)
    T = float(real(valtype(y)))
    return T(sum(values(y)) do p
        real(p) * imag(p)
    end)
end

"""
    walkernumber(w)

Compute the number of walkers in `w`. It is used for updating the shift. Overload this
function for modifying population control.

In most cases `walkernumber(w)` is identical to `norm(w,1)`. For `AbstractDVec`s with
complex coefficients it reports the one norm separately for the real and the imaginary part
as a `ComplexF64`. See [`Norm1ProjectorPPop`](@ref).
"""
walkernumber(w) = walkernumber(StochasticStyle(w), w)
# use StochasticStyle trait for dispatch
walkernumber(::StochasticStyle, w) = Norm1ProjectorPPop() ⋅ w
# complex walkers as two populations
# the following default is fast and generic enough to be good for real walkers and

"""
    localpart(dv) -> AbstractDVec

Get the part of `dv` that is located on this MPI rank. Returns `dv` itself for `DictVector`s.
"""
localpart(dv) = dv # default for local data

"""
    FrozenDVec

See: [`freeze`](@ref).
"""
struct FrozenDVec{K,V}
    pairs::Vector{Pair{K,V}}
end
Base.keytype(::FrozenDVec{K}) where {K} = K
Base.valtype(::FrozenDVec{<:Any,V}) where {V} = V
Base.eltype(::FrozenDVec{K,V}) where {K,V} = Pair{K,V}
Base.pairs(fd::FrozenDVec) = fd.pairs

"""
    freeze(dv)

Create a "frozen" version of `dv` which can no longer be modified or used in the
conventional manner, but supports faster dot products.
"""
freeze(dv) = FrozenDVec(collect(pairs(localpart(dv))))

function LinearAlgebra.dot(fd::FrozenDVec, dv)
    result = zero(promote_type(valtype(fd), valtype(dv)))
    for (k, v) in pairs(fd)
        result += dv[k] ⋅ v
    end
    return result
end
