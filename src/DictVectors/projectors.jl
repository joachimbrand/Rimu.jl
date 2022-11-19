# Define this type union for local (non-MPI) data
const DVecOrVec = Union{AbstractDVec,AbstractVector}

"""
Abstract supertype for projectors to be used in in lieu of DVecs or Vectors in `dot`
products. Implemented subtypes:

- [`UniformProjector`](@ref)
- [`NormProjector`](@ref)
- [`Norm2Projector`](@ref)
- [`Norm1ProjectorPPop`](@ref)

See also [`PostStepStrategy`](@ref Main.PostStepStrategy) for use of projectors in [`lomc!`](@ref Main.lomc!).

## Interface

Define a method for `LinearAlgebra.dot(projector, v)`.
"""
abstract type AbstractProjector end

"""
    UniformProjector() <: AbstractProjector
Represents a vector with all elements 1. To be used with [`dot()`](@ref).
Minimizes memory allocations.

```julia
UniformProjector()⋅v == sum(v)
dot(UniformProjector(), LO, v) == sum(LO*v)
```

See also [`PostStepStrategy`](@ref Main.PostStepStrategy), and [`AbstractProjector`](@ref) for use
of projectors in [`lomc!`](@ref Main.lomc!).
"""
struct UniformProjector <: AbstractProjector end

LinearAlgebra.dot(::UniformProjector, y::DVecOrVec) = sum(values(y))
Base.getindex(::UniformProjector, add) = 1

function LinearAlgebra.dot(::UniformProjector, op::AbstractHamiltonian, v::AbstractDVec)
    return sum(pairs(v)) do (key, val)
        diag = diagonal_element(op, key) * val
        offdiag = sum(offdiagonals(op, key)) do (add, elem)
            elem * val
        end
        diag + offdiag
    end
end

"""
    NormProjector() <: AbstractProjector
Results in computing the one-norm when used in `dot()`. E.g.
```julia
dot(NormProjector(),x)
-> norm(x,1)
```
`NormProjector()` thus represents the vector `sign.(x)`.

See also [`PostStepStrategy`](@ref Main.PostStepStrategy), and [`AbstractProjector`](@ref) for use
of projectors in [`lomc!`](@ref Main.lomc!).
"""
struct NormProjector <: AbstractProjector end

LinearAlgebra.dot(::NormProjector, y::DVecOrVec) = norm(y, 1)

"""
    Norm2Projector() <: AbstractProjector
Results in computing the two-norm when used in `dot()`. E.g.
```julia
dot(NormProjector(),x)
-> norm(x,2) # with type Float64
```

See also [`PostStepStrategy`](@ref Main.PostStepStrategy), and [`AbstractProjector`](@ref) for use
of projectors in [`lomc!`](@ref Main.lomc!).
"""
struct Norm2Projector <: AbstractProjector end

LinearAlgebra.dot(::Norm2Projector, y::DVecOrVec) = norm(y, 2)
# NOTE that this returns a `Float64` opposite to the convention for
# dot to return the promote_type of the arguments.

"""
    Norm1ProjectorPPop() <: AbstractProjector
Results in computing the one-norm per population when used in `dot()`. E.g.
```julia
dot(Norm1ProjectorPPop(),x)
-> norm(real.(x),1) + im*norm(imag.(x),1)
```

See also [`PostStepStrategy`](@ref Main.PostStepStrategy), and [`AbstractProjector`](@ref) for use
of projectors in [`lomc!`](@ref Main.lomc!).
"""
struct Norm1ProjectorPPop <: AbstractProjector end

function LinearAlgebra.dot(::Norm1ProjectorPPop, y::DVecOrVec)
    T = float(valtype(y))
    if T <: Complex
        return T(sum(values(y)) do p
            abs(real(p)) + im*abs(imag(p))
        end)
    else
        return dot(NormProjector(), y)
    end
end

# NOTE that this returns a `Float64` opposite to the convention for
# dot to return the promote_type of the arguments.
# NOTE: This operation should work for `MPIData` and is MPI synchronizing

"""
    PopsProjector() <: AbstractProjector
Results in computing the projection of one population on the other
when used in `dot()`. E.g.
```julia
dot(PopsProjector(),x)
-> real(x) ⋅ imag(x)
```

See also [`PostStepStrategy`](@ref Main.PostStepStrategy), and [`AbstractProjector`](@ref) for use
of projectors in [`lomc!`](@ref Main.lomc!).
"""
struct PopsProjector <: AbstractProjector end

function LinearAlgebra.dot(::PopsProjector, y::DVecOrVec)
    T = float(real(valtype(y)))
    return T(sum(values(y)) do p
        real(p) * imag(p)
    end)
end

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

freeze(dv) = FrozenDVec(collect(pairs(localpart(dv))))

freeze(p::AbstractProjector) = p

function LinearAlgebra.dot(fd::FrozenDVec, dv::AbstractDVec)
    result = zero(promote_type(valtype(fd), valtype(dv)))
    for (k, v) in pairs(fd)
        result += dv[k] ⋅ v
    end
    return result
end
LinearAlgebra.dot(dv::AbstractDVec, fd::FrozenDVec) = conj(dot(fd, dv))
