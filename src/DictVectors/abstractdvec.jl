###
### This file contains methods defined on `AbstractDVec`
### The type definition and relevant methods are found in the file "src/Interfaces/dictvectors.jl"
###
function Base.show(io::IO, dvec::AbstractDVec)
    summary(io, dvec)
    limit, _ = displaysize()
    for (i, p) in enumerate(pairs(localpart(dvec)))
        if length(dvec) > i > limit - 4
            print(io, "\n  ⋮   => ⋮")
            break
        else
            print(io, "\n  ")
            show(IOContext(io, :compact => true), p[1])
            print(io, " => ")
            show(IOContext(io, :compact => true), p[2])
        end
    end
end

###
### Types
###
Base.keytype(::Type{<:AbstractDVec{K}}) where {K} = K
Base.keytype(dv::AbstractDVec) = keytype(typeof(dv))
Base.valtype(::Type{<:AbstractDVec{<:Any,V}}) where {V} = V
Base.valtype(dv::AbstractDVec) = valtype(typeof(dv))
Base.eltype(::Type{<:AbstractDVec{K,V}}) where {K,V} = Pair{K,V}
Base.eltype(dv::AbstractDVec) = eltype(typeof(dv))
VectorInterface.scalartype(::Type{<:AbstractDVec{<:Any,V}}) where {V} = V

Base.isreal(v::AbstractDVec) = valtype(v)<:Real
Base.ndims(::AbstractDVec) = 1

###
### copy*, zero*
###
Base.zero(v::AbstractDVec) = empty(v)
VectorInterface.zerovector(v::AbstractDVec, ::Type{T}) where {T<:Number} = similar(v, T)
VectorInterface.zerovector!(v::AbstractDVec) = empty!(v)
VectorInterface.zerovector!!(v::AbstractDVec) = zerovector!(v)

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
### Higher level functions and linear algebra
###
Base.isequal(x::AbstractDVec{K1}, y::AbstractDVec{K2}) where {K1,K2} = false
function Base.isequal(x::AbstractDVec{K}, y::AbstractDVec{K}) where {K}
    x === y && return true
    length(x) != length(y) && return false
    all(pairs(x)) do (k, v)
        isequal(y[k], v)
    end
    return true
end

Base.:(==)(x::AbstractDVec, y::AbstractDVec) = isequal(x, y)

function Base.isapprox(v::AbstractDVec, w::AbstractDVec; kwargs...)
    # Length may be different, but vectors still approximately the same when `atol` is used.
    left = all(pairs(w)) do (key, val)
        isapprox(v[key], val; kwargs...)
    end
    right = all(pairs(v)) do (key, val)
        isapprox(w[key], val; kwargs...)
    end
    return left && right
end

function Base.sum(f, x::AbstractDVec)
    return sum(f, values(x))
end

function LinearAlgebra.mul!(w::AbstractDVec, v::AbstractDVec, α)
    zerovector!(w)
    sizehint!(w, length(v))
    if !iszero(α)
        for (key, val) in pairs(v)
            w[key] = α * val
        end
    end
    return w
end

function VectorInterface.scale!(v::AbstractDVec, α::Number)
    if iszero(α)
        zerovector!(v)
    elseif α ≠ one(α)
        for (key, val) in pairs(v)
            v[key] = α * val
        end
    end
    return v
end

function VectorInterface.scale(v::AbstractDVec, α::Number)
    if α == one(α)
        return copy(v)
    else
        result = zerovector(v, promote_type(typeof(α), scalartype(v)))
        mul!(result, v, α)
        return result
    end
end
VectorInterface.scale!!(v::AbstractDVec, α::Number) = scale!(v, α)

LinearAlgebra.lmul!(α, v::AbstractDVec) = scale!(v, α)
LinearAlgebra.rmul!(v::AbstractDVec, α) = scale!(v, α)

Base.:*(α, x::AbstractDVec) = scale(x, α)
Base.:*(x::AbstractDVec, α) = α * x

"""
    add!(x::AbstractDVec,y::AbstactDVec)

Inplace add `x+y` and store result in `x`.
"""
@inline function VectorInterface.add!(
    v::AbstractDVec{K}, w::AbstractDVec{K}, α::Number=true, β::Number=true
) where {K}
    for (key, val) in pairs(w)
        v[key] = β * v[key] + α * val
    end
    return v
end

function VectorInterface.add(
    v::AbstractDVec{K}, w::AbstractDVec{K}, α::Number=true, β::Number=true
) where {K}
    T = promote_type(scalartype(v), scalartype(w), typeof(α), typeof(β))
    result = scale(v, T(β))
    return add!(result, w, one(T), T(α))
end

function VectorInterface.add!!(
    x::AbstractDVec, y::AbstractDVec, α::Number=true, β::Number=true
)
    add!(x, y, α, β)
end

Base.:+(v::AbstractDVec, w::AbstractDVec) = add(v, w)
Base.:-(v::AbstractDVec, w::AbstractDVec) = add(v, w, -one(scalartype(w)))

# BLAS-like function: y = α*x + y
@inline function LinearAlgebra.axpy!(α, x::AbstractDVec, y::AbstractDVec)
    return add!(y, x, α)
end
# BLAS-like function: y = α*x + β*y
function LinearAlgebra.axpby!(α, x::AbstractDVec, β, y::AbstractDVec)
    return add!(y, x, α, β)
end

function VectorInterface.inner(x::AbstractDVec, y::AbstractDVec)
    # try to save time by looking for the smaller vec
    if isempty(x) || isempty(y)
        return zero(promote_type(valtype(x), valtype(y)))
    elseif length(x) < length(y)
        return sum(pairs(x)) do (key, val)
            conj(val) * y[key]
        end
    else
        return sum(pairs(y)) do (key, val)
            conj(x[key]) * val
        end
    end
end

LinearAlgebra.dot(x::AbstractDVec, y::AbstractDVec) = inner(x, y)

function LinearAlgebra.norm(x::AbstractDVec, p::Real=2)
    T = float(promote_type(valtype(x), typeof(p)))
    if p === 1
        return sum(abs, values(x); init=zero(T))
    elseif p === 2
        return sqrt(sum(abs2, values(x); init=zero(T)))
    elseif p === Inf
        return mapreduce(abs, max, values(x), init=real(zero(T)))
    else
        error("$p-norm of $(typeof(x)) is not implemented.")
    end
end

LinearAlgebra.normalize!(v::AbstractDVec, p::Real=2) = scale!(v, inv(norm(v, p)))
LinearAlgebra.normalize(v::AbstractDVec, p::Real=2) = normalize!(copy(v), p)

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
walkernumber(::StochasticStyle, w) = dot(Norm1ProjectorPPop(), w)
# complex walkers as two populations
# the following default is fast and generic enough to be good for real walkers and

###
### Vector-operator functions
###
function LinearAlgebra.mul!(w::AbstractDVec, h::AbstractHamiltonian, v::AbstractDVec)
    empty!(w)
    for (key, val) in pairs(v)
        w[key] += diagonal_element(h, key)*val
        for (add,elem) in offdiagonals(h, key)
            w[add] += elem*val
        end
    end
    return w
end

function Base.:*(h::AbstractHamiltonian, v::AbstractDVec)
    return mul!(similar(v, promote_type(eltype(h), valtype(v))), h, v)
end

"""
    dot(x, H::AbstractHamiltonian, v)

Evaluate `x⋅H(v)` minimizing memory allocations.
"""
function LinearAlgebra.dot(x::AbstractDVec, LO::AbstractHamiltonian, v::AbstractDVec)
    return dot(LOStructure(LO), x, LO, v)
end

LinearAlgebra.dot(::AdjointUnknown, x, LO::AbstractHamiltonian, v) = dot_from_right(x,LO,v)
# default for LOs without special structure: keep order

function LinearAlgebra.dot(::LOStructure, x, LO::AbstractHamiltonian, v)
    if length(x) < length(v)
        return conj(dot_from_right(v, LO', x)) # turn args around to execute faster
    else
        return dot_from_right(x,LO,v) # original order
    end
end

"""
    dot_from_right(x, LO, v)

Internal function evaluates the 3-argument `dot()` function in order from right
to left.
"""
function dot_from_right(x, op, v::AbstractDVec)
    result = zero(promote_type(valtype(x), eltype(op), valtype(v)))
    for (key, val) in pairs(v)
        result += conj(x[key]) * diagonal_element(op, key) * val
        for (add, elem) in offdiagonals(op, key)
            result += conj(x[add]) * elem * val
        end
    end
    return result
end
