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
VectorInterface.zerovector!!(v::AbstractDVec{<:Any,T}) where {T<:Number} = zerovector!!(v, T)

function VectorInterface.zerovector!!(v::AbstractDVec, ::Type{T}) where {T<:Number}
    if scalartype(v) ≡ T
        return zerovector!(v)
    else
        return zerovector(v, T)
    end
end

function Base.similar(dvec::AbstractDVec, args...; kwargs...)
    return sizehint!(empty(dvec, args...; kwargs...), length(dvec))
end

"""
    copyto!(w, v)

Copy contents of `v` to `w` without emptying `w`. See [`copy!`](@ref) for a version that
empties `w` first.
"""
@inline function Base.copyto!(w::AbstractDVec, v)
    sizehint!(w, length(v))
    for (key, val) in pairs(v)
        w[key] = val
    end
    return w
end

"""
    copy!(w::AbstractDVec, v)

Empty `w` and copy contents of `v` to it. See also [`copyto!`](@ref)
"""
@inline function Base.copy!(w::AbstractDVec, v)
    empty!(w)
    return copyto!(w, v)
end
Base.copy(v::AbstractDVec) = copyto!(empty(v), v)

###
### Higher level functions and linear algebra
###
Base.isequal(v::AbstractDVec{K1}, w::AbstractDVec{K2}) where {K1,K2} = false
function Base.isequal(v::AbstractDVec{K}, w::AbstractDVec{K}) where {K}
    v === w && return true
    length(v) != length(w) && return false
    return all(pairs(v)) do (key, val)
        w[key] == val
    end
end

Base.:(==)(v::AbstractDVec, w::AbstractDVec) = isequal(v, w)

function Base.isapprox(v::AbstractDVec, w::AbstractDVec; kwargs...)
    # Length may be different, but vectors still approximately the same when `atol` is used.
    left = all(pairs(w)) do (key, val)
        isapprox(v[key], val; kwargs...)
    end
    if left
        return all(pairs(v)) do (key, val)
            isapprox(w[key], val; kwargs...)
        end
    else
        return false
    end
end

function Base.sum(f, v::AbstractDVec)
    return sum(f, values(v))
end

function VectorInterface.scale!(w::AbstractDVec, v::AbstractDVec, α::Number)
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
    T = promote_type(typeof(α), scalartype(v))
    result = zerovector(v, T)
    scale!(result, v, α)
    return result
end
function VectorInterface.scale!!(v::AbstractDVec, α::T) where {T<:Number}
    U = scalartype(v)
    if promote_type(U, T) == U
        return scale!(v, α)
    else
        return scale(v, α)
    end
end
function VectorInterface.scale!!(y::AbstractDVec, v::AbstractDVec, α::T) where {T<:Number}
    Y = scalartype(y)
    U = scalartype(v)
    if promote_type(Y, U, T) == Y && keytype(y) == keytype(v)
        return scale!(y, v, α)
    else
        return scale(v, α)
    end
end

LinearAlgebra.mul!(w::AbstractDVec, v::AbstractDVec, α) = scale!(w, v, α)
LinearAlgebra.lmul!(α, v::AbstractDVec) = scale!(v, α)
LinearAlgebra.rmul!(v::AbstractDVec, α) = scale!(v, α)

Base.:*(α, x::AbstractDVec) = scale(x, α)
Base.:*(x::AbstractDVec, α) = α * x

@inline function VectorInterface.add!(
    w::AbstractDVec{K}, v::AbstractDVec{K}, α::Number=true, β::Number=true
) where {K}
    scale!(w, β)
    for (key, val) in pairs(v)
        w[key] += α * val
    end
    return w
end

function VectorInterface.add(
    w::AbstractDVec{K}, v::AbstractDVec{K}, α::Number=true, β::Number=true
) where {K}
    T = promote_type(scalartype(v), scalartype(w), typeof(α), typeof(β))
    result = scale(w, T(β))
    return add!(result, v, α)
end

function VectorInterface.add!!(
    v::AbstractDVec{K}, w::AbstractDVec{K}, α::Number=true, β::Number=true
) where {K}
    T = promote_type(scalartype(v), scalartype(w), typeof(α), typeof(β))
    if T ≡ scalartype(v)
        return add!(v, w, α, β)
    else
        return add(v, w, α, β)
    end
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

function VectorInterface.inner(v::AbstractDVec, w::AbstractDVec)
    # try to save time by looking for the smaller vec
    if isempty(v) || isempty(w)
        return zero(promote_type(valtype(v), valtype(w)))
    elseif length(v) < length(w)
        return sum(pairs(v)) do (key, val)
            conj(val) * w[key]
        end
    else
        return sum(pairs(w)) do (key, val)
            conj(v[key]) * val
        end
    end
end

LinearAlgebra.dot(v::AbstractDVec, w::AbstractDVec) = inner(v, w)

function LinearAlgebra.norm(v::AbstractDVec, p::Real=2)
    T = float(promote_type(valtype(v), typeof(p)))
    if p === 1
        return sum(abs, values(v); init=zero(T))
    elseif p === 2
        return sqrt(sum(abs2, values(v); init=zero(T)))
    elseif p === Inf
        return mapreduce(abs, max, values(v), init=real(zero(T)))
    else
        error("$p-norm of $(typeof(v)) is not implemented.")
    end
end

LinearAlgebra.normalize!(v::AbstractDVec, p::Real=2) = scale!(v, inv(norm(v, p)))
LinearAlgebra.normalize(v::AbstractDVec, p::Real=2) = normalize!(copy(v), p)

"""
    walkernumber(v)

Compute the number of walkers in `v`. It is used for updating the shift. Overload this
function for modifying population control.

In most cases `walkernumber(v)` is identical to `norm(v, 1)`. For `AbstractDVec`s with
complex coefficients it reports the one norm separately for the real and the imaginary part
as a `ComplexF64`. See [`Norm1ProjectorPPop`](@ref).
"""
walkernumber(v) = walkernumber(StochasticStyle(v), v)
# use StochasticStyle trait for dispatch
walkernumber(::StochasticStyle, v) = dot(Norm1ProjectorPPop(), v)
# complex walkers as two populations
# the following default is fast and generic enough to be good for real walkers and

"""
    walkernumber_and_length(v)

Compute [`walkernumber`](@ref) and `length` at the same time. When MPI is used, this is more
efficient than calling them separately.
"""
walkernumber_and_length(v) = walkernumber(v), length(v)

###
### Vector-operator functions
###
function LinearAlgebra.mul!(w::AbstractDVec, h::AbstractOperator, v::AbstractDVec)
    empty!(w)
    for (key, val) in pairs(v)
        w[key] += diagonal_element(h, key) * val
        for (add, elem) in offdiagonals(h, key)
            w[add] += elem * val
        end
    end
    return w
end

function Base.:*(h::AbstractOperator, v::AbstractDVec)
    T = promote_type(scalartype(h), scalartype(v))
    if eltype(h) ≠ scalartype(h)
        throw(ArgumentError("Operators with non-scalar eltype don't support "*
                            "multiplication with `*`. Use `mul!` or `dot` instead."))
    end
    # first argument in mul! requires IsDeterministic style
    w = empty(v, T; style=IsDeterministic{T}())
    return mul!(w, h, v)
end

# docstring in Interfaces
function LinearAlgebra.dot(w::AbstractDVec, op::AbstractOperator, v::AbstractDVec)
    return dot(LOStructure(op), w, op, v)
end
function LinearAlgebra.dot(::AdjointUnknown, w, op::AbstractOperator, v)
    return dot_from_right(w, op, v)
end
function LinearAlgebra.dot(::LOStructure, w, op::AbstractOperator, v)
    if length(w) < length(v)
        return conj(dot_from_right(v, op', w)) # turn args around to execute faster
    else
        return dot_from_right(w, op, v) # original order
    end
end

# docstring in Interfaces
function Interfaces.dot_from_right(w, op, v::AbstractDVec)
    T = typeof(zero(valtype(w)) * zero(eltype(op)) * zero(valtype(v)))
    result = zero(T)
    for (key, val) in pairs(v)
        result += conj(w[key]) * diagonal_element(op, key) * val
        for (add, elem) in offdiagonals(op, key)
            result += conj(w[add]) * elem * val
        end
    end
    return result
end
