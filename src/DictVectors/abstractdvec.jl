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

Base.isreal(v::AbstractDVec) = valtype(v)<:Real
Base.ndims(::AbstractDVec) = 1

###
### copy*, zero*
###
zero!(v::AbstractDVec) = empty!(v)

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
    empty!(w)
    sizehint!(w, length(v))
    for (key, val) in pairs(v)
        w[key] = val * α
    end
    return w
end
function LinearAlgebra.rmul!(x::AbstractDVec, α)
    for (k, v) in pairs(x)
        x[k] = v * α
    end
    return x
end
function LinearAlgebra.lmul!(α, x::AbstractDVec)
    for (k, v) in pairs(x)
        x[k] = α * v
    end
    return x
end

function Base.:*(α::T, x::AbstractDVec{<:Any,V}) where {T,V}
    result = similar(x, promote_type(T, V))
    if !iszero(α)
        mul!(result, x, α)
    end
    return result
end
Base.:*(x::AbstractDVec, α) = α * x

"""
    add!(x::AbstractDVec,y::AbstactDVec)

Inplace add `x+y` and store result in `x`.
"""
@inline function add!(x::AbstractDVec{K}, y::AbstractDVec{K}, α=true) where {K}
    for (k, v) in pairs(y)
        x[k] += α * v
    end
    return x
end
add!(x::AbstractVector, y) = x .+= values(y)

function add!(d::Dict, s, α=true)
    for (key, s_value) in s
        d_value = get(d, key, zero(valtype(d)))
        new_value = d_value + α * s_value
        if iszero(new_value)
            delete!(d, key)
        else
            d[key] = new_value
        end
    end
    return d
end

function Base.:+(v::AbstractDVec, w::AbstractDVec)
    result = similar(v, promote_type(valtype(v), valtype(w)))
    copy!(result, v)
    add!(result, w)
    return result
end
function Base.:-(v::AbstractDVec, w::AbstractDVec)
    result = similar(v, promote_type(valtype(v), valtype(w)))
    copy!(result, v)
    axpy!(-one(valtype(result)), w, result)
    return result
end

# BLAS-like function: y = α*x + y
@inline function LinearAlgebra.axpy!(α, x::AbstractDVec, y::AbstractDVec)
    return add!(y, x, α)
end
# BLAS-like function: y = α*x + β*y
function LinearAlgebra.axpby!(α, x::AbstractDVec, β, y::AbstractDVec)
    lmul!(β, y)
    axpy!(α, x, y)
    return y
end

function LinearAlgebra.dot(x::AbstractDVec, y::AbstractDVec)
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

function LinearAlgebra.normalize!(v::AbstractDVec, p=2)
    n = norm(v, p)
    rmul!(v, inv(n))
    return v
end
function LinearAlgebra.normalize(v::AbstractDVec, p=2)
    res = copy(v)
    return normalize!(res, p)
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
walkernumber(::StochasticStyle, w) = dot(Norm1ProjectorPPop(), w)
# complex walkers as two populations
# the following default is fast and generic enough to be good for real walkers and

###
### Vector-operator functions
###
function Base.:*(h::AbstractHamiltonian, v::AbstractDVec)
    return mul!(similar(v, promote_type(eltype(h), valtype(v))), h, v)
end

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
