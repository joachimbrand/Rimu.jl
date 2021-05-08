"""
    DVec{K,V,D<:AbstractDict{K,V},S}

Functionally equivalent to [`DVec`](@ref), but with the following changes:

* Any `AbstractDict` can be used for storage. To use this feature construct the `DVec` as
  `DVec(dict, capacity[, style])`.
* Supports a `style` keyword argument that sets the `StochasticStyle`.

## Constructors:

* `DVec(dict::AbstractDict, capacity[, style])`: create a `DVec` with `dict` for storage.
  Note that the data is not copied. Modifying the `DVec` will also modify `dict`.

* `DVec(args...; capacity[, style])`: `args...` are passed to the `Dict` constructor. The
  `Dict` is used for storage.

* `DVec{K,V}(capacity[, style])`: create an empty `DVec{K,V}`.

* `DVec(::AbstractVector{T}[, capacity, style])`: create a `DVec{Int,T}` from an array.
   Capacity defaults to the length of the array

* `DVec(adv::AbstractDVec[, capacity, style])`: create a `DVec` with the same contents as
   `adv`. `capacity` and `style` are inherited from `adv` by default.

In most cases, the default value for `style` is determined based on the resulting `DVec`'s
`eltype`. See also [`default_style`](@ref).
"""
struct DVec{K,V,S,D<:AbstractDict{K,V}} <: AbstractDVec{K,V}
    dict::D
end

# Constructors
function DVec(args...; capacity, style::Union{Nothing,StochasticStyle}=nothing)
    dict = Dict(args...)
    if isnothing(style)
        style = default_style(valtype(dict))
    end
    return DVec(dict, capacity, style)
end
function DVec(
    dict::AbstractDict{K,V}, capacity, style::StochasticStyle=default_style(valtype(dict))
) where {K,V}
    sizehint!(dict, capacity)
    return DVec{K,V,style,typeof(dict)}(dict)
end
function DVec{K,V}(capacity::Int, style::StochasticStyle=default_style(V)) where {K,V}
    return DVec(Dict{K,V}(), capacity, style)
end
function DVec(
    v::AbstractVector, capacity=length(v), style::StochasticStyle=default_style(eltype(v))
) where T
    return DVec(Dict(enumerate(v)), capacity, style)
end
function DVec(
    adv::AbstractDVec{K,V}, capacity=capacity(adv), style=StochasticStyle(adv)
) where {K,V}
    dvec = DVec{K,V}(capacity, style)
    return copyto!(dvec, adv)
end

function Base.empty(dvec::DVec, c=capacity(dvec), style=StochasticStyle(dvec))
    return DVec(empty(dvec.dict), c, style)
end
function Base.empty(dvec::DVec{K}, ::Type{V}) where {K,V}
    return DVec{K,V}(capacity(dvec), StochasticStyle(dvec))
end
function Base.empty(dvec::DVec, ::Type{K}, ::Type{V}) where {K,V}
    return DVec{K,V}(capacity(dvec), StochasticStyle(dvec))
end

Base.similar(dvec::DVec, args...) = empty(dvec, args...)

function Base.summary(io::IO, dvec::DVec{K,V,S}) where {K,V,S}
    cap = capacity(dvec)
    len = length(dvec)
    print(io, "DVec{$K,$V,$S} with $len entries, capacity $cap")
end

StochasticStyle(::DVec{<:Any,<:Any,S}) where {S} = S
capacity(dvec::DVec, args...) = capacity(dvec.dict, args...)

function Base.getindex(dvec::DVec{<:Any,V}, add) where V
    return get(dvec.dict, add, zero(V))
end
function Base.setindex!(dvec::DVec, v, k)
    if iszero(v)
        delete!(dvec, k)
    else
        dvec.dict[k] = v
    end
    return v
end
function Base.setindex!(dvec::DVec, v::AbstractFloat, k)
    if abs(v) ≤ eps(typeof(v))
        delete!(dvec, k)
    else
        dvec.dict[k] = v
    end
    return v
end

Base.pairs(dvec::DVec) = dvec.dict

function LinearAlgebra.rmul!(dvec::DVec, α::Number)
    rmul!(dvec.dict.vals, α)
    return dvec
end

@delegate DVec.dict [get, get!, haskey, getkey, pop!, isempty, length, values, keys]
@delegate_return_parent DVec.dict [delete!, empty!, sizehint!]

# several times faster than regular sum
function Base.sum(f::F, dvec::DVec{<:Any,V,<:Any,<:Dict}) where {F,V}
    if isempty(dvec)
        return f(zero(V))
    else
        vals = dvec.dict.vals
        slots = dvec.dict.slots
        result = f(vals[1] * (slots[1] == 0x1))
        @inbounds @simd for i in 2:length(vals)
            result += f(vals[i] * (slots[i] == 0x1))
        end
        return result
    end
end
