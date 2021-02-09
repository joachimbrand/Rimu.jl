"""
    DVec2{K,V,D<:AbstractDict{K,V},S}

Functionally equivalent to [`DVec`](@ref), but with the following changes:

* Any `AbstractDict` can be used for storage. To use this feature construct the `DVec2` as
  `DVec2(dict, capacity[, style])`.
* Supports a `style` keyword argument that sets the `StochasticStyle`.
"""
struct DVec2{K,V,S,D<:AbstractDict{K,V}} <: AbstractDVec{K,V}
    dict::D
end

# Need default for Complex{Float64}?
default_style(::Type{<:Integer}) = IsStochastic()
default_style(::Type{<:AbstractFloat}) = IsDeterministic()
default_style(::Type{<:Complex}) = IsStochastic2Pop()
default_style(::Type{T}) where T = error("Unable to pick default stochastic style for $T")

# Constructors
function DVec2(args...; capacity, style::Union{Nothing,StochasticStyle}=nothing)
    dict = Dict(args...)
    if isnothing(style)
        style = default_style(valtype(dict))
    end
    return DVec2(dict, capacity, style)
end
function DVec2(
    dict::AbstractDict{K,V}, capacity, style::StochasticStyle=default_style(valtype(dict))
) where {K,V}
    sizehint!(dict, (3 * capacity) ÷ 2 + 1)
    return DVec2{K,V,style,typeof(dict)}(dict)
end
function DVec2{K,V}(capacity::Int, style::StochasticStyle=default_style(V)) where {K,V}
    return DVec2(Dict{K,V}(), capacity, style)
end
function DVec2(
    v::AbstractVector, capacity=length(v), style::StochasticStyle=default_style(eltype(v))
) where T
    return DVec2(Dict(enumerate(v)), capacity, style)
end
function DVec2(
    adv::AbstractDVec{K,V}, capacity=capacity(adv), style=StochasticStyle(adv)
) where {K,V}
    dvec = DVec2{K,V}(capacity, style)
    return copyto!(dvec, adv)
end

function Base.empty(dvec::DVec2, c=capacity(dvec), style=StochasticStyle(dvec))
    return DVec2(empty(dvec.dict), c, style)
end
function Base.empty(dvec::DVec2{K}, ::Type{V}) where {K,V}
    return DVec2{K,V}(capacity(dvec), StochasticStyle(dvec))
end
function Base.empty(dvec::DVec2, ::Type{K}, ::Type{V}) where {K,V}
    return DVec2{K,V}(capacity(dvec), StochasticStyle(dvec))
end

Base.similar(dvec::DVec2, args...) = empty(dvec, args...)

function Base.summary(io::IO, dvec::DVec2{K,V,S}) where {K,V,S}
    cap = capacity(dvec)
    len = length(dvec)
    print(io, "DVec2{$K,$V,$S} with $len entries, capacity $cap")
end

# interface specification and stuff...
StochasticStyle(::Type{<:DVec2{<:Any,<:Any,S}}) where S = S

capacity(dvec::DVec2) = capacity(dvec.dict)

function Base.getindex(dvec::DVec2{<:Any,V}, add) where V
    return get(dvec.dict, add, zero(V))
end
function Base.setindex!(dvec::DVec2, v, k)
    if iszero(v)
        delete!(dvec, k)
    else
        dvec.dict[k] = v
    end
    return v
end
function Base.setindex!(dvec::DVec2, v::AbstractFloat, k)
    if abs(v) ≤ eps(typeof(v))
        delete!(dvec, k)
    else
        dvec.dict[k] = v
    end
    return v
end

Base.pairs(dvec::DVec2) = dvec.dict

function LinearAlgebra.rmul!(dvec::DVec2, α::Number)
    rmul!(dvec.dict.vals, α)
    return dvec
end

@delegate DVec2.dict [
    getindex, get, get!, haskey, getkey, pop!, isempty, length, values, keys
]
@delegate_return_parent DVec2.dict [delete!, empty!, sizehint!]
