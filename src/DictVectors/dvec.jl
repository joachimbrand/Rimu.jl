"""
    DVec{K,V,D<:AbstractDict{K,V},S}

Dictionary-based vector-like data structure for use with FCIQMC and
[KrylovKit](https://github.com/Jutho/KrylovKit.jl). While mostly behaving like a `Dict`, it
supports various linear algebra operations such as `norm` and `dot`. It has a
[`StochasticStyle`](@ref) that is used to select an appropriate spawning strategy in the
FCIQMC algorithm.

See also: [`AbstractDVec`](@ref).

# Constructors

* `DVec(dict::AbstractDict[, style, capacity])`: create a `DVec` with `dict` for storage.
  Note that the data may or may not be copied.

* `DVec(args...[; style, capacity])`: `args...` are passed to the `Dict` constructor. The
  `Dict` is used for storage.

* `DVec{K,V}([style, capacity])`: create an empty `DVec{K,V}`.

* `DVec(dv::AbstractDVec[, style, capacity])`: create a `DVec` with the same contents as
   `adv`. The `style` is inherited from `dv` by default.

The default `style` is selected based on the `DVec`'s `valtype` (see
[`default_style`](@ref)). If a style is given and the `valtype` does not match the `style`'s
`eltype`, the values are converted to an appropriate type.

The capacity argument is optional and sets the initial size of the `DVec` via `sizehint!`.

# Examples

```jldoctest
julia> dv = DVec(:a => 1)
DVec{Symbol,Int64,IsStochastic} with 1 entries
  :a => 1

julia> dv = DVec(:a => 2, :b => 3; style=IsDynamicSemistochastic())
DVec{Symbol,Float32,IsDynamicSemistochastic{true}} with 2 entries
  :a => 2.0f0
  :b => 3.0f0
```
"""
struct DVec{K,V,S<:StochasticStyle{V},D<:AbstractDict{K,V}} <: AbstractDVec{K,V}
    dict::D
    style::S
end

###
### Constructors
###
function DVec(args::Vararg{Pair{K,V}}; style=default_style(V), capacity=0) where {K,V}
    dict = Dict{K,V}()
    sizehint!(dict, max(length(args), capacity))
    for (k, v) in args
        dict[k] = convert(eltype(style), v)
    end
    return DVec(dict; style)
end
# In this constructor, the style matches the dict's valtype.
function DVec(
    dict::AbstractDict{K,V}; style::StochasticStyle{V}=default_style(V), capacity=0
) where {K,V}
    capacity > 0 && sizehint!(dict, capacity)
    return DVec{K,V,typeof(style),typeof(dict)}(dict, style)
end
# In this constructor, the dict has to be converted to the appropriate valtype.
function DVec(
    dict::Dict{K}; style::StochasticStyle{V}=default_style(valtype(dict)), capacity=0
) where {K,V}
    dict = convert(Dict{K,V}, dict)
    return DVec{K,V,typeof(style),typeof(dict)}(dict, style)
end
# Constructor from arbitrary iterator
function DVec(itr; style=nothing, capacity=0)
    dict = Dict(itr)
    if isnothing(style)
        return DVec(dict; capacity)
    else
        return DVec(dict; style, capacity)
    end
end
# Empty constructor.
function DVec{K,V}(; style::StochasticStyle=default_style(V), capacity=0) where {K,V}
    return DVec(Dict{K,V}(); style, capacity)
end
# From another DVec
function DVec(dv::AbstractDVec{K,V}, style=StochasticStyle(dv), capacity=0) where {K,V}
    dvec = DVec{K,V}(; style, capacity=max(capacity, length(dv)))
    return copyto!(dvec, dv)
end

function Base.empty(dvec::DVec{K,V}) where {K,V}
    return DVec{K,V}(; style=StochasticStyle(dvec))
end
function Base.empty(dvec::DVec{K}, ::Type{V}) where {K,V}
    return DVec{K,V}()
end
function Base.empty(dvec::DVec, ::Type{K}, ::Type{V}) where {K,V}
    return DVec{K,V}()
end

Base.similar(dvec::DVec, args...; kwargs...) = empty(dvec, args...; kwargs...)

###
### Show
###
function Base.summary(io::IO, dvec::DVec{K,V,S}) where {K,V,S}
    len = length(dvec)
    print(io, "DVec{$K,$V,$S} with $len entries")
end

###
### Interface
###
StochasticStyle(dv::DVec) = dv.style

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

import Base:
    get, get!, haskey, getkey, pop!, isempty, length, values, keys, delete!, empty!, sizehint!
@delegate DVec.dict [get, get!, haskey, getkey, pop!, isempty, length, values, keys]
@delegate_return_parent DVec.dict [delete!, empty!, sizehint!]

# simd sum for Dict
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
