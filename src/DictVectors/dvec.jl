"""
    DVec{K,V}(capacity) <: AbstractDVec{K,V}
    DVec(key => value; capacity)
    DVec(args...; capacity)
    DVec(d::Dict [, capacity])
    DVec(v::Vector{V} [, capacity])
Dictionary-based vector-like data structure with minimum capacity `capacity`
for storing values with keys.
The type of the values is `eltype(dv) == V`.
Indexing is done with an
arbitrary (in general non-integer) key with `keytype(dv) == K`.
If the keyword argument `capacity` is passed then args are parsed as for `Dict`.
When constructed from a `Vector`,
the keys will be integers ∈ `[1, length(v)]`. See [`AbstractDVec`](@ref). The
method [`capacity()`](@ref) is defined but not a strict upper limit as `Dict`
objects can expand.
"""
struct DVec{A,T} <: AbstractDVec{A,T}
    d::Dict{A,T}
end
# default constructor from `Dict` just wraps the dict, no copying or allocation:
# dv = DVec(Dict(k => v, ...))

# constructor like Dict with mandatory keyword capacity
DVec(args...; capacity) = DVec(Dict(args...), capacity)

function DVec(dict::D, capacity::Int) where D <: Dict
    if capacity*3 ≥ length(dict.keys)*2
        # requested capacity ≥ 2/3 of allocated memory to avoid rehashing
        # be triggered by `setindex!`
        sizehint!(dict, (capacity*3)>>1+1)
        # ensures memory allocation > 3/2 rquested capacity
    end # does nothing if dict is large enough (shrinking is not implemented)
    return DVec(dict)
end

# by specifying keytype, eltype, and capacity
DVec{K,V}(capacity::Int) where V where K = DVec(Dict{K,V}(), capacity)

# from Vector
function DVec(v::AbstractVector{T}, capacity = length(v)) where T
    indices = 1:length(v) # will be keys of dictionary
    ps = map(tuple,indices,v) # create iterator over pairs
    return DVec(Dict(ps), capacity)
end

# from AbstractDict; note that a new dict is constructed and data is copied
function DVec(d::AbstractDict{K,V},
              capacity = length(d)) where K where V
    dv = DVec{K,V}(capacity)
    for (k,v) in d
        dv[k] = v
    end
    return dv
end

# from AbstractDVec
function DVec(adv::AbstractDVec{K,V}, capacity = capacity(adv)) where K where V
    dv = DVec{K,V}(capacity) # allocate new DVec
    return copyto!(dv,adv) # generic for AbstractDVec
end

# the following also create and allocated new DVec objects
Base.empty(dv::DVec{K,V}, c::Integer = capacity(dv) ) where {K,V} = DVec{K,V}(c)
Base.empty(dv::DVec, ::Type{V}) where {V} = empty(dv,keytype(dv),V)
Base.empty(dv::DVec, ::Type{K}, ::Type{V}) where {K,V} = DVec{K,V}(capacity(dv))
Base.similar(dv::DVec, args...) = empty(dv, args...)

function Base.summary(io::IO, dvec::DVec{K,V}) where {K,V}
    cap = capacity(dvec)
    len = length(dvec)
    print(io, "DVec{$K,$V} with $len entries, capacity $cap")
end

capacity(dv::DVec, args...) = capacity(dv.d, args...)

# getindex returns the default value without adding it to dict
function Base.getindex(dv::DVec{K,V}, add) where {K,V}
    get(dv.d, add, zero(V))
end
function Base.getindex(dv::DVec{K,Tuple{V,F}}, add) where {K,V<:Number,F}
    get(dv.d, add, tuple(zero(V),zero(F)))
end

# iterator over pairs
Base.pairs(dv::DVec) = dv.d # just return the contained dictionary

# most functions are simply delegated to the wrapped dictionary
@delegate DVec.d [get, get!, haskey, getkey, pop!, isempty, length, values, keys]

# Some functions are delegated, but then need to return the main dictionary
# NOTE: push! is not included below, because the fallback version just
#       calls setindex!
@delegate_return_parent DVec.d [ delete!, empty!, sizehint! ]

function Base.setindex!(dv::DVec{K,V}, v::V, key::K) where K where V
    if v == zero(V)
        delete!(dv, key)
    else
        setindex!(dv.d, v, key)
    end
    return dv
end

function Base.setindex!(dv::DVec{K,V}, v::V, key::K) where {K, V<:AbstractFloat}
    if abs(v) ≤ eps(V)
        delete!(dv, key)
    else
        setindex!(dv.d, v, key)
    end
    return dv
end

# should be much faster than generic version from AbstractDVec
function LinearAlgebra.rmul!(w::DVec, α::Number)
    rmul!(w.d.vals,α)
    return w
end # rmul!
