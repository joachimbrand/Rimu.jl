"""
    DVec{K,T}(capacity) <: AbstractDVec{K,T}
    DVec(d::Dict [, capacity])
    DVec(v::Vector{T} [, capacity])
Construct a wrapped dictionary with minimum capacity `capacity` to
represent a vector-like object with `eltype(dv) == T`,
which corresponds to the values of the `Dict`. Indexing is done with an
arbitrary (in general non-integer) `keytype(dv) == K`.
When constructed from a `Vector`,
the keys will be integers ∈ `[0, length(v)]`. See [`AbstractDVec`](@ref). The
method [`capacity()`](@ref) is defined but not a strict upper limit as `Dict`
objects can expand.
"""
struct DVec{A,T} <: AbstractDVec{A,T}
    d::Dict{A,T}

    # function DVec(d::Dict{A,T}) where {A,T}
    #     T <: Number || throw(TypeError(:DVec,"construction from `Dict` for the value type",Number,T))
    #     return new{A,T}(d)
    # end
end
# default constructor from `Dict` just wraps the dict, no copying or allocation:
# dv = DVec(Dict(k => v, ...))


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
DVec{K,V}(capacity::Int) where V <: Number where K = DVec(Dict{K,V}(), capacity)

# from Vector
function DVec(v::Vector{T}, capacity = length(v)) where T
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
Base.empty(dv::DVec{K,V}) where {K,V} = DVec{K,V}(capacity(dv))
Base.empty(dv::DVec, ::Type{V}) where {V} = empty(dv,keytype(dv),V)
Base.empty(dv::DVec, ::Type{K}, ::Type{V}) where {K,V} = DVec{K,V}(capacity(dv))
# thus empty() and zero() will conserve the capacity of dv
Base.zero(dv::DVec) = empty(dv)
Base.similar(dv::DVec, ::Type{T}) where {T} = empty(dv, T)
Base.similar(dv::DVec) = empty(dv)

Base.keytype(::Type{DVec{K,T}}) where T where K = K
Base.eltype(::Type{DVec{K,T}}) where T where K = T
# for instances of AbstractDVec, eltype is already defined
# for the type we need to do it here because it has to be specific

capacity(dv::DVec) = (2*length(dv.d.keys))÷3
# 2/3 of the allocated storage size
# if number of elements grows larger, Dict will start rehashing and allocating
# new memory
function capacity(dv::DVec, s::Symbol)
    if s ==:allocated
        return length(dv.d.keys)
    elseif s == :effective
        return capacity(dv)
    else
        ArgumentError("Option symbol $s not recognized")
    end
end




# getindex returns the default value without adding it to dict
Base.getindex(dv::DVec, add) = get(dv.d, add, zero(eltype(dv)))

Base.iterate(dv::DVec) = iterate(dv.d)
Base.iterate(dv::DVec, state) = iterate(dv.d, state)

# most functions are simply delegated to the wrapped dictionary
@delegate DVec.d [ getindex, get, get!, haskey, getkey, pop!,
                              isempty, length ]
# used to do this with iterate, but this leads to type-unstable code

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

# should be much faster than generic version from AbstractDVec
function LinearAlgebra.rmul!(w::DVec, α::Number)
    rmul!(w.d.vals,α)
    return w
end # rmul!

function Base.show(io::IO, da::DVec{K,V}) where V where K
    print(io, "DVec{$K,$V}([")
    init = true
    for (key,val) in da
        if init
            init = false
        else
            print(io, ", ")
        end
        print(io, Pair(key,val))
    end
    print(io, "])")
end
