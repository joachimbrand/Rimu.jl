"""
    InitiatorValue{V}(; safe::V, unsafe::V, initiator::V) where V
Composite "walker" with three fields. For use with [`InitiatorDVec`](@ref)s.
"""
struct InitiatorValue{V}
    safe::V
    unsafe::V
    initiator::V
end
function InitiatorValue{V}(;safe=zero(V), unsafe=zero(V), initiator=zero(V)) where {V}
    return InitiatorValue{V}(V(safe), V(unsafe), V(initiator))
end
function InitiatorValue{V}(i::InitiatorValue) where {V}
    return InitiatorValue{V}(V(i.safe), V(i.unsafe), V(i.initiator))
end

function Base.:+(v::InitiatorValue, w::InitiatorValue)
    return InitiatorValue(v.safe + w.safe, v.unsafe + w.unsafe, v.initiator + w.initiator)
end
function Base.:-(v::InitiatorValue, w::InitiatorValue)
    return InitiatorValue(v.safe - w.safe, v.unsafe - w.unsafe, v.initiator - w.initiator)
end
function Base.:-(v::InitiatorValue{V}) where {V}
    return InitiatorValue{V}(-v.safe, -v.unsafe, -v.initiator)
end
Base.zero(::Union{I,Type{I}}) where {V,I<:InitiatorValue{V}} = InitiatorValue{V}()

Base.convert(::Type{<:InitiatorValue{V}}, x::InitiatorValue{V}) where {V} = x
function Base.convert(::Type{<:InitiatorValue{U}}, x::InitiatorValue{V}) where {U,V}
    return InitiatorValue{U}(safe=x.safe, unsafe=x.unsafe, initiator=x.initiator)
end
function Base.convert(::Type{<:InitiatorValue{V}}, x) where {V}
    return InitiatorValue{V}(safe=convert(V, x))
end

"""
    InitiatorRule{V}

Abstract type for defining initiator rules for [`InitiatorDVec`](@ref).
Concrete implementations:

* [`Initiator`](@ref)
* [`SimpleInitiator`](@ref)
* [`CoherentInitiator`](@ref)

When defining a new `InitiatorRule`, also define a corresponding method for [`value`](@ref)!
"""
abstract type InitiatorRule{V} end

"""
    value(i::InitiatorRule, v::InitiatorValue)
Convert the [`InitiatorValue`](@ref) `v` into a scalar value according to the
[`InitiatorRule`](@ref) `i`.

Internal function that implements functionality of [`InitiatorDVec`](@ref).
"""
value

"""
    Initiator(threshold) <: InitiatorRule

Initiator rule to be passed to [`InitiatorDVec`](@ref). An initiator is a configuration
`add` with a coefficient with magnitude `abs(v[add]) > threshold`. Rules:

* Initiators can spawn anywhere.
* Non-initiators can spawn to initiators.

See [`InitiatorRule`](@ref).
"""
struct Initiator{V} <: InitiatorRule{V}
    threshold::V
end

function value(i::Initiator, v::InitiatorValue)
    return v.safe + v.initiator + !iszero(v.initiator) * v.unsafe
end

"""
    SimpleInitiator(threshold) <: InitiatorRule

Simplified initiator rule to be passed to [`InitiatorDVec`](@ref).
An initiator is a configuration `add` with a coefficient with magnitude
`abs(v[add]) > threshold`. Rules:

* Initiators can spawn anywhere.
* Non-initiators cannot spawn.

See [`InitiatorRule`](@ref).
"""
struct SimpleInitiator{V} <: InitiatorRule{V}
    threshold::V
end

function value(i::SimpleInitiator, v::InitiatorValue)
    return v.safe + v.initiator
end

"""
    CoherentInitiator(threshold) <: InitiatorRule

Initiator rule to be passed to [`InitiatorDVec`](@ref).
An initiator is a configuration `add` with a coefficient with magnitude
`abs(v[add]) > threshold`. Rules:


* Initiators can spawn anywhere.
* Non-initiators can spawn to initiators.
* Multiple non-initiators can spawn to a single non-initiator if their contributions add up
  to a value greater than the initiator threshold.

  See [`InitiatorRule`](@ref).
"""
struct CoherentInitiator{V} <: InitiatorRule{V}
    threshold::V
end

function value(i::CoherentInitiator, v::InitiatorValue)
    if !iszero(v.initiator) || abs(v.unsafe) > i.threshold
        return v.initiator + v.safe + v.unsafe
    else
        return v.initiator + v.safe
    end
end

"""
    InitiatorDVec{K,V} <: AbstractDVec{K,V}

Dictionary-based vector-like data structure for use with [`lomc!`](@ref) and
[`KrylovKit.jl`](https://github.com/Jutho/KrylovKit.jl). See [`AbstractDVec`](@ref).
Functionally identical to [`DVec`](@ref), but contains [`InitiatorValue`](@ref)s internally
in order to facilitate initiator methods. How the initiators are handled is controlled by
the `initiator` keyword argument (see below).

## Constructors

* `InitiatorDVec(dict::AbstractDict[; style, initiator, capacity])`: create an
  `InitiatorDVec` with `dict` for storage.  Note that the data may or may not be copied.

* `InitiatorDVec(args...[; style, initiator, capacity])`: `args...` are passed to the `Dict`
  constructor. The `Dict` is used for storage.

* `InitiatorDVec{K,V}([; style, initiator, capacity])`: create an empty `InitiatorDVec{K,V}`.

* `InitiatorDVec(dv::AbstractDVec[; style, initiator, capacity])`: create an `InitiatorDVec`
   with the same contents as `dv`. The `style` is inherited from `dv` by default.

## Keyword  arguments

* `style`: A valid [`StochasticStyle`](@ref).  The default is selected based on the
  `InitiatorDVec`'s [`valtype`](@ref) (see [`default_style`](@ref)). If a style is given and
  the `valtype` does not match the `style`'s `eltype`, the values are converted to an
  appropriate type.

* `initiator = Initiator(1)`: A valid [`InitiatorRule`](@ref). See [`Initiator`](@ref).

* `capacity`: Indicative size as `Int`. Optional. Sets the initial size of the
  `InitiatorDVec` via [`sizehint!`](@ref).

"""
struct InitiatorDVec{
    K,V,D<:AbstractDict{K,InitiatorValue{V}},S<:StochasticStyle{V},I<:InitiatorRule
} <: AbstractDVec{K,V}
    storage::D
    style::S
    initiator::I
end

###
### Constructors
###
# Vararg
function InitiatorDVec(args...; kwargs...)
    storage = Dict(args...)
    return InitiatorDVec(storage; kwargs...)
end
# Dict with InitiatorValues
function InitiatorDVec(
    dict::AbstractDict{K,InitiatorValue{V}};
    style=default_style(V), initiator=Initiator(one(V)), capacity=0
) where {K,V}
    T = eltype(style)
    if T === V
        storage = dict
        sizehint!(storage, capacity)
    else
        storage = Dict{K,InitiatorValue{T}}()
        sizehint!(storage, capacity)
        for (k, v) in pairs(dict)
            storage[k] = InitiatorValue{T}(v.safe, v.unsafe, v.initiator)
        end
    end
    return InitiatorDVec(storage, style, initiator)
end
# Dict with regular values
function InitiatorDVec(
    dict::AbstractDict{K,V}; style=default_style(V), initiator=Initiator(one(V)), capacity=0
) where {K,V}
    T = eltype(style)
    storage = Dict{K,InitiatorValue{T}}()
    sizehint!(storage, capacity)
    for (k, v) in pairs(dict)
        storage[k] = InitiatorValue{T}(safe=v)
    end
    return InitiatorDVec(storage, style, initiator)
end
# Empty
function InitiatorDVec{K,V}(; kwargs...) where {K,V}
    return InitiatorDVec(Dict{K,InitiatorValue{V}}(); kwargs...)
end
# From another DVec
function InitiatorDVec(
    dv::AbstractDVec{K,V}; style=StochasticStyle(dv), initiator=Initiator(one(V)), capacity=0
) where {K,V}
    return InitiatorDVec(copy(storage(dv)); style, initiator)
end

function Base.empty(dvec::InitiatorDVec{K,V}) where {K,V}
    return InitiatorDVec{K,V}(; style=dvec.style, initiator=dvec.initiator)
end
function Base.empty(dvec::InitiatorDVec{K}, ::Type{V}) where {K,V}
    return InitiatorDVec{K,V}(; initiator=dvec.initiator)
end
function Base.empty(dvec::InitiatorDVec, ::Type{K}, ::Type{V}) where {K,V}
    return InitiatorDVec{K,V}(; initiator=dvec.initiator)
end

###
### Show
###
function Base.summary(io::IO, dvec::InitiatorDVec{K,V}) where {K,V}
    len = length(dvec)
    print(io, "InitiatorDVec{$K,$V} with $len entries, style = $(dvec.style), initiator = $(dvec.initiator)")
end

###
### Interface
###
StochasticStyle(dv::InitiatorDVec) = dv.style
storage(dv::InitiatorDVec) = dv.storage

function Base.getindex(dvec::InitiatorDVec{<:Any,V}, add) where {V}
    return value(dvec.initiator, get(dvec.storage, add, zero(InitiatorValue{V})))
end
function Base.setindex!(dvec::InitiatorDVec{<:Any,V}, v, k) where {V}
    if iszero(v)
        delete!(dvec.storage, k)
    else
        dvec.storage[k] = InitiatorValue{V}(safe=v)
    end
    return v
end

function Base.pairs(dvec::InitiatorDVec)
    return Iterators.map(pairs(dvec.storage)) do ((k, v))
        k => value(dvec.initiator, v)
    end
end
function Base.values(dvec::InitiatorDVec)
    return Iterators.map(values(dvec.storage)) do v
        value(dvec.initiator, v)
    end
end
function Base.get(dvec::InitiatorDVec, args...)
    return value(dvec.initiator, get(dvec.storage, args...))
end
function Base.get!(dvec::InitiatorDVec{<:Any,V}, key, default) where {V}
    return(value(dvec.initiator, get!(dvec.storage, key, InitiatorValue{V}(safe=default))))
end
function Base.get!(f::Function, dvec::InitiatorDVec{<:Any,V}, key) where {V}
    return(value(dvec.initiator, get!(dvec.storage, key, InitiatorValue{V}(safe=f()))))
end

@delegate InitiatorDVec.storage [haskey, getkey, pop!, isempty, length, keys]
@delegate_return_parent InitiatorDVec.storage [delete!, empty!, sizehint!]

"""
    deposit!(w::InitiatorDVec, add, val, p_add=>p_val)
Add `val` into `w` at address `add` as an [`InitiatorValue`](@ref).
"""
function deposit!(w::InitiatorDVec{<:Any,V}, add, val, (p_add, p_val)) where {V}
    i = w.initiator
    if p_add == add # diagonal death
        if abs(p_val) > i.threshold
            new_val = InitiatorValue{V}(initiator=val)
        else
            new_val = InitiatorValue{V}(safe=val)
        end
    else # offdiagonal spawn
        if abs(p_val) > i.threshold
            new_val = InitiatorValue{V}(safe=val)
        else
            new_val = InitiatorValue{V}(unsafe=val)
        end
    end
    w.storage[add] = get(w.storage, add, zero(InitiatorValue{V})) + new_val
end

function deposit!(w::InitiatorDVec{<:Any,V}, add, val::InitiatorValue{V}, _) where {V}
    dict = storage(w)
    prev_val = get(dict, add, zero(valtype(dict)))
    dict[add] = prev_val + val
end
