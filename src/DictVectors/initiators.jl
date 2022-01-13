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

Base.eltype(::InitiatorValue{V}) where V =  V

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
    InitiatorRule(v::InitiatorDVec)

Abstract type for defining initiator rules for [`InitiatorDVec`](@ref). Returns the
relevant rule when used as a function.

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
    PositivesAreInitiators() <: InitiatorRule

Initiator rule to be passed to [`InitiatorDVec`](@ref). Positive coeffcient values (real or
imaginary) qualify as initiators. Rules:

* Positive coefficients can spawn anywhere.
* Negative coefficients can spawn to positives.

See [`InitiatorRule`](@ref).
"""
struct PositivesAreInitiators <: InitiatorRule{Nothing} end

function value(i::PositivesAreInitiators, v::InitiatorValue)
    val = v.safe + v.initiator
    if  real(val) > 0
        val += real(v.unsafe)
    end
    if imag(val) > 0
        val += im*imag(v.unsafe)
    end
    return val
end

# function value(i::PositivesAreInitiators, v::InitiatorValue{<:Complex})
#     val = value
#     val = v.safe + v.initiator
#     if  real(v.initiator) > 0
#         val += real(v.unsafe)
#     end
#     if  imag(v.initiator) > 0
#         val += im*imag(v.unsafe)
#     end
#     return val
# end

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
  `InitiatorDVec`'s `valtype` (see [`default_style`](@ref)). If a style is given and
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

InitiatorRule(dv::InitiatorDVec) = dv.initiator

###
### Show
###
function Base.summary(io::IO, dvec::InitiatorDVec{K,V}) where {K,V}
    len = length(dvec)
    entries = length(dvec) == 1 ? "entry" : "entries"
    print(io, "InitiatorDVec{$K,$V} with $len $entries, style = $(dvec.style), initiator = $(dvec.initiator)")
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

function Base.get(dvec::InitiatorDVec, args...)
    return value(dvec.initiator, get(dvec.storage, args...))
end
function Base.get!(dvec::InitiatorDVec{<:Any,V}, key, default) where {V}
    return value(dvec.initiator, get!(dvec.storage, key, InitiatorValue{V}(safe=default)))
end
function Base.get!(f::Function, dvec::InitiatorDVec{<:Any,V}, key) where {V}
    return value(dvec.initiator, get!(dvec.storage, key, InitiatorValue{V}(safe=f())))
end

@delegate InitiatorDVec.storage [haskey, getkey, pop!, isempty, length, keys]
@delegate_return_parent InitiatorDVec.storage [delete!, empty!, sizehint!]

"""
    deposit!([r:InitiatorRule], w::InitiatorDVec, add, val, p_add=>p_val)
Add `val` into `w` at address `add` as an [`InitiatorValue`](@ref).

The default behaviour is to deposit a value as an `initiator` if it is a diagonal spawn
from a configuration that qualifies as an "initiator", i.e. exceeds the relevant threshold.
Other diagonal spawns and off-diagonal spwans to initiator parents are recorded as `safe`
(to be accepted by default) whereas off-diagonal spawns to non-initiator parents are
recorded as `unsafe`.

Specific [`InitiatorRule`](@ref)s can intercept and change this behaviour.
"""
function deposit!(w::InitiatorDVec, add, val, (p_add, p_val))
    return deposit!(InitiatorRule(w), w, add, val, (p_add, p_val))
end

function deposit!(::InitiatorRule, w::InitiatorDVec, add, val, (p_add, p_val))
    V = valtype(w)
    i = w.initiator
    old_val = get(w.storage, add, zero(InitiatorValue{V}))
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
    new_val += old_val
    if new_val == InitiatorValue{V}(0, 0, 0)
        delete!(w.storage, add)
    else
        w.storage[add] = new_val
    end
    return w
end


"""
    deposit!(::PositivesAreInitiators, w::InitiatorDVec, add, val, (p_add, p_val))
Deposit as [`InitiatorValue`](@ref) for the [`PositivesAreInitiators`](@ref) rule:

* positive values are stored as `safe`
* negative values are stored as `unsafe`
* this is done separately for `real` and `imag` parts of `Complex` values

The decision about whether or not to accept spawns will be made by [`value()`](@ref).
"""
function deposit!(::PositivesAreInitiators, w::InitiatorDVec, add, val, (p_add, p_val))
    V = valtype(w)
    old_val = get(w.storage, add, zero(InitiatorValue{V}))
    if p_add == add # diagonal deposit (could have negative values) regarded safe
        new_val = InitiatorValue{V}(;safe = val)
    else # offdiagonal spawn
        new_val = InitiatorValue{V}(;
            safe = ifelse(real(val)≥0, real(val), 0),
            unsafe = ifelse(real(val)<0,  real(val), 0)
        )
        if V <: Complex
            new_val += InitiatorValue{V}(;
                safe = ifelse(imag(val)≥0, im*imag(val), 0),
                unsafe = ifelse(imag(val)<0,  im*imag(val), 0)
            )
        end
    end

    new_val += old_val
    if new_val == InitiatorValue{V}(0, 0, 0)
        delete!(w.storage, add)
    else
        w.storage[add] = new_val
    end
    return w
end

function deposit!(w::InitiatorDVec{<:Any,V}, add, val::InitiatorValue{V}, _) where {V}
    dict = storage(w)
    old_val = get(dict, add, zero(valtype(dict)))
    dict[add] = old_val + val
    return w
end

###
### Iterators
###
# These are needed because `Iterators.map` does not infer `eltype` correctly and does not work
# with SplittablesBase.jl.
"""
    InitiatorIterator

Iterator over pairs or values of an `InitiatorDVec`. Supports the `SplittablesBase`
interface.
"""
struct InitiatorIterator{T,D,I}
    iter::D
    initiator::I

    InitiatorIterator{T}(iter::D, initiator::I) where {T,D,I} = new{T,D,I}(iter, initiator)
end
function SplittablesBase.halve(p::InitiatorIterator{T}) where {T}
    left, right = SplittablesBase.halve(p.iter)
    return InitiatorIterator{T}(left, p.initiator), InitiatorIterator{T}(right, p.initiator)
end
SplittablesBase.amount(p::InitiatorIterator) = SplittablesBase.amount(p.iter)

Base.length(p::InitiatorIterator) = length(p.iter)
Base.IteratorSize(::InitiatorIterator) = Base.HasLength()
Base.IteratorEltype(::InitiatorIterator) = Base.HasEltype()
Base.eltype(::InitiatorIterator{T}) where {T} = T

function Base.pairs(dvec::InitiatorDVec{K,V}) where {K,V}
    InitiatorIterator{Pair{K,V}}(pairs(storage(dvec)), dvec.initiator)
end
function Base.iterate(p::InitiatorIterator{<:Pair}, args...)
    it = iterate(p.iter, args...)
    isnothing(it) && return nothing
    (k, v), st = it
    return k => value(p.initiator, v), st
end

function Base.values(dvec::InitiatorDVec{<:Any,V}) where {V}
    InitiatorIterator{V}(values(storage(dvec)), dvec.initiator)
end
function Base.iterate(p::InitiatorIterator, args...)
    it = iterate(p.iter, args...)
    isnothing(it) && return nothing
    v, st = it
    return value(p.initiator, v), st
end
