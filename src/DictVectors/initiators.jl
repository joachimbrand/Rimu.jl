"""
    abstract type AbstractInitiatorValue{V}

A value equipped with additional information that enables a variation of the initiator
approximation. To be used with [`PDVec`](@ref), [`InitiatorDVec`](@ref) and
[`InitiatorRule`](@ref)s.

Must define:
* `Base.zero`, `Base.:+`, `Base.:-`, `Base.:*`
* [`from_initiator_value`](@ref) and [`to_initiator_value`](@ref)
"""
abstract type AbstractInitiatorValue{V} end
Base.zero(v::AbstractInitiatorValue) = zero(typeof(v))

Base.convert(::Type{T}, x) where {T<:AbstractInitiatorValue} = T(x)
Base.convert(::Type{T}, x::T) where {T<:AbstractInitiatorValue} = x

"""
    InitiatorValue{V}(; safe::V, unsafe::V, initiator::V) where V
Composite "walker" with three fields. For use with [`InitiatorDVec`](@ref)s.
"""
struct InitiatorValue{V} <: AbstractInitiatorValue{V}
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
function InitiatorValue{V}(x) where {V}
    return InitiatorValue{V}(safe=x)
end

Base.convert(::Type{V}, x::InitiatorValue{V}) where {V} = x.safe + x.unsafe + x.initiator

function Base.:+(v::InitiatorValue, w::InitiatorValue)
    return InitiatorValue(v.safe + w.safe, v.unsafe + w.unsafe, v.initiator + w.initiator)
end
function Base.:*(α, v::InitiatorValue)
    return InitiatorValue(α * v.safe, α * v.unsafe, α * v.initiator)
end
Base.zero(::Type{InitiatorValue{V}}) where {V} = InitiatorValue{V}()


"""
    NonInitiatorValue{V}

Value that does not contain any additional information - used with [`NonInitiator`](@ref),
the default initiator rule for [`PDVec`](@ref).
"""
struct NonInitiatorValue{V} <: AbstractInitiatorValue{V}
    value::V

    NonInitiatorValue{V}(v) where {V} = new{V}(V(v))
    NonInitiatorValue{V}(v::NonInitiatorValue{V}) where {V} = v
end

NonInitiatorValue(v::V) where {V} = NonInitiatorValue{V}(v)

Base.convert(::Type{V}, x::NonInitiatorValue{V}) where {V} = x.value

function Base.:+(v::NonInitiatorValue, w::NonInitiatorValue)
    return NonInitiatorValue(v.value + w.value)
end
function Base.:*(α, v::NonInitiatorValue)
    return NonInitiatorValue(α * v.value)
end
Base.zero(::Type{NonInitiatorValue{V}}) where {V} = NonInitiatorValue(zero(V))

"""
    InitiatorRule{V}

Abstract type for defining initiator rules for [`InitiatorDVec`](@ref).
Concrete implementations:

* [`Initiator`](@ref)
* [`SimpleInitiator`](@ref)
* [`CoherentInitiator`](@ref)
* [`NonInitiator`](@ref)

# Extended Help

`InitiatorRule`s define how to store and retrieve data from associated
[`AbstractInitiatorValue`](@ref)s. When defining a new `InitiatorRule`, also define the
following:

* [`initiator_valtype`](@ref)
* [`from_initiator_value`](@ref)
* [`to_initiator_value`](@ref)

"""
abstract type InitiatorRule end

"""
    initiator_valtype(rule::InitiatorRule, T)

Return the [`AbstractInitiatorValue{T}`](@ref) that is employed by the `rule`.
"""
initiator_valtype

"""
    from_initiator_value(i::InitiatorRule, v::AbstractInitiatorValue)

Convert the [`AbstractInitiatorValue`](@ref) `v` into a scalar value according to the
[`InitiatorRule`](@ref) `i`.
"""
from_initiator_value

"""
    to_initiator_value(::InitiatorRule, k::K, v::V, parent)

Convert `v` to an [`AbstractInitiatorValue`](@ref), taking the initiator rule and the
`parent` that spawned it into account.
"""
to_initiator_value

"""
    Initiator(threshold = 1.0) <: InitiatorRule

Initiator rule to be passed to [`PDVec`](@ref) or [`InitiatorDVec`](@ref). An initiator is a
configuration `add` with a coefficient with magnitude `abs(v[add]) > threshold`. The
`threshold` can be passed as a keyword argument. Rules:

* Initiators can spawn anywhere.
* Non-initiators can spawn to initiators.

See [`InitiatorRule`](@ref).
"""
struct Initiator{T} <: InitiatorRule
    threshold::T
end
Initiator(;threshold=1.0) = Initiator(threshold)
initiator_valtype(::Initiator, ::Type{V}) where {V} = InitiatorValue{V}

function from_initiator_value(::Initiator, v::InitiatorValue)
    return v.safe + v.initiator + !iszero(v.initiator) * v.unsafe
end

function _default_to_initiator_value(rule::InitiatorRule, add, val::V, parent) where {V}
    p_add, p_val = parent
    if p_add == add
        if abs(p_val) > rule.threshold
            return InitiatorValue(zero(V), zero(V), V(val))
        else
            return InitiatorValue(V(val), zero(V), zero(V))
        end
    else
        if abs(p_val) > rule.threshold
            return InitiatorValue(V(val), zero(V), zero(V))
        else
            return InitiatorValue(zero(V), V(val), zero(V))
        end
    end
end

function to_initiator_value(rule::Initiator, add, val, parent)
    _default_to_initiator_value(rule, add, val, parent)
end

"""
    SimpleInitiator(threshold = 1.0) <: InitiatorRule

Initiator rule to be passed to [`PDVec`](@ref) or [`InitiatorDVec`](@ref). An initiator is
a configuration `add` with a coefficient with magnitude `abs(v[add]) > threshold`. The
`threshold` can be passed as a keyword argument. Rules:

* Initiators can spawn anywhere.
* Non-initiators cannot spawn.

See [`InitiatorRule`](@ref).
"""
struct SimpleInitiator{T} <: InitiatorRule
    threshold::T
end
SimpleInitiator(;threshold=1.0) = SimpleInitiator(threshold)
initiator_valtype(::SimpleInitiator, ::Type{V}) where {V} = InitiatorValue{V}

function from_initiator_value(i::SimpleInitiator, v::InitiatorValue)
    return v.safe + v.initiator
end
function to_initiator_value(rule::SimpleInitiator, add, val, parent)
    _default_to_initiator_value(rule, add, val, parent)
end

"""
    CoherentInitiator(threshold = 1.0) <: InitiatorRule

Initiator rule to be passed to [`PDVec`](@ref) or [`InitiatorDVec`](@ref). An initiator is
a configuration `add` with a coefficient with magnitude `abs(v[add]) > threshold`. The
`threshold` can be passed as a keyword argument. Rules:

* Initiators can spawn anywhere.
* Non-initiators can spawn to initiators.
* Multiple non-initiators can spawn to a single non-initiator if their contributions add up
  to a value greater than the initiator threshold.

See [`InitiatorRule`](@ref).
"""
struct CoherentInitiator{T} <: InitiatorRule
    threshold::T
end
CoherentInitiator(;threshold=1.0) = CoherentInitiator(threshold)
initiator_valtype(::CoherentInitiator, ::Type{V}) where {V} = InitiatorValue{V}

function from_initiator_value(i::CoherentInitiator, v::InitiatorValue)
    if !iszero(v.initiator) || abs(v.unsafe) > i.threshold
        return v.initiator + v.safe + v.unsafe
    else
        return v.initiator + v.safe
    end
end
function to_initiator_value(rule::CoherentInitiator, add, val, parent)
    _default_to_initiator_value(rule, add, val, parent)
end

"""
    NonInitiator() <: InitiatorRule

Initiator rule that disables the approximation. This is the default setting for
[`PDVec`](@ref).

See [`InitiatorRule`](@ref).
"""
struct NonInitiator <: InitiatorRule end
initiator_valtype(::NonInitiator, ::Type{V}) where {V} = NonInitiatorValue{V}

function to_initiator_value(::NonInitiator, _, val, _)
    return NonInitiatorValue(val)
end

function from_initiator_value(::NonInitiator, v::NonInitiatorValue)
    return v.value
end
