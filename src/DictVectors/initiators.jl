"""
    InitiatorValue{V}(; safe::V, unsafe::V, initiator::V) where V
Composite "walker" with three fields. For use with [`Initiator`](@ref).
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
Base.zero(::Union{V,Type{InitiatorValue{V}}}) where {V} = InitiatorValue{V}()

###
### Styles
###
abstract type AbstractInitiator{V} <: StochasticStyle{InitiatorValue{V}} end

struct Initiator{V,S<:StochasticStyle{V}} <: AbstractInitiator{V}
    style::S
    threshold::V
end

value(::Initiator, v::InitiatorValue) = v.safe + v.initiator + !iszero(v.initiator) * v.unsafe

struct SimpleInitiator{V,S<:StochasticStyle{V}} <: AbstractInitiator{V}
    style::S
    threshold::V
end

value(::SimpleInitiator, v::InitiatorValue) = v.safe + v.initiator

struct CoherentInitiator{V,S<:StochasticStyle{V}} <: AbstractInitiator{V}
    style::S
    threshold::V
end

function value(i::SimpleInitiator, v::InitiatorValue)
    if !iszero(v.initiator) || abs(v.unsafe) > i.threshold
        return v.initiator + v.safe + v.unsafe
    else
        return v.initiator + v.safe
    end
end
