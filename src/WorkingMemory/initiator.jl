"""
    InitiatorValue{V}(; safe::V, unsafe::V, initiator::V) where V
Composite "walker" with three fields. For use with [`InitiatorMemory`](@ref).
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

# rule for combining the fields of `InitiatorValue` to a simple number:
# only add v.unsafe if v.initiator is populated as that indicates an initiator origin
function value(v::InitiatorValue)
    return v.safe + v.unsafe * !iszero(v.initiator) + v.initiator
end

"""
    InitiatorMemory(mem, threshold) <: AbstractWorkingMemory
Working memory for initiator methods with `threshold` and
`valtype(mem)::InitiatorValue`. See [`InitiatorValue`](@ref) and
[`AbstractWorkingMemory`](@ref).
"""
struct InitiatorMemory{
    K,V,M<:AbstractWorkingMemory{K,InitiatorValue{V}}
} <: AbstractWorkingMemory{K,V}
    mem::M
    threshold::V
end

Base.empty!(mem::InitiatorMemory) = empty!(mem.mem)
# pairs() combines `InitiatorValue`s to numbers with `value()`
Base.pairs(mem::InitiatorMemory) = Iterators.map(p -> p[1] => value(p[2]), pairs(mem.mem))
DictVectors.capacity(mem::InitiatorMemory) = capacity(mem.mem)
DictVectors.StochasticStyle(mem::InitiatorMemory) = StochasticStyle(mem.mem)

# spawning rules for `InitiatorMemory`:
# populate val.initiator only for diagonal spawn from initiator (remembers inititor status)
# populate val.unsafe only for off-diagonal spawn from non-initiators
# everything else goes to val.safe
function spawn!(mem::InitiatorMemory, k, v, (pk, pv))
    V = valtype(mem)
    is_initiator = abs(pv) > mem.threshold
    if k == pk # diagonal spawn
        if is_initiator
            val = InitiatorValue{V}(initiator=v)
        else
            val = InitiatorValue{V}(safe=v)
        end
    else # offdiagonal spawn
        if is_initiator
            val = InitiatorValue{V}(safe=v)
        else
            # spawns from non-initiators might be rejected later
            val = InitiatorValue{V}(unsafe=v)
        end
    end
    spawn!(mem.mem, k, val, (pk, pv))
end

# Temporary hack to track number of initiators.
using StaticArrays
function sort_into_targets!(target, mem::InitiatorMemory, stats)
    empty!(target)
    num_initiators = 0
    for (k, v) in pairs(mem.mem)
        num_initiators += !iszero(v.initiator)
        target[k] = value(v)
    end
    stats = setindex(stats, num_initiators, 3)
    return target, mem, stats
end
