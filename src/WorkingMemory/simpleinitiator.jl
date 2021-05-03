"""
    SimpleInitiatorMemory(mem, threshold) <: AbstractWorkingMemory
Working memory for a simplified initiator method with `threshold`. See also
[`InitiatorMemory`](@ref) and [`AbstractWorkingMemory`](@ref).
"""
struct SimpleInitiatorMemory{
    K,V,M<:AbstractWorkingMemory{K,SimpleInitiatorValue{V}}
} <: AbstractWorkingMemory{K,V}
    mem::M
    threshold::V
end

Base.empty!(mem::SimpleInitiatorMemory) = empty!(mem.mem)
Base.pairs(mem::SimpleInitiatorMemory) = pairs(mem.mem)
DictVectors.capacity(mem::SimpleInitiatorMemory) = capacity(mem.mem)
DictVectors.StochasticStyle(mem::SimpleInitiatorMemory) = StochasticStyle(mem.mem)

# spawning rules for `SimpleInitiatorMemory`:
# Only allow diagonal spawns and offdiagonal spawns from initiators.
function spawn!(mem::SimpleInitiatorMemory, k, v, (pk, pv))
    if k == pk || abs(pv) > mem.threshold
        spawn!(mem.mem, k, v, (pk, pv))
    end
end
