struct DVecMemory{K,V,D<:AbstractDVec{K,V}} <: AbstractWorkingMemory{K,V}
    dvec::D
end

Base.empty!(mem::DVecMemory) = empty!(mem.dvec)
DictVectors.capacity(mem::DVecMemory) = capacity(mem.dvec)
DictVectors.StochasticStyle(mem::DVecMemory) = StochasticStyle(mem.dvec)

spawn!(mem::DVecMemory, k, v, _) = mem.dvec[k] += v

# notes: this is where stats should be produced.
function sort_into_targets!(target, mem::DVecMemory, stats)
    empty!(target)
    for (k, v) in pairs(mem.dvec)
        target[k] = v
    end
    return target, mem, stats
end
