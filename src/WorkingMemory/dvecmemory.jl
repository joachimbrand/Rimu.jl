struct DVecMemory{K,V,D<:AbstractDVec{K,V}} <: AbstractWorkingMemory{K,V}
    dvec::D
end

spawn!(mem::DVecMemory, k, v, _) = mem.dvec[k] += v
DictVectors.capacity(mem::DVecMemory) = capacity(mem.dvec)

#DictVectors.StochasticStyle(::Type{<:DVecMemory{<:Any,<:Any,D}}) where D = StochasticStyle(D)
DictVectors.StochasticStyle(mem::DVecMemory) = StochasticStyle(mem.dvec)

Base.empty!(mem::DVecMemory) = empty!(mem.dvec)

# notes: this is where stats should be produced.
function sort_into_targets!(target, mem::DVecMemory, stats)
    empty!(target)
    for (k, v) in pairs(mem.dvec)
        target[k] = v
    end
    return target, mem, stats
end
