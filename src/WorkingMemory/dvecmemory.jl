"""
    DVecMemory(dvec::AbstractDVec) <: AbstractWorkingMemory
Wraps an `AbstractDVec` for use as working memory. See [`AbstractWorkingMemory`](@ref).
"""
struct DVecMemory{K,V,D<:AbstractDVec{K,V}} <: AbstractWorkingMemory{K,V}
    dvec::D
end

Base.empty!(mem::DVecMemory) = empty!(mem.dvec)
Base.pairs(mem::DVecMemory) = pairs(mem.dvec)
DictVectors.capacity(mem::DVecMemory) = capacity(mem.dvec)
DictVectors.StochasticStyle(mem::DVecMemory) = StochasticStyle(mem.dvec)

spawn!(mem::DVecMemory, k, v, _) = mem.dvec[k] += v
