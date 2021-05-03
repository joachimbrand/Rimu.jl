"""
    AbstractWorkingMemory
Iteration over `mem::AbstractWorkingMemory` produces pairs of `address` and `value`.
To be used with
* [`spawn!(::AbstractWorkingMemory, address, value, parent)`](@ref)
"""
abstract type AbstractWorkingMemory{K,V} end

Base.valtype(::AbstractWorkingMemory{<:Any,V}) where {V} = V
Base.keytype(::AbstractWorkingMemory{K}) where {K} = K

# Why is this method here in this file?
"""
    spawn!(mem, new_add, new_val, add => val)
Add `new_val` to `mem` at address `new_add`, given `add => val` as information where the
spawn originated.
"""
spawn!(dv::AbstractDVec, k, v, _) = dv[k] += v

# notes: this is where stats should be produced.
function sort_into_targets!(target, mem::AbstractWorkingMemory, stats)
    empty!(target)
    for (k, v) in pairs(mem)
        target[k] = v
    end
    return target, mem, stats
end
