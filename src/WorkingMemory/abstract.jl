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
spawn!(dv::AbstractDVec, k, v, _) = dv[k] += v

# notes: this is where stats should be produced.
function sort_into_targets!(target, mem::AbstractWorkingMemory, stats)
    empty!(target)
    for (k, v) in pairs(mem)
        target[k] = v
    end
    return target, mem, stats
end

#=
Idea:

The following functions are used in place of wm[key] += val:

offdiagonal_spawn!(::AbstractWorkingMemory{K,V}, ::K, ::V, ::Pair{K,V})
diagonal_spawn!(::AbstractWorkingMemory{K,V}, ::K, ::V)

The working memory type determines how it stores the values. Projections are done here.

Possible implementations:

DVecWorkingMemory - what we have right now
VectorWorkingMemory - probably useful with MPI - less shuffling around
MPIWorkingMemory (?)

InitiatorWokringMemory{K,V,M<:AbstractWorkingMemory{K,Tuple{V,V,V}}} - new initiator idea
ThreadedWorkingMemory{K,V,N,M<:AbstractWorkingMemory{K,V}} - in place of the NTuple we use now

Question marks:

* Perhaps StochasticStyles should be set by setting a keyword argument - no longer a part of
  DVec.
=#
