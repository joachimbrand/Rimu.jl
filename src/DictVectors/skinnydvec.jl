"""
    SkinnyDVec(keys, vals; style) <: AbstractDVec
Implementation of an [`AbstractDVec`](@ref) using lists implemented as `Vector`s for
internal storage. Doesn't comply with the full [`AbstractDVec`](@ref) interface.

Supports fast iteration. Adding elements with `setindex!` just appends them to the list.
"""
struct SkinnyDVec{K,V,S<:StochasticStyle{V}} <: AbstractDVec{K,V}
    keys::Vector{K}
    vals::Vector{V}
    style::S
end
function SkinnyDVec(keys, vals; style=default_style(eltype(vals)))
    length(keys)==length(vals) || throw(ArgumentError("keys and vals must have the same length"))
    return SkinnyDVec(collect(keys), collect(vals), style)
end
StochasticStyle(w::SkinnyDVec) = w.style

Base.length(w::SkinnyDVec) = length(w.keys)
Base.values(w::SkinnyDVec) = w.vals
Base.keys(w::SkinnyDVec) = w.keys
function Base.setindex!(w::SkinnyDVec, val, add) # always append to list
    push!(w.keys, add)
    push!(w.vals, val)
    return w
end
Base.getindex(w::SkinnyDVec, key) = zero(valtype(w)) # always return zero

@inline function add!(w::SkinnyDVec, v::AbstractDVec)
    for (add, val) in pairs(v)
        w[add] = val
    end
    return w
end
