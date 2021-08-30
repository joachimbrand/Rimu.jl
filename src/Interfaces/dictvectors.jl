"""
    deposit!(w::AbstractDVec, add, val, parent::Pair)

Add `val` into `w` at address `add`, taking into account initiator rules if applicable.
`parent` contains the `address => value` pair from which the pair `add => val`
was created. [`InitiatorDVec`](@ref) can intercept this and add its own functionality.
"""
function deposit!(w, add, val, _)
    w[add] += convert(valtype(w), val)
end

"""
    localpart(dv) -> AbstractDVec

Get the part of `dv` that is located on this MPI rank. Returns `dv` itself for `DictVector`s.
"""
localpart(dv) = dv # default for local data

"""
    storage(dvec) -> AbstractDict

Return the raw storage associated with `dvec` as an `AbstractDict`. Used in MPI
communication.
"""
storage(v::AbstractVector) = Dict(pairs(v))

"""
    zero!(v)

Replace `v` by a zero vector as an inplace operation. For `AbstractDVec` types it means
removing all non-zero elements. For `AbstractArrays`, it sets all of the values to zero.
"""
zero!(v::AbstractVector{T}) where {T} = v .= zero(T)

StochasticStyle(::AbstractArray{T}) where {T} = default_style(T)
