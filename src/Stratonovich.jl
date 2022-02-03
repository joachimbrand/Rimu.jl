"""
Experimental module for correcting the stochastic vector compression to more resemble
a Stratonovich stochastic differential equation.
"""
module Stratonovich

using ..Interfaces, ..ConsistentRNG
using ..DictVectors: InitiatorRule, InitiatorValue, InitiatorDVec
using ..StochasticStyles: ThresholdCompression

import ..StochasticStyles: diagonal_step!, compress!
import ..DictVectors: deposit!, value

export StratonovichCorrection

"""
    StratonovichCorrection() <: InitiatorRule

Rule to be passed to [`InitiatorDVec`](@ref) with the `initiator` keyword argument.

`StratonovichCorrection` triggers a correction to coefficients that will be subjected to
stochastic vector compression in order to emulate a stochastic differential equation of the
Stratonovich type.

To be used with [`IsDynamicSemistochastic`](@ref) and [`ThresholdCompression`](@ref).

### Example usage:
```julia-doctest
julia> v = InitiatorDVec(
           BoseFS((1,2,1)) => 3.4;
           style=IsDynamicSemistochastic(),
           initiator = StratonovichCorrection()
       )
InitiatorDVec{BoseFS{4, 3, BitString{6, 1, UInt8}},Float64} with 1 entry, style = IsDynamicSemistochastic{Float64,ThresholdCompression,DynamicSemistochastic}(), initiator = StratonovichCorrection()
  fs"|1 2 1⟩" => 3.4
```
See [`InitiatorRule`](@ref).
"""
struct StratonovichCorrection{V} <: InitiatorRule{V}
    threshold::V # use it as scaling factor
end

StratonovichCorrection() = StratonovichCorrection(1)

value(::StratonovichCorrection, v::InitiatorValue) = v.safe
# `v.safe` to hold the combined value from all diagonal and off-diagonal spawns
# `v.unsafe` to hold only the diagonal matrix element (to be used for the correction)


# We are dispatching on the type of the coefficient vector. This will not yet work
# with MPIData!

function deposit!(
    w::InitiatorDVec{<:Any,<:Any,<:Any,<:Any, <:StratonovichCorrection},
    add, val, (p_add, p_val)
)
    # for `StratonovichCorrection` we do not need initiator behaviour but simply add
    # values from off-diagonal elements to `safe`
    @assert p_add ≠ add "`deposit!` was called for diagonal step"
    # this should not happen as diagonal deposits are dealt with in `diagonal_step!`
    V = valtype(w)

    old_val = get(w.storage, add, zero(InitiatorValue{V}))
    new_val = InitiatorValue{V}(safe=val)
    new_val += old_val
    if new_val == InitiatorValue{V}(0, 0, 0)
        delete!(w.storage, add)
    else
        w.storage[add] = new_val
    end
    return w
end


@inline function diagonal_step!(
    w::InitiatorDVec{<:Any,<:Any,<:Any,<:Any, <:StratonovichCorrection},
    ham, add, val, dτ, shift, threshold=0, report_stats=false
)
    pd = dτ * (diagonal_element(ham, add) - shift)
    new_val = (1 - pd) * val
    # now deposit
    V = valtype(w)
    old_ival = get(w.storage, add, zero(InitiatorValue{V}))
    new_ival = InitiatorValue{V}(safe = old_ival.safe + new_val, unsafe = -pd)
    # add all contributions in `safe`, remember diagonal contribution `-pd` in `unsafe`
    if new_ival == InitiatorValue{V}(0, 0, 0)
        delete!(w.storage, add)
    else
        w.storage[add] = new_ival
    end

    # for now we are ignoring the `threshold` and `report_stats` arguments
    z = zero(V)
    return (z, z, z, z)
end

function compress!(t::ThresholdCompression,
    v::InitiatorDVec{<:Any,<:Any,<:Any,<:Any, <:StratonovichCorrection}
)
    # w = localpart(v) # fix MPI later!
    s_factor = v.initiator.threshold # the parameter stored in `StratonovichCorrection`
    ws = storage(v)
    for (add, ival) in pairs(ws)
        val = ival.safe + sign(ival.safe) * ival.unsafe/2 * s_factor
        # apply Stratonovich correction
        prob = abs(val) / t.threshold
        if prob < 1 # projection is only necessary if abs(val) < s.threshold
            val = ifelse(prob > cRand(), t.threshold * sign(val), zero(val))
            v[add] = val
        end
    end
    return v
end

end
