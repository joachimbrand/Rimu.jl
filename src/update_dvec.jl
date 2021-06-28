function threshold_project!(v, threshold)
    w = localpart(v)
    for (add, val) in pairs(w)
        prob = abs(val) / threshold
        if prob < 1 # projection is only necessary if abs(val) < s.threshold
            val = ifelse(prob > cRand(), threshold * sign(val), zero(val))
            w[add] = val
        end
    end
    return v
end

"""
    update_dvec!([::StochasticStyle,] dvec) -> dvec, nt

Perform an arbitrary transformation on `dvec` after the spawning step is completed and
report statistics to the `DataFrame`.

Returns the new `dvec` and a `NamedTuple` `nt` of statistics to be reported.

When extending this function for a custom [`StochasticStyle`](@ref), define a method
for the two-argument call signature!
"""
update_dvec!(::StochasticStyle, v) = v, NamedTuple()

update_dvec!(v) = update_dvec!(StochasticStyle(v), v)

function update_dvec!(s::IsDynamicSemistochastic{<:Any,true}, v)
    len_before = length(v)
    return threshold_project!(v, s.proj_threshold), (; len_before)
end

function update_dvec!(s::IsExplosive, v)
    len_before = length(v)
    return threshold_project!(v, s.proj_threshold), (; len_before)
end
