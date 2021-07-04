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

function update_dvec!(s::IsDeterministic, v)
    len_before = length(v)
    return compress!(s.compression, v), (; len_before)
end

function update_dvec!(s::IsDynamicSemistochastic, v)
    len_before = length(v)
    return compress!(s.compression, v), (; len_before)
end

function update_dvec!(s::IsExplosive, v)
    len_before = length(v)
    return compress!(s.compression, v), (; len_before)
end

function update_dvec!(s::IsDynamicSemistochasticPlus, v)
    ζ = 1e-3
    ξ = 0.04^2/4
    len_before = length(v)
    # DoubleLog for len_before
    δ = s.target_len_before - len_before
    strength = s.strength + ζ * δ
    s = @set s.strength = strength
    v = @set v.style = s

    threshold_project!(v, s.threshold), (; len_before, strength)
end
