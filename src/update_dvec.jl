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

update_dvec!(v, shift) = update_dvec!(StochasticStyle(v), v, shift)
update_dvec!(::StochasticStyle, v, _) = v

function update_dvec!(s::IsDynamicSemistochastic{true}, v, _)
    return threshold_project!(v, s.proj_threshold)
end
