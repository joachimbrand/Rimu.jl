"""
    ThresholdCompression(threshold=1) <: CompressionStrategy

[`CompressionStrategy`](@ref) that compresses a vector by threshold projection. Every entry
in the vector with a value below the threshold is either set to zero, or increased to the
threshold. The probabilty of setting it to zero is equal to `abs(value) / threshold`.
"""
struct ThresholdCompression{T} <: CompressionStrategy
    threshold::T
end
ThresholdCompression() = ThresholdCompression(1)

function compress!(t::ThresholdCompression, v)
    w = localpart(v)
    for (key, val) in pairs(w)
        prob = abs(val) / t.threshold
        if prob < 1 # projection is only necessary if abs(val) < s.threshold
            val = ifelse(prob > rand(), t.threshold * sign(val), zero(val))
            w[key] = val
        end
    end
    return v
end
function move_and_compress!(t::ThresholdCompression, target, source)
    for (key, val) in source
        prob = abs(val) / t.threshold
        if prob < 1 && prob > rand()
            target[key] = t.threshold * sign(val)
        elseif prob ≥ 1
            target[key] = val
        end
    end
    return target
end
