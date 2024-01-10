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

_set_value!(w, key, val) = w[key] = val
_set_value!(w::Dict, key, val) = val ≠ 0 ? w[key] = val : delete!(w, key)

step_stats(::ThresholdCompression) = (:len_before,), (0,)

function _threshold_compress!(t::ThresholdCompression, w, pairs)
    for (add, val) in pairs
        prob = abs(val) / t.threshold
        if prob < 1 # projection is only necessary if abs(val) < s.threshold
            val = ifelse(prob > rand(), t.threshold * sign(val), zero(val))
        end
        _set_value!(w, add, val)
    end
end

function compress!(t::ThresholdCompression, v)
    len_before = length(v)
    v_local = localpart(v)
    _threshold_compress!(t, v_local, pairs(v_local))
    return (len_before,)
end

function compress!(t::ThresholdCompression, w, v)
    len_before = length(v)
    w_local = localpart(w)
    v_local = localpart(v)
    empty!(w_local)
    _threshold_compress!(t, w_local, pairs(v_local))
    return (len_before,)
end
