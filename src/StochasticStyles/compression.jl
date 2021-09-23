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
    for (add, val) in pairs(w)
        prob = abs(val) / t.threshold
        if prob < 1 # projection is only necessary if abs(val) < s.threshold
            val = ifelse(prob > cRand(), t.threshold * sign(val), zero(val))
            w[add] = val
        end
    end
    return v
end

"""
    DoubleOrNothing(; threshold=1, prob=0.5) <: CompressionStrategy

[`CompressionStrategy`](@ref) that compresses a vector by a double or nothing approach. For
each value in the vector below the `threshold`, a coin is flipped. If heads, the value is
doubled, if tails, the value is removed from the vector. The fairness of the coin is
controlled with `prob` - lower values indicate lower probabilties to keep.
"""
struct DoubleOrNothing{T} <: CompressionStrategy
    threshold::T
    prob::Float64
end
DoubleOrNothing(; threshold=1, prob=0.5) = DoubleOrNothing(threshold, prob)

function compress!(d::DoubleOrNothing, v)
    w = localpart(v)
    for (add, val) in pairs(w)
        if abs(val) < d.threshold
            val = ifelse(cRand() > d.prob, zero(valtype(w)), val / d.prob)
            w[add] = val
        end
    end
    return v
end

"""
    DoubleOrNothingWithTarget(; target, threshold=1.0) <: CompressionStrategy

Like [`DoubleOrNothing`](@ref), but attempts to keep the number of elements in the vector
after compression as close to `target` as possible. Works like `DoubleOrNothing`, but the
`prob` parameter is chosen dynamically to try to reach the `target` length.
"""
struct DoubleOrNothingWithTarget{T} <: CompressionStrategy
    target::Int
    threshold::T
end
function DoubleOrNothingWithTarget(; target, threshold=1)
    return DoubleOrNothingWithTarget(target, threshold)
end

function compress!(d::DoubleOrNothingWithTarget, v)
    w = localpart(v)
    prob = d.target / length(v)
    if prob < 1
        compress!(DoubleOrNothing(d.threshold, prob), v)
    end
    return v
end

"""
    DoubleOrNothingWithThreshold(; threshold_hi=1.0, threshold_lo=1e-3, prob=0.5) <: CompressionStrategy

A combination of [`ThresholdProjection`](@ref) and [`DoubleOrNothing`](@ref). If a value is
below `threshold_lo`, [`ThresholdProjection`](@ref) is applied, and if it is below
`threshold_hi`, [`DoubleOrNothing`](@ref) is. The `prob` parameter has the same function as
in [`DoubleOrNothing`](@ref).
"""
struct DoubleOrNothingWithThreshold{T} <: CompressionStrategy
    threshold_hi::T
    threshold_lo::T
    prob::Float64
end
function DoubleOrNothingWithThreshold(; threshold_hi=1.0, threshold_lo=1e-3, prob=0.5)
    return DoubleOrNothingWithThreshold(promote(threshold_hi, threshold_lo)..., prob)
end

function compress!(d::DoubleOrNothingWithThreshold, v)
    w = localpart(v)
    for (add, val) in pairs(w)
        if abs(val) < d.threshold_lo
            # Threshold projection
            prob = abs(val) / d.threshold_lo
            w[add] = ifelse(prob > cRand(), d.threshold_lo * sign(val), zero(val))
        elseif abs(val) < d.threshold_hi
            # Double or nothing projection
            val = ifelse(cRand() > d.prob, zero(valtype(w)), val / d.prob)
            w[add] = val
        end
    end
    return v
end
