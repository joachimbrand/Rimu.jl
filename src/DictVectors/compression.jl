using ..ConsistentRNG

"""
    CompressionStrategy

The `CompressionStrategy` controls how a vector is compressed after a step. To use, define
`CompressionStrategy(::StochasticStyle)`. The default implementation returns
[`NoCompression`](@ref).
"""
abstract type CompressionStrategy end

"""
    NoCompression <: CompressionStrategy end

Default [`CompressionStrategy`](@ref). Leaves the vector intact.
"""
struct NoCompression <: CompressionStrategy end

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
    DoubleOrNothing(threshold) <: CompressionStrategy

[`CompressionStrategy`](@ref) that compresses a vector by a double or nothing approach. For
each value in the vector below the threshold, a coin is flipped. If heads, the value is
doubled, if tails, the value is removed from the vector.
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
            if cRand() > d.prob
                delete!(w, add)
            else
                w[add] /= d.prob
            end
        end
    end
    return v
end

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
            if cRand() > d.prob
                delete!(w, add)
            else
                w[add] /= d.prob
            end
        end
    end
    return v
end
