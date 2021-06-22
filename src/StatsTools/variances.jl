"""
    autocovariance(v::Vector,h::Int; corrected::Bool=true)
``\\hat{\\gamma}(h) =\\frac{1}{n}\\sum_{t=1}^{n-h}(v_{t+h}-\\bar{v})(v_t-\\bar{v})^*``
Calculate the autocovariance of dataset `v` with a delay `h`. If `corrected` is `true`
(the default) then the sum is scaled with `n-h`, whereas the sum
is scaled with `n` if corrected is `false` where `n = length(v)`.
"""
function autocovariance(v::AbstractVector,h::Int; corrected::Bool=true, mean = mean(v))
    n = length(v)
    covsum = zero(mean)
    for i in 1:n-h
         covsum += (v[i]-mean)*conj(v[i+h]-mean)
    end
    gamma = covsum/(n - Int(corrected))
    return gamma
end

@doc raw"""
    pseudo_cov(x, y; xmean = mean(x), ymean = mean(y), corrected = true)
Compute the pseudo covariance between collections `x` and `y` returning a scalar:
```math
\frac{1}{n}\sum_{i=1}^{n} (x_i - \bar{x})(y_{i} - \bar{y})
```
Optionally,
precomputed means can be passed as keyword arguments. `pseudo_cov(x,y)` is functionally
equivalent to `Statistics.cov(x, conj(y); corrected = false)` but it is found to be
significantly faster and avoids allocations.
"""
@inline function pseudo_cov(x, y; xmean = mean(x), ymean = mean(y), corrected = true)
    n = length(x)
    @assert length(y) == n
    res = zero(promote_type(eltype(x), eltype(y))) / 1
    for i = 1:n
         res += (x[i] - xmean) * (y[i] - ymean)
    end
    return res / (n - Int(corrected))
end
