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
        @inbounds covsum += (v[i]-mean)*conj(v[i+h]-mean)
    end
    gamma = covsum/(n - Int(corrected))
    return gamma
end
