# blocking of single time series

"""
    BlockingResult(mean, err, err_err, p_cov, k)
Result of [`block_and_test()`](@ref).

### Fields:
- `mean`: sample mean
- `err`: standard error (estimated standard deviation of the mean)
- `err_err`: estimated uncertainty of `err`
- `p_cov`: estimated pseudo covariance of `mean`, relevant for complex time series
- `k::Int`: k-1 blocking steps were used to uncorrelate time series
"""
@with_kw struct BlockingResult{T}
    mean::T
    err::Float64
    err_err::Float64
    p_cov::T # pseudo covariance for complex normal distribution
    k::Int
end
# constructor from NamedTuple of vectors
function BlockingResult(nt, k)
    T = eltype(nt.mean)
    if k < 0 # blocking failed
        return BlockingResult{T}(nt.mean[1], NaN, NaN, T(NaN), k)
    end
    return BlockingResult{T}(nt.mean[1], nt.std_err[k], nt.std_err_err[k], nt.p_cov[k], k)
end

"""
    cov(r::BlockingResult{<:Complex})
Return the covariance matrix of the multivariate normal distribution approximating
the uncertainty of the blocking result `r` of a complex time series.
"""
function Statistics.cov(r::BlockingResult{<:Complex})
    v_xx = real(r.err^2 + r.p_cov)/2
    v_xy = imag(r.p_cov)/2
    v_yy = real(r.err^2 - r.p_cov)/2
    return [v_xx v_xy; v_xy v_yy]
end

"""
    measurement(r::BlockingResult)
    Measurements.±(r::BlockingResult)
Convert a `BlockingResult` into a `Measurement` for linear error propagation with
[`Measurements`](@ref).

Limitation: Does not account for covariance in complex `BlockingResult`.
"""
function Measurements.measurement(r::BlockingResult{<:Real})
    return measurement(r.mean, r.err)
end
function Measurements.measurement(r::BlockingResult{<:Complex})
    Σ = cov(r) # real valued covariance matrix
    cm = complex(measurement(real(r.mean), √Σ[1,1]), measurement(imag(r.mean), √Σ[2,2]))
    return cm
end

"""
    Particles(r::BlockingResult; mc_samples = 2000)
    MonteCarloMeasurements.±(r::BlockingResult; mc_samples = 2000)
Convert a `BlockingResult` into a `Particles` object for nonlinear error propagation with
[`MonteCarloMeasurements`](@ref).
"""
function MonteCarloMeasurements.Particles(r::BlockingResult{<:Real}; mc_samples = 2000)
    return Particles(r.mean, r.err)
end
function MonteCarloMeasurements.Particles(r::BlockingResult{<:Complex}; mc_samples = 2000)
    Σ = cov(r) # real valued covariance matrix
    ps = Particles(mc_samples, MvNormal([real(r.mean), imag(r.mean)], Σ))
    return complex(ps[1], ps[2])
end


"""
    block_and_test(v::AbstractVector; corrected = true)
    -> BlockingResult(mean, err, err_err, p_cov, k)
Compute the sample mean `mean` and estimate the standard deviation of the mean
(standard error) `err` of a correlated time series using the blocking algorithm from
Flyvberg and Peterson [JCP (1989)](http://aip.scitation.org/doi/10.1063/1.457480)
and the M test of Jonsson
[PRE (2018)](https://link.aps.org/doi/10.1103/PhysRevE.98.043304). `k` is the number of
blocking transformations required to pass the hypothesis test for an uncorrelated time
series and `err_err` the estimated standard error or `err`. If decorrelating the
time series fails according to the M test, `NaN` is returned as the standard error and `-1`
for `k`.
`corrected` controls whether
bias correction for variances is used.
"""
function block_and_test(v::AbstractVector; corrected::Bool=true)
    T = float(eltype(v))
    if length(v) == 0
        @error "Attempted blocking on an empty vector"
        return BlockingResult(zero(T), NaN, NaN, T(NaN) -1)
    elseif length(v) == 1 # treat like failed M test
        return BlockingResult(T(v[1]), NaN, NaN, T(NaN) -1)
    end
    nt = blocks_with_m(v; corrected)
    k = mtest(nt.mj; warn=false)
    return BlockingResult(nt, k)
end

# function block_and_test(v::AbstractVector{<:Complex}; corrected::Bool=true)
#     if length(v) == 0
#         @error "Attempted blocking on an empty vector"
#     elseif length(v) == 1 # treat like failed M test
#         return (
#             BlockingResult(get_real(v[1]), NaN, NaN, -1),
#             BlockingResult(get_imag(v[1]), NaN, NaN, -1),
#         )
#     else
#         nt = blocks_with_m(v; corrected)
#         k_re = mtest(get_real(nt.mj); warn=false)
#         if k_re > 0
#             err_re = real(nt.std_err[k_re])
#             err_err_re = imag(nt.std_err_err[k_re])
#         else
#             err_re = NaN
#             err_err_re = NaN
#         end
#         k_im = mtest(get_imag(nt.mj); warn=false)
#         if k_im > 0
#             err_im = imag(nt.std_err[k_im])
#             err_err_im = imag(nt.std_err_err[k_im])
#         else
#             err_im = NaN
#             err_err_im = NaN
#         end
#         mean = nt.mean[1]
#         return (
#             BlockingResult(real(mean), err_re, err_err_re, k_re),
#             BlockingResult(imag(mean), err_im, err_err_im, k_im),
#         )
#     end
#     return (
#         BlockingResult(0.0, NaN, NaN, -1),
#         BlockingResult(0.0, NaN, NaN, -1),
#     )
# end

# Do we need this?
"""
    blocker(v::Vector) -> new_v::Vector
Reblock the data by successively taking the mean of two adjacent data points to
form a new vector with a half of the `length(v)`. The last data point will be
discarded if `length(v)` is odd.
"""
function blocker(v::AbstractVector{T}) where T
    P = typeof(zero(T)/1)
    new_v = Array{P}(undef,(length(v)÷2))
    for i  in 1:length(v)÷2
        @inbounds new_v[i] = (v[2i-1]+v[2i])/2
    end
    return new_v
end
function blocker(v::AbstractVector{T}) where T <: Complex
    P = typeof(zero(T)/1)
    new_v = StructArray{P}(undef,(length(v)÷2))
    for i  in 1:length(v)÷2
        @inbounds new_v[i] = (v[2i-1]+v[2i])/2
    end
    return new_v
end

# """
#     blocker!(v::Vector)
# Perform a single blocking step on `v` inplace. The length of `v` will reduce to
# `length(v)÷2`.
# """
# function blocker!(v::Vector)
#     new_len = length(v)÷2
#     for i in 1:new_len
#         @inbounds v[i]  = (v[2i-1]+v[2i])/2
#     end
#     return resize!(v, new_len)
# end


"""
    mtest(mj::AbstractVector; warn = true) -> k
The "M test" based on Jonsson, M. Physical Review E, 98(4), 043304, (2018).
Expects `mj` to be a vector relevant M_j values from a blocking analysis,
which are compared to a χ^2 distribution.
Returns the row number `k` where the M-test is passed.
If the M-test has failed `mtest()` returns the value `-1` and optionally prints
a warning message.
"""
function mtest(mj::AbstractVector; warn = true)
    # the χ^2 99 percentiles
    q = [6.634897,  9.210340,  11.344867, 13.276704, 15.086272,
        16.811894, 18.475307, 20.090235, 21.665994, 23.209251,
        24.724970, 26.216967, 27.688250, 29.141238, 30.577914,
        31.999927, 33.408664, 34.805306, 36.190869, 37.566235,
        38.932173, 40.289360, 41.638398, 42.979820, 44.314105,
        45.641683, 46.962942, 48.278236, 49.587884, 50.892181]
    m = reverse(cumsum(reverse(mj)))
    k = 1
    while k <= length(m)-1
       if m[k] < q[k]
           return k
       else
           k += 1
       end
    end
    if warn
        @warn "M test failed, more data needed"
    end
    return -1 # indicating the the M-test has failed
end
mtest(table) = mtest(table.mj)

# get_real(v) = real(v)
# get_real(v::StructArray{<:Complex}) = v.re
#
# get_imag(v) = imag(v)
# get_imag(v::StructArray{<:Complex}) = v.im

"""
    blocks_with_m(v; corrected = true) -> (;blocks, mean, std_err, std_err_err, p_cov, mj)
Perform the blocking algorithm from Flyvberg and Peterson
[JCP (1989)](http://aip.scitation.org/doi/10.1063/1.457480).
Returns named tuple with the results from all blocking steps.
"""
@inline function blocks_with_m(v; corrected::Bool=true)
    T = float(eltype(v))
    R = real(T)
    n_steps = floor(Int,log2(length(v)))

    # initialise arrays to be returned
    blocks = Vector{Int}(undef,n_steps)
    mean_arr = Vector{T}(undef,n_steps)
    std_err = Vector{R}(undef,n_steps)
    std_err_err = Vector{R}(undef,n_steps)
    p_cov = Vector{T}(undef,n_steps)
    mj = Vector{R}(undef,n_steps)

    for i in 1:n_steps
        n = length(v)
        blocks[i] = n
        mean_v = mean(v)
        mean_arr[i] = mean_v
        variance = var(v; corrected, mean=mean_v) # variance
        # sample covariance ŷ(1) Eq. (6) [Jonsson]
        gamma = real(autocovariance(v,1; corrected, mean=mean_v))
        # the M value Eq. (12) [Jonsson]
        mj[i] = n*((n-1)*variance/(n^2)+gamma)^2/(variance^2)
        stderr_v = sqrt(variance/n) # standard error
        std_err[i] = stderr_v
        std_err_err[i] = stderr_v/sqrt(2*(n-1)) # error on standard error Eq. (28) [F&P]
        p_cov[i] = pseudo_cov(v,v; xmean=mean_v, ymean=mean_v, corrected)/n
        v = blocker(v) # re-blocking the dataset
    end
    (length(v)≤ 0 || length(v)>2) && @error "Something went wrong in `blocks_with_m`."
    return (;blocks, mean=mean_arr, std_err, std_err_err, p_cov, mj)
end

# @inline function blocks_with_m(v::AbstractVector{<:Real}; corrected::Bool=true)
#     n_steps = floor(Int,log2(length(v)))
#
#     # initialise arrays to be returned
#     blocks = Vector{Int}(undef,n_steps)
#     mean_arr = Vector{Float64}(undef,n_steps)
#     std_err = Vector{Float64}(undef,n_steps)
#     std_err_err = Vector{Float64}(undef,n_steps)
#     mj = Vector{Float64}(undef,n_steps)
#
#     for i in 1:n_steps
#         n = length(v)
#         blocks[i] = n
#         mean_v = mean(v)
#         mean_arr[i] = mean_v
#         variance = var(v; corrected=corrected, mean=mean_v) # variance
#         # sample covariance ŷ(1) Eq. (6) [Jonsson]
#         gamma = autocovariance(v,1; corrected=corrected, mean=mean_v)
#         mj[i] = n*((n-1)*variance/(n^2)+gamma)^2/(variance^2) # the M value Eq. (12) [Jonsson]
#         stderr_v = sqrt(variance/n)# standard error
#         std_err[i] = stderr_v
#         std_err_err[i] = stderr_v/sqrt(2*(n-1)) # error on standard error Eq. (28) [F&P]
#         v = blocker(v) # re-blocking the dataset
#     end
#     (length(v)≤ 0 || length(v)>2) && @error "Something went wrong in `blocks_with_m`."
#     return (;blocks, mean=mean_arr, std_err, std_err_err, mj)
# end
#
# @inline function blocks_with_m(v::AbstractVector{T}; corrected::Bool=true) where T<:Complex
#     n_steps = floor(Int,log2(length(v)))
#     C = typeof(zero(T)/1)
#
#     # initialise arrays to be returned
#     blocks = Vector{Int}(undef,n_steps)
#     mean_arr = Vector{C}(undef,n_steps)
#     std_err = Vector{C}(undef,n_steps)
#     std_err_err = Vector{C}(undef,n_steps)
#     mj = StructArray{C}(undef,n_steps)
#
#     for i in 1:n_steps
#         n = length(v)
#         blocks[i] = n
#         mean_v = mean(v)
#         mean_arr[i] = mean_v
#         var_re = var(get_real(v); corrected=corrected, mean=real(mean_v)) # variance
#         var_im = var(get_imag(v); corrected=corrected, mean=imag(mean_v)) # variance
#         # sample covariance ŷ(1) Eq. (6) [Jonsson]
#         gamma_re = autocovariance(get_real(v),1; corrected=corrected, mean=real(mean_v))
#         gamma_im = autocovariance(get_imag(v),1; corrected=corrected, mean=imag(mean_v))
#         # the M value Eq. (12) [Jonsson]
#         mj[i] =
#             n * ((n - 1) * var_re / (n^2) + gamma_re)^2 / (var_re^2) +
#             im * n * ((n - 1) * var_im / (n^2) + gamma_im)^2 / (var_im^2)
#         stderr_v = sqrt(var_re/n) + im*sqrt(var_im/n) # standard error
#         std_err[i] = stderr_v
#         std_err_err[i] = stderr_v/sqrt(2*(n-1)) # error on standard error Eq. (28) [F&P]
#         v = blocker(v) # re-blocking the dataset
#     end
#     (length(v)≤ 0 || length(v)>2) && @error "Something went wrong in `blocks_with_m`."
#     return (;blocks, mean=mean_arr, std_err, std_err_err, mj)
# end
#
# @inline function blocks_with_m!(v; corrected::Bool=true)
#     table = []
#     while true
#         n = length(v)
#         n < 2 && break
#         μ = mean(v)
#         var_v = var(v; corrected, mean=μ) # real
#         γ = real(autocovariance(v, 1; corrected, mean=μ)) # real
#         mj = n * ((n - 1) * var_v / (n^2) + γ)^2 / (var_v^2) # real
#         std_err = √(var_v/n)
#         std_err_err = std_err/√(2*(n-1))
#         push!(table, (; blocks=n, mean=μ, std_err, std_err_err, mj))
#         blocker!(v)
#     end
#     return DataFrame(table)
# end
# @inline function blocks_with_m_new(vo; corrected::Bool=true)
#     v = copy!(float(eltype(vo))[], vo) # copy input data to vector as working memory
#     return blocks_with_m!(v; corrected)
# end
#
