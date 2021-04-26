"""
`Blocking`

Module that contains functions performing the Flyvbjerg-Petersen
(J. Chem. Phys. 91, 461 (1989)) blocking analysis for evaluating
the standard error on a correlated data set. A "M-test" is also
implemented based on Jonsson (Phys. Rev. E 98, 043304, (2018)).
"""
module Blocking

using DataFrames, Statistics
using StructArrays, Parameters
import Measurements

export autocovariance, covariance
export blocker, blocking, blockingErrorEstimation, mtest
export autoblock, blockAndMTest
export growthWitness, gW, smoothen
export block_and_test, mean_and_se, mean_pm_se
export crosscov_FP, cov_bare, fidelity_and_se

# """
# Calculate the variance of the dataset v
# """
# function variance(v::Vector)
#     n = 0::Int
#     sum = 0.0::Float64
#     sumsq = 0.0::Float64
#     for x in v
#         n += 1
#         sum += x
#         sumsq += x^2
#     end
#     return (sumsq-sum^2/n)/(n-1)
# end
#
# """
# Calculate the standard deviation of the dataset v
# """
# sd(v::Vector) = sqrt(variance(v))
#
"""
    se(v::Vector;corrected::Bool=true)
Calculate the standard error of the dataset `v`. If `corrected` is `true`
(the default) then the sum in `std` is scaled with `n-1`, whereas the sum
is scaled with `n` if corrected is `false` where `n = length(v)`.
"""
se(v; corrected::Bool=true) = std(v;corrected=corrected)/sqrt(length(v))

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

"""
    blocking(v::Vector; typos = nothing) -> df
Perform a blocking analysis according to Flyvberg and Peterson
[JCP (1989)](http://aip.scitation.org/doi/10.1063/1.457480)
for single data set and return a `DataFrame` with
statistical data for each blocking step. M-test data according to Jonsson
[PRE (2018)](https://link.aps.org/doi/10.1103/PhysRevE.98.043304) is also
provided.

Keyword argument `typos`
* `typos = nothing` - correct all presumed typos.
* `typos = :FP` - use Flyvberg and Peterson (correct) standard error and Jonsson formul for M.
* `typos = :Jonsson` - calculate `M` and standard error as written in Jonsson.
"""
function blocking_old(v::Vector; typos = nothing)
    df = DataFrame(blocks = Int[], mean = Float64[], stdev = Float64[],
                    std_err = Float64[], std_err_err = Float64[], gamma = Float64[], M = Float64[])
    while length(v) >= 2
        n = length(v)
        mean = sum(v)/n
        var = v .- mean
        sigmasq = sum(var.^2)/n # uncorrected sample variance
        if typos === nothing
            gamma = sum(var[1:n-1].*var[2:n])/(n-1)
        else # typos ∈ {:FP, :Jonsson}
            gamma = sum(var[1:n-1].*var[2:n])/n
            # sample covariance ŷ(1) Eq. (6) [Jonsson]
            # but why is the denominator n and not (n-1)????
        end
        mj = n*((n-1)*sigmasq/(n^2)+gamma)^2/(sigmasq^2)
        stddev = sqrt(sigmasq)
        if typos === nothing || typos == :FP
            stderr = stddev/sqrt(n-1) # [F&P] Eq. (28)
        else
            stderr = stddev/sqrt(n) # [Jonsson] Fig. 2
        end
        stderrerr = stderr*1/sqrt(2*(n-1)) # [F&P] Eq. (28)
        v = blocker(v)
        #println(n, mean, stddev, stderr)
        push!(df,(n, mean, stddev, stderr, stderrerr, gamma, mj))
    end
    return df
end

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

"""
    blocking(v::Vector; corrected::Bool=true) -> df
Perform a blocking analysis according to Flyvberg and Peterson
[JCP (1989)](http://aip.scitation.org/doi/10.1063/1.457480)
for single data set and return a `DataFrame` with
statistical data for each blocking step. M-test data according to Jonsson
[PRE (2018)](https://link.aps.org/doi/10.1103/PhysRevE.98.043304) is also
provided.
If `corrected` is `true` (the default) then the sum in `var` is scaled
with `n-1` and in `autocovariance` is scaled with `n-h`, whereas the sum
is scaled with `n` for both if corrected is `false` where `n = length(v)`.
"""
function blocking(v::AbstractVector{T}; corrected::Bool=true) where T
    P = promote_type(T, Float64)
    df = DataFrame(blocks = Int[], mean = P[], stdev = Float64[],
                    std_err = Float64[], std_err_err = Float64[], gamma = P[], M = P[])
    while length(v) >= 2
        n = length(v) # size of current dataset
        mean_v = mean(v)
        variance = var(v; corrected=corrected) # variance
        gamma = autocovariance(v,1; corrected=corrected) # sample covariance ŷ(1) Eq. (6) [Jonsson]
        mj = n*((n-1)*variance/(n^2)+gamma)^2/(variance^2) # the M value Eq. (12) [Jonsson]
        stddev = sqrt(variance) # standard deviation
        stderr = stddev/sqrt(n) # standard error
        stderrerr = stderr/sqrt(2*(n-1)) # error on standard error Eq. (28) [F&P]
        v = blocker(v) # re-blocking the dataset
        push!(df,(n, mean_v, stddev, stderr, stderrerr, gamma, mj))
    end
    return df
end

get_real(v) = real(v)
get_real(v::StructArray{<:Complex}) = v.re

get_imag(v) = imag(v)
get_imag(v::StructArray{<:Complex}) = v.im

"""
    blocks_with_m(v; corrected = true) -> (;blocks, mean, std_err, std_err_err, mj)
Perform the blocking algorithm from Flyvberg and Peterson
[JCP (1989)](http://aip.scitation.org/doi/10.1063/1.457480).
Returns named tuple with the results from all blocking steps.
"""
@inline function blocks_with_m(v::AbstractVector{<:Real}; corrected::Bool=true)
    n_steps = floor(Int,log2(length(v)))

    # initialise arrays to be returned
    blocks = Vector{Int}(undef,n_steps)
    mean_arr = Vector{Float64}(undef,n_steps)
    std_err = Vector{Float64}(undef,n_steps)
    std_err_err = Vector{Float64}(undef,n_steps)
    mj = Vector{Float64}(undef,n_steps)

    for i in 1:n_steps
        n = length(v)
        blocks[i] = n
        mean_v = mean(v)
        mean_arr[i] = mean_v
        variance = var(v; corrected=corrected, mean=mean_v) # variance
        # sample covariance ŷ(1) Eq. (6) [Jonsson]
        gamma = autocovariance(v,1; corrected=corrected, mean=mean_v)
        mj[i] = n*((n-1)*variance/(n^2)+gamma)^2/(variance^2) # the M value Eq. (12) [Jonsson]
        stderr_v = sqrt(variance/n)# standard error
        std_err[i] = stderr_v
        std_err_err[i] = stderr_v/sqrt(2*(n-1)) # error on standard error Eq. (28) [F&P]
        v = blocker(v) # re-blocking the dataset
    end
    (length(v)≤ 0 || length(v)>2) && @error "Something went wrong in `blocks_with_m`."
    return (;blocks, mean=mean_arr, std_err, std_err_err, mj)
end

@inline function blocks_with_m(v::AbstractVector{T}; corrected::Bool=true) where T<:Complex
    n_steps = floor(Int,log2(length(v)))
    C = typeof(zero(T)/1)

    # initialise arrays to be returned
    blocks = Vector{Int}(undef,n_steps)
    mean_arr = Vector{C}(undef,n_steps)
    std_err = Vector{C}(undef,n_steps)
    std_err_err = Vector{C}(undef,n_steps)
    mj = StructArray{C}(undef,n_steps)

    for i in 1:n_steps
        n = length(v)
        blocks[i] = n
        mean_v = mean(v)
        mean_arr[i] = mean_v
        var_re = var(get_real(v); corrected=corrected, mean=real(mean_v)) # variance
        var_im = var(get_imag(v); corrected=corrected, mean=imag(mean_v)) # variance
        # sample covariance ŷ(1) Eq. (6) [Jonsson]
        gamma_re = autocovariance(get_real(v),1; corrected=corrected, mean=real(mean_v))
        gamma_im = autocovariance(get_imag(v),1; corrected=corrected, mean=imag(mean_v))
        # the M value Eq. (12) [Jonsson]
        mj[i] =
            n * ((n - 1) * var_re / (n^2) + gamma_re)^2 / (var_re^2) +
            im * n * ((n - 1) * var_im / (n^2) + gamma_im)^2 / (var_im^2)
        stderr_v = sqrt(var_re/n) + im*sqrt(var_im/n) # standard error
        std_err[i] = stderr_v
        std_err_err[i] = stderr_v/sqrt(2*(n-1)) # error on standard error Eq. (28) [F&P]
        v = blocker(v) # re-blocking the dataset
    end
    (length(v)≤ 0 || length(v)>2) && @error "Something went wrong in `blocks_with_m`."
    return (;blocks, mean=mean_arr, std_err, std_err_err, mj)
end

@with_kw struct BlockingResult
    mean::Float64
    err::Float64
    err_err::Float64
    k::Int
end

get_ks(br::BlockingResult) = (br.k,)
get_ks(t::Tuple) = (br.k for br in t)

"""
    measurement(r::BlockingResult)
    ±(r::BlockingResult)
Convert a `BlockingResult` into a `Measurement`.
"""
Measurements.measurement(r::BlockingResult) = Measurements.measurement(r.mean, r.err)

"""
    block_and_test(v; corrected = true) -> BlockingResult(mean, err, err_err, k)
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
function block_and_test(v::AbstractVector{<:Real}; corrected::Bool=true)
    if length(v) == 0
        @error "Attempted blocking on an empty vector"
    elseif length(v) == 1 # treat like failed M test
        return BlockingResult(v[1], NaN, NaN, -1)
    else
        nt = blocks_with_m(v; corrected)
        k = mtest(nt.mj; warn=false)
        mean = nt.mean[1]
        if k > 0
            err = nt.std_err[k]
            err_err = nt.std_err_err[k]
        else
            err = NaN
            err_err = NaN
        end
        return BlockingResult(mean, err, err_err, k)
    end
    return BlockingResult(0.0, NaN, NaN, -1)
end

function block_and_test(v::AbstractVector{<:Complex}; corrected::Bool=true)
    if length(v) == 0
        @error "Attempted blocking on an empty vector"
    elseif length(v) == 1 # treat like failed M test
        return (
            BlockingResult(get_real(v[1]), NaN, NaN, -1),
            BlockingResult(get_imag(v[1]), NaN, NaN, -1),
        )
    else
        nt = blocks_with_m(v; corrected)
        k_re = mtest(get_real(nt.mj); warn=false)
        if k_re > 0
            err_re = real(nt.std_err[k_re])
            err_err_re = imag(nt.std_err_err[k_re])
        else
            err_re = NaN
            err_err_re = NaN
        end
        k_im = mtest(get_imag(nt.mj); warn=false)
        if k_im > 0
            err_im = imag(nt.std_err[k_im])
            err_err_im = imag(nt.std_err_err[k_im])
        else
            err_im = NaN
            err_err_im = NaN
        end
        mean = nt.mean[1]
        return (
            BlockingResult(real(mean), err_re, err_err_re, k_re),
            BlockingResult(imag(mean), err_im, err_err_im, k_im),
        )
    end
    return (
        BlockingResult(0.0, NaN, NaN, -1),
        BlockingResult(0.0, NaN, NaN, -1),
    )
end

"""
    mean_and_se(v; corrected = true) -> (mean, err)
Return the mean and estimated standard error from an automated blocking analysis as a tuple.
See [`block_and_test`](@ref).
`corrected` controls whether
bias correction for variances is used.
"""
function mean_and_se(v::AbstractVector{<:Real}; corrected::Bool=true)
    br = block_and_test(v; corrected)
    return (br.mean, br.err)
end
function mean_and_se(v::AbstractVector{<:Complex}; corrected::Bool=true)
    t = block_and_test(v; corrected) # returns tuple with real and complex results
    return (t[1].mean + im*t[2].mean, t[1].err, t[2].err)
end

"""
    mean_pm_se(v; corrected = true) -> mean ± err
Return the mean and estimated standard error from an automated blocking analysis as a
[`Measurement`](@ref). See [`block_and_test`](@ref).
`corrected` controls whether
bias correction for variances is used.
"""
function mean_pm_se(v::AbstractVector{<:Real}; corrected::Bool=true)
    return Measurements.measurement(block_and_test(v; corrected))
end
function mean_pm_se(v::AbstractVector{<:Complex}; corrected::Bool=true)
    t = block_and_test(v; corrected) # returns tuple with real and complex results
    return complex(Measurements.measurement(t[1]), Measurements.measurement(t[2]))
end

"""
    fidelity_and_se(rr::Tuple; p_field = :hproj, steps_equilibrate = 0) -> (;fid, fid_err)
Compute the fidelity of the average coefficient vector and the projector defined in
`p_field` and its standard error obtained from blocking analysis with mtest.
Expects the result of a replica run of [`fciqmc!()`](@ref) passed as argument `rr` (a
tuple of DataFrames). The fidelity of states `|ψ⟩` and `|ϕ⟩` is defined as
```math
F(ψ,ϕ) = \\frac{|⟨ψ|ϕ⟩|^2}{⟨ψ|ψ⟩⟨ϕ|ϕ⟩} .
```
Specifically, `fidelity_and_se` computes
```math
F(v,⟨c⟩) = \\frac{⟨(c₁⋅v)(v⋅c₂)⟩}{⟨c₁⋅c₂⟩} ,
```
where `v` is the projector specified by `p_field`, which is assumed to be normalised to
unity with the two-norm (i.e. `v⋅v == 1`), and `c₁` and `c₂` are two replica coefficient
vectors.
"""
function fidelity_and_se(rr::Tuple; p_field = :hproj, steps_equilibrate = 0)
    df1 = rr[2][1] :: DataFrame # first replica DataFrame
    df2 = rr[2][2] :: DataFrame # second replica DataFrame
    dfr = rr[1] :: DataFrame # joint results DataFrame
    # numerator for fidelity calculation as time series (array)
    fid_num = conj(getproperty(df1, p_field)) .* getproperty(df2, p_field)
    fid_num = fid_num[steps_equilibrate+1:end]
    # denominator
    fid_den = dfr.xdoty[steps_equilibrate+1:end]
    bt_num = block_and_test(fid_num)
    bt_den = block_and_test(fid_den)
    # choose largst k from mtest on blocking analyses on numerator and denominator
    ks = (get_ks(bt_num)..., get_ks(bt_den)...)
    any(k->k<0, ks) && @warn "m-test failed at least once for fidelity" ks
    k = max(ks...)
    qblocks = blocking(fid_num, fid_den)
    fid = qblocks.mean_x[1]/qblocks.mean_y[1]
    fid_err = k>0 ? qblocks.SE_f[k] : missing
    return (;fid, fid_err)
end
function fidelity_and_se(
    df::DataFrame;
    v_field = :hproj, # v ⋅ c with complex coefficients
    s_field = :vproj, # real(c) ⋅ imag(c)
    steps_equilibrate = 0,
)
    complex_v_dot_cn = getproperty(df, v_field)[steps_equilibrate+1:end]

    # numerator for fidelity calculation as time series (array)
    fid_num = real(complex_v_dot_cn) .* imag(complex_v_dot_cn)
    # denominator
    fid_den = getproperty(df, s_field)[steps_equilibrate+1:end]
    bt_num = block_and_test(fid_num)
    bt_den = block_and_test(fid_den)
    # choose largst k from mtest on blocking analyses on numerator and denominator
    ks = (get_ks(bt_num)..., get_ks(bt_den)...)
    any(k->k<0, ks) && @warn "m-test failed at least once for fidelity" ks
    k = max(ks...)
    qblocks = blocking(fid_num, fid_den)
    fid = qblocks.mean_x[1]/qblocks.mean_y[1]
    fid_err = k>0 ? qblocks.SE_f[k] : missing
    return (;fid, fid_err)
end

"""
    covariance(x::Vector,y::Vector; corrected::Bool=true)
Calculate the covariance between the two data sets `x` and `y` with equal length.
If `corrected` is `true` (the default) then the sum is scaled with
 `n-1`, whereas the sum is scaled with `n` if corrected is `false`
 where `n = length(x) = length(y)`.
"""
function covariance(vi::AbstractVector,vj::AbstractVector; corrected::Bool=true)
    # if length(vi) != length(vj)
    #     @warn "Two data sets with non-equal length! Truncating the longer one."
    #     if length(vi) > length(vj)
    #         vi = vi[1:length(vj)]
    #     else
    #         vj = vj[1:length(vi)]
    #     end
    # end
    n = length(vi)
    meani = mean(vi)
    meanj = mean(vj)
    covsum = zero(meani)
    for i in 1:n
        covsum += (vi[i]-meani)*conj(vj[i]-meanj)
    end
    cov = covsum/(n - Int(corrected))
    return cov
end


"""
    combination_division(x::Vector,y::Vector; corrected::Bool=true)
Find the standard error on the quotient of means `x̄/ȳ` from two data sets,
note that the standard errors are different on ``(x̄/ȳ) \\neq \\bar{(\\frac{x}{y})}``.
If `corrected` is `true` (the default) then the sums in both variance and covariance
are scaled with `n-1`, whereas the sums are scaled with `n` if corrected is `false`
 where `n = length(x) = length(y)`.
"""
function combination_division(vi::AbstractVector,vj::AbstractVector; corrected::Bool=true)
    # if length(vi) != length(vj)
    #     @warn "Two data sets with non-equal length! Truncating the longer one."
    #     if length(vi) > length(vj)
    #         vi = vi[1:length(vj)]
    #     else
    #         vj = vj[1:length(vi)]
    #     end
    # end
    n = length(vi)
    meani = mean(vi)
    meanj = mean(vj)
    meanf = meani/meanj
    sei = se(vi;corrected=corrected)
    sej = se(vj;corrected=corrected)
    cov = covariance(vi,vj;corrected=corrected)
    sef = abs(meanf*sqrt((sei/meani)^2 + (sej/meanj)^2 - 2.0*cov/(n*meani*meanj)))
    return sef
end


"""
    blocking(x::Vector,y::Vector) -> df::DataFrame
Perform a blocking analysis for the quotient of means `x̄/ȳ` from two data sets.
If `corrected` is `true` (the default) then the sums in both variance and covariance
are scaled with `n-1`, whereas the sums are scaled with `n` if corrected is `false`
 where `n = length(x) = length(y)`.
Entries in returned dataframe:
* `blocks` = number of blocks in current blocking step;
* `mean_x`, `SD_x`, `SE_x`, `SE_SE_x` = the mean, standard deviation, standard error and error on standard error estimated for dataset `x`;
* `mean_y`, `SD_y`, `SE_y`, `SE_SE_y` = ditto. for dataset `y`;
* `Covariance` = the covariance between data in `x` and `y`;
* `mean_f` = `x̄/ȳ`;
* `SE_f` = standard error estimated for `x̄/ȳ`.
"""
function blocking(vi::Vector{T1},vj::Vector{T2}; corrected::Bool=true) where {T1, T2}
    P = promote_type(T1, T2, Float64)
    df = DataFrame(blocks=Int[], mean_x=P[], SD_x=Float64[], SE_x=Float64[], SE_SE_x=Float64[],
            mean_y=P[], SD_y=Float64[], SE_y=Float64[], SE_SE_y=Float64[], Covariance=P[],
            mean_f=P[], SE_f=Float64[])
    # if length(vi) != length(vj)
    #     @warn "Two data sets with non-equal length! Truncating the longer one."
    #     if length(vi) > length(vj)
    #         vi = vi[1:length(vj)]
    #     else
    #         vj = vj[1:length(vi)]
    #     end
    # end
    while length(vi) >= 2
        n = length(vi)
        meani = mean(vi)
        meanj = mean(vj)
        meanf = meani/meanj
        sdi = std(vi;corrected=corrected)
        sdj = std(vj;corrected=corrected)
        sei = sdi/sqrt(n)
        sej = sdj/sqrt(n)
        sesei = sei*1/sqrt(2*(n-1))
        sesej = sej*1/sqrt(2*(n-1))
        cov = covariance(vi,vj;corrected=corrected)
        #sef = sei/sej
        sef = combination_division(vi,vj;corrected=corrected)
        vi = blocker(vi)
        vj = blocker(vj)
        #println(n, mean, stddev, stderr)
        push!(df,(n, meani, sdi, sei, sesei, meanj, sdj, sej, sesej, cov, meanf, sef))
    end
    return df
end

# no longer needed, using the M test now
# """
# estimating stnadard error from blocking analysis based on the overlapping of
# error bars, if all the error bars (or more than 3 on a roll) behind current
# one are overlapping with it, return the current standard error with error bar.
# """
# function blockingErrorEstimation(df::DataFrame)
#     e = df.std_err[1:end-1] # ignoring the last data point
#     ee = df.std_err_err[1:end-1] # ignoring the last data point
#     n = length(e)
#     ind = collect(1:length(e))
#     e_upper = map(x->e[x]+ee[x],ind) # upper bounds
#     e_lower = map(x->e[x]-ee[x],ind) # lower bounds
#     i = 1 # start from the first data point
#     plateau = false
#     while i < n
#         count = 0 # set up a counter for checking overlapped error bars
#         for j in (i+1):n # j : all data points after i
#             if e_lower[i] >= e_lower[j] && e_upper[i] <= e_upper[j]
#                 count += 1
#                 #println("i: ",i," j: ",j," c: ",count)
#                 # some tolerance, say if there are 3 overlaps on a roll could be a plateau
#                 if count > 3 && (i + count) == j
#                     plateau = true
#                     println("\x1b[32mplateau detected\x1b[0m")
#                     return e[i], ee[i], plateau
#                 end
#             end
#         end # for
#         if count == (n-i)
#             println("\x1b[32mNO plateau is detected, take the best estimation\x1b[0m")
#             return e[i], ee[i], plateau
#         else
#             i += 1 # move on to next point
#         end
#     end # while
#     println("\x1b[32mNO plateau, NO error bar overlap, take the second last point\x1b[0m")
#     return e[i], ee[i], plateau # return the last ponit
# end

"""
    mtest(df::DataFrame; warn = true) -> k
The "M test" based on Jonsson, M. Physical Review E, 98(4), 043304, (2018).
Expects `df` to be output of a blocking analysis with column `df.M` containing
relevant M_j values, which are compared to a χ^2 distribution.
Returns the row number `k` where the M-test is passed.
If the M-test has failed `mtest()` returns the value `-1` and optionally prints
a warning message.
"""
function mtest(df::DataFrame; warn = true)
    return mtest(df.M; warn)
end
function mtest(mj::AbstractVector; warn = true)
    # the χ^2 99 percentiles
    q = [6.634897,  9.210340,  11.344867, 13.276704, 15.086272,
        16.811894, 18.475307, 20.090235, 21.665994, 23.209251,
        24.724970, 26.216967, 27.688250, 29.141238, 30.577914,
        31.999927, 33.408664, 34.805306, 36.190869, 37.566235,
        38.932173, 40.289360, 41.638398, 42.979820, 44.314105,
        45.641683, 46.962942, 48.278236, 49.587884, 50.892181]
    m = reverse(cumsum(reverse(mj)))
    #println(M)
    k = 1
    while k <= length(m)-1
       if m[k] < q[k]
           # if info
           #     stder = round(df.std_err[k],digits=3)
           #     stderer = round(df.std_err_err[k],digits=3)
           #     println("\x1b[32mM test passed, the smallest k is $k\x1b[0m")
           #     println("\x1b[32mStandard error estimation: $stder ± $stderer\x1b[0m")
           # end
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

# using Statistics, StatsBase
#
# function mm(x::Vector, x̄)
#     n = length(x)
#     σ2 = varm(x,x̄, corrected = false)
#     γ = cov(x[1:n-1],x[2:n], corrected = false)
#     m = n * ((n - 1) * σ2 + γ)^2 / σ2
#     return m
# end
#
# """
#     autoblock(x,y)
# Perform automated blocking analysis for `x̄/ȳ`.
# """
# function block2(x::Vector,y::Vector)
#     n = length(x)
#     @assert length(y)==n "Vectors do not have the same length"
#     ms = Vector{Float64}(undef,trunc(Int,log2(n)))
#     blocking_step = 1
#     while n ≥ 2
#         x̄ = mean(x)
#         ȳ = mean(y)
#         f̄ = x̄/ȳ
#         σ2x = varm(x,x̄, corrected = false)
#         γx = cov(x[1:n-1],x[2:n], corrected = false)
#         mx = mm(x, x̄)
#         my = mm(y, ȳ)
#         sef = combination_division(x,y)
#         x = blocker(x)
#         y = blocker(y)
#         n = length(x)
#         blocking_step += 1
#     end
# end

"""
    v̄, σ, σσ, k, df = blockAndMTest(v::Vector)
Perform a blocking analysis and M-test on `v` returning the mean `v̄`,
standard error `σ`, its error `σσ`, the number of blocking steps `k`, and
the `DataFrame` `df` with blocking data.
"""
function blockAndMTest(v::Vector; corrected::Bool=true)
    df = blocking(v;corrected=corrected)
    k = mtest(df, warn=false)
    v̄ = df.mean[1]
    if k>0
        σ = df.std_err[k]
        σσ = df.std_err_err[k]
    else
        @warn "M test failed, more data needed"
        σ = maximum(df.std_err)
        σσ = maximum(df.std_err_err)
    end
    return v̄, σ, σσ, k, df
end

function blockTestShiftAndProjected(df::DataFrame; start = 1, stop = size(df)[1], corrected::Bool=true)
    s̄, σs, σσs, ks, dfs = blockAndMTest(df.shift[start:stop];corrected=corrected)
    v̄, σv, σσv, kv, dfv = blockAndMTest(df.vproj[start:stop];corrected=corrected)
    h̄, σh, σσh, kh, dfh = blockAndMTest(df.hproj[start:stop];corrected=corrected)
    dfp = blocking(df.hproj[start:stop], df.vproj[start:stop];corrected=corrected)
    k = max(ks, kv, kh)
    @show ks, kv, kh
    ks==kv==kh || @warn "k values are not the same."
    return s̄, σs, dfp.mean_f[1], dfp.SE_f[k], ks, kv, kh
end

"""
    autoblock(df::DataFrame; start = 1, stop = size(df)[1])
    -> s̄, σs, ē, σe, k
Determine mean shift `s̄` and projected energy `ē` with respective standard
errors `σs` and `σe` by blocking analsis from the `DataFrame` `df` returned
from `fciqmc!()`. The number `k` of blocking
steps and decorrelation time `2^k` are obtained from the M-test for the
shift and also applied to the projected energy, assuming that the projected
quantities decorrelate on the same time scale. Only the real part of the shift
is considered. Returns a named tuple.
"""
function autoblock(df::DataFrame; start = 1, stop = size(df)[1], corrected::Bool=true)
    s̄, σs, σσs, ks, dfs = blockAndMTest(real.(df.shift[start:stop]);corrected=corrected) # shift
    if eltype(df.hproj) == Missing
        return (s̄ = s̄, σs = σs, ē = missing, σe = missing, k = ks)
    end
    dfp = blocking(df.hproj[start:stop], df.vproj[start:stop];corrected=corrected) # projected
    return (s̄ = s̄, σs = σs, ē = dfp.mean_f[1], σe = dfp.SE_f[ks], k = ks)
end

# call signature for chaining
autoblock(nt::NamedTuple{(:df,:eqsteps)}) = autoblock(nt.df; start = nt.eqsteps)

# version for replica run
"""
    autoblock(dftup::Tuple; start = 1, stop = size(dftup[1])[1])
    -> s̄1, σs1, s̄2, σs2, ē1, σe1, ē2, σe2, ēH, σeH, k
Replica version. `dftup` is the tuple of `DataFrame`s returned from replica
`fciqmc!()`. Returns a named tuple with shifts and three variational energy
estimators and respective errors obtained from blocking analysis. The larger
of the `k` values from M-tests on the two shift time series is used.
"""
function autoblock(dftup::Tuple; start = 1, stop = size(dftup[1])[1], corrected::Bool=true)
    (df_mix, (df_1, df_2)) = dftup # unpack the three DataFrames
    s̄1, σs1, σσs1, ks1, dfs1 = blockAndMTest(real.(df1.shift[start:stop]);corrected=corrected) # shift 1
    s̄2, σs2, σσs2, ks2, dfs2 = blockAndMTest(real.(df2.shift[start:stop]);corrected=corrected) # shift 2
    xdy = df_mix.xdy[start:stop]
    s1_xdy = real.(dfs1.shift[start:stop]).*xdy
    s2_xdy = real.(dfs2.shift[start:stop]).*xdy
    xHy = df_mix.xHy[start:stop]
    df_var_1 = blocking(s1_xdy, xdy;corrected=corrected)
    df_var_2 = blocking(s2_xdy, xdy;corrected=corrected)
    df_var_H = blocking(xHy, xdy;corrected=corrected)
    dfp = blocking(df.hproj[start:stop], df.vproj[start:stop];corrected=corrected)
    ks = max(ks1, ks2)
    return (s̄1=s̄1, σs1=σs1, s̄2=s̄2, σs2=σs2,
        ē1 = df_var_1.mean_f[1], σe1 = df_var_1.SE_f[ks],
        ē2 = df_var_2.mean_f[1], σe2 = df_var_2.SE_f[ks],
        ēH = df_var_H.mean_f[1], σeH = df_var_H.SE_f[ks], k = ks)

end

#G_b^{(n)} = \\bar{S}^{(n)} - \\frac{\\log\\vert\\mathbf{c}^{(n+b)}\\vert - \\log\\vert\\mathbf{c}^{(n)}\\vert}{b d\\tau},
"""
    growthWitness(norm::AbstractArray, shift::AbstractArray, dt; b = 30, pad = :true) -> g
    growthWitness(df::DataFrame; b = 30, pad = :true) -> g
Compute the growth witness
```math
G_b^{(n)} = S̄^{(n)} - \\frac{\\log\\vert\\mathbf{c}^{(n+b)}\\vert - \\log\\vert\\mathbf{c}^{(n)}\\vert}{b d\\tau},
```
where `S̄` is an average of the `shift` over `b` time steps and \$\\vert\\mathbf{c}^{(n)}\\vert ==\$ `norm[n]`.
The parameter `b ≥ 1` averages the derivative quantity over `b` time steps and helps suppress noise.

If `pad` is set to `:false` then the returned array `g` has the length `length(norm) - b`.
If set to `:true` then `g` will be padded up to the same length as `norm` and `shift`.
"""
function growthWitness(norm::AbstractArray, shift::AbstractArray, dt; b = 30, pad = :true)
    l = length(norm)
    @assert length(shift) == l "`norm` and `shift` arrays need to have the same length."
    n = l - b
    g = Vector{eltype(shift)}(undef, pad ? l : n)
    offset = pad ? b÷2 : 0 # use offset only if pad == :true
    for i in 1:n
        g[i + offset] = -(1/(b*dt) * log(norm[i+b]/norm[i]) - 1/(b+1) * sum(shift[i:i+b]))
    end
    if pad # pad the vector g at both ends
        g[1:offset] .= g[offset]
        g[offset+n+1 : end] .= g[offset+n]
    end
    return g
end
growthWitness(df::DataFrame; b = 30, pad = :true) = growthWitness(df.norm, df.shift, df.dτ[1]; b=b, pad=pad)

"""
    smoothen(noisy::AbstractVector, b; pad = :true)
Smoothen the array `noisy` by averaging over a sliding window of length `b`.
Pad to `length(noisy)` if `pad == true`. Otherwise, the returned array will have
the length `length(noisy) - b`.
"""
function smoothen(noisy::AbstractVector, b; pad = :true)
    l = length(noisy)
    n = l - b
    smooth = Vector{promote_type(eltype(noisy),Float64)}(undef, pad ? l : n)
    offset = pad ? b÷2 : 0 # use offset only if pad == :true
    for i in 1:n
        smooth[i + offset] = 1/b * sum(noisy[i:i+b-1])
    end
    if pad # pad the vector g at both ends
        smooth[1:offset] .= smooth[offset+1]
        smooth[offset+n+1 : end] .= smooth[offset+n]
    end
    return smooth
end

"""
    gW(norm::AbstractArray, shift::AbstractArray, dt, [b]; pad = :true) -> g
    gW(df::DataFrame, [b]; pad = :true) -> g
Compute the growth witness
```math
G^{(n)} = S^{(n)} - \\frac{\\vert\\mathbf{c}^{(n+1)}\\vert - \\vert\\mathbf{c}^{(n)}\\vert}{\\vert\\mathbf{c}^{(n)}\\vert d\\tau},
```
where `S` is the `shift` and \$\\vert\\mathbf{c}^{(n)}\\vert ==\$ `norm[n, 1]`.
Setting `b ≥ 1` a sliding average over `b` time steps is computed.

If `pad` is set to `:false` then the returned array `g` has the length `length(norm) - b`.
If set to `:true` then `g` will be padded up to the same length as `norm` and `shift`.
"""
function gW(norm::AbstractArray, shift::AbstractArray, dt)
    l = length(norm)
    @assert length(shift) == l "`norm` and `shift` arrays need to have the same length."
    n = l - 1
    g = Vector{eltype(shift)}(undef, l)
    for i in 1:n
        g[i] = shift[i] - (norm[i+1] - norm[i])/(dt*norm[i])
    end
    # pad the vector g at the end
    g[n+1] = g[n]
    return g
end
function gW(norm::AbstractArray, shift::AbstractArray, dt, b; pad = :true)
    g_raw = gW(norm, shift, dt)
    return smoothen(g_raw, b; pad)
end
gW(df::DataFrame) = gW(df.norm, df.shift, df.dτ[1])
gW(df::DataFrame, b; pad = :true) = gW(df.norm, df.shift, df.dτ[1], b; pad=pad)


@doc raw"""
    cov_bare(x, y; xmean = mean(x), ymean = mean(y))
Compute the bare covariance between collections `x` and `y` returning a scalar:
```math
\frac{1}{n}\sum_{i=1}^{n} (x_i - \bar{x})(y_{i} - \bar{y})
```
Optionally,
precomputed means can be passed as keyword arguments. `cov_bare(x,y)` is functionally
equivalent to `Statistics.cov(x, conj(y); corrected = false)` but it is found to be
significantly faster and avoids allocations.
"""
@inline function cov_bare(x, y; xmean = mean(x), ymean = mean(y))
    n = length(x)
    @assert length(y) == n
    res = zero(promote_type(eltype(x), eltype(y))) / 1
    for i = 1:n
        @inbounds res += (x[i] - xmean) * (y[i] - ymean)
    end
    return res / n
end


@doc raw"""
    crosscov_FP(x,y; [maxlag])
Variant of a cross-covariance function inspired by estimator for a correlation function
in [Flyvberg & Petersen (1989)](http://aip.scitation.org/doi/10.1063/1.457480). The optional
keyword argument `maxlag` determines the length of the returned array as `maxlag+1`.
The function accepts complex-valued vectors for `x` and `y`, which must have the same
length ``n``. The returned vector the contains the values of
```math
\tilde{\gamma}(h) = \frac{1}{n-h}\sum_{i=1}^{n-h} (x_i - \bar{x})(y_{i+h} - \bar{y})
```
for lags of `h = 0:maxlag`.
"""
function crosscov_FP(x, y; maxlag = min(length(x), 10 * floor(Int, log10(length(x)))))
    n = length(x)
    @assert length(y) == n
    lags = 0:maxlag # possible values of h
    xmean = mean(x)
    ymean = mean(y)
    T = typeof(zero(promote_type(eltype(x), eltype(y))) / 1)
    ccv = zeros(T, length(lags))
    for (ih, h) in enumerate(lags)
        ccv[ih] = cov_bare(@view(x[1:n-h]), @view(y[1+h:n]); xmean, ymean)
    end
    return ccv
end

end # module Blocking
