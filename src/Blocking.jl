"""
`Blocking`

Module that contains functions performing the Flyvbjerg-Petersen
(J. Chem. Phys. 91, 461 (1989)) blocking analysis for evaluating
the standard error on a correlated data set. A "M-test" is also
implemented based on Jonsson (Phys. Rev. E 98, 043304, (2018)).
"""
module Blocking

using DataFrames, Statistics

export autocovariance, covariance
export blocker, blocking, blockingErrorEstimation, mtest
export autoblock, blockAndMTest
export growthWitness, gW

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
se(v::Vector;corrected::Bool=true) = std(v;corrected=corrected)/sqrt(length(v))

"""
    blocker(v::Vector) -> new_v::Vector
Reblock the data by successively taking the mean of two adjacent data points to
form a new vector with a half of the `length(v)`. The last data point will be
discarded if `length(v)` is odd.
"""
function blocker(v::Vector{T}) where T
    P = promote_type(T, Float64)
    new_v = Array{P}(undef,(length(v)÷2))
    for i  in 1:length(v)÷2
        new_v[i] = 0.5*(v[2i-1]+v[2i])
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
function autocovariance(v::Vector,h::Int; corrected::Bool=true)
    n = length(v)
    mean_v = mean(v)
    covsum = zero(mean_v)
    for i in 1:n-h
        covsum += (v[i]-mean_v)*conj(v[i+h]-mean_v)
    end
    gamma = covsum/ifelse(corrected, (n-h), n)
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
function blocking(v::Vector{T}; corrected::Bool=true) where T
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

"""
    covariance(x::Vector,y::Vector; corrected::Bool=true)
Calculate the covariance between the two data sets `x` and `y` with equal length.
If `corrected` is `true` (the default) then the sum is scaled with
 `n-1`, whereas the sum is scaled with `n` if corrected is `false`
 where `n = length(x) = length(y)`.
"""
function covariance(vi::Vector,vj::Vector; corrected::Bool=true)
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
    cov = covsum/ifelse(corrected, (n-1), n)
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
function combination_division(vi::Vector,vj::Vector; corrected::Bool=true)
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
    # the χ^2 99 percentiles
    q = [6.634897,  9.210340,  11.344867, 13.276704, 15.086272,
        16.811894, 18.475307, 20.090235, 21.665994, 23.209251,
        24.724970, 26.216967, 27.688250, 29.141238, 30.577914,
        31.999927, 33.408664, 34.805306, 36.190869, 37.566235,
        38.932173, 40.289360, 41.638398, 42.979820, 44.314105,
        45.641683, 46.962942, 48.278236, 49.587884, 50.892181]
    Mj = df.M
    M = reverse(cumsum(reverse(Mj)))
    #println(M)
    k = 1
    while k <= length(M)-1
       if M[k] < q[k]
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
    dfp = blocking(df.hproj[start:stop], df.vproj[start:stop];corrected=corrected) # projected
    return (s̄ = s̄, σs = σs, ē = dfp.mean_f[1], σe = dfp.SE_f[ks], k = ks)
end

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
    l = length(norm)
    @assert length(g_raw) == l "`norm` and `shift` arrays need to have the same length."
    n = l - b
    g = Vector{eltype(shift)}(undef, pad ? l : n)
    offset = pad ? b÷2 : 0 # use offset only if pad == :true
    for i in 1:n
        g[i + offset] = 1/b * sum(g_raw[i:i+b-1])
    end
    if pad # pad the vector g at both ends
        g[1:offset] .= g[offset]
        g[offset+n+1 : end] .= g[offset+n]
    end
    return g
end
gW(df::DataFrame) = gW(df.norm, df.shift, df.dτ[1])
gW(df::DataFrame, b; pad = :true) = gW(df.norm, df.shift, df.dτ[1], b; pad=pad)

end # module Blocking
