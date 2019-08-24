"""
`Blocking`

Module that contains functions performing the Flyvbjerg-Petersen blocking
analysis for evaluating the standard error on a correlated data set.
"""
module Blocking

using DataFrames

export blocker, blocking, blockingErrorEstimation, mtest

"""
Reblock the data by successively taking the mean of adjacent data points
"""
function blocker(v::Vector)
    new_v = Array{Float64}(undef,(length(v)÷2))
    for i  in 1:length(v)÷2
        new_v[i] = 0.5*(v[2i-1]+v[2i])
    end
    return new_v
end

"""
perform a blocking analysis for single data set
"""
function blocking(v::Vector)
    df = DataFrame(blocks = Int[], mean = Float64[], stdev = Float64[],
                    std_err = Float64[], std_err_err = Float64[], gamma = Float64[], M = Float64[])
    while length(v) >= 2
        n = length(v)
        mean = sum(v)/n
        var = v .- mean
        sigmasq = sum(var.^2)/n
        gamma = sum(var[1:n-1].*var[2:n])/n
        mj = n*((n-1)*sigmasq/(n^2)+gamma)^2/(sigmasq^2)
        stddev = sqrt(sigmasq)
        stderr = stddev/sqrt(n-1)
        stderrerr = stderr*1/sqrt(2*(n-1))
        v = blocker(v)
        #println(n, mean, stddev, stderr)
        push!(df,(n, mean, stddev, stderr, stderrerr, gamma, mj))
    end
    return df
end

"""
Calculate the covariance between the two data sets vi and vj.
"""
function covariance(vi::Vector,vj::Vector)
    if length(vi) != length(vj)
        error("Two data sets with non-equal length!")
    else
        n = length(vi)
        meani = sum(vi)/n
        meanj = sum(vj)/n
        covsum = 0
        for i in range(1,length=n)
            covsum += (vi[i]-meani)*(vj[i]-meanj)
        end
        cov = covsum/(n-1)
        return cov
    end
end


"""
find the standard error on standard errors on two datasets
"""
function combination_division(vi::Vector,vj::Vector)
    if length(vi) != length(vj)
        error("Two data sets with non-equal length!")
    else
        n = length(vi)
        meani = sum(vi)/n
        meanj = sum(vj)/n
        meanf = meani/meanj
        sei = se(vi)
        sej = se(vj)
        cov = covariance(vi,vj)
        sef = abs(meanf*sqrt((sei/meani)^2 + (sej/meanj)^2 - 2.0*cov/(n*meani*meanj)))
        return sef
    end
end


"""
perform a blocking analysis for two data sets
"""
function blocking(vi::Vector,vj::Vector)
    df = DataFrame(blocks=Int[], mean_i=Float64[], SD_i=Float64[], SE_i=Float64[], SE_SE_i=Float64[],
            mean_j=Float64[], SD_j=Float64[], SE_j=Float64[], SE_SE_j=Float64[], Covariance=Float64[],
            mean_f=Float64[], SE_f=Float64[])
    if length(vi) != length(vj)
        error("Two data sets with non-equal length!")
    else
        while length(vi) >= 2
            n = length(vi)
            meani = sum(vi)/n
            meanj = sum(vj)/n
            meanf = meani/meanj
            sdi = sd(vi)
            sdj = sd(vj)
            sei = se(vi)
            sej = se(vj)
            sesei = sei*1/sqrt(2*(n-1))
            sesej = sej*1/sqrt(2*(n-1))
            cov = covariance(vi,vj)
            #sef = sei/sej
            sef = combination_division(vi,vj)
            vi = blocker(vi)
            vj = blocker(vj)
            #println(n, mean, stddev, stderr)
            push!(df,(n, meani, sdi, sei, sesei, meanj, sdj, sej, sesej, cov, meanf, sef))
        end
        return df
    end
end

"""
estimating stnadard error from blocking analysis based on the overlapping of
error bars, if all the error bars (or more than 3 on a roll) behind current
one are overlapping with it, return the current standard error with error bar.
"""
function blockingErrorEstimation(df::DataFrame)
    e = df.std_err[1:end-1] # ignoring the last data point
    ee = df.std_err_err[1:end-1] # ignoring the last data point
    n = length(e)
    ind = collect(1:length(e))
    e_upper = map(x->e[x]+ee[x],ind) # upper bounds
    e_lower = map(x->e[x]-ee[x],ind) # lower bounds
    i = 1 # start from the first data point
    plateau = false
    while i < n
        count = 0 # set up a counter for checking overlapped error bars
        for j in (i+1):n # j : all data points after i
            if e_lower[i] >= e_lower[j] && e_upper[i] <= e_upper[j]
                count += 1
                #println("i: ",i," j: ",j," c: ",count)
                # some tolerance, say if there are 3 overlaps on a roll could be a plateau
                if count > 3 && (i + count) == j
                    plateau = true
                    println("\x1b[32mplateau detected\x1b[0m")
                    return e[i], ee[i], plateau
                end
            end
        end # for
        if count == (n-i)
            println("\x1b[32mNO plateau is detected, take the best estimation\x1b[0m")
            return e[i], ee[i], plateau
        else
            i += 1 # move on to next point
        end
    end # while
    println("\x1b[32mNO plateau, NO error bar overlap, take the second last point\x1b[0m")
    return e[i], ee[i], plateau # return the last ponit
end

"""
The "M test" based on Jonsson, M. Physical Review E, 98(4), 043304, (2018).
If the blocking analysis (BA) has passed the M test, an error estimation will be
given based on the smallest k (i.e. meaningful results at the k-th data point
on a BA plot).
"""
function mtest(df::DataFrame)
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
           stder = round(df.std_err[k],digits=5)
           stderer = round(df.std_err_err[k],digits=5)
           println("\x1b[32mM test passed, the smallest k is $k\x1b[0m")
           println("\x1b[32mStandard error estimation: $stder ± $stderer\x1b[0m")
           return k
       else
           k += 1
       end
    end
    if k > length(M)-1
        println("\x1b[32mM test failed, more data needed\x1b[0m")
    end
end

end # module Blocking
