# plotting scripts

using PyPlot
# pygui(true)

function plotQMCStats(df::DataFrame; newfig=true)
    newfig && figure()
    title("QMC stats")
    plot(df.steps,df.norm,"xb")
    plot(df.steps,df.len,".g")
    plot(df.steps,df.spawns,"or")
    plot(df.steps,df.deaths,"+k")
    plot(df.steps,df.clones,"+m")
    plot(df.steps,df.antiparticles,"xk")
    legend(["1 norm","configs","spawns","deaths","clones","antiparticles"])
    xlabel("time steps")
end

function plotQMCEnergy(df::DataFrame, refE::Real = NaN; newfig=true)
    newfig && figure()
    title("QMC energy")
    plot(df.steps,ones(length(df.steps)).*refE,"--g")
    plot(df.steps,df.shift,".b")
    legend(["exact","shift"])

    # if isnan(refE)
    #     legend("shift")
    # else
    #     legend("exact","shift")
    # end
    xlabel("time steps")
end

function plotCombinedStats(rs)
    nreplicas = length(rs)
    f = figure()
    title("combined  stats from $nreplicas replicas")
    # todo: make this work for arbitrary number of replcas
    df1 = rs[1]
    df2 = rs[2]
    plot(df1.steps,df1.norm,"xb")
    plot(df2.steps,df2.norm,"xk")
    plot(df1.steps,df1.len,".g")
    plot(df2.steps,df2.len,".k")
    legend(["norm r1","norm r2","configs r1","configs r2"])
    xlabel("time steps")
    return f
end
function plotCombinedEnergy(mdf::DataFrame, rs, refE = NaN)
    f = figure()
    title("combined QMC energy")
    df1 = rs[1]
    df2 = rs[2]
    plot(df1.steps,ones(length(df1.steps)).*refE,"--g")
    plot(df1.steps,df1.shift,".b")
    plot(df2.steps,df2.shift,"+k")
    plot(mdf.steps,mdf.aveH,"xm")
    legend(["exact","shift 1","shift 2","mixed"])
    xlabel("time steps")
    return f
end

"""
    linear_fit(d, t, w = ones(t))
Fit `d` to the linear relationship
`d = α + β t`
by minimising the least square error weighted by `w`.
"""
function linear_fit(d, t, w = ones(length(t)))
    a = sum(t.*d.*w)
    b = sum(d.*w)
    c = sum(w)
    d = sum(t.*w)
    e = sum(t.^2 .* w)
    f = sum(d.^2 .* w)
    β = (a*c - d*b)/(e*c - d^2)
    α = (a - β*e)/d
    return α, β
end

"""
plotting the blocking analysis results from a dataframe
"""
function plotBlockingAnalysisDF(df::DataFrame)
    println(df)
    blocks = df.blocks
    se = df.std_err
    se_err = df.std_err_err
    figure()
    semilogx(blocks, se, "go-", basex=2, markersize=3.5)
    errorbar(blocks, se, ecolor="g", yerr=se_err,capsize=3,fmt="none")
    xlim(maximum(blocks),minimum(blocks))
    xlabel("Number of blocks")
    ylabel("Standard error")
    legend(["σ std.err","σ(σ) std.err.err"])
    autoscale()
end
