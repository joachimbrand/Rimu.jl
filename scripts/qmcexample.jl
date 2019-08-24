using Rimu

include("plotting.jl")
pygui(true)

ham = BoseHubbardReal1D(
    n = 9,
    m = 9,
    u = 6.0,
    t = 1.0,
    AT = BSAdd64)

ham(:dim)
# bit-string address of initial state: choose default
aIni = nearUniform(ham)

# Exact energy with Lanczos method from KrylovKit
cIni = DVec(Dict(aIni=>1.0),ham(:dim))
capacity(cIni)
length(cIni)
using KrylovKit
@time allresults = eigsolve(ham, cIni, 1, :SR; issymmetric = true)
exactEnergy = allresults[1][1]

# set up parameters for simulations
walkernumber = 20_000
steps = 800
dτ = 0.005

# Deterministic FCIQMC
svec2 = DVec(Dict(aIni => 2.0), ham(:dim))
StochasticStyle(svec2)

pa = FCIQMCParams(laststep = steps,  dτ = dτ)
τ_strat = ConstantTimeStep()
s_strat = LogUpdateAfterTargetWalkers(targetwalkers = walkernumber)
v2 = copy(svec2)
@time rdf = fciqmc!(v2, pa, ham, s_strat, τ_strat)

plotQMCStats(rdf)

# stochastic with small walker number
svec = DVec(Dict(aIni => 2), ham(:dim))
StochasticStyle(svec)

pas = FCIQMCParams(laststep = steps, dτ = dτ)
vs = copy(svec)
@time rdfs = fciqmc!(vs, pas, ham, s_strat, τ_strat)

plotQMCStats(rdfs, newfig = false)

# plot energies
# deterministic
plotQMCEnergy(rdf,exactEnergy)

norm_ratio = rdf[2:end,:norm] ./ rdf[1:(end-1),:norm]
Ẽ = rdf[1:end-1,:shift] + (1 .- norm_ratio)./ pa.dτ
plot(rdf[2:end,:steps],Ẽ,".r")

plotQMCEnergy(rdfs, newfig=false)

start_blocking = 400
dfshift = blocking(rdfs[start_blocking:end,:shift])
qmcEnergy = dfshift[1,:mean]
qmcEnergyError = dfshift[mtest(dfshift),:std_err]
block_steps = rdfs[start_blocking:end,:steps]
fill_between(block_steps,qmcEnergy-qmcEnergyError,
    qmcEnergy+qmcEnergyError,facecolor="m",alpha=0.3)
plot(block_steps,ones(length(block_steps))*qmcEnergy,"--m")

using DSP
# define kernel for smoothing
w = 3
gausskernel = [exp(-i^2/(2*w^2)) for i = -3w:3w]
gausskernel ./= sum(abs.(gausskernel))
shift = rdfs[:,:shift]
smooth_shift = conv(shift, gausskernel)[3w+1:end-3w]
plot(rdfs.steps, smooth_shift, ".-g")

# plot energy difference from exact for determinstic
Ediff = abs.(Ẽ .- exactEnergy)
figure(); title("log-lin plot of Ediff")
semilogy(rdf[2:end,:steps],Ediff,"xg")

# plot blocking analysis for manual checking
plotBlockingAnalysisDF(dfshift)
title("Blocking analysis for `shift`")
