writetofile = false
showplots = true

using Feather, DataFrames, Rimu
showplots && using PyPlot
showplots && pygui(true)

filename = "ba.out"
io = stdout
writetofile && (io = open(filename,"w"))


# look for "plotting.jl" in the same folder as this script
scriptsdir = @__DIR__
include(joinpath(scriptsdir,"plotting.jl"))

# read data
df = Feather.read("fciqmcdata.feather")
println(io, "Number of time steps: ", size(df,1)-1)

# plot stats and energy
showplots && plotQMCStats(df)
showplots && plotQMCEnergy(df)

# analyse energy data
start_blocking = 1_000 # put a meaningful value in
println(io, "Starting blocking analysis at time step ", start_blocking)
ba_shift  = blocking(df[start_blocking:end,:shift])
qmcEnergy = ba_shift[1,:mean]
qmcEnergyError = ba_shift[mtest(ba_shift),:std_err]
block_steps = df[start_blocking:end,:steps]
showplots && fill_between(block_steps,qmcEnergy-qmcEnergyError,
    qmcEnergy+qmcEnergyError,facecolor="m",alpha=0.3)
showplots && plot(block_steps,ones(length(block_steps))*qmcEnergy,"--m")
println(io, "Final result for ground state eigenvalue (from shift): $qmcEnergy ± $qmcEnergyError")

writetofile && close(io)
