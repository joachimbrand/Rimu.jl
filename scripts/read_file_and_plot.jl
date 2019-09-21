using Feather, DataFrames, Rimu
using PyPlot
pygui(true)

# look for "plotting.jl" in the same folder as this script
scriptsdir = @__DIR__
include(joinpath(scriptsdir,"plotting.jl"))

# read data
df = Feather.read("fciqmcdata.feather")
println("Number of time steps: ", size(df,1)-1)

# plot stats and energy
plotQMCStats(df)
plotQMCEnergy(df)

# analyse energy data 
start_blocking = 2_000 # put a meaningful value in
println("Starting blocking analysis at time step ", start_blocking)
ba_shift  = blocking(df[start_blocking:end,:shift])
qmcEnergy = ba_shift[1,:mean]
qmcEnergyError = ba_shift[mtest(ba_shift),:std_err]
block_steps = df[start_blocking:end,:steps]
fill_between(block_steps,qmcEnergy-qmcEnergyError,
    qmcEnergy+qmcEnergyError,facecolor="m",alpha=0.3)
plot(block_steps,ones(length(block_steps))*qmcEnergy,"--m")
println("Final result for ground state eigenvalue (from shift): $qmcEnergy ± $qmcEnergyError")
