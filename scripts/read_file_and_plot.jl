using Feather, DataFrames
using PyPlot
pygui(true)

# look for "plotting.jl" in the same folder as this script
scriptsdir = @__DIR__
include(joinpath(scriptsdir,"plotting.jl"))

df = Feather.read("fciqmcdata.feather")

plotQMCStats(df)
plotQMCEnergy(df)
