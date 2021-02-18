module BenchBoseHubbardReal1D

using Rimu
using BenchmarkTools

include("helpers.jl")
suite = BenchmarkGroup()

addr1 = nearUniform(BoseFS{32,32})
H1 = BoseHubbardReal1D(addr1; u=6.0, t=1.0)
suite["small"] = @benchmarkable run_lomc(
    $H1, $addr1; n_steps=5000, Ntw=2000, dτ=1e-3
) samples=3 seconds=100

addr2 = nearUniform(BoseFS{100,200})
H2 = BoseHubbardReal1D(addr2; u=6.0, t=1.0)
suite["big"] = @benchmarkable run_lomc(
    $H2, $addr2; n_steps=3000, Ntw=2000, dτ=1e-5
) samples=3 seconds=100

end
BenchBoseHubbardReal1D.suite
