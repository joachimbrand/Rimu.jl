module BenchHubbardMom1D

using Rimu
using BenchmarkTools

include("helpers.jl")
suite = BenchmarkGroup()

addr1 = BoseFS((0, 8, 0, 0, 0, 0, 0, 0))
H1 = HubbardMom1D(addr1; u=6.0, t=1.0)
suite["small"] = @benchmarkable run_lomc(
    $H1, $addr1, n_steps=10000, Ntw=4_000, dτ=1e-3
) samples=3 seconds=100

addr2 = nearUniform(BoseFS{100,100})
H2 = HubbardMom1D(addr2; u=6.0, t=1.0)
suite["big"] = @benchmarkable run_lomc(
    $H2, $addr2, n_steps=500, Ntw=1000, dτ=1e-6
) samples=3 seconds=100

end
BenchHubbardMom1D.suite
