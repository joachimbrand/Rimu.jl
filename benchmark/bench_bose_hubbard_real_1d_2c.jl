module BenchBoseHubbardMom1D2C

using Rimu
using BenchmarkTools

include("helpers.jl")
suite = BenchmarkGroup()

addr1 = BoseFS2C(nearUniform(BoseFS{16,16}),nearUniform(BoseFS{1,16}))
H1 = BoseHubbardReal1D2C(addr1; ua=6.0, ub=6.0, ta=1.0, tb=1.0, v=6.0)
suite["small"] = @benchmarkable run_lomc(
    $H1, $addr1; n_steps=2000, Ntw=2000, dτ=1e-3
) samples=3 seconds=100

addr2 = BoseFS2C(nearUniform(BoseFS{64,128}), nearUniform(BoseFS{16,128}))
H2 = BoseHubbardReal1D2C(addr2; ua=6.0, ub=6.0, ta=1.0, tb=1.0, v=6.0)
suite["big"] = @benchmarkable run_lomc(
    $H2, $addr2; n_steps=300, Ntw=2000, dτ=1e-5
) samples=3 seconds=100

end
BenchBoseHubbardMom1D2C.suite
