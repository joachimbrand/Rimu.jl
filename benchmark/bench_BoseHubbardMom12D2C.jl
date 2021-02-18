module BenchBoseHubbardMom1D2C

using Rimu
using BenchmarkTools

include("helpers.jl")
suite = BenchmarkGroup()

addr1 = BoseFS2C(BoseFS((0, 8, 0, 0, 0, 0, 0, 0)),BoseFS((0, 8, 0, 0, 0, 0, 0, 0)))
H1 = BoseHubbardMom1D2C(addr1; ua=6.0, ub=6.0, ta=1.0, tb=1.0, v=6.0)
suite["small"] = @benchmarkable run_lomc(
    $H1, $addr1; n_steps=6000, Ntw=2000, dτ=1e-5
) samples=3 seconds=100

addr2 = BoseFS2C(nearUniform(BoseFS{64,128}), nearUniform(BoseFS{16,128}))
H2 = BoseHubbardMom1D2C(addr2; ua=6.0, ub=6.0, ta=1.0, tb=1.0, v=6.0)
suite["big"] = @benchmarkable run_lomc(
    $H2, $addr2; n_steps=200, Ntw=2000, dτ=1e-5
) samples=3 seconds=100

end
BenchBoseHubbardMom1D2C.suite
