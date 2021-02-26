module BenchBoseHubbardReal1D

using Rimu
using BenchmarkTools


suite = BenchmarkGroup()

function run_lomc(vectype, H, addr, n_steps=5_000)
    Nw_init = 10
    Ntw = 2_000
    dτ = 0.001
    k = 1

    c = vectype(Dict(addr => Nw_init), Ntw)
    params = RunTillLastStep(; step=0, dτ=dτ, laststep=n_steps)
    s_strat = DoubleLogUpdate(targetwalkers=Ntw, ζ=0.08)
    τ_strat = ConstantTimeStep()
    r_strat = EveryTimeStep(; projector=copytight(c))
    Rimu.ConsistentRNG.seedCRNG!(17)

    return lomc!(
        H, c;
        params=params,
        laststep=n_steps,
        s_strat=s_strat,
        r_strat=r_strat,
        τ_strat=τ_strat
    )
end

addr = nearUniform(BoseFS{8,16})
H = BoseHubbardReal1D(addr; u=6.0, t=1.0)

suite["DVec small"] = @benchmarkable run_lomc(
    $DVec, $H, $addr, $20000
) samples=3 seconds=100
suite["DVec2 small"] = @benchmarkable run_lomc(
    $DVec2, $H, $addr, $20000
) samples=3 seconds=100

addr = nearUniform(BoseFS{32,64})
H = BoseHubbardReal1D(addr; u=6.0, t=1.0)

suite["DVec big"] = @benchmarkable run_lomc(
    $DVec, $H, $addr, $6000
) samples=3 seconds=100
suite["DVec2 big"] = @benchmarkable run_lomc(
    $DVec2, $H, $addr, $6000
) samples=3 seconds=100

end
BenchBoseHubbardReal1D.suite
