module BenchBoseHubbardMom1D2C

using Rimu
using BenchmarkTools


suite = BenchmarkGroup()

function run_lomc(vectype, H, addr, n_steps, dτ)
    Nw_init = 10
    Ntw = 2_000
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

addr = BoseFS2C(BoseFS((0, 4, 0, 0)),BoseFS((0, 4, 0, 0)))
H = BoseHubbardMom1D2C(addr; ua=6.0, ub=6.0, ta=1.0, tb=1.0, v=6.0)

suite["DVec small"] = @benchmarkable run_lomc(
    $DVec, $H, $addr, $20000, 1e-3
) samples=4 seconds=100
suite["DVec2 small"] = @benchmarkable run_lomc(
    $DVec2, $H, $addr, $20000, 1e-3
) samples=3 seconds=100

addr = BoseFS2C(BoseFS((0, 8, 0, 0, 0, 0, 0, 0)),BoseFS((0, 8, 0, 0, 0, 0, 0, 0)))
H = BoseHubbardMom1D2C(addr; ua=6.0, ub=6.0, ta=1.0, tb=1.0, v=6.0)

suite["DVec big"] = @benchmarkable run_lomc(
    $DVec, $H, $addr, $8000, 1e-5
) samples=4 seconds=100
suite["DVec2 big"] = @benchmarkable run_lomc(
    $DVec2, $H, $addr, $8000, 1e-5
) samples=3 seconds=100

end
BenchBoseHubbardMom1D2C.suite
