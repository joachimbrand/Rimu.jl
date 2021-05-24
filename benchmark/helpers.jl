using Rimu

"""
    run_lomc(H, addr; n_steps, Ntw, dτ)

Function used to run lomc! benchmarks. Does the necessary setup and runs.
"""
function run_lomc(H, addr; n_steps, Ntw, dτ)
    Nw_init = 10
    k = 1

    c = DVec(Dict(addr => Nw_init))
    params = RunTillLastStep(; step=0, dτ=dτ, laststep=n_steps)
    s_strat = DoubleLogUpdate(targetwalkers=Ntw, ζ=0.08)
    τ_strat = ConstantTimeStep()
    r_strat = EveryTimeStep(; projector=copy(c))
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
