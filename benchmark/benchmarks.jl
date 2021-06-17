using Rimu
using BenchmarkTools

const SUITE = @benchmarkset "Rimu" begin
    @case "(10, 20) Mom space with projected energy and initiator" begin
        add = BoseFS(ntuple(i -> ifelse(i == 10, 10, 0), 20))
        ham = HubbardMom1D(add, u=6.0)
        dv = InitiatorDVec(add => 1.0; style=IsDynamicSemistochastic())
        post_step = ProjectedEnergy(ham, dv)
        s_strat = DoubleLogUpdate(targetwalkers=20_000)

        lomc!(ham, dv; s_strat, post_step, dτ=1e-4, laststep=8000)
    end
    @case "(4+1, 11) 2C Mom space with G2Correlators" begin
        add = BoseFS2C(ntuple(i -> ifelse(i == 5, 4, 0), 11), ntuple(==(5), 11))
        ham = BoseHubbardMom1D2C(add, v=0.1)
        dv = DVec(add => 1.0; style=IsDynamicSemistochastic())
        s_strat = DoubleLogUpdate(targetwalkers=10_000)
        replica = AllOverlaps(2, ntuple(i -> G2Correlator(i - 1), 7))

        lomc!(ham, dv; s_strat, replica, laststep=4000)
    end
    @case "(50, 50) Real space" begin
        add = nearUniform(BoseFS{50,50})
        ham = HubbardReal1D(add, u=6.0)
        dv = DVec(add => 1.0; style=IsDynamicSemistochastic())
        s_strat = DoubleLogUpdate(targetwalkers=50_000)

        lomc!(ham, dv; s_strat, dτ=1e-4, laststep=4000)
    end
end
