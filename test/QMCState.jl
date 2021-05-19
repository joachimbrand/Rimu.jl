using Rimu
using Test
using Rimu.DictVectors: Initiator, SimpleInitiator, CoherentInitiator, IsStochastic2Pop
using Rimu.StatsTools
using Rimu.ConsistentRNG: seedCRNG!
using KrylovKit
using Suppressor
using Statistics

@testset "lomc!/QMCState" begin
    @testset "Setting laststep" begin
        add = BoseFS{5,2}((2,3))
        H = HubbardReal1D(add; u=0.1)
        dv = DVec(add => 1; style=IsStochastic())

        df, state = lomc!(H, copy(dv); laststep=100)
        @test size(df, 1) == 100
        @test state.replicas[1].params.step == 100

        df, state = lomc!(H, copy(dv); laststep=200)
        @test size(df, 1) == 200
        @test state.replicas[1].params.step == 200

        df, state = lomc!(H, copy(dv); laststep=13)
        @test size(df, 1) == 13
        @test state.replicas[1].params.step == 13
    end

    @testset "Setting walkernumber" begin
        add = BoseFS{2,5}((0,0,2,0,0))
        H = HubbardMom1D(add; u=0.5)
        dv = DVec(add => 1; style=IsStochasticWithThreshold(1.0))

        s_strat = DoubleLogUpdate(ζ=0.05, ξ=0.05^2/4, targetwalkers=100)
        walkers = lomc!(H, copy(dv); s_strat, laststep=1000).df.norm
        @test median(walkers) ≈ 100 rtol=0.1

        s_strat = DoubleLogUpdate(ζ=0.05, ξ=0.05^2/4, targetwalkers=200)
        walkers = lomc!(H, copy(dv); s_strat, laststep=1000).df.norm
        @test median(walkers) ≈ 200 rtol=0.1

        s_strat = DoubleLogUpdate(ζ=0.05, ξ=0.05^2/4, targetwalkers=1000)
        walkers = lomc!(H, copy(dv); s_strat, laststep=1000).df.norm
        @test median(walkers) ≈ 1000 rtol=0.1
    end

    @testset "Replicas" begin
        add = nearUniform(BoseFS{5,15})
        H = HubbardReal1D(add)
        dv = DVec(add => 1, style=IsDynamicSemistochastic())

        df, state = lomc!(H, dv; num_replicas=2, operator=H)
        @test ("xHy", "xdoty") ⊆ names(df)
        @test any(!iszero, df.xdoty)
        @test any(!iszero, df.xHy)
        @test length(state.replicas) == 2
        @test state.replicas[1].params.step == state.replicas[2].params.step
        @test state.replicas[1].v ≠ state.replicas[2].v
        @test df.shift_1 ≠ df.shift_2
    end

    @testset "Dead population" begin
        add = BoseFS{5,2}((2,3))
        H = HubbardReal1D(add; u=20)
        dv = DVec(add => 1; style=IsStochastic())

        # Only population is dead.
        df = @suppress_err lomc!(H, copy(dv); laststep=100).df
        @test size(df, 1) < 100

        # Populations in replicas are dead.
        df = @suppress_err lomc!(H, copy(dv); laststep=100, num_replicas=5).df
        @test size(df, 1) < 100
    end

    @testset "Setting `maxlength`" begin
        add = BoseFS{15,10}((0,0,0,0,0,15,0,0,0,0))
        H = HubbardMom1D(add; u=6.0)
        dv = DVec(add => 1; style=IsDynamicSemistochastic())

        df = @suppress_err lomc!(H, copy(dv); maxlength=10, dτ=1e-4).df
        @test all(df.len[1:end-1] .≤ 10)
        @test df.len[end] > 10

        df = @suppress_err lomc!(H, copy(dv); maxlength=10, dτ=1e-4, num_replicas=6).df
        @test all(df.len_1[1:end-1] .≤ 10)
        @test all(df.len_2[1:end-1] .≤ 10)
        @test all(df.len_3[1:end-1] .≤ 10)
        @test all(df.len_4[1:end-1] .≤ 10)
        @test all(df.len_5[1:end-1] .≤ 10)
        @test all(df.len_6[1:end-1] .≤ 10)
    end

    @testset "Continuations" begin
        add = BoseFS{5,5}((1,1,1,1,1))
        H = HubbardReal1D(add; u=0.5)
        # Using Deterministic to get exact same result
        dv = DVec(add => 1, style=IsDeterministic())

        # Run lomc!, then change laststep and continue.
        df, state = lomc!(H, copy(dv); threading=false)
        state.replicas[1].params.laststep = 200
        df1 = lomc!(state, df).df

        # Run lomc! with laststep already set.
        df2 = lomc!(H, copy(dv); laststep=200).df

        @test df1.len ≈ df2.len
        @test df1.norm ≈ df2.norm
        @test df1.shift ≈ df2.shift
    end

    @testset "Errors" begin
        add = BoseFS{5,2}((2,3))
        H = HubbardReal1D(add; u=20)
        dv = DVec(add => 1; style=IsStochastic())

        @test_throws ErrorException lomc!(H, dv; num_replicas=1, operator=H)
        @test_throws ErrorException lomc!(H, dv; num_replicas=-1)
    end
end

@testset "Ground state energy estimates" begin
    for H in (
        HubbardReal1D(BoseFS((1,1,2))),
        BoseHubbardReal1D2C(BoseFS2C((1,2,2), (0,1,0))),
        BoseHubbardMom1D2C(BoseFS2C((0,1), (1,0))),
    )
        @testset "$H, $style" begin
            dv = DVec(starting_address(H) => 2; style=IsDynamicSemistochastic())
            r_strat = EveryTimeStep(projector=copy(dv))

            E0 = eigsolve(H, copy(dv), 1, :SR; issymmetric=true)[1][1]

            df = lomc!(H, dv; r_strat, laststep=3000).df

            # Shift estimate.
            Es, σs = mean_and_se(df.shift)
            s_low, s_high = Es - 2σs, Es + 2σs
            # Projected estimate.
            r = ratio_of_means(df.hproj, df.vproj)
            p_low, p_high = quantile(r, [0.0015, 0.9985])

            @test s_low < E0 < s_high
            @test p_low < E0 < p_high
        end
    end

    @testset "Stochastic style comparison" begin
        add = BoseFS{5,5}((1,1,1,1,1))
        H = HubbardReal1D(add)
        E0 = -8.280991746582686

        seedCRNG!(1234)
        dv_st = DVec(add => 1; style=IsStochastic())
        dv_th = DVec(add => 1; style=IsStochasticWithThreshold(1.0))
        dv_cx = DVec(add => 1; style=IsStochastic2Pop())
        dv_dy = DVec(add => 1; style=IsDynamicSemistochastic())
        dv_de = DVec(add => 1; style=IsDeterministic())

        s_strat = DoubleLogUpdate(ζ=0.05, ξ=0.05^2/4, targetwalkers=100)
        df_st = lomc!(H, dv_st; s_strat, laststep=2500).df
        df_th = lomc!(H, dv_th; s_strat, laststep=2500).df
        df_cx = lomc!(H, dv_cx; s_strat, laststep=2500).df
        df_dy = lomc!(H, dv_dy; s_strat, laststep=2500).df
        df_de = lomc!(H, dv_de; s_strat, laststep=2500).df

        @test ("spawns", "deaths", "clones", "antiparticles", "annihilations") ⊆ names(df_st)
        @test ("spawns", "deaths", "clones", "antiparticles", "annihilations") ⊆ names(df_cx)
        @test ("spawns", "deaths") ⊆ names(df_th)
        @test ("exact_steps", "inexact_steps", "spawns") ⊆ names(df_dy)
        @test ("exact_steps",) ⊆ names(df_de)

        E_st, σ_st = mean_and_se(df_st.shift[500:end])
        E_th, σ_th = mean_and_se(df_th.shift[500:end])
        E_cx, σ_cx = mean_and_se(df_cx.shift[500:end])
        E_dy, σ_dy = mean_and_se(df_dy.shift[500:end])
        E_de, σ_de = mean_and_se(df_de.shift[500:end])

        # Stochastic noise depends on the method.
        @test σ_cx > σ_st > σ_th > σ_dy > σ_de
        # All estimates are fairly good.
        @test E_st ≈ E0 atol=3σ_st
        @test E_th ≈ E0 atol=3σ_th
        @test E_cx ≈ E0 atol=3σ_cx
        @test E_dy ≈ E0 atol=3σ_dy
        @test E_de ≈ E0 atol=3σ_de
    end

    @testset "Initiator energies" begin
        add = BoseFS{10,10}((0,0,0,0,10,0,0,0,0,0))
        H = HubbardMom1D(add)
        E0 = -16.36048582876015

        dv_no = DVec(
            add => 1;
            style=IsDynamicSemistochastic()
        )
        dv_i1 = InitiatorDVec(
            add => 1;
            initiator=Initiator(1),
            style=IsDynamicSemistochastic(),
        )
        dv_i2 = InitiatorDVec(
            add => 1;
            initiator=SimpleInitiator(1),
            style=IsDynamicSemistochastic(),
        )
        dv_i3 = InitiatorDVec(
            add => 1;
            initiator=CoherentInitiator(1),
            style=IsDynamicSemistochastic(),
        )

        @testset "Energies below the plateau & initiator bias" begin
            seedCRNG!(8008)

            s_strat = DoubleLogUpdate(ζ=0.05, ξ=0.05^2/4, targetwalkers=50)
            laststep = 2500
            dτ = 1e-3
            df_no = lomc!(H, copy(dv_no); s_strat, laststep, dτ).df
            df_i1 = lomc!(H, copy(dv_i1); s_strat, laststep, dτ).df
            df_i2 = lomc!(H, copy(dv_i2); s_strat, laststep, dτ).df
            df_i3 = lomc!(H, copy(dv_i3); s_strat, laststep, dτ).df

            E_no, σ_no = mean_and_se(df_no.shift[500:end])
            E_i1, σ_i1 = mean_and_se(df_i1.shift[500:end])
            E_i2, σ_i2 = mean_and_se(df_i2.shift[500:end])
            E_i3, σ_i3 = mean_and_se(df_i3.shift[500:end])

            # Garbage energy from no initiator.
            @test E_no < E0
            # Initiator has a bias.
            @test E_i1 > E0 - σ_i1
            @test E_i2 > E0 - σ_i2
            @test E_i3 > E0 - σ_i3

            # Simple initiator has the largest bias.
            @test E_i2 > E_i1
            # Normal and coherent initiators are about the same.
            @test E_i1 ≈ E_i3 atol=max(3σ_i1, 3σ_i3)
        end

        @testset "Energies above the plateau" begin
            seedCRNG!(1337)

            s_strat = DoubleLogUpdate(ζ=0.05, ξ=0.05^2/4, targetwalkers=3000)
            laststep = 2500
            dτ = 1e-2
            df_no = lomc!(H, copy(dv_no); s_strat, laststep, dτ).df
            df_i1 = lomc!(H, copy(dv_i1); s_strat, laststep, dτ).df
            df_i2 = lomc!(H, copy(dv_i2); s_strat, laststep, dτ).df
            df_i3 = lomc!(H, copy(dv_i3); s_strat, laststep, dτ).df

            E_no, σ_no = mean_and_se(df_no.shift[500:end])
            E_i1, σ_i1 = mean_and_se(df_i1.shift[500:end])
            E_i2, σ_i2 = mean_and_se(df_i2.shift[500:end])
            E_i3, σ_i3 = mean_and_se(df_i3.shift[500:end])

            # All estimates should be fairly good.
            @test E_no ≈ E0 atol=3σ_no
            @test E_i1 ≈ E0 atol=3σ_i1
            @test E_i2 ≈ E0 atol=3σ_i2
            @test E_i3 ≈ E0 atol=3σ_i3
        end
    end
end
