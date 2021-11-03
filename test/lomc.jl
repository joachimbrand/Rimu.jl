using Rimu
using Test
using Rimu.DictVectors: Initiator, SimpleInitiator, CoherentInitiator
using Rimu.StochasticStyles: IsStochastic2Pop, Bernoulli, WithoutReplacement, IsExplosive
using Rimu.StochasticStyles: DoubleOrNothing, ThresholdCompression
using Rimu.StatsTools
using Rimu.ConsistentRNG: seedCRNG!
using Rimu.RMPI
using KrylovKit
using Suppressor
using Statistics

@testset "lomc!/QMCState" begin
    @testset "Setting laststep" begin
        add = BoseFS{5,2}((2,3))
        H = HubbardReal1D(add; u=0.1)
        dv = DVec(add => 1; style=IsStochasticInteger())

        df, state = lomc!(H, copy(dv); laststep=100)
        @test size(df, 1) == 100
        @test state.replicas[1].params.step == 100

        df, state = lomc!(H, copy(dv); laststep=200)
        @test size(df, 1) == 200
        @test state.replicas[1].params.step == 200

        df, state = lomc!(H, copy(dv); laststep=13)
        @test size(df, 1) == 13
        @test state.replicas[1].params.step == 13

        state.laststep = 100
        df = lomc!(state, df).df
        @test size(df, 1) == 100

        state.step = 0
        df = lomc!(state, df).df
        @test size(df, 1) == 200
        @test df.steps == [1:100; 1:100]
    end

    @testset "Setting walkernumber" begin
        add = BoseFS{2,5}((0,0,2,0,0))
        H = HubbardMom1D(add; u=0.5)
        dv = DVec(add => 1; style=IsStochasticWithThreshold(1.0))

        s_strat = DoubleLogUpdate(ζ=0.05, ξ=0.05^2/4, targetwalkers=100)
        v = copy(dv)
        walkers = lomc!(H, v; s_strat, laststep=1000).df.norm
        @test median(walkers) ≈ 100 rtol=0.1
        s_strat = LogUpdate(0.05)
        walkers = lomc!(H, v; s_strat, laststep=1000).df.norm # continuation run
        @test median(walkers) > 10 # essentially just test that it does not error

        s_strat = DoubleLogUpdate(ζ=0.05, ξ=0.05^2/4, targetwalkers=200)
        walkers = lomc!(H, copy(dv); s_strat, laststep=1000).df.norm
        @test median(walkers) ≈ 200 rtol=0.1

        s_strat = DoubleLogUpdate(ζ=0.05, ξ=0.05^2/4, targetwalkers=1000)
        walkers = lomc!(H, copy(dv); s_strat, laststep=1000).df.norm
        @test median(walkers) ≈ 1000 rtol=0.1
    end

    @testset "Replicas" begin
        add = near_uniform(BoseFS{5,15})
        H = HubbardReal1D(add)
        dv = DVec(add => 1, style=IsDynamicSemistochastic())

        @testset "NoStats" begin
            df, state = lomc!(H, dv; replica=NoStats(1))
            @test state.replica == NoStats(1)
            @test length(state.replicas) == 1
            @test "shift" ∈ names(df)
            @test "shift_1" ∉ names(df)

            df, state = lomc!(H, dv; replica=NoStats(3))
            @test state.replica == NoStats(3)
            @test length(state.replicas) == 3
            @test df.shift_1 ≠ df.shift_2 && df.shift_2 ≠ df.shift_3
            @test "shift_4" ∉ names(df)
        end

        @testset "AllOverlaps" begin
            # column names are of the form c{i}_dot_c{j} and c{i}_Op{k}_c{j}.
            num_stats(df) = length(filter(x -> match(r"^c[0-9]", x) ≠ nothing, names(df)))

            # No operator: N choose 2 reports.
            df, _ = lomc!(H, dv; replica=AllOverlaps(4))
            @test num_stats(df) == binomial(4, 2)
            df, _ = lomc!(H, dv; replica=AllOverlaps(5))
            @test num_stats(df) == binomial(5, 2)

            # One operator: 2 * N choose 2 reports.
            df, _ = lomc!(H, dv; replica=AllOverlaps(4, H))
            @test num_stats(df) == 2 * binomial(4, 2)
            df, _ = lomc!(H, dv; replica=AllOverlaps(5, H))
            @test num_stats(df) == 2 * binomial(5, 2)

            # Two operators: 3 * N choose 2 reports.
            df, _ = lomc!(H, dv; replica=AllOverlaps(2, (GutzwillerSampling(H, 1), H)))
            @test num_stats(df) == 3 * binomial(2, 2)
            df, _ = lomc!(H, dv; replica=AllOverlaps(7, (GutzwillerSampling(H, 1), H)))
            @test num_stats(df) == 3 * binomial(7, 2)

            # Complex operator
            v = DVec(1 => 1)
            G = MatrixHamiltonian(rand(5, 5))
            O = MatrixHamiltonian(rand(ComplexF64, 5, 5))
            df, _ = lomc!(G, v, replica=AllOverlaps(2, O))
            @test df.c1_dot_c2 isa Vector{ComplexF64}
            @test df.c1_Op1_c2 isa Vector{ComplexF64}

            # MPIData
            df, _ = lomc!(H, MPIData(dv); replica=AllOverlaps(4, H))
            @test num_stats(df) == 2 * binomial(4, 2)
            df, _ = lomc!(H, MPIData(dv); replica=AllOverlaps(5, H))
            @test num_stats(df) == 2 * binomial(5, 2)
        end
    end

    @testset "Dead population" begin
        add = BoseFS{5,2}((2,3))
        H = HubbardReal1D(add; u=20)
        dv = DVec(add => 1; style=IsStochasticInteger())

        # Only population is dead.
        params = RunTillLastStep(shift = 0.0)
        df = @suppress_err lomc!(H, copy(dv); params, laststep=100).df
        @test size(df, 1) < 100

        # population does not die with sensible default shift
        df = lomc!(H, copy(dv); laststep=100).df
        @test size(df, 1) == 100

        # Populations in replicas are dead.
        params = RunTillLastStep(shift = 0.0)
        df = @suppress_err lomc!(H, copy(dv); params, laststep=100, replica=NoStats(5)).df
        @test size(df, 1) < 100
    end

    @testset "Default DVec" begin
        add = BoseFS{5,2}((2,3))
        H = HubbardReal1D(add; u=20)
        df, state = lomc!(H; laststep=100)
        @test StochasticStyle(state.replicas[1].v) isa IsStochasticInteger

        df, state = lomc!(H; laststep=100, style = IsDeterministic())
        @test StochasticStyle(state.replicas[1].v) isa IsDeterministic
    end

    @testset "ShiftStrategy" begin
        add = BoseFS{5,2}((2,3))
        H = HubbardReal1D(add; u=20)

        # DontUpdate
        s_strat = DontUpdate(targetwalkers = 100)
        df = lomc!(H; s_strat, laststep=100).df
        @test size(df, 1) < 100 # finish early without error

        # LogUpdateAfterTargetWalkers
        s_strat = LogUpdateAfterTargetWalkers(targetwalkers = 100)
        df, state  = lomc!(H; s_strat, laststep=100)
        @test size(df, 1) == 100
        @test df.shiftMode[end] # finish in variable shift mode
        @test df.norm[end] > 100

        # LogUpdate
        v = state.replicas[1].v
        params = state.replicas[1].params
        s_strat = LogUpdate()
        df = lomc!(H, v; df, params, s_strat, laststep=200).df
        @test size(df, 1) == 200
        @test 500 > df.norm[end] > 100

        # DoubleLogUpdateAfterTargetWalkers
        s_strat = DoubleLogUpdateAfterTargetWalkers(targetwalkers = 100)
        df, state  = lomc!(H; s_strat, laststep=100)
        @test size(df, 1) == 100
        @test df.shiftMode[end] # finish in variable shift mode
        @test df.norm[end] > 100

        # test unexported strategies
        # DoubleLogSumUpdate
        s_strat = Rimu.DoubleLogSumUpdate(targetwalkers = 100)
        df, state  = lomc!(H; s_strat, laststep=100)
        @test size(df, 1) == 100
        @test df.shiftMode[end] # finish in variable shift mode

        # TripleLogUpdate
        s_strat = Rimu.TripleLogUpdate(targetwalkers = 100)
        df, state  = lomc!(H; s_strat, laststep=100)
        @test size(df, 1) == 100
        @test df.shiftMode[end] # finish in variable shift mode

        # DoubleLogProjected
        s_strat = Rimu.DoubleLogProjected(target = 100.0, projector=UniformProjector())
        df, state  = lomc!(H; s_strat, laststep=100)
        @test size(df, 1) == 100
        @test df.shiftMode[end] # finish in variable shift mode

    end

    @testset "deprecated" begin
        @test_throws ErrorException DelayedLogUpdateAfterTargetWalkers()
        @test_throws ErrorException HistoryLogUpdate()
        @test_throws ErrorException DelayedLogUpdate()
        @test_throws ErrorException DelayedDoubleLogUpdateAfterTW()
        @test_throws ErrorException DoubleLogUpdateAfterTargetWalkersSwitch()
        @test_throws ErrorException DelayedDoubleLogUpdate()
    end

    @testset "Setting `maxlength`" begin
        add = BoseFS{15,10}((0,0,0,0,0,15,0,0,0,0))
        H = HubbardMom1D(add; u=6.0)
        dv = DVec(add => 1; style=IsDynamicSemistochastic())

        seedCRNG!(1336)

        df = @suppress_err lomc!(H, copy(dv); maxlength=10, dτ=1e-4).df
        @test all(df.len[1:end-1] .≤ 10)
        @test df.len[end] > 10

        df, state = @suppress_err lomc!(H, copy(dv); maxlength=10, dτ=1e-4, replica=NoStats(6))
        @test all(df.len_1[1:end-1] .≤ 10)
        @test all(df.len_2[1:end-1] .≤ 10)
        @test all(df.len_3[1:end-1] .≤ 10)
        @test all(df.len_4[1:end-1] .≤ 10)
        @test all(df.len_5[1:end-1] .≤ 10)
        @test all(df.len_6[1:end-1] .≤ 10)

        state.maxlength += 1000
        df_cont = lomc!(state).df
        @test size(df_cont, 1) == 100 - size(df, 1)
    end

    @testset "Continuations" begin
        add = BoseFS{5,5}((1,1,1,1,1))
        H = HubbardReal1D(add; u=0.5)
        # Using Deterministic to get exact same result
        dv = DVec(add => 1.0, style=IsDeterministic())

        # Run lomc!, then change laststep and continue.
        df, state = lomc!(H, copy(dv); threading=false)
        state.laststep = 200
        df1 = lomc!(state, df).df

        # Run lomc! with laststep already set.
        df2 = lomc!(H, copy(dv); laststep=200).df

        @test df1.len ≈ df2.len
        @test df1.norm ≈ df2.norm
        @test df1.shift ≈ df2.shift
    end

    @testset "Reporting" begin
        add = BoseFS((1,2,1,1))
        H = HubbardReal1D(add; u=2)
        dv = DVec(add => 1, style=IsDeterministic())

        @testset "ReportDFAndInfo" begin
            r_strat = ReportDFAndInfo(k=5, i=20, io=devnull, writeinfo=true)
            df = lomc!(H, copy(dv); r_strat, laststep=100).df
            @test size(df, 1) == 20

            out = @capture_out begin
                r_strat = ReportDFAndInfo(k=5, i=20, io=stdout, writeinfo=true)
                lomc!(H, copy(dv); r_strat, laststep=100)
            end
            @test length(split(out, '\n')) == 6 # (last line is empty)
        end
        @testset "ReportToFile" begin
            # Clean up.
            rm("test-report.arrow"; force=true)
            rm("test-report-1.arrow"; force=true)
            rm("test-report-2.arrow"; force=true)

            r_strat = ReportToFile(filename="test-report.arrow", io=devnull, save_if=false)
            df = lomc!(H, copy(dv); r_strat, laststep=100).df
            @test !isfile("test-report.arrow")

            r_strat = ReportToFile(filename="test-report.arrow", io=devnull)
            df = lomc!(H, copy(dv); r_strat, laststep=100).df
            @test isempty(df)
            df1 = RimuIO.load_df("test-report.arrow")

            r_strat = ReportToFile(filename="test-report.arrow", io=devnull, chunk_size=5)
            df = lomc!(H, copy(dv); r_strat, laststep=100).df
            @test isempty(df)
            df2 = RimuIO.load_df("test-report-1.arrow")

            r_strat = ReportToFile(filename="test-report.arrow", io=devnull, return_df=true)
            df3 = lomc!(H, copy(dv); r_strat, laststep=100).df
            @test isempty(df)
            df4 = RimuIO.load_df("test-report-2.arrow")

            @test df1.shift ≈ df2.shift
            @test df2.norm ≈ df3.norm
            @test df3 == df4

            # Clean up.
            rm("test-report.arrow"; force=true)
            rm("test-report-1.arrow"; force=true)
            rm("test-report-2.arrow"; force=true)
        end
    end

    @testset "Post step" begin
        add = BoseFS((0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0))
        H = HubbardMom1D(add; u=4)
        dv = DVec(add => 1)

        @testset "Projector, ProjectedEnergy" begin
            ConsistentRNG.seedCRNG!(1337)

            post_step = (
                Projector(p1=NormProjector()),
                Projector(p2=copy(dv)),
                ProjectedEnergy(H, dv),
                ProjectedEnergy(H, dv, vproj=:vproj2, hproj=:hproj2),
            )
            df, _ = lomc!(H, copy(dv); post_step)
            @test df.vproj == df.vproj2 == df.p2
            @test df.norm ≈ df.p1

            @test_throws ArgumentError lomc!(
                H, dv; post_step=(Projector(a=dv), Projector(a=dv))
            )
            @test_throws ArgumentError Projector(a=dv, b=dv)
            @test_throws ArgumentError Projector()
        end

        @testset "SignCoherence" begin
            ConsistentRNG.seedCRNG!(1337)

            ref = eigsolve(H, dv, 1, :SR; issymmetric=true)[2][1]
            post_step = (SignCoherence(ref), SignCoherence(dv * -1, name=:single_coherence))
            df, _ = lomc!(H, copy(dv); post_step)
            @test df.coherence[1] == 1.0
            @test all(-1.0 .≤ df.coherence .≤ 1.0)
            @test all(in.(df.single_coherence, Ref((-1, 0, 1))))

            cdv = DVec(add => 1 + im)
            df, _ = lomc!(H, cdv; post_step)
            @test df.coherence isa Vector{ComplexF64}
        end

        @testset "WalkerLoneliness" begin
            ConsistentRNG.seedCRNG!(1337)

            post_step = WalkerLoneliness()
            df, _ = lomc!(H, copy(dv); post_step)
            @test df.loneliness[1] == 1
            @test all(1 .≥ df.loneliness .≥ 0)

            cdv = DVec(add => 1 + im)
            df, _ = lomc!(H, cdv; post_step)
            @test df.loneliness isa Vector{ComplexF64}
        end

        @testset "Timer" begin
            post_step = Rimu.Timer()
            time_before = time()
            df, _ = lomc!(H, copy(dv); post_step)
            time_after = time()
            @test df.time[1] > time_before
            @test df.time[end] < time_after
            @test issorted(df.time)
        end
    end

    # only test threading if more than one thread is available
    Threads.nthreads() > 1 && @testset "Threading" begin
        add = BoseFS((0,0,0,0,10,0,0,0,0,0))
        H = HubbardMom1D(add)
        s_strat = DoubleLogUpdate(targetwalkers=10_000)
        laststep = 1000
        for dv in (DVec(add => 1), InitiatorDVec(add => 1), MPIData(DVec(add => 1)))
            @test_throws ArgumentError lomc!(H; threading=:something, laststep, s_strat)

            df1, s1 = lomc!(
                H, deepcopy(dv); threading=:auto, laststep, s_strat
            )
            df2, s2 = lomc!(
                H, deepcopy(dv); threading=true, laststep, s_strat
            )
            df3, s3 = lomc!(
                H, deepcopy(dv); threading=false, laststep, s_strat
            )
            df4, s4 = lomc!(
                H, deepcopy(dv); threading=Rimu.NoThreading(), laststep, s_strat
            )
            df5, s5 = lomc!(
                H, deepcopy(dv); threading=Rimu.ThreadsThreading(), laststep, s_strat
            )
            df6, s6 = lomc!(
                H, deepcopy(dv); threading=Rimu.SplittablesThreading(), laststep, s_strat
            )
            df7, s7 = lomc!(
                H, deepcopy(dv); threading=Rimu.ThreadsXSumThreading(), laststep, s_strat
            )
            df8, s8 = lomc!(
                H, deepcopy(dv); threading=Rimu.ThreadsXForeachThreading(), laststep, s_strat
            )

            @test s1.threading == Rimu.SplittablesThreading()
            @test s2.threading == Rimu.SplittablesThreading()
            @test s3.threading == Rimu.NoThreading()

            # Check that working memory was selected correctly.
            N = Threads.nthreads()
            D = typeof(localpart(dv))
            @test s1.replicas[1].w isa NTuple{N,D}
            @test s2.replicas[1].w isa NTuple{N,D}
            @test s3.replicas[1].w isa D
            @test s4.replicas[1].w isa D
            @test s5.replicas[1].w isa NTuple{N,D}
            @test s6.replicas[1].w isa NTuple{N,D}
            @test s7.replicas[1].w isa NTuple{N,D}
            @test s8.replicas[1].w isa NTuple{N,D}

            # Check energies.
            E1, _ = mean_and_se(df1.shift[900:end])
            E2, _ = mean_and_se(df2.shift[900:end])
            E3, _ = mean_and_se(df3.shift[900:end])
            E4, _ = mean_and_se(df4.shift[900:end])
            E5, _ = mean_and_se(df5.shift[900:end])
            E6, _ = mean_and_se(df6.shift[900:end])
            E7, _ = mean_and_se(df7.shift[900:end])
            E8, _ = mean_and_se(df8.shift[900:end])

            @test E1 ≈ E2 rtol=0.1
            @test E2 ≈ E3 rtol=0.1
            @test E3 ≈ E4 rtol=0.1
            @test E4 ≈ E5 rtol=0.1
            @test E5 ≈ E6 rtol=0.1
            @test E6 ≈ E7 rtol=0.1
            @test E7 ≈ E8 rtol=0.1
        end
    end
end

@testset "Ground state energy estimates" begin
    for H in (
        HubbardReal1D(BoseFS((1,1,2))),
        BoseHubbardReal1D2C(BoseFS2C((1,2,2), (0,1,0))),
        BoseHubbardMom1D2C(BoseFS2C((0,1), (1,0))),
    )
        @testset "$H" begin
            dv = DVec(starting_address(H) => 2; style=IsDynamicSemistochastic())
            post_step = ProjectedEnergy(H, dv)

            E0 = eigsolve(H, copy(dv), 1, :SR; issymmetric=true)[1][1]

            df = lomc!(H, dv; post_step, laststep=3000).df

            # Shift estimate.
            Es, σs = mean_and_se(df.shift)
            s_low, s_high = Es - 2σs, Es + 2σs
            # Projected estimate.
            r = ratio_of_means(df.hproj, df.vproj)
            p_low, p_high = pquantile(r, [0.0015, 0.9985])

            @test s_low < E0 < s_high
            @test p_low < E0 < p_high
        end
    end

    @testset "Stochastic style comparison" begin
        add = BoseFS{5,5}((1,1,1,1,1))
        H = HubbardReal1D(add)
        E0 = -8.280991746582686

        seedCRNG!(1234)
        dv_st = DVec(add => 1; style=IsStochasticInteger())
        dv_th = DVec(add => 1; style=IsStochasticWithThreshold(1.0))
        dv_cx = DVec(add => 1 + im; style=IsStochastic2Pop())
        dv_dy = DVec(add => 1; style=IsDynamicSemistochastic())
        dv_de = DVec(add => 1; style=IsDeterministic())
        dv_dp = DVec(add => 1; style=IsDeterministic(ThresholdCompression()))
        dv_ex = DVec(add => 1; style=IsExplosive())

        dv_dn = DVec(add => 10; style=IsDynamicSemistochastic(compression=DoubleOrNothing()))
        dv_nr = DVec(add => 1; style=IsDynamicSemistochastic(spawning=WithoutReplacement()))
        dv_br = DVec(add => 1; style=IsDynamicSemistochastic(spawning=Bernoulli()))

        s_strat = DoubleLogUpdate(ζ=0.05, ξ=0.05^2/4, targetwalkers=100)
        s_strat_cx = DoubleLogUpdate(ζ=0.05, ξ=0.05^2/4, targetwalkers=100 + 100im)
        df_st = lomc!(H, dv_st; s_strat, laststep=2500).df
        df_th = lomc!(H, dv_th; s_strat, laststep=2500).df
        df_cx = lomc!(H, dv_cx; s_strat=s_strat_cx, laststep=2500).df
        df_dy = lomc!(H, dv_dy; s_strat, laststep=2500).df
        df_de = lomc!(H, dv_de; s_strat, laststep=2500).df
        df_dp = lomc!(H, dv_dp; s_strat, laststep=2500).df
        df_ex = lomc!(H, dv_ex; s_strat, laststep=2500).df

        df_dn = lomc!(H, dv_dn; s_strat, laststep=2500).df
        df_nr = lomc!(H, dv_nr; s_strat, laststep=2500).df
        df_br = lomc!(H, dv_br; s_strat, laststep=2500).df

        @test ("spawns", "deaths", "clones", "zombies", "annihilations") ⊆ names(df_st)
        @test ("spawns", "deaths", "clones", "zombies", "annihilations") ⊆ names(df_cx)
        @test "spawns" ∈ names(df_th)
        @test ("exact_steps", "inexact_steps", "spawns") ⊆ names(df_dy)
        @test "exact_steps" ∈ names(df_de)

        @test ("explosions", "ticks", "normal_steps",
               "explosive_spawns", "normal_spawns") ⊆ names(df_ex)
        @test ("exact_steps", "len_before") ⊆ names(df_dp)
        @test ("exact_steps", "len_before") ⊆ names(df_br)
        @test ("exact_steps", "len_before") ⊆ names(df_nr)
        @test ("exact_steps", "len_before") ⊆ names(df_dn)
        @test "len_before" ∉ names(df_st)
        @test "len_before" ∉ names(df_th)
        @test "len_before" ∉ names(df_cx)
        @test "len_before" ∉ names(df_de)

        E_st, σ_st = mean_and_se(df_st.shift[500:end])
        E_th, σ_th = mean_and_se(df_th.shift[500:end])
        E_cx, σ_cx = mean_and_se(real.(df_cx.shift[500:end]))
        E_dy, σ_dy = mean_and_se(df_dy.shift[500:end])
        E_de, σ_de = mean_and_se(df_de.shift[500:end])
        E_dp, σ_dp = mean_and_se(df_dp.shift[500:end])
        E_ex, σ_ex = mean_and_se(df_dp.shift[500:end])

        E_dn, σ_dn = mean_and_se(df_dn.shift[500:end])
        E_nr, σ_nr = mean_and_se(df_nr.shift[500:end])
        E_br, σ_br = mean_and_se(df_br.shift[500:end])

        # Stochastic noise depends on the method. Sampling without replacement makes a
        # small difference and is not consistently lower, so is not included here. A similar
        # thing happens with deterministic with compression and explosive spawns.
        @test σ_st > σ_th > σ_dn > σ_dy > σ_de
        # All estimates are fairly good.
        @test E_st ≈ E0 atol=3σ_st
        @test E_th ≈ E0 atol=3σ_th
        @test E_cx ≈ E0 atol=3σ_cx
        @test E_dy ≈ E0 atol=3σ_dy
        @test E_de ≈ E0 atol=3σ_de
        @test E_dp ≈ E0 atol=3σ_dp
        @test E_ex ≈ E0 atol=3σ_ex
        @test E_nr ≈ E0 atol=3σ_nr
        @test E_dn ≈ E0 atol=3σ_dn

        # For some reason, Bernoulli requires more walkers
        @test_broken E_br ≈ E0 atol=3σ_br
    end

    @testset "Initiator energies" begin
        add = BoseFS{10,10}((0,0,0,0,10,0,0,0,0,0))
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

            H = HubbardMom1D(add; u=4.0)
            E0 = -9.251592973178997

            s_strat = DoubleLogUpdate(ζ=0.05, ξ=0.05^2/4, targetwalkers=300)
            laststep = 6_000
            dτ = 5e-4
            df_no = lomc!(H, copy(dv_no); s_strat, laststep, dτ).df
            df_i1 = lomc!(H, copy(dv_i1); s_strat, laststep, dτ).df
            df_i2 = lomc!(H, copy(dv_i2); s_strat, laststep, dτ).df
            df_i3 = lomc!(H, copy(dv_i3); s_strat, laststep, dτ).df

            E_no, σ_no = mean_and_se(df_no.shift[2000:end])
            E_i1, σ_i1 = mean_and_se(df_i1.shift[2000:end])
            E_i2, σ_i2 = mean_and_se(df_i2.shift[2000:end])
            E_i3, σ_i3 = mean_and_se(df_i3.shift[2000:end])

            # Garbage energy from no initiator.
            @test E_no < E0
            # Initiator has a bias.
            @test E_i1 > E0
            @test E_i2 > E0
            @test E_i3 > E0

            # Simple initiator has the largest bias.
            @test E_i2 > E_i1
            # Normal and coherent initiators are about the same.
            @test E_i1 ≈ E_i3 atol=max(3σ_i1, 3σ_i3)
        end

        @testset "Energies above the plateau" begin
            seedCRNG!(1337)

            H = HubbardMom1D(add)
            E0 = -16.36048582876015

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
