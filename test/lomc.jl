using Rimu
using Test
using Rimu.DictVectors: Initiator, SimpleInitiator, CoherentInitiator, NonInitiator
using Rimu.StochasticStyles: IsStochastic2Pop, Bernoulli, WithoutReplacement
using Rimu.StochasticStyles: ThresholdCompression
using Rimu.StatsTools
using Random
using KrylovKit
using Suppressor
using Statistics
using Logging
using DataFrames
using Setfield
import Tables

Random.seed!(1234)
@testset "lomc!/ReplicaState" begin
    @testset "AbstractMatrix" begin
        @test_throws ArgumentError lomc!([1 2; 3 4])
    end

    @testset "Setting laststep + working memory" begin
        address = BoseFS{5,2}((2,3))
        H = HubbardReal1D(address; u=0.1)
        dv = DVec(address => 1; style=IsStochasticInteger())

        # test passing working memory
        v = copy(dv)
        wm = copy(dv)
        df, state = lomc!(H, v; wm, laststep=9)
        @test_broken state.spectral_states[1].single_states[1].wm === wm # after number of steps divisible by 3
        @test state_vectors(state)[1] === v
        @test state.spectral_states[1].single_states[1].pv !== v
        @test state.spectral_states[1].single_states[1].pv !== wm

        df = lomc!(state, df, laststep=10).df
        @test_broken state.spectral_states[1].single_states[1].v === wm
        @test state.spectral_states[1].single_states[1].pv === v

        @test size(df, 1) == 10
        @test state.step[] == 10

        df, state = lomc!(state, df, laststep=100)
        @test size(df, 1) == 100

        state.step[] = 0
        df, state = lomc!(state, df)
        @test size(df, 1) == 200
        @test df.step == [1:100; 1:100]
    end

    @testset "Setting dτ and shift" begin
        address = BoseFS{5,2}((2,3))
        H = HubbardReal1D(address; u=0.1)
        dv = DVec(address => 1; style=IsStochasticInteger())
        df, state = @test_logs (:warn, Regex("(Simulation)")) lomc!(H, dv; laststep=0, shift=23.1, dτ=0.002)
        @test state.spectral_states[1].single_states[1].shift_parameters.time_step  == 0.002
        @test state.spectral_states[1].single_states[1].shift_parameters.shift == 23.1
        @test state.replica_strategy == NoStats{1}() # uses getfield method
    end
    @testset "default_starting_vector" begin
        addr = BoseFS{5,2}((2,3))
        H = HubbardReal1D(addr; u=0.1)
        @test default_starting_vector(H) == default_starting_vector(addr)
        addr2 = BoseFS{5,2}((3, 2))
        @test default_starting_vector(H, address=addr2) == default_starting_vector(addr2)
        @test default_starting_vector(addr; threading=false) isa DVec
        @test default_starting_vector(addr; threading=true) isa PDVec
        v = default_starting_vector(addr; threading=true)
        # @test_logs (:warn, Regex("(Starting)")) lomc!(H, v; laststep=1, threading=false)
        # @test_logs (:warn, Regex("(Starting)")) lomc!(H, v; laststep=1, style=IsStochasticInteger())
    end
    @testset "Setting walkernumber" begin
        address = BoseFS{2,5}((0,0,2,0,0))
        H = HubbardMom1D(address; u=0.5)
        dv = DVec(address => 1; style=IsStochasticWithThreshold(1.0))

        s_strat = DoubleLogUpdate(ζ=0.05, ξ=0.05^2/4, target_walkers=100)
        v = copy(dv)
        walkers = lomc!(H, v; s_strat, laststep=1000).df.norm
        @test median(walkers) ≈ 100 rtol=0.1
        s_strat = LogUpdate(0.05)
        walkers = lomc!(H, v; s_strat, laststep=1000).df.norm # continuation run
        @test median(walkers) > 10 # essentially just test that it does not error

        s_strat = DoubleLogUpdate(ζ=0.05, ξ=0.05^2/4, target_walkers=200)
        walkers = lomc!(H, copy(dv); s_strat, laststep=1000).df.norm
        @test median(walkers) ≈ 200 rtol=0.1

        s_strat = DoubleLogUpdate(ζ=0.05, ξ=0.05^2/4, target_walkers=1000)
        walkers = lomc!(H, copy(dv); s_strat, laststep=1000).df.norm
        @test median(walkers) ≈ 1000 rtol=0.1

        _, state = @test_logs (:warn, Regex("(Simulation)")) lomc!(H, copy(dv); targetwalkers=500, laststep=0)
        @test only(state).algorithm.shift_strategy.target_walkers == 500
    end

    @testset "Replicas" begin
        address = near_uniform(BoseFS{5,15})
        H = HubbardReal1D(address)
        G = GutzwillerSampling(H, g=1)
        @testset "NoStats" begin
            dv = DVec(address => 1, style=IsDynamicSemistochastic())
            df, state = lomc!(H, dv; replica_strategy=NoStats(1))
            @test state.replica_strategy == NoStats(1)
            @test length(state.spectral_states) == 1
            @test "shift" ∈ names(df)
            @test "shift_1" ∉ names(df)

            df, state = lomc!(H, dv; replica_strategy=NoStats(3))
            @test state.replica_strategy == NoStats(3)
            @test length(state.spectral_states) == 3
            @test df.shift_1 ≠ df.shift_2 && df.shift_2 ≠ df.shift_3
            @test "shift_4" ∉ names(df)

            @test isnothing(Rimu.check_transform(NoStats(), H))
        end

        # column names are of the form c{i}_dot_c{j} and c{i}_Op{k}_c{j}.
        function num_stats(df)
            return length(filter(x -> match(r"^c[0-9]", x) ≠ nothing, names(df)))
        end
        @testset "AllOverlaps" begin
            for dv in (
                DVec(address => 1, style=IsDynamicSemistochastic()),
                PDVec(address => 1, style=IsDynamicSemistochastic()),
            )

                # No operator: N choose 2 reports.
                df, _ = lomc!(H, dv; replica_strategy=AllOverlaps(4))
                @test num_stats(df) == binomial(4, 2)
                df, _ = lomc!(H, dv; replica_strategy=AllOverlaps(5))
                @test num_stats(df) == binomial(5, 2)

                # No vector norm: N choose 2 reports.
                df, _ = lomc!(H, dv; replica_strategy=AllOverlaps(4; operator=H, vecnorm=false))
                @test num_stats(df) == binomial(4, 2)
                df, _ = lomc!(H, dv; replica_strategy=AllOverlaps(5; operator=H, vecnorm=false))
                @test num_stats(df) == binomial(5, 2)

                # No operator, no vector norm: 0 reports.
                df, _ = lomc!(H, dv; replica_strategy=AllOverlaps(4; vecnorm=false))
                @test num_stats(df) == 0
                df, _ = lomc!(H, dv; replica_strategy=AllOverlaps(5; vecnorm=false))
                @test num_stats(df) == 0

                # One operator: 2 * N choose 2 reports.
                df, _ = lomc!(H, dv; replica_strategy=AllOverlaps(4; operator=H))
                @test num_stats(df) == 2 * binomial(4, 2)
                df, _ = lomc!(H, dv; replica_strategy=AllOverlaps(5; operator=H))
                @test num_stats(df) == 2 * binomial(5, 2)

                # Two operators: 3 * N choose 2 reports.
                df, _ = lomc!(H, dv; replica_strategy=AllOverlaps(2; operator=(G, H)))
                @test num_stats(df) == 3 * binomial(2, 2)
                df, _ = lomc!(H, dv; replica_strategy=AllOverlaps(7; operator=(G, H)))
                @test num_stats(df) == 3 * binomial(7, 2)
                df, _ = lomc!(H, dv; replica_strategy=AllOverlaps(7; operator=[G, H]))
                @test num_stats(df) == 3 * binomial(7, 2)

                # Transformed operators: (3 + 1) * N choose 2 reports.
                df, _ = lomc!(G, dv; replica_strategy=AllOverlaps(2; operator=(H, G), transform=G))
                @test num_stats(df) == 4 * binomial(2, 2)
                df, _ = lomc!(G, dv; replica_strategy=AllOverlaps(7; operator=(H, G), transform=G))
                @test num_stats(df) == 4 * binomial(7, 2)

                # Check transformation
                # good transform - no warning
                @test_logs min_level=Logging.Warn Rimu.check_transform(AllOverlaps(; operator=H, transform=G), G)
                # no operators - no warning
                @test_logs min_level=Logging.Warn Rimu.check_transform(AllOverlaps(;), H)
                # Hamiltonian transformed and operators not transformed
                @test_logs (:warn, Regex("(Expected overlaps)")) Rimu.check_transform(AllOverlaps(; operator=H), G)
                # Hamiltonian not transformed and operators transformed
                @test_logs (:warn, Regex("(Expected overlaps)")) Rimu.check_transform(AllOverlaps(; operator=H, transform=G), H)
                # Different transformations
                @test_logs (:warn, Regex("(not consistent)")) Rimu.check_transform(AllOverlaps(; operator=H, transform=GutzwillerSampling(H, 0.5)), G)
            end
        end
        @testset "AllOverlaps special cases" begin
            # Complex operator
            v = DVec(1 => 1)
            G = MatrixHamiltonian(rand(5, 5))
            O = MatrixHamiltonian(rand(ComplexF64, 5, 5))
            df, _ = lomc!(G, v, replica_strategy=AllOverlaps(2; operator=O))
            @test df.c1_dot_c2 isa Vector{ComplexF64}
            @test df.c1_Op1_c2 isa Vector{ComplexF64}
        end
    end

    @testset "Dead population" begin
        address = BoseFS{5,2}((2,3))
        H = HubbardReal1D(address; u=20)
        dv = DVec(address => 10; style=IsStochasticInteger())

        # Only population is dead.
        params = RunTillLastStep(shift = 0.0)
        df = @suppress_err lomc!(H, copy(dv); params, laststep=100).df
        @test size(df, 1) < 100

        # population does not die with sensible default shift
        df = lomc!(H, copy(dv); laststep=100).df
        @test size(df, 1) == 100

        # Populations in replicas are dead.
        params = RunTillLastStep(shift = 0.0)
        df = @suppress_err lomc!(H, copy(dv); params, laststep=100, replica_strategy=NoStats(5)).df
        @test size(df, 1) < 100
    end

    @testset "Default DVec" begin
        address = BoseFS{5,2}((2,3))
        H = HubbardReal1D(address; u=20)
        df, state = lomc!(H; laststep=100)
        @test StochasticStyle(state_vectors(state)[1]) isa IsStochasticInteger

        df, state = lomc!(H; laststep=100, style = IsDeterministic())
        @test StochasticStyle(state_vectors(state)[1]) isa IsDeterministic

        df, state = lomc!(H; laststep=1, threading=false, initiator=Initiator())
        @test state_vectors(state)[1] isa InitiatorDVec
    end

    @testset "ShiftStrategy" begin
        address = BoseFS{5,2}((2,3))
        H = HubbardReal1D(address; u=20)

        # DontUpdate
        s_strat = DontUpdate(target_walkers = 100)
        df = lomc!(H; s_strat, laststep=100).df
        @test size(df, 1) < 100 # finish early without error

        # LogUpdateAfterTargetWalkers
        s_strat = LogUpdateAfterTargetWalkers(target_walkers = 100)
        df, state  = lomc!(H; s_strat, laststep=100)
        @test size(df, 1) == 100
        @test df.shift_mode[end] # finish in variable shift mode
        @test df.norm[end] > 100

        # LogUpdate
        s_strat = DoubleLogUpdate(target_walkers=100)
        df, state = lomc!(H; s_strat, laststep=100)
        @test size(df, 1) == 100

        v = state_vectors(state)[1]
        step = state.step[]
        s_strat = LogUpdate()
        df = lomc!(H, v; df, step, s_strat, laststep=200).df
        @test size(df, 1) == 200
        @test 500 > df.norm[end] > 100

        # DoubleLogUpdateAfterTargetWalkers
        s_strat = DoubleLogUpdateAfterTargetWalkers(target_walkers = 100)
        df, state  = lomc!(H; s_strat, laststep=100)
        @test size(df, 1) == 100
        @test df.shift_mode[end] # finish in variable shift mode
        @test df.norm[end] > 100

        # test unexported strategies
        # DoubleLogSumUpdate
        s_strat = Rimu.DoubleLogSumUpdate(target_walkers = 100)
        df, state  = lomc!(H; s_strat, laststep=100)
        @test size(df, 1) == 100

        # DoubleLogProjected
        s_strat = Rimu.DoubleLogProjected(target = 100.0, projector=UniformProjector())
        df, state  = lomc!(H; s_strat, laststep=100)
        @test size(df, 1) == 100
    end

    @testset "Setting `maxlength`" begin
        address = BoseFS{15,10}((0,0,0,0,0,15,0,0,0,0))
        H = HubbardMom1D(address; u=6.0)
        dv = PDVec(address => 1; style=IsDynamicSemistochastic())

        Random.seed!(1336)

        df = @suppress_err lomc!(H, copy(dv); maxlength=10, dτ=1e-4).df
        @test all(df.len[1:end-1] .≤ 10)
        @test df.len[end] > 10

        df, state = @suppress_err lomc!(H, copy(dv); maxlength=10, dτ=1e-4, replica_strategy=NoStats(6))
        @test all(df.len_1[1:end-1] .≤ 10)
        @test all(df.len_2[1:end-1] .≤ 10)
        @test all(df.len_3[1:end-1] .≤ 10)
        @test all(df.len_4[1:end-1] .≤ 10)
        @test all(df.len_5[1:end-1] .≤ 10)
        @test all(df.len_6[1:end-1] .≤ 10)

        state.maxlength[] += 1000
        df_cont = lomc!(state).df
        @test size(df_cont, 1) == 100 - size(df, 1)
    end

    @testset "Continuations" begin
        address = BoseFS{5,5}((1,1,1,1,1))
        H = HubbardReal1D(address; u=0.5)
        # Using Deterministic to get exact same result
        dv = PDVec(address => 1.0, style=IsDeterministic())

        # Run lomc!, then change laststep and continue.
        df, state = lomc!(H, copy(dv))
        # @set state.simulation_plan.last_step = 200
        df1 = lomc!(state, df, laststep=200).df

        # Run lomc! with laststep already set.
        df2 = lomc!(H, copy(dv); laststep=200).df

        @test df1.len ≈ df2.len
        @test df1.norm ≈ df2.norm
        @test df1.shift ≈ df2.shift
    end

    @testset "Reporting" begin
        address = BoseFS((1,2,1,1))
        H = HubbardReal1D(address; u=2)
        dv = PDVec(address => 1, style=IsDeterministic())

        @testset "ReportDFAndInfo" begin
            reporting_strategy = ReportDFAndInfo(reporting_interval=5, info_interval=10, io=devnull, writeinfo=true)
            df = lomc!(H, copy(dv); reporting_strategy, laststep=100).df
            @test size(df, 1) == 20
            @test metadata(df, "Rimu.PACKAGE_VERSION") == string(Rimu.PACKAGE_VERSION)

            out = @capture_out begin
                reporting_strategy = ReportDFAndInfo(reporting_interval=5, info_interval=10, io=stdout, writeinfo=true)
                lomc!(H, copy(dv); reporting_strategy, laststep=100)
            end
            @test length(split(out, '\n')) == 3 # (last line is empty)
        end
        @testset "ReportToFile" begin
            # Clean up.
            rm("test-report.arrow"; force=true)
            rm("test-report-1.arrow"; force=true)
            rm("test-report-2.arrow"; force=true)
            rm("test-report-3.arrow"; force=true)
            rm("test-report-nc.arrow"; force=true)
            rm("test-report-lz4.arrow"; force=true)

            reporting_strategy = ReportToFile(filename="test-report.arrow", io=devnull, save_if=false)
            df = lomc!(H, copy(dv); reporting_strategy, laststep=100).df
            @test !isfile("test-report.arrow")
            @test Rimu._isopen(reporting_strategy) == false

            reporting_strategy = ReportToFile(filename="test-report.arrow", io=devnull)
            df = lomc!(H, copy(dv); reporting_strategy, laststep=100, metadata=(;u=6.0)).df
            @test isempty(df)
            @test Rimu._isopen(reporting_strategy) == false
            df1 = RimuIO.load_df("test-report.arrow")
            @test metadata(df1, "u") == "6.0" # custom metadata is saved
            @test metadata(df1, "filename") == "test-report.arrow" # filename in metadata

            reporting_strategy = ReportToFile(filename="test-report.arrow", io=devnull, chunk_size=5)
            df = lomc!(H, copy(dv); reporting_strategy, laststep=100).df
            @test isempty(df)
            @test Rimu._isopen(reporting_strategy) == false
            df2 = RimuIO.load_df("test-report-1.arrow")

            reporting_strategy = ReportToFile(filename="test-report.arrow", io=devnull, return_df=true)
            df3 = lomc!(H, copy(dv); reporting_strategy, laststep=100).df
            @test isempty(df)
            @test Rimu._isopen(reporting_strategy) == false
            df4 = RimuIO.load_df("test-report-2.arrow")

            @test df1.shift ≈ df2.shift
            @test df2.norm ≈ df3.norm
            @test df3 == df4

            # ReportToFile with skipping interval
            df5 = df1[10:10:100,:]
            reporting_strategy = ReportToFile(filename="test-report.arrow", reporting_interval=10, io=devnull, chunk_size=10)
            df = lomc!(H, copy(dv); reporting_strategy, laststep=100).df
            @test isempty(df)
            df6 = RimuIO.load_df("test-report-3.arrow")

            @test df6.shift ≈ df5.shift
            @test df6.norm ≈ df5.norm

            # ReportToFile with compression
            @test_throws ArgumentError ReportToFile(compress=false)

            reporting_strategy = ReportToFile(
                filename="test-report-nc.arrow", io=devnull, return_df=true,
                compress=nothing
            )
            df7 = lomc!(H, copy(dv); reporting_strategy, laststep=100).df
            @test isempty(df)
            @test Rimu._isopen(reporting_strategy) == false
            @test df7 == RimuIO.load_df("test-report-nc.arrow")


            reporting_strategy = ReportToFile(
                filename="test-report-lz4.arrow", io=devnull, return_df=true,
                compress=:lz4
            )
            df8 = lomc!(H, copy(dv); reporting_strategy, laststep=100).df
            @test isempty(df)
            @test Rimu._isopen(reporting_strategy) == false
            @test df8 == RimuIO.load_df("test-report-lz4.arrow")

            @test filesize("test-report-lz4.arrow") < filesize("test-report-nc.arrow")
            @test filesize("test-report.arrow") < filesize("test-report-lz4.arrow")
            # The default compression `:zstd` produces the smallest files.

            # Clean up.
            rm("test-report.arrow"; force=true)
            rm("test-report-1.arrow"; force=true)
            rm("test-report-2.arrow"; force=true)
            rm("test-report-3.arrow"; force=true)
            rm("test-report-nc.arrow"; force=true)
            rm("test-report-lz4.arrow"; force=true)
        end
        @testset "Report" begin
            rp = Rimu.Report()
            Rimu.report!(rp, :b, 4)
            Rimu.report!(rp, :b, 6)
            Rimu.report_metadata!(rp, :a, 1)
            @test Rimu.get_metadata(rp, "a") == "1"
            @test sprint(show, rp) == "Report:\n  b => [4, 6]\n metadata:\n  a => 1"

            # Tables integration
            NamedTuple(first(Tables.rows(rp))) == (b=4,)
            Tables.schema(rp) isa Tables.Schema
        end
    end

    @testset "Post step" begin
        address = BoseFS((0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0))
        H = HubbardMom1D(address; u=4)
        dv = DVec(address => 1)

        @testset "Projector, ProjectedEnergy" begin
            Random.seed!(1337)

            post_step_strategy = (
                Projector(p1=NormProjector()),
                Projector(p2=copy(dv)),
                ProjectedEnergy(H, dv),
                ProjectedEnergy(H, dv, vproj=:vproj2, hproj=:hproj2),
                ProjectedEnergy(H, UniformProjector(), vproj=:vproj3, hproj=:hproj3),
            )
            df, _ = lomc!(H, copy(dv); post_step_strategy)
            @test df.vproj == df.vproj2 == df.p2
            @test df.norm ≈ df.p1
            @test df.norm ≥ df.vproj3

            @test_throws ArgumentError lomc!(
                H, dv; post_step_strategy=(Projector(a=dv), Projector(a=dv))
            )
            @test_throws ArgumentError Projector(a=dv, b=dv)
            @test_throws ArgumentError Projector()
        end

        @testset "SignCoherence" begin
            Random.seed!(1337)

            ref = eigsolve(H, dv, 1, :SR; issymmetric=true)[2][1]
            post_step_strategy = (SignCoherence(ref), SignCoherence(dv * -1, name=:single_coherence))
            df, _ = lomc!(H, copy(dv); post_step_strategy)
            @test df.coherence[1] == 1.0
            @test all(-1.0 .≤ df.coherence .≤ 1.0)
            @test all(in.(df.single_coherence, Ref((-1, 0, 1))))

            cdv = DVec(address => 1 + im)
            df, _ = lomc!(H, cdv; post_step_strategy)
            @test df.coherence isa Vector{ComplexF64}
        end

        @testset "WalkerLoneliness" begin
            Random.seed!(1337)

            post_step_strategy = WalkerLoneliness()
            df, _ = lomc!(H, copy(dv); post_step_strategy)
            @test df.loneliness[1] == 1
            @test all(1 .≥ df.loneliness .≥ 0)

            cdv = DVec(address => 1 + im)
            df, _ = lomc!(H, cdv; post_step_strategy)
            @test df.loneliness isa Vector{ComplexF64}
        end

        @testset "Timer" begin
            post_step_strategy = Rimu.Timer()
            time_before = time()
            df, _ = lomc!(H, copy(dv); post_step_strategy)
            time_after = time()
            @test df.time[1] > time_before
            @test df.time[end] < time_after
            @test issorted(df.time)
        end

        @testset "SingleParticleDensity" begin
            post_step_strategy = (
                SingleParticleDensity(save_every=2),
            )
            df, st = lomc!(H, copy(dv); post_step_strategy)
            @test all(==(ntuple(_ -> 0, num_modes(address))), df.single_particle_density[1:2:end])
            @test all(≈(3), sum.(df.single_particle_density[2:2:end]))

            @test df.single_particle_density[end] == single_particle_density(
                st.spectral_states[1].single_states[1].v
            )

            for address in (
                BoseFS2C((1,2,3), (0,1,0)),
                CompositeFS(BoseFS((1,2,3)), FermiFS((0,1,0)))
            )
                @test single_particle_density(address) == (1, 3, 3)
                @test single_particle_density(address; component=1) == (1, 2, 3)
                @test single_particle_density(address; component=2) == (0, 1, 0)
                @test single_particle_density(DVec(address => 1); component=2) == (0, 7, 0)
            end
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
            post_step_strategy = ProjectedEnergy(H, dv)

            E0 = eigsolve(H, copy(dv), 1, :SR; issymmetric=true)[1][1]

            df = lomc!(H, dv; post_step_strategy, laststep=3000).df

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
        address = BoseFS{5,5}((1,1,1,1,1))
        H = HubbardReal1D(address)
        E0 = -8.280991746582686

        Random.seed!(1234)
        dv_st = DVec(address => 1; style=IsStochasticInteger())
        dv_th = DVec(address => 1; style=IsStochasticWithThreshold(1.0))
        dv_cx = DVec(address => 1 + im; style=IsStochastic2Pop())
        dv_dy = DVec(address => 1; style=IsDynamicSemistochastic())
        dv_de = DVec(address => 1; style=IsDeterministic())
        dv_dp = DVec(address => 1; style=IsDeterministic(ThresholdCompression()))

        dv_nr = DVec(address => 1; style=IsDynamicSemistochastic(spawning=WithoutReplacement()))
        dv_br = DVec(address => 1; style=IsDynamicSemistochastic(spawning=Bernoulli()))

        s_strat = DoubleLogUpdate(ζ=0.05, ξ=0.05^2/4, target_walkers=100)
        s_strat_cx = DoubleLogUpdate(ζ=0.05, ξ=0.05^2/4, target_walkers=100 + 100im)
        df_st = lomc!(H, dv_st; s_strat, laststep=2500).df
        df_th = lomc!(H, dv_th; s_strat, laststep=2500).df
        df_cx = lomc!(H, dv_cx; s_strat=s_strat_cx, laststep=2500).df
        df_dy = lomc!(H, dv_dy; s_strat, laststep=2500).df
        df_de = lomc!(H, dv_de; s_strat, laststep=2500).df
        df_dp = lomc!(H, dv_dp; s_strat, laststep=2500).df

        df_nr = lomc!(H, dv_nr; s_strat, laststep=2500).df
        df_br = lomc!(H, dv_br; s_strat, laststep=2500).df

        @test ("spawns", "deaths", "clones", "zombies") ⊆ names(df_st)
        @test ("spawns", "deaths", "clones", "zombies") ⊆ names(df_cx)
        @test "spawns" ∈ names(df_th)
        @test ("exact_steps", "inexact_steps", "spawns") ⊆ names(df_dy)
        @test "exact_steps" ∈ names(df_de)

        @test ("exact_steps", "len_before") ⊆ names(df_dp)
        @test ("exact_steps", "len_before") ⊆ names(df_br)
        @test ("exact_steps", "len_before") ⊆ names(df_nr)
        @test "len_before" ∉ names(df_st)
        @test "len_before" ∉ names(df_th)
        @test "len_before" ∉ names(df_cx)
        @test "len_before" ∉ names(df_de)
        @test all(>(0), df_dp.len_before)
        @test all(df_dp.len_before .≥ df_dp.len)

        E_st, σ_st = mean_and_se(df_st.shift[500:end])
        E_th, σ_th = mean_and_se(df_th.shift[500:end])
        E_cx, σ_cx = mean_and_se(real.(df_cx.shift[500:end]))
        E_dy, σ_dy = mean_and_se(df_dy.shift[500:end])
        E_de, σ_de = mean_and_se(df_de.shift[500:end])
        E_dp, σ_dp = mean_and_se(df_dp.shift[500:end])

        E_nr, σ_nr = mean_and_se(df_nr.shift[500:end])
        E_br, σ_br = mean_and_se(df_br.shift[500:end])

        # Stochastic noise depends on the method. Sampling without replacement makes a
        # small difference and is not consistently lower, so is not included here. A similar
        # thing happens with deterministic with compression and explosive spawns.
        @test σ_st > σ_th > σ_dy
        # All estimates are fairly good.
        @test E_st ≈ E0 atol=3σ_st
        @test E_th ≈ E0 atol=3σ_th
        @test E_cx ≈ E0 atol=3σ_cx
        @test E_dy ≈ E0 atol=3σ_dy
        @test E_de ≈ E0 atol=3σ_de
        @test E_dp ≈ E0 atol=3σ_dp
        @test E_nr ≈ E0 atol=3σ_nr
        @test E_br ≈ E0 atol=3σ_br
    end

    @testset "Initiator energies" begin
        address = BoseFS{10,10}((0,0,0,0,10,0,0,0,0,0))
        dv_no = DVec(
            address => 1;
            style=IsDynamicSemistochastic()
        )
        dv_i1 = InitiatorDVec(
            address => 1;
            initiator=Initiator(1),
            style=IsDynamicSemistochastic(),
        )
        dv_i2 = InitiatorDVec(
            address => 1;
            initiator=SimpleInitiator(1),
            style=IsDynamicSemistochastic(),
        )
        dv_i3 = InitiatorDVec(
            address => 1;
            initiator=CoherentInitiator(1),
            style=IsDynamicSemistochastic(),
        )
        dv_ni = InitiatorDVec(
            address => 1;
            initiator=NonInitiator(),
            style=IsDynamicSemistochastic(),
        )

        @testset "Energies below the plateau & initiator bias" begin
            Random.seed!(8008)

            H = HubbardMom1D(address; u=4.0)
            E0 = -9.251592973178997

            s_strat = DoubleLogUpdate(ζ=0.05, ξ=0.05^2/4, target_walkers=300)
            laststep = 6_000
            dτ = 5e-4
            df_no = lomc!(H, copy(dv_no); s_strat, laststep, dτ).df
            df_i1 = lomc!(H, copy(dv_i1); s_strat, laststep, dτ).df
            df_i2 = lomc!(H, copy(dv_i2); s_strat, laststep, dτ).df
            df_i3 = lomc!(H, copy(dv_i3); s_strat, laststep, dτ).df
            df_ni = lomc!(H, copy(dv_ni); s_strat, laststep, dτ).df

            E_no, σ_no = mean_and_se(df_no.shift[2000:end])
            E_i1, σ_i1 = mean_and_se(df_i1.shift[2000:end])
            E_i2, σ_i2 = mean_and_se(df_i2.shift[2000:end])
            E_i3, σ_i3 = mean_and_se(df_i3.shift[2000:end])
            E_ni, σ_ni = mean_and_se(df_ni.shift[2000:end])

            # Garbage energy from no initiator.
            @test E_no < E0
            @test E_ni < E0
            @test E_no ≈ E_ni atol=3 * σ_no
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
            Random.seed!(1337)

            H = HubbardMom1D(address)
            E0 = -16.36048582876015

            s_strat = DoubleLogUpdate(ζ=0.05, ξ=0.05^2/4, target_walkers=3000)
            laststep = 2500
            dτ = 1e-2
            df_no = lomc!(H, copy(dv_no); s_strat, laststep, dτ).df
            df_i1 = lomc!(H, copy(dv_i1); s_strat, laststep, dτ).df
            df_i2 = lomc!(H, copy(dv_i2); s_strat, laststep, dτ).df
            df_i3 = lomc!(H, copy(dv_i3); s_strat, laststep, dτ).df
            df_ni = lomc!(H, copy(dv_ni); s_strat, laststep, dτ).df

            E_no, σ_no = mean_and_se(df_no.shift[500:end])
            E_i1, σ_i1 = mean_and_se(df_i1.shift[500:end])
            E_i2, σ_i2 = mean_and_se(df_i2.shift[500:end])
            E_i3, σ_i3 = mean_and_se(df_i3.shift[500:end])
            E_ni, σ_ni = mean_and_se(df_ni.shift[500:end])

            # All estimates should be fairly good.
            @test E_no ≈ E0 atol=3σ_no
            @test E_i1 ≈ E0 atol=3σ_i1
            @test E_i2 ≈ E0 atol=3σ_i2
            @test E_i3 ≈ E0 atol=3σ_i3
            @test E_ni ≈ E0 atol=3σ_ni
        end
    end
end
