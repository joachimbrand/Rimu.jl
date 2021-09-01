using Rimu
using Test
using Rimu.StochasticStyles

using Rimu.StochasticStyles: projected_deposit!, diagonal_step!, spawn!
using Rimu.StochasticStyles:
    Exact, SingleSpawn, WithReplacement, WithoutReplacement, Bernoulli, DynamicSemistochastic

@testset "Generic Hamiltonian-free functions" begin
    matrix = [1 2 3; 4 5 6; 7 8 9]
    @test diagonal_element(matrix, 3) == 9
    @test offdiagonals(matrix, 1) == [2 => 4, 3 => 7]
    @test offdiagonals(matrix, 2) == [1 => 2, 3 => 8]

    add, prob, val = random_offdiagonal(matrix, 1)
    @test add in (2, 3)
    @test prob == 1/2
    @test val in (4, 7)

    vec = [1, 2, 3, 4, 5]
    deposit!(vec, 1, 2, 1 => 2)
    deposit!(vec, 4, -2, 1 => 2)
    @test vec == [3, 2, 3, 2, 5]

    @test StochasticStyle(vec) == IsStochasticInteger{Int64}()
    @test StochasticStyle(Float32.(vec)) == IsDeterministic{Float32}()

    names, values = step_stats(vec, Val(2))
    @test names == (:spawn_attempts, :spawns, :deaths, :clones, :zombies, :annihilations)
    @test length(values) == 2
    @test values[1] == values[2] == Rimu.MultiScalar((0, 0, 0, 0, 0, 0))

    w = [1.0, 2.0, 3.0]

    @test fciqmc_col!(w, matrix, 1, 2, 0.1, 0.2) == (1, )
    @test w[1] == 1.0 + (1 + 0.2 * (0.1 - matrix[1, 1])) * 2
    @test w[2] == 2.0 - 0.2 * matrix[2, 1] * 2
    @test w[3] == 3.0 - 0.2 * matrix[3, 1] * 2
end

@testset "projected_deposit!" begin
    @testset "Integer" begin
        for _ in 1:20
            dv = DVec(:a => 1)
            @test projected_deposit!(dv, :a, 0.5, :a => 1) isa Tuple{Int, Int}
            @test dv[:a] == 1 || dv[:a] == 2

            @test projected_deposit!(dv, :b, -5.5, :a => 1) isa Tuple{Int, Int}
            @test dv[:b] == -5 || dv[:b] == -6

            for i in 1:13
                @test projected_deposit!(dv, :c, 1, :a => 1) isa Tuple{Int, Int}
            end
            @test dv[:c] == 13
        end
    end

    @testset "Exact" begin
        for _ in 1:20
            dv = DVec(:a => 1.0)
            projected_deposit!(dv, :a, -1, :a => 1.0)
            @test isempty(dv)

            projected_deposit!(dv, :a, 1e-9, :a => 1.0)
            projected_deposit!(dv, :b, -1 - 1e-9, :a => 1.0)
            @test dv[:a] == 1e-9
            @test dv[:b] == -1 - 1e-9
        end
    end

    @testset "Projected" begin
        for _ in 1:20
            dv = DVec(:a => 1.0)
            projected_deposit!(dv, :a, 0.5, :a => 1.0, 1.0)
            projected_deposit!(dv, :b, -0.5, :a => 1.0, 1.0)
            projected_deposit!(dv, :c, 1.1, :a => 1.0, 1.0)
            projected_deposit!(dv, :d, 0.1, :a => 1.0, 0.685)
            @test dv[:a] == 1.0 || dv[:a] == 2.0
            @test dv[:b] == 0.0 || dv[:b] == -1.0
            @test dv[:c] == 1.1
            @test dv[:d] == 0.685 || dv[:d] == 0
        end
    end
end

@testset "diagonal_step!" begin
    add = BoseFS((1,1,1))
    H = HubbardReal1D(add)
    @testset "Integer" begin
        # nothing happens - one annihilation
        dv = DVec(add => -1)
        @test diagonal_step!(dv, H, add, 1, 1e-5, 0) == (0, 0, 0, 1)
        @test dv[add] == 0

        # clones
        for _ in 1:20
            dv = DVec(add => 0)
            st = diagonal_step!(dv, H, BoseFS((2,0,1)), 1, 0.5, 10)
            @test st[1] == 4 || st[1] == 5
            @test dv[BoseFS((2,0,1))] == 5 || dv[BoseFS((2,0,1))] == 6
        end
        # deaths
        for _ in 1:20
            dv = DVec(add => 0)
            st = diagonal_step!(dv, H, BoseFS((2,0,1)), 1, 0.5, -0.5)
            @test st[2] == 0 || st[2] == 1
            @test dv[BoseFS((2,0,1))] == 0 || dv[BoseFS((2,0,1))] == 1
        end
        # zombies
        for _ in 1:20
            dv = DVec(add => 0)
            st = diagonal_step!(dv, H, BoseFS((2,0,1)), 1, 0.5, -10)
            @test st[2] == 1
            @test st[3] == 4 || st[3] == 5
            @test dv[BoseFS((2,0,1))] == -4 || dv[BoseFS((2,0,1))] == -5
        end
    end
    @testset "Exact" begin
        # nothing happens - one annihilation
        dv = DVec(add => -1.0)
        @test diagonal_step!(dv, H, add, 1, 1e-5, 0) == (0, 0, 0, 0)
        @test dv[add] == 0
        # clones
        dv = DVec(add => 0.0)
        st = diagonal_step!(dv, H, BoseFS((2,0,1)), 1, 0.5, 10)
        @test st[1] == 4.5
        @test dv[BoseFS((2,0,1))] == 5.5
        # deaths
        dv = DVec(add => 0.0)
        st = diagonal_step!(dv, H, BoseFS((2,0,1)), 1, 0.5, -0.5)
        @test st[2] == 0.75
        @test dv[BoseFS((2,0,1))] == 0.25
        # zombies
        dv = DVec(add => 0.0)
        st = diagonal_step!(dv, H, BoseFS((2,0,1)), 1, 0.5, -10)
        @test st[2] == 1
        #@test st[3] == 4.5 # <- annihilations disabled for now
        @test dv[BoseFS((2,0,1))] == -4.5
    end
    @testset "Projected" begin
        # nothing happens - but may be projected anyway
        for _ in 1:20
            dv = DVec(add => 0.0)
            st = diagonal_step!(dv, H, add, 1, 1e-5, 0, 1.5)
            @test st[2] == 1.0 || st[2] == 0.5
            @test dv[add] == 0.0 || dv[add] == 1.5
        end

        # clones
        dv = DVec(add => 0.0)
        st = diagonal_step!(dv, H, BoseFS((2,0,1)), 1, 0.5, 10, 1)
        @test st[1] == 4.5
        @test dv[BoseFS((2,0,1))] == 5.5
        # deaths
        for _ in 1:20
            dv = DVec(add => 0.0)
            st = diagonal_step!(dv, H, BoseFS((2,0,1)), 1, 0.5, -0.5, 1)
            @test st[2] == 0.0 || st[2] == 1.0
            @test dv[BoseFS((2,0,1))] == 0.0 || dv[BoseFS((2,0,1))] == 1.0
        end
        # zombies
        dv = DVec(add => 0.0)
        st = diagonal_step!(dv, H, BoseFS((2,0,1)), 1, 0.5, -10)
        @test st[2] == 1
        @test st[3] == 4.5
        @test dv[BoseFS((2,0,1))] == -4.5
    end
end

@testset "spawn!" begin
    dss_r = DynamicSemistochastic(WithReplacement(), 1.0, Inf)
    dss_w = DynamicSemistochastic(WithoutReplacement(), 1.0, Inf)
    dss_b = DynamicSemistochastic(Bernoulli(), 1.0, Inf)
    dss_ws = DynamicSemistochastic(WithoutReplacement(0, 1.1), 1.0, Inf)
    dss_bs = DynamicSemistochastic(Bernoulli(0, 1.5), 1.0, Inf)
    dss_s = DynamicSemistochastic(SingleSpawn(), 1.0, Inf)

    # The expected value for all spawning strategies should be the same. This tests makes
    # enough spawns to hopefully reach that average consistently.
    for (add, H) in [
        (BoseFS((0,0,0,3,1,1)), HubbardMom1D(BoseFS((0,0,0,3,1,1)); u=6.0)),
        (BoseFS((5,0,0,5,0,0,0)), HubbardReal1D(near_uniform(BoseFS{10,7}); u=1.0)),
    ]
        exact = DVec(add => 1.0)
        vanilla = DVec(add => 1.0)
        strong = DVec(add => 1.0)
        single = DVec(add => 1.0)
        semi_rep = DVec(add => 1.0)
        semi_bern = DVec(add => 1.0)
        semi_wo = DVec(add => 1.0)
        semi_bern_strong = DVec(add => 1.0)
        semi_wo_strong = DVec(add => 1.0)
        semi_single = DVec(add => 1.0)

        for _ in 1:10000
            val = rand() * num_offdiagonals(H, add) * 1.2
            spawn!(Exact(), exact, H, add, val, 1e-5)
            spawn!(WithReplacement(), vanilla, H, add, val, 1e-5)
            spawn!(WithReplacement(0.0, 2.0), strong, H, add, val, 1e-5)
            spawn!(SingleSpawn(), single, H, add, val, 1e-5)
            spawn!(dss_r, semi_rep, H, add, val, 1e-5)
            spawn!(dss_w, semi_wo, H, add, val, 1e-5)
            spawn!(dss_b, semi_bern, H, add, val, 1e-5)
            spawn!(dss_ws, semi_wo_strong, H, add, val, 1e-5)
            spawn!(dss_bs, semi_bern_strong, H, add, val, 1e-5)
            spawn!(dss_s, semi_single, H, add, val, 1e-5)
        end

        for k in keys(exact)
            @test exact[k] ≈ vanilla[k] rtol=0.1
            @test exact[k] ≈ strong[k] rtol=0.1
            @test exact[k] ≈ single[k] rtol=0.3 # ← very noisy.
            @test exact[k] ≈ semi_rep[k] rtol=0.1
            @test exact[k] ≈ semi_bern[k] rtol=0.1
            @test exact[k] ≈ semi_wo[k] rtol=0.1
            @test exact[k] ≈ semi_bern_strong[k] rtol=0.1
            @test exact[k] ≈ semi_wo_strong[k] rtol=0.1
            @test exact[k] ≈ semi_single[k] rtol=0.3 # ← very noisy.
        end
    end
end

@testset "Compression does not change 1-norm on average." begin
    add = BoseFS((0,0,0,10,0,0,0))
    H = HubbardMom1D(add)
    dv = DVec(add => 1.0, style=IsDeterministic())
    lomc!(H, dv)

    for compression in (
        StochasticStyles.ThresholdCompression(),
        StochasticStyles.ThresholdCompression(2),
        StochasticStyles.DoubleOrNothing(),
        StochasticStyles.DoubleOrNothing(prob=0.1),
        StochasticStyles.DoubleOrNothingWithThreshold(),
        StochasticStyles.DoubleOrNothingWithTarget(target=500),
    )
        target = similar(dv)
        for _ in 1:1000
            compressed = StochasticStyles.compress!(compression, copy(dv))
            add!(target, compressed)
        end
        @test walkernumber(target * (1/1000)) ≈ walkernumber(dv) rtol=0.1
    end
end
