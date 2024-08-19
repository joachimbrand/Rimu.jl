using Rimu
using Test
using Rimu.StochasticStyles

using Rimu.StochasticStyles: projected_deposit!, diagonal_step!, spawn!
using Rimu.StochasticStyles:
    Exact, SingleSpawn, WithReplacement, WithoutReplacement, Bernoulli,
    DynamicSemistochastic, IsStochastic2Pop

@testset "default_style" begin
    default_style_of_typeof(x) = default_style(typeof(x))
    @test default_style_of_typeof(1) == IsStochasticInteger{Int}()
    @test default_style_of_typeof(1.0) == IsDeterministic{Float64}()
    @test default_style_of_typeof(1.0f0) == IsDeterministic{Float32}()
    @test default_style_of_typeof(1im) == IsStochastic2Pop{Complex{Int}}()
    @test default_style_of_typeof(1.0 + 1im) == IsDeterministic{ComplexF64}()
    @test default_style_of_typeof([1.0f0 + 1im, 1.0f0 + 1im]) isa StyleUnknown
end
@testset "Generic Hamiltonian-free functions" begin
    matrix = [1 2 3; 4 5 6; 7 8 9]
    @test diagonal_element(matrix, 3) == 9
    @test offdiagonals(matrix, 1) == [(2, 4), (3, 7)]
    @test offdiagonals(matrix, 2) == [(1, 2), (3, 8)]

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

    names, values = step_stats(vec)
    @test names == (:spawn_attempts, :spawns, :deaths, :clones, :zombies)
    @test values == Rimu.MultiScalar((0, 0, 0, 0, 0))

    w = [1.0, 2.0, 3.0]

    @test apply_column!(w, matrix, 1, 2) == (1, )
    @test w[1] == 1.0 + 2 * matrix[1, 1]
    @test w[2] == 2.0 + 2 * matrix[2, 1]
    @test w[3] == 3.0 + 2 * matrix[3, 1]
end

@testset "projected_deposit!" begin
    @testset "Integer" begin
        for _ in 1:20
            dv = DVec(:a => 1)
            @test projected_deposit!(dv, :a, 0.5, :a => 1, 0) in (0, 1)
            @test dv[:a] == 1 || dv[:a] == 2

            @test projected_deposit!(dv, :b, -5.5, :a => 1, 0) in (-5, -6)
            @test dv[:b] == -5 || dv[:b] == -6

            for i in 1:13
                @test projected_deposit!(dv, :c, 1.0, :a => 1, 0) == 1
            end
            @test dv[:c] == 13
        end
    end

    @testset "Exact" begin
        dv = DVec(:a => 1.0)
        @test projected_deposit!(dv, :a, -1, :a => 1.0, 0) ≡ -1.0
        @test isempty(dv)

        @test projected_deposit!(dv, :a, 1e-9, :a => 1.0, 0) ≡ 1e-9
        @test projected_deposit!(dv, :b, -1 - 1e-9, :a => 1.0, 0) ≡ -1 - 1e-9
        @test dv[:a] == 1e-9
        @test dv[:b] == -1 - 1e-9
    end

    @testset "Projected" begin
        for _ in 1:20
            dv = DVec(:a => 1.0)
            @test projected_deposit!(dv, :a, 0.5, :a => 1.0, 1.0) in (0, 1)
            @test projected_deposit!(dv, :b, -0.5, :a => 1.0, 1.0) in (-1, 0)
            @test projected_deposit!(dv, :c, 1.1, :a => 1.0, 1.0) == 1.1
            @test projected_deposit!(dv, :d, 0.1, :a => 1.0, 0.685) in (0, 0.685)
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
        # Single death
        dv = DVec(add => 0)
        @test diagonal_step!(dv, H, add, 1, 0) == (0, 1, 0)
        @test dv[add] == 0

        # clones
        for _ in 1:20
            dv = DVec(add => 0)
            T = Rimu.FirstOrderTransitionOperator(H, 10, 0.5)
            st = diagonal_step!(dv, T, BoseFS((2,0,1)), 1, 0)
            @test st[1] == 4 || st[1] == 5
            @test dv[BoseFS((2,0,1))] == st[1] + 1 # original value + clones
        end
        # deaths
        for _ in 1:20
            dv = DVec(BoseFS((2,0,1)) => 0)
            T = Rimu.FirstOrderTransitionOperator(H, -1.0, 0.125)
            st = diagonal_step!(dv, T, BoseFS((2,0,1)), 2, 0)
            @test st[2] == 0 || st[2] == 1
            @test dv[BoseFS((2,0,1))] == 2 - st[2] # original value - deaths
        end
        # zombies
        for _ in 1:20
            dv = DVec(BoseFS((2,0,1)) => 0)
            T = Rimu.FirstOrderTransitionOperator(H, -10, 0.5)
            st = diagonal_step!(dv, T, BoseFS((2,0,1)), 1, 0)
            @test st[2] == 1
            @test st[3] == 4 || st[3] == 5
            @test dv[BoseFS((2,0,1))] == 1 - st[2] - st[3] # original value - death - zombie
        end
    end
    @testset "Exact" begin
        # nothing happens - one annihilation
        dv = DVec(add => -1.0)
        T = Rimu.FirstOrderTransitionOperator(H, 0, 1)
        @test diagonal_step!(dv, T, add, 1.0, 0) == (0, 0, 0)
        @test dv[add] == 0
        # clones
        dv = DVec(add => 0.0)
        T = Rimu.FirstOrderTransitionOperator(H, 10, 1)
        st = diagonal_step!(dv, T, BoseFS((2,0,1)), 2.5, 0)
        @test st[1] == 22.5
        @test dv[BoseFS((2,0,1))] == 25.0
        # deaths
        dv = DVec(add => 0.0)
        T = Rimu.FirstOrderTransitionOperator(H, -0.5, 0.5)
        st = diagonal_step!(dv, T, BoseFS((2,0,1)), 1.0, 0)
        @test st[2] == 0.75
        @test dv[BoseFS((2,0,1))] == 0.25
        # zombies
        dv = DVec(add => 0.0)
        T = Rimu.FirstOrderTransitionOperator(H, -10, 0.5)
        st = diagonal_step!(dv, T, BoseFS((2,0,1)), 1.0, 0)
        @test st[2] == 1
        @test st[3] == 4.5
        @test dv[BoseFS((2,0,1))] == -4.5
    end
    @testset "Projected" begin
        # nothing happens but may be projected anyway resulting in either a clone or a death
        for _ in 1:20
            dv = DVec(add => 0.0)
            st = diagonal_step!(dv, H, BoseFS((2,0,1)), 1.0, 1.5) # diagonal element is 1
            @test st[1] == 0.5 || st[2] == 1
            @test dv[BoseFS((2,0,1))] == 1 + st[1] - st[2]
        end

        # clones - above projection threshold
        dv = DVec(add => 0.0)
        T = Rimu.FirstOrderTransitionOperator(H, 10, 0.5)
        st = diagonal_step!(dv, T, BoseFS((2,0,1)), 1, 1)
        @test st[1] == 4.5
        @test dv[BoseFS((2,0,1))] == 5.5
        # deaths - below threshold
        for _ in 1:20
            dv = DVec(add => 0.0)
            T = Rimu.FirstOrderTransitionOperator(H, -0.5, 0.5)
            st = diagonal_step!(dv, T, BoseFS((2,0,1)), 1.0, 1)
            @test st[2] == 0.0 || st[2] == 1.0
            @test dv[BoseFS((2,0,1))] == 1 - st[2]
        end
        # zombies
        dv = DVec(add => 0.0)
        T = Rimu.FirstOrderTransitionOperator(H, -10, 0.5)
        st = diagonal_step!(dv, T, BoseFS((2,0,1)), 1, 1)
        @test st[2] == 1
        @test st[3] == 4.5
        @test dv[BoseFS((2,0,1))] == -4.5
    end
end

@testset "spawn!" begin
    dss_r = DynamicSemistochastic(WithReplacement(), 1.0, Inf)
    dss_w = DynamicSemistochastic(WithoutReplacement(), 1.0, Inf)
    dss_b = DynamicSemistochastic(Bernoulli(), 1.0, Inf)
    dss_ws = DynamicSemistochastic(WithoutReplacement(0), 1.0, Inf)
    dss_bs = DynamicSemistochastic(Bernoulli(0), 1.0, Inf)
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
            spawn!(Exact(), exact, H, add, val)
            spawn!(WithReplacement(), vanilla, H, add, val)
            spawn!(WithReplacement(0.0), strong, H, add, val, 2.0)
            spawn!(SingleSpawn(), single, H, add, val)
            spawn!(dss_r, semi_rep, H, add, val)
            spawn!(dss_w, semi_wo, H, add, val)
            spawn!(dss_b, semi_bern, H, add, val)
            spawn!(dss_ws, semi_wo_strong, H, add, val, 1.1)
            spawn!(dss_bs, semi_bern_strong, H, add, val, 1.5)
            spawn!(dss_s, semi_single, H, add, val)
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

@testset "Compression does not change the vector on average." begin
    add = BoseFS((0,0,0,10,0,0,0))
    H = HubbardMom1D(add)
    dv = DVec(add => 1.0, style=IsDeterministic())
    lomc!(H, dv)
    normalize!(dv)

    for compression in (
        StochasticStyles.ThresholdCompression(),
        StochasticStyles.ThresholdCompression(2),
    )
        target_1 = similar(dv)
        target_2 = similar(dv)
        dv_before = copy(dv)
        for _ in 1:1000
            compressed_1 = copy(dv)
            compressed_2 = similar(dv)
            StochasticStyles.compress!(compression, compressed_1)
            StochasticStyles.compress!(compression, compressed_2, dv)
            @test length(compressed_1) < length(dv)
            @test length(compressed_2) < length(dv)
            add!(target_1, compressed_1)
            add!(target_2, compressed_2)
        end
        scale!(target_1, 1/1000)
        scale!(target_2, 1/1000)
        @test walkernumber(target_1) ≈ walkernumber(dv) rtol=0.1
        @test walkernumber(target_2) ≈ walkernumber(dv) rtol=0.1
        @test dot(target_1, dv) ≈ 1 rtol=0.1
        @test dot(target_2, dv) ≈ 1 rtol=0.1

        # double check that the dv didn't change during compression.
        @test dv_before == dv
    end
end

@testset "Exact substrategy to DynamicSemistochastic" begin
    dss_e = DynamicSemistochastic(Exact(), 1.0, Inf)
    add, H = (BoseFS((0,0,0,3,1,1)), HubbardMom1D(BoseFS((0,0,0,3,1,1)); u=6.0))
    val = rand() * num_offdiagonals(H, add) * 1.2
    exact = DVec(add => 1.0)
    ds_exact = DVec(add => 1.0)
    spawn!(Exact(), exact, H, add, val)
    spawn!(dss_e, ds_exact, H, add, val)
    @test keys(exact) == keys(ds_exact)
    for k in keys(exact)
        @test exact[k] ≈ ds_exact[k]
    end
end
