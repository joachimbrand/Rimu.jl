using Rimu
using Test
using LinearAlgebra

using Rimu: projected_deposit!, diagonal_step!, spawns!, exact_spawns!, semistochastic_spawns!

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
        @test diagonal_step!(dv, H, add, 1, 1e-5, 0) == (0, 0, 0, 1)
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
        @test st[3] == 4.5
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

@testset "spawns!" begin
    add = BoseFS((0,0,0,3,0,0))
    H = HubbardMom1D(add; u=6.0)
    @testset "Integer" begin
        for _ in 1:20
            dv = empty(DVec(add => 1))
            spawns, _ = spawns!(dv, H, add, 100, 1e-2)
            @test norm(dv, 1) == spawns
        end
    end
    @testset "Projected" begin
        for _ in 1:20
            dv = empty(DVec(add => 1))
            spawns, _ = spawns!(dv, H, add, 10.5, 1e-2, 1)
            @test norm(dv, 1) == spawns
        end
    end
    @testset "Semistochastic" begin
        dv1 = empty(DVec(add => 1.0))
        dv2 = empty(DVec(add => 1.0))

        exact, inexact, spawns1, _ = semistochastic_spawns!(dv1, H, add, 105, 1e-2)
        spawns2, _ = exact_spawns!(dv2, H, add, 10.5, 1e-2)

        @test exact == 1
        @test inexact == 0
        @test spawns1 ≈ 10 * spawns2
        @test all(values(dv1) .≈ values(dv2 * 10))

        dv1 = empty(DVec(add => 1.0))
        dv2 = empty(DVec(add => 1.0))
        seedCRNG!(1000)
        exact, inexact, spawns1, _ = semistochastic_spawns!(dv1, H, add, 4, 1e-2)
        seedCRNG!(1000)
        spawns2, _ = spawns!(dv2, H, add, 4, 1e-2)

        @test exact == 0
        @test inexact == 1
        @test spawns1 == spawns2
        @test dv1 == dv2
    end
end
