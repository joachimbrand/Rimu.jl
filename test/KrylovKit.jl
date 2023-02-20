using KrylovKit
using LinearAlgebra
using Rimu
using Test

@testset "Krylov eigsolve with BoseFS{6,6}" begin
    ham = HubbardReal1D(near_uniform(BoseFS{6,6}); u=6.0, t=1.0)

    a_init = near_uniform(ham)
    c_init = DVec(a_init => 1.0)

    all_results = eigsolve(ham, c_init, 1, :SR; issymmetric = true)
    energy = all_results[1][1]

    @test energy ≈ -4.0215 atol=0.0001
end

if VERSION ≥ v"1.9"
    @testset "KrylovKit Extension" begin
        add = FermiFS2C((0,1,0,1,0), (0,0,1,0,0))
        ham_bm = HubbardMom1D(add)
        ham_tc = Transcorrelated1D(add)

        true_bm = eigen(Matrix(ham_bm)).values[1]
        res1 = eigsolve(ham_bm, DVec(add => 1.0), 1, :SR)
        res2 = eigsolve(ham_bm, PDVec(add => 1.0), 1, :SR)

        @test res1[1][1] ≈ true_bm
        @test res1[1][1] ≈ res2[1][1]
        @test res1[1][1] isa Real
        @test res2[1][1] isa Real

        true_tc = eigen(Matrix(ham_tc)).values[1]
        res3 = eigsolve(ham_tc, DVec(add => 1), 1, :SR)
        res4 = eigsolve(ham_tc, PDVec(add => 1), 1, :SR)

        @test res3[1][1] ≈ true_tc
        @test res3[1][1] ≈ res4[1][1]
        @test res3[1][1] isa ComplexF64
        @test res4[1][1] isa ComplexF64
    end
end
