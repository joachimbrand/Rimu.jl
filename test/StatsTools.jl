using Rimu, Rimu.StatsTools
using Statistics, DataFrames
using Test

@testset "smoothen" begin
    noisy = randn(1000)
    b = 20
    smooth = smoothen(noisy, b)
    @test length(smooth) == length(noisy)
    @test mean(smooth) ≈ mean(noisy)
    @test var(noisy) > var(smooth)
    @test length(smoothen(noisy, b)) == length(noisy)
end

@testset "growth_witness" begin
    @test_throws AssertionError growth_witness(rand(10),rand(11),0.01)
    nor = rand(200)
    shift = rand(200)
    dτ = 0.01*ones(200)
    df = DataFrame(; norm=nor, shift, dτ)
    @test mean(growth_witness(df, 10)) ≈ mean(growth_witness(nor, shift, dτ[1]))
end

using Rimu.StatsTools: blocker
@testset "blocking" begin
    # real
    v = randn(2^10)
    br = block_and_test(v)
    @test 0.01 < br.err < 0.04
    @test br.k ≤ 4 # should be 0+1 (independent random variables)
    brs = block_and_test(smoothen(v, 2^5)) # introduce correlation
    @test mean(v) ≈ br.mean ≈ brs.mean
    @test 0.01 < brs.err < 0.04
    @test 5 < brs.k ≤ 7 # should be 5+1
    bor = block_and_test(ones(2000)) # blocking fails
    @test bor.k == -1
    @test isnan(bor.err)

    # complex
    w = randn(ComplexF64, 2^10)
    bc = block_and_test(w)
    @test bc.k ≤ 4  # should be 0+1 (independent random variables)
    bcs = block_and_test(smoothen(w, 2^5))
    @test mean(w) ≈ bc.mean ≈ bcs.mean
    @test 5 < bcs.k ≤ 7 # should be 5+1

    @test block_and_test([1]).k == -1 == block_and_test(Int[]).k
end

@testset "ratio_of_means" begin
    n_samples = 2000
    br= ratio_of_means(rand(n_samples),rand(n_samples); α=0.05)
    @test br.blocks * 2^(br.k-1) ≤ n_samples
end
