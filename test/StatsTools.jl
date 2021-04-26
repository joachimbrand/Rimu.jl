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
