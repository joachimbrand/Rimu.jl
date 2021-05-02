using Rimu, Rimu.StatsTools
using Statistics, DataFrames, Random
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

using Rimu.StatsTools: x_by_y_linear
@testset "ratio_of_means" begin
    Random.seed!(17) # make sure the tests don't trip over rare fluctuations
    n_samples = 2000
    br= ratio_of_means(rand(n_samples),rand(n_samples); α=0.05)
    @test br.blocks * 2^(br.k-1) ≤ n_samples

    # badly behaved example of Cauchy distribution for ratio
    r = ratio_of_means(randn(2000),randn(2000))
    q95 = quantile(r.ratio, [0.025,0.975])
    @test q95[1] < 0 < q95[2]

    r = ratio_of_means(randn(ComplexF64,2000),randn(ComplexF64,2000))
    qr95 = quantile(real(r.ratio), [0.025,0.975])
    @test qr95[1] < 0 < qr95[2]
    qi95 = quantile(imag(r.ratio), [0.025,0.975])
    @test qi95[1] < 0 < qi95[2]

    # well behaved real example
    n_samples = 2000
    μ_a, μ_b = 2.0, 3.0
    σ_a, σ_b = 0.5, 0.1 # std of sample means
    f, σ_f = x_by_y_linear(μ_a, μ_b, σ_a, σ_b, 0) # expected ratio and std, uncorrelated
    @test f ≈ μ_a/μ_b
    # consistency with error propagation from Measurement.jl
    am = Measurements.:±(μ_a, σ_a)
    bm = Measurements.:±(μ_b, σ_b)
    @test σ_f ≈ (am / bm).err
    # consistency with error propagation from MonteCarloMeasurements.jl
    amc = MonteCarloMeasurements.:±(μ_a, σ_a)
    bmc = MonteCarloMeasurements.:±(μ_b, σ_b)
    @test isapprox([f-σ_f, f, f+σ_f], quantile(amc / bmc, [0.16,0.5,0.84]), rtol = 2/√2000)
    # add correlation for testing `ratio_of_means`
    ρ = 0.02
    f, σ_f = x_by_y_linear(μ_a, μ_b, σ_a, σ_b, ρ) # expected ratio and std
    @test f ≈ μ_a/μ_b
    ab = rand(MvNormal([μ_a, μ_b], [σ_a^2 ρ; ρ σ_b^2]*n_samples), n_samples)
    a = ab[1,:]
    b = ab[2,:]
    r = ratio_of_means(a,b)
    @test r.k < 3 # uncorrelated samples
    @test isapprox(μ_a/μ_b, r.f; atol = 2σ_f)
    @test isapprox(μ_a/μ_b, median(r.ratio); atol = 2σ_f)
    q = quantile(r.ratio, [0.16,0.5,0.84])
    @test isapprox(q[3]-q[2], σ_f; rtol = 4/√n_samples)
    @test isapprox(q[2]-q[1], σ_f; rtol = 4/√n_samples)
    @test isapprox(r.σ_f, σ_f; rtol = 4/√n_samples)

    # correlated time series
    rs = ratio_of_means(smoothen(a, 2^3), smoothen(b, 2^6))
    @test 6 ≤ rs.k ≤ 8 # ideally k==7 for 6 blocking steps to decorrelate
    # quantiles after blocking stay the same
    @test isapprox(q, quantile(rs.ratio, [0.16,0.5,0.84]); atol = 2σ_f)
    @test isapprox(μ_a/μ_b, rs.f; atol = 2σ_f)
    @test isapprox(rs.σ_f, σ_f; atol = 2σ_f)

    # well behaved complex example
    n_samples = 2000
    μ_a, μ_b = 2.0 + 1.0im, 3.0 - 2.0im
    σ_a, σ_b = 0.5, 0.1 # std of sample means
    f, σ_f = x_by_y_linear(μ_a, μ_b, σ_a, σ_b, 0) # expected ratio and std
    @test f ≈ μ_a/μ_b

    a = μ_a .+ √n_samples * σ_a * randn(ComplexF64, n_samples)
    b = μ_b .+ √n_samples * σ_b * randn(ComplexF64, n_samples)
    r = ratio_of_means(a,b)

    @test r.k < 3 # uncorrelated samples
    @test isapprox(μ_a/μ_b, r.f; atol = abs(2σ_f))
    @test isapprox(real(μ_a/μ_b), median(real(r.ratio)); atol = abs(2σ_f))
    qr = quantile(real(r.ratio), [0.16,0.5,0.84])
    qi = quantile(imag(r.ratio), [0.16,0.5,0.84])
    # Is this the correct way to test against linear error propagation?
    @test qr[3]-qr[2] < 2abs(σ_f)
    @test qi[3]-qi[2] < 2abs(σ_f)
    @test qr[2]-qr[1] < 2abs(σ_f)
    @test qi[2]-qi[1] < 2abs(σ_f)
    @test isapprox(r.σ_f, σ_f; rtol = 4/√n_samples)
end
