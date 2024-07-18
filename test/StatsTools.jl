using Rimu, Rimu.StatsTools
using Statistics, DataFrames, Random, Distributions
using Suppressor
using Test
using KrylovKit, LinearAlgebra
using MonteCarloMeasurements

import Measurements

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
    @test_throws AssertionError growth_witness(rand(10), rand(11), 0.01)
    @test_throws AssertionError growth_witness(rand(5), rand(5), 0.01; skip=10)
    nor = rand(200)
    shift = rand(200)
    time_step = 0.01 * ones(200)
    df = DataFrame(; norm=nor, shift, time_step)
    m = mean(growth_witness(shift, nor, time_step[1]; skip=10))
    @test mean(growth_witness(df, 10; skip=10)) ≈ m
    @test mean(growth_witness(df; skip=10)) ≈ m
end

using Rimu.StatsTools: blocker
@testset "blocking" begin
    Random.seed!(13) # make sure we don't trip over rare fluctuations
    # real
    v = randn(2^10)
    br = blocking_analysis(v)
    @test br == blocking_analysis_data(v).br
    @test MonteCarloMeasurements.:±(br) ≈ Measurements.:±(br)
    @test 0.01 < br.err < 0.04
    @test br.k ≤ 4 # should be 0+1 (independent random variables)
    brs = blocking_analysis(smoothen(v, 2^5)) # introduce correlation
    @test mean(v) ≈ br.mean ≈ brs.mean
    @test 0.01 < brs.err < 0.04
    @test 5 < brs.k ≤ 7 # should be 5+1
    bor = @test_logs (:warn,) blocking_analysis(ones(2000)) # blocking fails
    @test bor.k == -1
    @test isnan(bor.err)

    # complex
    w = randn(ComplexF64, 2^10)
    bc = blocking_analysis(w)
    @test real(MonteCarloMeasurements.:±(bc)) ≈ real(Measurements.:±(bc))
    @test imag(MonteCarloMeasurements.:±(bc)) ≈ imag(Measurements.:±(bc))
    @test bc.k ≤ 4  # should be 0+1 (independent random variables)
    bcs = blocking_analysis(smoothen(w, 2^5))
    @test mean(w) ≈ bc.mean ≈ bcs.mean
    @test 5 < bcs.k ≤ 7 # should be 5+1

    @test -1 == (@test_logs (:warn,) blocking_analysis([1]).k)
    @test -1 == (@test_logs (:error,) blocking_analysis(Int[]).k)
end

using Rimu.StatsTools: x_by_y_linear, ratio_estimators, particles
@testset "ratio_of_means" begin
    Random.seed!(17) # make sure the tests don't trip over rare fluctuations
    n_samples = 2000
    br = ratio_of_means(rand(n_samples), rand(n_samples); α=0.05)
    @test br.blocks * 2^(br.k - 1) ≤ n_samples
    @test begin
        show(devnull, br)
        true
    end # does not throw error

    # Make sure all other exported functions work.
    for f in (
        pmedian, pmiddle, piterate, pextrema, pminimum, pmaximum, pmean, pcov
    )
        @test f(br) == f(br.ratio)
    end

    # badly behaved example of Cauchy distribution for ratio
    r = @suppress ratio_of_means(randn(2000), randn(2000))
    q95 = pquantile(r.ratio, [0.025, 0.975])
    @test q95[1] < 0 < q95[2]

    # complex time series
    r = @suppress ratio_of_means(randn(ComplexF64, 2000), randn(ComplexF64, 2000))
    qr95 = pquantile(real(r.ratio), [0.025, 0.975])
    @test qr95[1] < 0 < qr95[2]
    qi95 = pquantile(imag(r.ratio), [0.025, 0.975])
    @test qi95[1] < 0 < qi95[2]
    @test_throws ErrorException pquantile(r, [0.025, 0.975])
    @test begin
        show(devnull, r)
        true
    end # does not throw error

    # mixed real and complex
    r = @suppress @inferred ratio_of_means(randn(1000), randn(ComplexF64, 1000))
    @test r.k ≤ 4
    r = @suppress @inferred ratio_of_means(randn(ComplexF64, 1000), randn(1000))
    @test r.k ≤ 4

    # zero variance example
    r = @suppress ratio_of_means([0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 2])
    @test Tuple(val_and_errs(r)) == (0, 0, 0)
    r = @suppress ratio_of_means(complex.(ones(2000)), complex.(ones(2000)))
    @test pvar(real(r.ratio)) == 0 == pvar(imag(r.ratio))

    # well behaved real example
    Random.seed!(17) # make sure the tests don't trip over rare fluctuations
    n_samples = 2000
    μ_a, μ_b = 2.0, 3.0
    σ_a, σ_b = 0.5, 0.1 # std of sample means
    f, σ_f = x_by_y_linear(μ_a, μ_b, σ_a, σ_b, 0) # expected ratio and std, uncorrelated
    @test f ≈ μ_a / μ_b
    # consistency with error propagation from Measurement.jl
    am = Measurements.:±(μ_a, σ_a)
    bm = Measurements.:±(μ_b, σ_b)
    @test σ_f ≈ (am / bm).err
    # consistency with error propagation from MonteCarloMeasurements.jl
    amc = MonteCarloMeasurements.:±(μ_a, σ_a)
    bmc = MonteCarloMeasurements.:±(μ_b, σ_b)
    @test isapprox(
        [f - σ_f, f, f + σ_f], pquantile(amc / bmc, [0.16, 0.5, 0.84]),
        rtol=2 / √2000
    )

    # add correlation for testing `ratio_of_means`
    Random.seed!(13) # make sure the tests don't trip over rare fluctuations
    ρ = 0.02
    f, σ_f = x_by_y_linear(μ_a, μ_b, σ_a, σ_b, ρ) # expected ratio and std
    @test f ≈ μ_a / μ_b
    ab = rand(MvNormal([μ_a, μ_b], [σ_a^2 ρ; ρ σ_b^2] * n_samples), n_samples)
    a = ab[1, :]
    b = ab[2, :]
    r = ratio_of_means(a, b) #; mc_samples = Val(10_000))
    @test 1 < r.k ≤ 4 # weakly correlated samples
    @test isapprox(μ_a / μ_b, r.f; atol=2σ_f)
    @test isapprox(μ_a / μ_b, pmedian(r.ratio); atol=2σ_f)
    q = pquantile(r.ratio, [0.16, 0.5, 0.84])
    @test isapprox(q[3] - q[2], σ_f; rtol=8 / √n_samples)
    @test isapprox(q[2] - q[1], σ_f; rtol=8 / √n_samples)
    @test isapprox(r.σ_f, σ_f; rtol=8 / √n_samples)

    # correlated time series
    rs = ratio_of_means(smoothen(a, 2^3), smoothen(b, 2^6))
    @test 6 ≤ rs.k ≤ 8 # ideally k==7 for 6 blocking steps to decorrelate
    # quantiles after blocking stay the same
    @test isapprox(q, pquantile(rs.ratio, [0.16, 0.5, 0.84]); atol=2σ_f)
    @test isapprox(μ_a / μ_b, rs.f; atol=2σ_f)
    @test isapprox(rs.σ_f, σ_f; atol=2σ_f)

    Random.seed!(1234) # make sure the tests don't trip over rare fluctuations
    # well behaved complex example
    n_samples = 2000
    μ_a, μ_b = 2.0 + 1.0im, 3.0 - 2.0im
    σ_a, σ_b = 0.5, 0.1 # std of sample means
    f, σ_f = x_by_y_linear(μ_a, μ_b, σ_a, σ_b, 0) # expected ratio and std
    @test f ≈ μ_a / μ_b

    a = μ_a .+ √n_samples * σ_a * randn(ComplexF64, n_samples)
    b = μ_b .+ √n_samples * σ_b * randn(ComplexF64, n_samples)
    r = ratio_of_means(a, b)

    @test r.k < 3 # uncorrelated samples
    @test isapprox(μ_a / μ_b, r.f; atol=abs(2σ_f))
    @test isapprox(μ_a / μ_b, pmedian(r); atol=abs(2σ_f))
    qr = pquantile(real(r.ratio), [0.16, 0.5, 0.84])
    qi = pquantile(imag(r.ratio), [0.16, 0.5, 0.84])
    # Is this the correct way to test against linear error propagation?
    @test qr[3] - qr[2] < 2abs(σ_f)
    @test qi[3] - qi[2] < 2abs(σ_f)
    @test qr[2] - qr[1] < 2abs(σ_f)
    @test qi[2] - qi[1] < 2abs(σ_f)
    @test isapprox(r.σ_f, σ_f; rtol=4 / √n_samples)

    # type stability of Particles
    d = MvNormal([1.0, 1.0], [0.1 0.01; 0.01 0.1])
    @test typeof(particles(100, d)) == typeof(particles(Val(100), d))
    @test typeof(particles(100, d)) == typeof(particles(100, [1.0, 1.0], [0.1 0.01; 0.01 0.1]))
    @inferred particles(nothing, d)
    @inferred particles(Val(100), d)
    @inferred ratio_estimators(rand(100), rand(100), 2; mc_samples=Val(100))
    @inferred ratio_estimators(
        rand(ComplexF64, 100), rand(ComplexF64, 100), 2;
        mc_samples=Val(100)
    )
    @inferred ratio_of_means(rand(1000), 100 .+ rand(1000))
    p = @inferred particles(nothing, [1.0, 1.0], [0.1 0.01; 0.01 0.0])
    @test pvar.(p)[2] == 0
end

@testset "Reweighting" begin
    Random.seed!(133)
    ham = HubbardReal1D(BoseFS((1, 1, 1, 1)), u=6.0, t=1.0)
    # using KrylovKit
    # fv = DVec(starting_address(ham)=>1.0; capacity=dimension(ham))
    # kkresults = eigsolve(ham, fv, 1, :SR; issymmetric = true)
    # exact_energy = kkresults[1][1]
    exact_energy = -2.869739978337469
    # run integer walker FCIQMC to get significant bias
    v = DVec(starting_address(ham) => 2; capacity=dimension(ham))
    steps_equi = 200
    steps_meas = 2^10
    p = RunTillLastStep(laststep=steps_equi + steps_meas)
    post_step = ProjectedEnergy(ham, v)
    s_strat = DoubleLogUpdate(target_walkers=10)
    df = lomc!(ham, v; params=p, s_strat, post_step).df
    @test_throws ArgumentError variational_energy_estimator(df) # see next testset
    bs = shift_estimator(df; skip=steps_equi)
    @test bs == blocking_analysis(df.shift[steps_equi+1:end])
    pcb = bs.mean - exact_energy

    @test pcb > 0.0 # the shift has a large population control bias
    # test growth_estimator
    h = 2^(bs.k - 1) # approximate number of steps to decorrelate the shift
    E_r = bs.mean # set up reference energy
    ge = @suppress growth_estimator(df, h; E_r, skip=steps_equi)
    pcb_est = E_r - ge.ratio # estimated PCB in the shift from reweighting
    @test 0.2 < pmedian(pcb_est) < pcb
    @inferred growth_estimator(rand(1000), 100 .+ rand(1000), 20, 0.01; change_type=to_measurement)
    @inferred growth_estimator(rand(1000), 100 .+ rand(1000), 20, 0.01)
    # fails due to type instability in MonteCarloMeasurements
    # test w_lin()
    @test @suppress ge.ratio ≈ growth_estimator(df, h; E_r, skip=steps_equi, weights=w_lin).ratio
    # test growth_estimator_analysis
    df_ge, correlation_estimate, se, se_l, se_u = @suppress begin
        growth_estimator_analysis(df; skip=steps_equi)
    end
    @test correlation_estimate == h
    @test se ≈ bs.mean
    df_nt = @suppress growth_estimator_analysis(df; skip=steps_equi).df_ge
    @test all(abs.(df_nt.val .- df_ge.val) .< df_nt.val_l)
    # projected energy
    bp = @suppress projected_energy(df; skip=steps_equi)
    time_step = Rimu.StatsTools.determine_constant_time_step(df)
    me = @suppress @inferred mixed_estimator(
        df.hproj, df.vproj, df.shift, h, time_step;
        skip=steps_equi, E_r
    )
    @test me.ratio ≈ bp.ratio # reweighting has not significantly improved the projected energy
    @test val(projected_energy(df; skip=steps_equi)) ≈
        val(mixed_estimator(df, 0; skip=steps_equi)) rtol=0.1
    # test mixed_estimator_analysis
    df_me, correlation_estimate, se, se_l, se_u = @suppress begin
        mixed_estimator_analysis(df; skip=steps_equi)
    end
    @test correlation_estimate == h
    @test se ≈ bs.mean
    df_met = @suppress mixed_estimator_analysis(df; skip=steps_equi).df_me
    @test all(abs.(df_met.val .- df_me.val) .< df_met.val_l)
end

@testset "Rayleigh quotient reweighting" begin
    ham = HubbardReal1D(BoseFS((1, 1, 1, 1)), u = 6.0, t = 1.0)
    dv = DVec(starting_address(ham) => 2; capacity = dimension(ham))
    dvals = [0,1]
    best_g2 = [0.220679, 0.907466]    # results for tw = 10K, 2^18 steps, no reweighting

    skipsteps = 2^8
    runsteps = 2^10
    num_reps = 2
    tw = 10

    params = RunTillLastStep(laststep = skipsteps + runsteps)
    s_strat = DoubleLogUpdate(target_walkers = tw)
    G2list = ([G2RealCorrelator(i) for i in dvals]...,)

    Random.seed!(174)
    df = lomc!(ham, dv; params, s_strat, replica_strategy = AllOverlaps(num_reps; operator = G2list)).df

    for d in dvals
        # without reweighting
        r = rayleigh_replica_estimator(df; op_name = "Op$(d+1)", skip = skipsteps)
        g2_h0 = val_and_errs(r).val
        # with reweighting
        df_rre, _ = @suppress rayleigh_replica_estimator_analysis(
            df; op_name = "Op$(d+1)", skip = skipsteps
        )
        g2_rw = df_rre[end,:val]
        # reweighting improves the estimate
        @test abs(g2_h0 - best_g2[d+1]) > abs(g2_rw - best_g2[d+1])
    end
end

using Rimu.StatsTools: replica_fidelity
@testset "Fidelity and variational energy" begin
    ham = HubbardReal1D(BoseFS((1,1,1,1)), u=6.0, t=1.0)

    # get exact eigenvectors with KrylovKit
    fv = DVec(starting_address(ham) => 1.0; capacity=dimension(ham))
    kkresults = eigsolve(ham, fv, 2, :SR; issymmetric=true)
    gs = kkresults[2][1] # ground state; norm(gs) ≈ 1
    es = kkresults[2][2] # 1st excited state; norm(es) ≈ 1
    @test norm(gs) ≈ 1 ≈ norm(es) # they are normalised
    @test isapprox(gs ⋅ es, 0; atol=√eps(Float64)) # and orthogonal

    # Set up oblique vector at angle to ground state
    α = π / 3 # 60° angle
    os = add!(cos(α) * gs, sin(α) * es)
    @test norm(os) ≈ 1

    # set up replica MC
    v = DVec(starting_address(ham) => 2; capacity=dimension(ham))
    steps_equi = 200
    steps_meas = 2^10
    p = RunTillLastStep(laststep=steps_equi + steps_meas)
    post_step = (Projector(vproj=gs), Projector(hproj=os))
    s_strat = DoubleLogUpdate(target_walkers=10)

    # run replica fciqmc
    Random.seed!(170)
    rr = lomc!(ham, v; params=p, s_strat, post_step, replica_strategy=AllOverlaps()).df

    # check fidelity with ground state
    fid_gs = replica_fidelity(rr; p_field=:vproj, skip=steps_equi)
    @test fid_gs.ratio ≈ 1
    _, val_l, val_u = val_and_errs(fid_gs) # extract errors from quantiles
    @test val_l < 0.03 && val_u < 0.03 # errors are small
    # TODO
    #=
    # check fidelity with oblique state
    fid_os = StatsTools.replica_fidelity(rr; p_field=:hproj, skip=steps_equi)
    @test fid_os.ratio ≈ cos(α)^2
    γ = acos(√fid_os.ratio) # quantum angle or Fubini-Study metric
    @test γ ≈ α
    =#

    # test variational energy directly on the result of the replica run
    ve = variational_energy_estimator(rr; skip=steps_equi, max_replicas=5)
    # the `max_replicas` option has no effect in this case
    @test kkresults[1][1] < pmedian(ve)
    @test pmedian(ve) < shift_estimator(rr; shift = :shift_1, skip=steps_equi).mean
    @test_throws ArgumentError variational_energy_estimator(DataFrame()) # empty df
    vs = [rand(5) for _ in 1:4]
    @test_throws ArgumentError variational_energy_estimator(vs,vs) # wrong length arrays
end

comp_tuples(a, b; atol=0) = mapreduce((x, y) -> isapprox(x, y; atol), &, Tuple(a), Tuple(b))
@testset "convenience" begin
    v, σ = 2.0, 0.2
    m = Measurements.measurement(v, σ)
    @test comp_tuples(val_and_errs(m), (v, σ, σ))
    # @test comp_tuples(med_and_errs(m), (v, σ, σ, 2σ, 2σ))
    # @test map((x,y)->isapprox(x,y),Tuple(med_and_errs(m)),(v, σ, σ, 2σ, 2σ))|>all
    # @test med_and_errs(m) == (med = v, err1_l = σ, err1_u = σ, err2_l = σ, err2_u = σ)
    mp = MonteCarloMeasurements.Particles(m)
    @test comp_tuples(val_and_errs(mp), (v, σ, σ); atol=0.01)
    # @test comp_tuples(med_and_errs(mp), (v, σ, σ, 2σ, 2σ); atol=0.01)
    # @test map((x,y)->isapprox(x,y; atol=0.01),Tuple(med_and_errs(mp)),(v, σ, σ, 2σ, 2σ))|>all
    @test m ≈ to_measurement(mp)

    @test val(2.0) == 2.0
    @test val(m) == v
    @test val(mp) == pmedian(mp)
    @test errs(2.0) == (err_l=0, err_u=0)
    @test val_and_errs(2; name="x") == (x=2, x_l=0, x_u=0)
    @test comp_tuples(val_and_errs(m, p=0.954499736104), (v, 2σ, 2σ))
    @test comp_tuples(val_and_errs(m, n=2), (v, 2σ, 2σ))
    @test comp_tuples(val_and_errs(mp, n=2), (v, 2σ, 2σ); atol=0.01)

    r = @suppress ratio_of_means(randn(2000), randn(2000))
    @test comp_tuples(val_and_errs(r), (v=val(r), errs(r)...))
    @test NamedTuple(r).val_k == r.k
    @test NamedTuple(r).val_δ_y == r.δ_y
    @test NamedTuple(r; name=:frodo).frodo == pmedian(r)
    br = blocking_analysis(rand(2000))
    @test comp_tuples(val_and_errs(br), (v=val(br), errs(br)...))
    @test NamedTuple(br).val_blocks == br.blocks

    # complex time series
    r = @suppress ratio_of_means(randn(ComplexF64, 2000), randn(ComplexF64, 2000))
    nt = val_and_errs(r)
    @test imag(nt.val_l) ≠ 0 ≠ real(nt.val_l)
    r = blocking_analysis(rand(ComplexF64, 2000))
    nt = val_and_errs(r)
    @test imag(nt.val_l) ≠ 0 ≠ real(nt.val_l)
end
