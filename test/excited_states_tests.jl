using Rimu
using Test


@testset "excited state energies" begin
    ham = HubbardReal1D(BoseFS(1,1,1,1,1))
    pr = ExactDiagonalizationProblem(ham)
    vals = solve(pr).values

    spectral_strategy = GramSchmidt(3)
    last_step=2000
    style = IsDeterministic()
    p = ProjectorMonteCarloProblem(ham; spectral_strategy, last_step, style)
    s = solve(p)
    df = DataFrame(s)
    energy1 = shift_estimator(df, shift="shift_s1_1", skip=1000)
    energy2 = shift_estimator(df, shift="shift_s2_1", skip=1000)
    energy3 = shift_estimator(df, shift="shift_s3_1", skip=1000)

    @test energy1.mean ≈ vals[1]
    @test energy2.mean ≈ vals[2]
    @test energy3.mean ≈ vals[3]

    n_replicas = 2
    p = ProjectorMonteCarloProblem(ham; spectral_strategy, last_step, style, n_replicas)
    s = solve(p)
    df = DataFrame(s)
    energy1 = shift_estimator(df, shift="shift_s1_1", skip=1000)
    energy2 = shift_estimator(df, shift="shift_s2_1", skip=1000)
    energy3 = shift_estimator(df, shift="shift_s3_1", skip=1000)
    energy4 = shift_estimator(df, shift="shift_s1_2", skip=1000)
    energy5 = shift_estimator(df, shift="shift_s2_2", skip=1000)
    energy6 = shift_estimator(df, shift="shift_s3_2", skip=1000)
    
    @test energy1.mean ≈ vals[1]
    @test energy2.mean ≈ vals[2]
    @test energy3.mean ≈ vals[3]
    @test energy4.mean ≈ vals[1]
    @test energy5.mean ≈ vals[2]
    @test energy6.mean ≈ vals[3]
end
