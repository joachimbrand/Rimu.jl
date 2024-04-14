using Rimu
using Test
using Random

Random.seed!(1234) # for reproducibility, as some solvers start with random vectors

# first we do tests that don't require KrylovKit and the extension
@testset "LinearAlgebraEigen" begin
    # LinearAlgebraEigen
    lae = LinearAlgebraEigen(; permute=true, scale=true)
    @test eval(Meta.parse(repr(lae))) == lae

    p = ExactDiagonalizationProblem(HubbardMom1D(BoseFS(1, 2, 3)))
    @test eval(Meta.parse(repr(p))) == p
    solver = init(p)
    @test solver.algorithm isa LinearAlgebraEigen
    @test dimension(solver.basissetrep) == size(solver.basissetrep.sm)[1] ≤ dimension(p.h)
    res = solve(solver)
    @test res.values[1] ≈ -3.045633163020568
end

VERSION ≥ v"1.9" && @testset "extension not loaded" begin
    # Can only test this when KrylovKit is not loaded
    ext = Base.get_extension(Rimu, :KrylovKitExt)
    if ext === nothing
        @test_throws ErrorException KrylovKitMatrix()
        @test_throws ErrorException KrylovKitDirect()
    end
    ext2 = Base.get_extension(Rimu, :ArpackExt)
    if ext2 === nothing
        @test_throws ErrorException ArpackEigs()
    end
    ext3 = Base.get_extension(Rimu, :IterativeSolversExt)
    if ext3 === nothing
        @test_throws ErrorException LOBPCG()
    end
end

using KrylovKit, Arpack, IterativeSolvers

VERSION ≥ v"1.9" && @testset "ExactDiagonalizationProblem" begin
    # KrylovKitMatrix
    km = KrylovKitMatrix(howmany=2, which=:SR)
    @test eval(Meta.parse(repr(km))) == km

    # KrylovKitDirect
    kd = KrylovKitDirect(howmany=2, which=:SR)
    @test eval(Meta.parse(repr(kd))) == kd

    # ArpackEigs
    ae = ArpackEigs(howmany=2, which=:SR)
    @test eval(Meta.parse(repr(ae))) == ae

    # LOBPCG
    lobpcg = LOBPCG(howmany=2, which=:SR)
    @test eval(Meta.parse(repr(lobpcg))) == lobpcg

    algs = [km, kd, ae, lobpcg]
    hamiltonians = [
        HubbardReal1D(BoseFS(1, 2, 3)),
        HubbardMom1D(BoseFS(1, 2, 3)),
        FroehlichPolaron(OccupationNumberFS(0,0,0); mode_cutoff=3)
    ]
    for h in hamiltonians
        p = ExactDiagonalizationProblem(h)
        @test eval(Meta.parse(repr(p))) == p
        energies = map(algs) do alg
            solver = init(p, alg)
            @test solver.problem == p
            res = solve(solver)
            @test res.success
            res.values[1]
        end
        @test energies[1] ≈ energies[2] ≈ energies[3]
    end

    # solve with KrylovKitMatrix
    p = ExactDiagonalizationProblem(HubbardReal1D(BoseFS(1,2,3)); which=:SR)
    @test eval(Meta.parse(repr(p))) == p
    solver = init(p, KrylovKitMatrix(); howmany=2)
    @test dimension(solver.basissetrep) == dimension(p.h) == size(solver.basissetrep.sm)[1]

    res_km = solve(solver)
    values, vectors, info = res_km
    @test length(values) == length(vectors) == info.converged ≥ 2

    # solve with KrylovKitDirect
    solver = init(p, KrylovKitDirect(); howmany=2)
    va_kd, ve_kd, info_kd = solve(solver)
    @test values ≈ va_kd
    addr = starting_address(res_km.problem.h)
    factor = vectors[1][addr] / ve_kd[1][addr]
    @test vectors[1] ≈ scale(ve_kd[1], factor)

    # solve with LinearAlgebraEigen
    res = @test_logs((:warn, "Unused keyword arguments in `solve`: (which = :SR,)"),
        solve(p, LinearAlgebraEigen()))
    va_la, ve_la, info_la = res

    @test values[1:2] ≈ va_la[1:2]
    factor = vectors[1][addr] / ve_la[1][addr]
    @test vectors[1] ≈ scale(ve_la[1], factor)

    # solve with ArpackEigs
    solver = init(p, ArpackEigs(); howmany=2)
    va_ae, ve_ae, info_ae = solve(solver)
    @test values[1:2] ≈ va_ae[1:2]
    factor = vectors[1][addr] / ve_ae[1][addr]
    @test vectors[1] ≈ scale(ve_ae[1], factor)

    p2 = ExactDiagonalizationProblem(
        HubbardReal1D(BoseFS(1, 2, 3)), DVec(BoseFS(1, 2, 3) => 2.3)
    )
    @test eval(Meta.parse(repr(p2))) == p2
    s2 = init(p2, KrylovKitMatrix(; howmany=3))

    res = solve(s2)
    @test length(res.values) == length(res.vectors) == res.info.converged ≥ 3
    res_full = solve(p2)
    @test res.values[1:3] ≈ res_full.values[1:3]

    p3 = ExactDiagonalizationProblem(
        HubbardMom1D(BoseFS(1, 2, 3)), BoseFS(1, 2, 3)
    )
    @test eval(Meta.parse(repr(p3))) == p3
    s3 = init(p3, KrylovKitDirect(); howmany=5)
    r3 = solve(s3)
    @test r3.success
end
