using Rimu
using Test
using Random
using Suppressor

@testset "BasisSetRepresentation" begin
    @testset "basics" begin
        m = 100
        n = 100
        addr = BoseFS(Tuple(i == 1 ? n : 0 for i in 1:m))
        ham = HubbardReal1D(addr)
        @test_throws ArgumentError BasisSetRepresentation(ham) # dimension too large
        m = 2
        n = 10
        addr = near_uniform(BoseFS{n,m})
        ham = HubbardReal1D(addr)
        bsr = BasisSetRepresentation(ham; nnzs=dimension(ham))
        @test length(bsr.basis) == dimension(bsr) ≤ dimension(ham)
        @test_throws ArgumentError BasisSetRepresentation(ham, BoseFS((1, 2, 3))) # wrong address type
        @test Matrix(bsr) == Matrix(bsr.sm) == Matrix(ham)
        @test sparse(bsr) == bsr.sm == sparse(ham)
        addr2 = bsr.basis[2]
        @test starting_address(BasisSetRepresentation(ham, addr2)) == addr2
        @test isreal(ham) == (eltype(ham) <: Real)
        @test isdiag(ham) == (LOStructure(ham) ≡ IsDiagonal())
        @test ishermitian(ham) == (LOStructure(ham) ≡ IsHermitian())
        @test issymmetric(ham) == (ishermitian(ham) && isreal(ham))
    end

    @testset "filtering" begin
        ham = HubbardReal1D(near_uniform(BoseFS{10,2}))
        bsr_orig = BasisSetRepresentation(ham; sort=true)
        mat_orig = Matrix(bsr_orig)
        mat_cut_index = diag(mat_orig) .< 30
        mat_cut_manual = mat_orig[mat_cut_index, mat_cut_index]
        bsr = BasisSetRepresentation(ham; cutoff=30, sort=true)
        mat_cut = Matrix(bsr)
        @test mat_cut == mat_cut_manual
        # pass a basis and generate truncated BasisSetRepresentation
        bsrt = BasisSetRepresentation(ham, bsr.basis; filter=Returns(false), sort=true)
        @test bsrt.basis == bsr.basis
        @test bsr.sm == bsrt.sm
        # pass addresses and generate reachable basis
        @test BasisSetRepresentation(ham, bsr.basis, sort=true).basis == bsr_orig.basis

        filterfun(fs) = maximum(onr(fs)) < 8
        mat_cut_index = filterfun.(BasisSetRepresentation(ham; sort=true).basis)
        mat_cut_manual = mat_orig[mat_cut_index, mat_cut_index]
        mat_cut = Matrix(ham; filter=filterfun, sort=true)
        @test mat_cut == mat_cut_manual
    end

    @testset "getindex" begin
        ham = HubbardReal1D(near_uniform(BoseFS{10,2}))
        bsr = BasisSetRepresentation(ham; sort=true)
        b = bsr.basis
        @test [ham[i, j] for i in b, j in b] == Matrix(bsr)
    end

    @testset "momentum blocking" begin
        add1 = BoseFS((2, 0, 0, 0))
        add2 = BoseFS((0, 1, 0, 1))
        ham = HubbardMom1D(add1)

        @test Matrix(ham, add1; sort=true) == Matrix(ham, add2; sort=true)
        @test Matrix(ham, add1) ≠ Matrix(ham, add2)

        add1 = BoseFS((2, 0, 0, 0, 0))
        add2 = BoseFS((0, 1, 0, 0, 1))
        ham = HubbardMom1D(add1)

        @test Matrix(ham, add1; sort=true) == Matrix(ham, add2; sort=true)
        @test Matrix(ham, add1) ≠ Matrix(ham, add2)
    end

    using Rimu.ExactDiagonalization: fix_approx_hermitian!, isapprox_enforce_hermitian!
    using Rimu.ExactDiagonalization: build_sparse_matrix_from_LO
    using Random
    @testset "fix_approx_hermitian!" begin
        # generic `Matrix`
        Random.seed!(17)
        mat = rand(5, 5)
        @test !ishermitian(mat)
        @test_throws ArgumentError fix_approx_hermitian!(mat; test_approx_symmetry=true)
        @test !ishermitian(mat) # still not hermitian
        fix_approx_hermitian!(mat; test_approx_symmetry=false)
        @test ishermitian(mat) # now it is hermitian

        # sparse matrix
        Random.seed!(17)
        mat = sparse(rand(5, 5))
        @test !ishermitian(mat)
        @test_throws ArgumentError fix_approx_hermitian!(mat; test_approx_symmetry=true)
        @test !ishermitian(mat) # still not hermitian

        # subtle symmetry violation due to `ParitySymmetry` wrapper
        ham = HubbardMom1D(BoseFS((1, 0, 1, 2, 0)))
        even = ParitySymmetry(ham; odd=false)
        odd = ParitySymmetry(ham; even=false)

        even_sm, _ = build_sparse_matrix_from_LO(even)
        even_m = Matrix(even) # symmetrised version via BasisSetRepresentation

        @test !issymmetric(even_sm) # not symmetric due to floating point errors
        @test issymmetric(even_m) # because it was passed through `fix_approx_hermitian!`
        @test even_sm ≈ even_m # still approximately the same!
    end

    @testset "basis-only" begin
        m = 5
        n = 5
        add = near_uniform(BoseFS{n,m})
        ham = HubbardReal1D(add)
        @test_throws ArgumentError build_basis(ham, BoseFS((1, 2, 3))) # wrong address type
        # same basis as BSR
        bsr = BasisSetRepresentation(ham)
        basis = build_basis(ham)
        @test basis == bsr.basis
        @test basis == build_basis(ham, basis) # passing multiple addresses
        # sorting
        basis = build_basis(ham, add; sort=true)
        @test basis == sort!(bsr.basis)
        # filtering
        @test_throws ArgumentError build_basis(ham, add; sizelim=100)
        @test length(build_basis(ham, add; cutoff=-1)) == 1 # no new addresses added
        cutoff = n * (n - 1) / 4  # half maximum energy
        bsr = BasisSetRepresentation(ham, add; cutoff)
        basis = build_basis(ham, add; cutoff)
        @test basis == bsr.basis
    end
end

Random.seed!(123) # for reproducibility, as some solvers start with random vectors

# first we do tests that don't require KrylovKit and the extension
@testset "LinearAlgebraSolver" begin
    # LinearAlgebraSolver
    lae = LinearAlgebraSolver(; permute=true, scale=true)
    @test eval(Meta.parse(repr(lae))) == lae

    p = ExactDiagonalizationProblem(HubbardMom1D(BoseFS(1, 2, 3)))
    @test eval(Meta.parse(repr(p))) == p
    solver = init(p)
    @test solver.algorithm isa LinearAlgebraSolver
    @test dimension(solver.basissetrep) == size(solver.basissetrep.sm)[1] ≤ dimension(p.h)
    res = solve(solver)
    @test res.values[1] ≈ -3.045633163020568
end

@testset "extension not loaded" begin
    # Can only test this when KrylovKit is not loaded
    ext = Base.get_extension(Rimu, :KrylovKitExt)
    if ext === nothing
        @test_throws ErrorException KrylovKitSolver()
    end
    ext2 = Base.get_extension(Rimu, :ArpackExt)
    if ext2 === nothing
        @test_throws ErrorException ArpackSolver()
    end
    ext3 = Base.get_extension(Rimu, :IterativeSolversExt)
    if ext3 === nothing
        @test_throws ErrorException LOBPCGSolver()
    end
end

using KrylovKit, Arpack, IterativeSolvers

@testset "ExactDiagonalizationProblem" begin
    # KrylovKitSolver matrix
    km = KrylovKitSolver(matrix_free=false, howmany=2, which=:SR)
    @test eval(Meta.parse(repr(km))) == km

    # KrylovKitSolver matrix free
    kd = KrylovKitSolver(matrix_free = true, howmany=2, which=:SR)
    @test eval(Meta.parse(repr(kd))) == kd

    # ArpackSolver
    ae = ArpackSolver(howmany=2, which=:SR)
    @test eval(Meta.parse(repr(ae))) == ae

    # LOBPCGSolver
    lobpcg = LOBPCGSolver(howmany=2, which=:SR)
    @test eval(Meta.parse(repr(lobpcg))) == lobpcg

    # LinearAlgebraSolver
    lae = LinearAlgebraSolver(; permute=true, scale=true)
    @test eval(Meta.parse(repr(lae))) == lae

    algs = [km, kd, ae, lobpcg, lae]
    hamiltonians = [
        HubbardReal1D(BoseFS(1, 2, 3)),
        HubbardMom1D(BoseFS(1, 2, 3)),
        FroehlichPolaron(OccupationNumberFS(0,0,0); mode_cutoff=3)
    ]
    for h in hamiltonians
        p = ExactDiagonalizationProblem(h)
        @test eval(Meta.parse(repr(p))) == p
        @test dimension(p) == dimension(h)
        energies = map(algs) do alg
            solver = init(p, alg)
            @test solver.problem == p
            res = @suppress solve(solver; verbose=false)
            @test res.success
            @test res isa Rimu.ExactDiagonalization.EDResult
            @test length(res.values) == length(res.vectors)
            @test length(res.values) == length(res.coefficient_vectors) ≥ res.howmany
            @test length(res.basis) == length(res.vectors[1]) ≤ dimension(p.h)
            for (i, dv) in enumerate(res.vectors)
                @test DVec(zip(res.basis, res.coefficient_vectors[i])) ≈ dv
            end
            # algorithm can take starting vector
            p2 = ExactDiagonalizationProblem(h, starting_address(h))
            res2 = @suppress solve(p, alg; verbose=true)
            @test repr(res2.info) isa String
            @test res.values[1] ≈ res2.values[1]
            res.values[1]
        end
        @test all(energies[1] .≈ energies)
    end

    # wrong starting vector
    h = HubbardReal1D(BoseFS(1, 2, 3))
    p = ExactDiagonalizationProblem(h, [1, 2, 3])
    @test_throws ArgumentError init(p, KrylovKitSolver(true))
    @test_throws ArgumentError init(p, ArpackSolver())

    # DVec starting vector
    v = DVec(BoseFS(1, 2, 3) => 1.0)
    p = ExactDiagonalizationProblem(h, v)
    @test eval(Meta.parse(repr(p))) == p
    solver = init(p, KrylovKitSolver(true))
    @test solver.v0 == v
    solver2 = init(p, ArpackSolver())
    @test DVec(solver2.v0) == v

    # solve with KrylovKitSolver matrix
    p = ExactDiagonalizationProblem(HubbardReal1D(BoseFS(1,2,3)); which=:SR)
    @test eval(Meta.parse(repr(p))) == p
    solver = init(p, KrylovKitSolver(false); howmany=2)
    @test dimension(solver.basissetrep) == dimension(p.h) == size(solver.basissetrep.sm)[1]
    @test startswith(repr(solver),"MatrixEDSolver")

    res_km = solve(solver)
    values, vectors, info = res_km
    @test length(values) == length(vectors) == info.converged ≥ 2

    # solve with KrylovKitSolver
    solver = init(p, KrylovKitSolver(true); howmany=2)
    va_kd, ve_kd, info_kd = solve(solver)
    @test values ≈ va_kd
    addr = starting_address(res_km.problem.h)
    factor = vectors[1][addr] / ve_kd[1][addr]
    @test vectors[1] ≈ scale(ve_kd[1], factor)
    @test startswith(repr(solver), "KrylovKitDirectEDSolver")

    # solve with LinearAlgebraSolver
    res = @test_logs((:warn, "The keyword(s) \"which\" are unused and will be ignored."),
        solve(p, LinearAlgebraSolver()))
    va_la, ve_la, info_la = res

    @test values[1:2] ≈ va_la[1:2]
    factor = vectors[1][addr] / ve_la[1][addr]
    @test vectors[1] ≈ scale(ve_la[1], factor)

    # solve with ArpackSolver
    solver = init(p, ArpackSolver(); howmany=2)
    va_ae, ve_ae, info_ae = solve(solver)
    @test values[1:2] ≈ va_ae[1:2]
    factor = vectors[1][addr] / ve_ae[1][addr]
    @test vectors[1] ≈ scale(ve_ae[1], factor)

    p2 = ExactDiagonalizationProblem(
        HubbardReal1D(BoseFS(1, 2, 3)), DVec(BoseFS(1, 2, 3) => 2.3)
    )
    @test eval(Meta.parse(repr(p2))) == p2
    s2 = init(p2, KrylovKitSolver(; howmany=3))

    res = solve(s2)
    @test length(res.values) == length(res.vectors) == res.info.converged ≥ 3
    res_full = solve(p2)
    @test res.values[1:3] ≈ res_full.values[1:3]

    p3 = ExactDiagonalizationProblem(
        HubbardMom1D(BoseFS(1, 2, 3)), BoseFS(1, 2, 3)
    )
    @test eval(Meta.parse(repr(p3))) == p3
    s3 = init(p3, KrylovKitSolver(); howmany=5)
    r3 = solve(s3)
    @test r3.success
end
