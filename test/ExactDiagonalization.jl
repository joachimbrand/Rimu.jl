using Rimu
using Test
using Random
using Suppressor
using SparseArrays

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
        bsr = BasisSetRepresentation(ham)
        @test length(bsr.basis) == dimension(bsr) ≤ dimension(ham)
        @test_throws ArgumentError BasisSetRepresentation(ham, BoseFS((1, 2, 3))) # wrong address type
        @test Matrix(bsr) == Matrix(bsr.sparse_matrix) == Matrix(ham)
        @test sparse(bsr) == bsr.sparse_matrix == sparse(ham)
        addr2 = bsr.basis[2]
        @test starting_address(BasisSetRepresentation(ham, addr2)) == addr2
        @test isreal(ham) == (eltype(ham) <: Real)
        @test isdiag(ham) == (LOStructure(ham) ≡ IsDiagonal())
        @test ishermitian(ham) == (LOStructure(ham) ≡ IsHermitian())
        @test issymmetric(ham) == (ishermitian(ham) && isreal(ham))

        # Test for non-Hamiltonian
        g2 = G2RealCorrelator(1)
        bsr_g2 = BasisSetRepresentation(g2, bsr.basis)
        @test isdiag(bsr_g2.sparse_matrix)

        bsr_g2 = BasisSetRepresentation(g2, bsr.basis[1])
        @test size(bsr_g2.sparse_matrix) == (1, 1)
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
        @test bsr.sparse_matrix == bsrt.sparse_matrix
        # pass addresses and generate reachable basis
        @test BasisSetRepresentation(ham, bsr.basis, sort=true).basis == bsr_orig.basis

        filterfun(fs) = maximum(onr(fs)) < 8
        mat_cut_index = filterfun.(BasisSetRepresentation(ham; sort=true).basis)
        mat_cut_manual = mat_orig[mat_cut_index, mat_cut_index]
        mat_cut = Matrix(ham; filter=filterfun, sort=true)
        @test mat_cut == mat_cut_manual
    end

    @testset "max_depth and minimum_size" begin
        addr = BoseFS(5, 1 => 1)
        ham = HubbardRealSpace(addr; geometry=CubicGrid((5,), (false,)))

        basis_addr = build_basis(addr)

        @test build_basis(ham; max_depth=0) == [addr]
        @test build_basis(ham; max_depth=1) == basis_addr[1:2]
        @test build_basis(ham; max_depth=2) == basis_addr[1:3]
        @test build_basis(ham; max_depth=3) == basis_addr[1:4]
        @test build_basis(ham; max_depth=4) == basis_addr

        @test build_basis(ham; minimum_size=0) == [addr]
        @test build_basis(ham; minimum_size=1) == basis_addr[1:2]
        @test build_basis(ham; minimum_size=2) == basis_addr[1:3]
        @test build_basis(ham; minimum_size=3) == basis_addr[1:4]
        @test build_basis(ham; minimum_size=4) == basis_addr
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

    @testset "isapprox_enforce_hermitian!" begin
        matrix = sprand(100, 100, 0.2)
        matrix .+= matrix'
        for _ in 1:1000
            matrix[rand(1:100), rand(1:100)] += 1e-9
        end
    
        matrix1 = copy(matrix)
        @test !ishermitian(matrix1)
        @test isapprox_enforce_hermitian!(matrix1)
        @test ishermitian(matrix1)
    
        matrix2 = copy(matrix)
        @test !isapprox_enforce_hermitian!(matrix2; atol=1e-12)
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

        # build_basis for HardcoreBoseFS and FermiFS
        hbas = build_basis(HardcoreBoseFS{2, 4})
        @test length(hbas) == dimension(HardcoreBoseFS{2, 4})
        fbas = build_basis(FermiFS{2, 4})
        @test all(f.bs == h.bs for (f, h) in zip(fbas, hbas))


        # build_basis with missing N for FermiFS and HardcoreBoseFS
        fbasis = build_basis(FermiFS{missing}(1, 0, 1))
        @test fbasis == build_basis(FermiFS{missing,3})
        @test eltype(fbasis) <: FermiFS{missing,3}
        @test length(fbasis) == 8 == dimension(FermiFS{missing}(1, 0, 1))
        hbasis = build_basis(HardcoreBoseFS{missing}(1, 0, 1))
        @test eltype(hbasis) <: HardcoreBoseFS{missing,3}
        @test length(hbasis) == 8 == dimension(HardcoreBoseFS{missing}(1, 0, 1))
        @test all(f.bs == h.bs for (f, h) in zip(fbasis, hbasis))
        @test_throws ArgumentError build_basis(FermiFS{missing, 64})
    end

    @testset "fock build basis" begin
        for addr in (
            BoseFS(10, 10),
            FermiFS(1, 1, 0),
            FermiFS(1, 0),
            BoseFS(1, 1, 1, 1, 1, 2, 1, 1),
            FermiFS(1, 1, 1, 0, 0, 0),
            FermiFS2C((1, 1, 0, 0, 0, 0), (0, 1, 0, 1, 1, 0)),
            CompositeFS(BoseFS(2, 0, 0), FermiFS(1, 0, 1), BoseFS(0, 2, 0), BoseFS(1, 0, 0)),
        )
            H = HubbardRealSpace(addr)
            @test build_basis(addr) == build_basis(H; sort=true)
        end
        # These don't work with HubbardRealSpace
        for addr in (
            FermiFS(1),
            BoseFS(1),
        )
            @test build_basis(addr) == [addr]
        end
    end
end

@testset "OperatorAsMap" begin
    @testset "Fail for AdjointUnknown()" begin
        @test_throws ArgumentError LinearMap(Transcorrelated1D(FermiFS2C((1,),(1,))))
    end
    for ham in (
        HubbardMom1D(FermiFS2C((0,0,1,0,0), (0,1,0,1,0))),       # real symmetric
        GutzwillerSampling(HubbardReal1D(BoseFS(1,1,1,1)), 0.5), # real non-hermitian
        ExtendedHubbardReal1D(FermiFS(1,0,1,0,0); t=im),         # complex hermitian
        MatrixHamiltonian(rand(ComplexF64, 10, 10)),             # complex non-hermitian
        )
        @testset "on $(typeof(ham))" begin
            basis = build_basis(ham; sort=true)
            op = LinearMap(ham, basis)

            @testset "basic properties" begin
                @test_throws ArgumentError LinearMap(ham, [:a, :b, :c])
                @test eltype(op) == eltype(ham)
                @test size(op) == (length(basis), length(basis))
                @test isreal(op) == isreal(ham)
                @test ishermitian(op) == ishermitian(ham)
                @test issymmetric(op) == issymmetric(ham)
                @test op' == LinearMap(ham', basis)
            end
            @testset "Other constructors" begin
                op_alt1 = LinearMap(ham; full_basis=false)
                @test size(op_alt1) == size(op)
                @test sort(op_alt1.basis) == basis

                op_alt2 = LinearMap(ham; basis)
                @test size(op_alt2) == size(op)
                @test op_alt2.basis == basis

                if !isa(ham, MatrixHamiltonian)
                    op_alt3 = LinearMap(ham; full_basis=true)
                    @test size(op_alt3, 1) ≥ size(op, 1)
                    @test issubset(basis, op_alt3.basis)

                    op_alt4 = LinearMap(ham, basis[1])
                    @test size(op_alt4, 1) == size(op, 1)
                    @test sort(op_alt4.basis) == basis
                end
            end
            @testset "BasisSetRepresentation" begin
                bsr = BasisSetRepresentation(op)
                @test bsr.basis == op.basis
                @test bsr.sparse_matrix == sparse(ham, basis)
            end
            @testset "*, mul!, dot, LinearMaps stuff" begin
                matrix = sparse(op)
                matrix_dense = Matrix(op)
                @test issparse(matrix)
                @test matrix_dense isa Matrix
                @test matrix == matrix_dense

                v = rand(length(basis)) + rand(length(basis)) .* im
                w = rand(length(basis)) + rand(length(basis)) .* im

                @test matrix * v ≈ op * v
                @test op * v == op(v)
                @test dot(v, matrix, w) ≈ dot(v, op, w)
                @test dot(v, matrix, w) ≈ dot(v, op, w)

                α, β = rand(2)
                w1 = copy(w)
                w2 = copy(w)
                mul!(w1, matrix, v, α, β)
                @test mul!(w2, op, v, α, β) ≡ w2
                @test w1 ≈ w2
            end
        end
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
    @test dimension(solver.basis_set_rep) == size(solver.basis_set_rep.sparse_matrix)[1] ≤ dimension(p.hamiltonian)
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

Random.seed!(1234) # for reproducibility, as some solvers start with random vectors

@testset "ExactDiagonalizationProblem" begin
    hams = (
        HubbardMom1D(FermiFS2C((0,0,1,0,0), (0,1,0,1,0))),       # real symmetric
        GutzwillerSampling(HubbardReal1D(BoseFS(1,1,1,1)), 0.5), # real non-hermitian
        ExtendedHubbardReal1D(FermiFS(0,1,0,1,0,1,0); t=im),     # complex hermitian
        MatrixHamiltonian(rand(ComplexF64, 10, 10)),               # complex non-hermitian
    )
    algs = (
        LinearAlgebraSolver(; permute=true),
        KrylovKitSolver(true; howmany=3),
        KrylovKitSolver(false; krylovdim=10, maxiter=1000),
        ArpackSolver(true),
        ArpackSolver(false; howmany=2, maxiters=10_000),
        LOBPCGSolver(true, howmany=2, maxiters=10_000),
        LOBPCGSolver(false, maxiters=10_000),
    )

    for ham in hams, alg in algs
        @testset "$(typeof(ham)) with $alg" begin
            if !ishermitian(ham) && alg isa LOBPCGSolver
                prob = ExactDiagonalizationProblem(ham)
                @test_throws ArgumentError init(prob, alg)
                continue
            end
            @testset "show and problem basics" begin
                prob = ExactDiagonalizationProblem(ham, howmany=5)

                # parsing show does not quite work for that one due to roundinge errors
                if ham ≠ hams[4]
                    @test eval(Meta.parse(repr(alg))) == alg
                    @test eval(Meta.parse(repr(prob))) == prob
                end
                @test dimension(prob) == dimension(ham)

                solver = init(prob, alg)
                @test repr(solver) isa String # no error on print
                @test solver.problem == prob
            end
            @testset "Sanity checks" begin
                prob = ExactDiagonalizationProblem(ham)
                result = solve(prob)
                @test result isa Rimu.ExactDiagonalization.EDResult
                @test result.success
                @test length(result.values) == length(result.vectors)
                @test length(result.coefficient_vectors) == length(result.vectors)

                for (i, dv) in enumerate(result.vectors)
                    @test DVec(zip(result.basis, result.coefficient_vectors[i])) ≈ dv
                end

                # Make sure printing doesn't error
                @test repr(result) isa String
                @test repr(result.info) isa String
            end

            @testset "Setting initial_vector" begin
                eig = eigen(Matrix(ham))
                addr = starting_address(ham)
                for v0 in (
                    addr, [addr], (addr,),
                    DVec(addr=>1.0), PDVec(addr=>1.0),
                    freeze(DVec([addr=>1])),
                )
                    prob = ExactDiagonalizationProblem(ham, v0)
                    result = solve(prob, alg)
                    @test result.values[1] ≈ eig.values[1]
                end
            end
            if alg ∉ algs[[1, 2, 4]]
                # break here so that testing doesn't take too long. We test the
                # DenseEDSolver and two versions of IterativeEDSolver.
                continue
            end
            @testset "Set algorithm in different places" begin
                prob = ExactDiagonalizationProblem(ham; algorithm=alg)
                @test init(prob).algorithm == alg
                @test init(prob, LinearAlgebraSolver()).algorithm == LinearAlgebraSolver()

                @test_logs(
                    (:warn, "The keyword(s) \"algorithm\" are unused and will be ignored."),
                    solve(prob, KrylovKitSolver())
                )
            end
            @testset "Unused kwargs" begin
                prob = ExactDiagonalizationProblem(ham; one=1)
                solver = init(prob, alg; two=2)
                @test_logs(
                    (:warn, "The keyword(s) \"one\", \"two\", \"three\" are unused and will be ignored."),
                    solve(solver; three=3)
                )
            end
            @testset "verbose" begin
                prob = ExactDiagonalizationProblem(ham; verbose=true)
                err = @capture_err solve(prob, alg)
                @test err ≠ ""
                @test !occursin("Warning", err)
            end
        end
    end
    @testset "Energy comparisons" begin
        for ham in (
            HubbardReal1D(BoseFS(1, 2, 3)),
            HubbardMom1D(BoseFS(1, 2, 3)),
        )
            prob = ExactDiagonalizationProblem(ham)
            eigvals = solve(prob).values

            smallest = map(algs[2:end]) do alg
                solve(prob, alg).values[1]
            end
            @test all(≈(eigvals[1]), smallest)

            first_excited = map(algs[2:end]) do alg
                prob = ExactDiagonalizationProblem(ham; howmany=2)
                solve(prob, alg).values[2]
            end
            @test all(≈(eigvals[2]), first_excited)

            largest = map(algs[2:end]) do alg
                prob = ExactDiagonalizationProblem(ham; which=:LR)
                solve(prob, alg).values[1]
            end
            @test all(≈(eigvals[end]), largest)
        end
    end
    @testset "Unsuccessful solve warning" begin
        ham = HubbardMom1D(BoseFS(10, 5 => 10); u=6.0)
        for alg in algs[2:end]
            prob = ExactDiagonalizationProblem(ham; howmany=10)
            result = @test_logs((:warn,), solve(prob, alg; maxiters=1))
            @test !result.success
        end
    end

    @testset "General" begin
        @testset "Bad starting vector" begin
            ham = HubbardReal1D(BoseFS(1, 2, 3))
            prob = ExactDiagonalizationProblem(ham, [1, 2, 3])
            @test_throws ArgumentError init(prob, KrylovKitSolver(true))
            @test_throws ArgumentError init(prob, ArpackSolver())
        end

        @testset "LOBPCG which errors" begin
            prob = ExactDiagonalizationProblem(HubbardMom1D(BoseFS(0, 2, 0)))
            @test_throws ArgumentError solve(prob, LOBPCGSolver(); which=:LM)
        end
    end
end
