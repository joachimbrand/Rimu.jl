using DataFrames
using Rimu
using LinearAlgebra
using SafeTestsets
using StaticArrays
using Statistics
using Suppressor
using Logging, TerminalLoggers
using TOML
using Test
using Rimu.StatsTools
using ExplicitImports: check_no_implicit_imports


@test Rimu.PACKAGE_VERSION == VersionNumber(TOML.parsefile(pkgdir(Rimu, "Project.toml"))["version"])

@safetestset "Interfaces" begin
    include("Interfaces.jl")
end

@safetestset "ExactDiagonalization" begin
    include("ExactDiagonalization.jl")
end

@safetestset "BitStringAddresses" begin
    include("BitStringAddresses.jl")
end

@safetestset "StochasticStyles" begin
    include("StochasticStyles.jl")
end

@safetestset "DictVectors" begin
    include("DictVectors.jl")
end

@testset "Hamiltonians" begin
    include("Hamiltonians.jl")
end

@safetestset "projector_monte_carlo_problem" begin
    include("projector_monte_carlo_problem.jl")
end

@safetestset "lomc!" begin
    include("lomc.jl")
end

@safetestset "RimuIO" begin
    include("RimuIO.jl")
end

@safetestset "StatsTools" begin
    include("StatsTools.jl")
end

using Rimu: replace_keys, delete_and_warn_if_present, clean_and_warn_if_others_present
@testset "helpers" begin
    @testset "walkernumber" begin
        v = [1,2,3]
        @test walkernumber(v) == norm(v,1)
        dvc = DVec(:a => 2-5im)
        @test StochasticStyle(dvc) isa StochasticStyles.IsStochastic2Pop
        @test walkernumber(dvc) == 2.0 + 5.0im
        dvi= DVec(:a=>Complex{Int32}(2-5im))
        @test StochasticStyle(dvi) isa StochasticStyles.IsStochastic2Pop
        dvr = DVec(i => randn() for i in 1:100; capacity = 100)
        @test walkernumber(dvr) ≈ norm(dvr,1)
    end
    @testset "MultiScalar" begin
        a = Rimu.MultiScalar(1, 1.0, SVector(1))
        @test a[1] ≡ 1
        @test a[2] ≡ 1.0
        @test a[3] ≡ SVector(1)
        @test length(a) == 3
        @test collect(a) == [1, 1.0, SVector(1)]
        b = Rimu.MultiScalar(SVector(2, 3.0, SVector(4)))
        for op in (+, min, max)
            c = op(a, b)
            @test op(a[1], b[1]) == c[1]
            @test op(a[2], b[2]) == c[2]
            @test op(a[2], b[2]) == c[2]
        end
        @test_throws MethodError a + Rimu.MultiScalar(1, 1, 1)
    end

    @testset "keyword helpers" begin
        nt = (; a=1, b=2, c = 3, d = 4)
        nt2 = replace_keys(nt, (:a => :x, :b => :y, :u => :v))
        @test nt2 == (c=3, d=4, x=1, y=2)
        nt3 = @test_logs((:warn, "The keyword(s) \"a\", \"b\" are unused and will be ignored."),
            delete_and_warn_if_present(nt, (:a, :b, :u)))
        @test nt3 == (; c = 3, d = 4)
        nt4 = @test_logs((:warn, "The keyword(s) \"c\", \"d\" are unused and will be ignored."),
            clean_and_warn_if_others_present(nt, (:a, :b, :u)))
        @test nt4 == (; a = 1, b = 2)
    end
end

@testset "BoseFS2C" begin
    bfs2c = BoseFS2C(BoseFS((1,2,0,4)),BoseFS((4,0,3,1)))
    @test typeof(bfs2c) <: BoseFS2C{7,8,4}
    @test num_occupied_modes(bfs2c.bsa) == 3
    @test num_occupied_modes(bfs2c.bsb) == 3
    @test onr(bfs2c.bsa) == [1,2,0,4]
    @test onr(bfs2c.bsb) == [4,0,3,1]
    @test Hamiltonians.bose_hubbard_2c_interaction(bfs2c) == 8 # n_a*n_b over all sites
end

@testset "TwoComponentBosonicHamiltonian" begin
    aIni2cReal = BoseFS2C(BoseFS((1,1,1,1)),BoseFS((1,1,1,1))) # real space two-component
    Ĥ2cReal = BoseHubbardReal1D2C(aIni2cReal; ua = 6.0, ub = 6.0, ta = 1.0, tb = 1.0, v= 6.0)
    hamA = HubbardReal1D(BoseFS((1,1,1,1)); u=6.0, t=1.0)
    hamB = HubbardReal1D(BoseFS((1,1,1,1)); u=6.0)
    @test hamA == Ĥ2cReal.ha
    @test hamB == Ĥ2cReal.hb
    @test num_offdiagonals(Ĥ2cReal,aIni2cReal) == 16
    @test num_offdiagonals(Ĥ2cReal,aIni2cReal) == num_offdiagonals(Ĥ2cReal.ha,aIni2cReal.bsa)+num_offdiagonals(Ĥ2cReal.hb,aIni2cReal.bsb)
    @test dimension(Ĥ2cReal) == 1225
    @test dimension(Float64, Ĥ2cReal) == 1225.0

    hp2c = offdiagonals(Ĥ2cReal,aIni2cReal)
    @test length(hp2c) == 16
    @test hp2c[1][1] == BoseFS2C(BoseFS((0,2,1,1)), BoseFS((1,1,1,1)))
    @test hp2c[1][2] ≈ -1.4142135623730951
    @test diagonal_element(Ĥ2cReal,aIni2cReal) ≈ 24.0 # from the V term

    aIni2cMom = BoseFS2C(BoseFS((0,4,0,0)),BoseFS((0,4,0,0))) # momentum space two-component
    Ĥ2cMom = BoseHubbardMom1D2C(aIni2cMom; ua = 6.0, ub = 6.0, ta = 1.0, tb = 1.0, v= 6.0)
    @test num_offdiagonals(Ĥ2cMom,aIni2cMom) == 9
    @test dimension(Ĥ2cMom) == 1225
    @test dimension(Float64, Ĥ2cMom) == 1225.0

    hp2cMom = offdiagonals(Ĥ2cMom,aIni2cMom)
    @test length(hp2cMom) == 9
    @test hp2cMom[1][1] == BoseFS2C(BoseFS((1,2,1,0)), BoseFS((0,4,0,0)))
    @test hp2cMom[1][2] ≈ 2.598076211353316

    smat2cReal, adds2cReal = ExactDiagonalization.build_sparse_matrix_from_LO(Ĥ2cReal,aIni2cReal)
    eig2cReal = eigen(Matrix(smat2cReal))
    smat2cMom, adds2cMom = ExactDiagonalization.build_sparse_matrix_from_LO(Ĥ2cMom, aIni2cMom)
    eig2cMom = eigen(Matrix(smat2cMom))
    @test eig2cReal.values[1] ≈ eig2cMom.values[1]
end

@safetestset "KrylovKit" begin
    include("KrylovKit.jl")
end

@testset "Logging" begin
    default_logger()
    l = Base.global_logger()
    @test l isa Logging.ConsoleLogger
    sl = smart_logger()
    if isdefined(Main, :IJulia) && Main.IJulia.inited
        @test sl isa ConsoleProgressMonitor.ProgressLogRouter
        @info "Jupyter progress bar" sl
    elseif isa(stderr, Base.TTY) && (get(ENV, "CI", nothing) ≠ true)
        @test sl isa TerminalLoggers.TerminalLogger
        @info "Terminal progress bar" sl
    else
        @test sl isa Logging.ConsoleLogger
        @info "No progress bar" sl
    end
    @test default_logger() isa Logging.ConsoleLogger
end

@safetestset "doctests" begin
    include("doctests.jl")
end

@safetestset "ExplicitImports" begin
    using Rimu
    using ExplicitImports
    # Check that no implicit imports are used in the Rimu module.
    # See https://ericphanson.github.io/ExplicitImports.jl/stable/
    @test check_no_implicit_imports(Rimu; skip=(Rimu, Base, Core, VectorInterface)) === nothing
    # If this test fails, make your import statements explicit.
    # For example, replace `using Foo` with `using Foo: bar, baz`.
end

# Note: Running Rimu with several MPI ranks is tested seperately on GitHub CI and not here.
