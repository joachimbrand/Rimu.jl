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
using Rimu.StatsTools, Rimu.RimuIO


# assuming VERSION ≥ v"1.6"
# the following is needed because random numbers of collections are computed
# differently after version 1.6, and thus the results of many tests change
# for Golden Master Testing (@https://en.wikipedia.org/wiki/Characterization_test)
@assert VERSION ≥ v"1.6"

@test Rimu.PACKAGE_VERSION == VersionNumber(TOML.parsefile(pkgdir(Rimu, "Project.toml"))["version"])

@safetestset "Interfaces" begin
    include("Interfaces.jl")
end

@safetestset "StatsTools" begin
    include("StatsTools.jl")
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

@safetestset "lomc!" begin
    include("lomc.jl")
end

@testset "IsDeterministic with Vector" begin
    ham = HubbardReal1D(BoseFS((1, 1, 1, 1)))
    dim = dimension(ham)
    sm, basis = Rimu.Hamiltonians.build_sparse_matrix_from_LO(ham, starting_address(ham))
    @test dim == length(basis)
    # run lomc! in deterministic mode with Matrix and Vector
    a = lomc!(sm, ones(dim); threading=true).df # no actual threading is done, though
    b = lomc!(sm, ones(dim); threading=false).df
    @test a.shift ≈ b.shift
    # run lomc! in deterministic mode with Hamiltonian and DVec
    v = DVec(k=>1.0 for k in basis) # corresponds to `ones(dim)`
    c = lomc!(ham, v).df
    @test a.shift ≈ c.shift
end

@testset "helpers" begin
    @testset "walkernumber" begin
        v = [1,2,3]
        @test walkernumber(v) == norm(v,1)
        dvc = DVec(:a => 2-5im)
        @test StochasticStyle(dvc) isa StochasticStyles.IsStochastic2Pop
        @test walkernumber(dvc) == 2.0 + 5.0im
        Rimu.purge_negative_walkers!(dvc)
        @test walkernumber(dvc) == 2.0 + 0.0im
        dvi= DVec(:a=>Complex{Int32}(2-5im))
        @test StochasticStyle(dvi) isa StochasticStyles.IsStochastic2Pop
        dvr = DVec(i => randn() for i in 1:100; capacity = 100)
        @test walkernumber(dvr) ≈ norm(dvr,1)
    end
    @testset "MultiScalar" begin
        a = Rimu.MultiScalar(1, 1.0, SVector(1))
        b = Rimu.MultiScalar(SVector(2, 3.0, SVector(4)))
        c = Rimu.MultiScalar((3, 4.0, SVector(5)))
        @test a + b == c
        @test_throws MethodError a + Rimu.MultiScalar(1, 1, 1)

        @test Rimu.combine_stats(a) == a
        @test Rimu.combine_stats([a, b]) == c
    end
end

@testset "RimuIO" begin
    @testset "save_df, load_df" begin
        file = joinpath(@__DIR__, "tmp.arrow")
        rm(file; force=true)

        df = DataFrame(a=[1, 2, 3], b=Complex{Float64}[1, 2, 3+im], d=rand(Complex{Int}, 3))
        RimuIO.save_df(file, df)
        df2 = RimuIO.load_df(file)
        @test df == df2

        rm(file)
    end
    @testset "save_dvec, load_dvec" begin
        # BSON is currently broken on 1.8
        if VERSION ≤ v"1.7"
            file1 = joinpath(@__DIR__, "tmp1.bson")
            file2 = joinpath(@__DIR__, "tmp2.bson")
            rm(file1; force=true)
            rm(file2; force=true)

            add = BoseFS2C((1,1,0,1), (1,1,0,0))
            dv = InitiatorDVec(add => 1.0, style=IsDynamicSemistochastic(abs_threshold=3.5))
            H = BoseHubbardMom1D2C(add)

            _, state = lomc!(H, dv; replica=NoStats(2))
            RimuIO.save_dvec(file1, state.replicas[1].v)
            RimuIO.save_dvec(file2, state.replicas[2].v)

            dv1 = RimuIO.load_dvec(file1)
            dv2 = RimuIO.load_dvec(file2)

            @test dv1 == state.replicas[1].v
            @test typeof(dv2) == typeof(state.replicas[1].v)
            @test StochasticStyle(dv1) == StochasticStyle(state.replicas[1].v)
            @test storage(dv2) == storage(state.replicas[2].v)

            rm(file1; force=true)
            rm(file2; force=true)
        end
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

    smat2cReal, adds2cReal = Hamiltonians.build_sparse_matrix_from_LO(Ĥ2cReal,aIni2cReal)
    eig2cReal = eigen(Matrix(smat2cReal))
    smat2cMom, adds2cMom = Hamiltonians.build_sparse_matrix_from_LO(Ĥ2cMom,aIni2cMom)
    eig2cMom = eigen(Matrix(smat2cMom))
    @test eig2cReal.values[1] ≈ eig2cMom.values[1]
end

@safetestset "KrylovKit" begin
    include("KrylovKit.jl")
end
@safetestset "RMPI" begin
    include("RMPI.jl")
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

# Note: This test is only for local testing, as MPI is tested separately on CI
@testset "MPI" begin
    # read name of mpi executable from environment variable if defined
    # necessary for allow-run-as root workaround for Pipelines
    mpiexec = haskey(ENV, "JULIA_MPIEXEC") ? ENV["JULIA_MPIEXEC"] : "mpirun"
    is_local = !haskey(ENV, "CI")

    juliaexec = Base.julia_cmd()

    if is_local
        mpi_test_filename = isfile("mpi_runtests.jl") ?  "mpi_runtests.jl" : "test/mpi_runtests.jl"
        if isfile(mpi_test_filename)
            rr = run(`$mpiexec -np 2 $juliaexec -t 1 $mpi_test_filename`)
            @test rr.exitcode == 0
        else
            @warn "Could not find mpi_runtests.jl. Not testing MPI."
        end
    else
        @info "not testing MPI on CI"
    end
end
