using Rimu
using Test
using LinearAlgebra
using Statistics, DataFrames
using SafeTestsets

# assuming VERSION ≥ v"1.6"
# the following is needed because random numbers of collections are computed
# differently after version 1.6, and thus the results of many tests change
# for Golden Master Testing (@https://en.wikipedia.org/wiki/Characterization_test)
@assert VERSION ≥ v"1.6"

@safetestset "StatsTools" begin
    include("StatsTools.jl")
end

@safetestset "BitStringAddresses" begin
    include("BitStringAddresses.jl")
end

@safetestset "DictVectors.jl" begin
    include("DictVectors.jl")
end

using Rimu.ConsistentRNG
@testset "ConsistentRNG.jl" begin
    seedCRNG!(127) # uses `RandomNumbers.Xorshifts.Xoshiro256StarStar()`
    @test cRand(UInt128) == 0x31b845fb73fcb6aa392bdbbeabf3ce50

    @test rand(ConsistentRNG.CRNGs[][1],UInt128) == 0xad825b6d8cdd554f3deee102bf4d5704
    # Only looks at first element of the `NTuple`. This should be reproducible
    # regardless of `numthreads()`.
    @test rand(trng(),UInt16) == 0xcd4d
    @test rand(newChildRNG(),UInt16) == 0x6e4c
    @test ConsistentRNG.check_crng_independence(0) == Threads.nthreads()
end

@testset "Hamiltonians.jl" begin
    include("Hamiltonians.jl")
end

@testset "BoseHubbardMom1D" begin
    bfs= BoseFS((1,0,2,1,2,1,1,3))
    @test Hamiltonians.numberoccupiedsites(bfs) == 7
    @test Hamiltonians.num_singly_doubly_occupied_sites(bfs) == (7,3)
    @test Hamiltonians.num_singly_doubly_occupied_sites(onr(bfs)) == Hamiltonians.num_singly_doubly_occupied_sites(bfs)

    ham = Hamiltonians.BoseHubbardMom1D(bfs)
    @test num_offdiagonals(ham,bfs) == 273
    @test get_offdiagonal(ham, bfs, 205) == (BoseFS((1,0,2,1,3,0,0,4)), 0.21650635094610965)
    @test diagonal_element(ham,bfs) ≈ 14.296572875253808
    m = momentum(ham)
    @test diagonal_element(m,bfs) ≈ -1.5707963267948966
    v = DVec(Dict(bfs => 10))
    @test rayleigh_quotient(m, v) ≈ -1.5707963267948966

    ham = Hamiltonians.HubbardMom1D(bfs)
    @test num_offdiagonals(ham,bfs) == 273
    @test get_offdiagonal(ham, bfs, 205) == (BoseFS((1,0,2,1,3,0,0,4)), 0.21650635094610965)
    @test diagonal_element(ham,bfs) ≈ 14.296572875253808
    m = momentum(ham)
    @test diagonal_element(m,bfs) ≈ -1.5707963267948966
    v = DVec(Dict(bfs => 10))
    @test rayleigh_quotient(m, v) ≈ -1.5707963267948966

    fs = BoseFS((1,2,1,0)) # should be zero momentum
    ham = BoseHubbardMom1D(fs,t=1.0)
    m=momentum(ham) # define momentum operator
    mom_fs = diagonal_element(m, fs) # get momentum value as diagonal matrix element of operator
    @test isapprox(mom_fs, 0.0, atol = sqrt(eps())) # check whether momentum is zero
    @test reduce(&,[isapprox(mom_fs, diagonal_element(m,h[1]), atol = sqrt(eps())) for h in offdiagonals(ham, fs)]) # check that momentum does not change for offdiagonals
    # construct full matrix
    smat, adds = Hamiltonians.build_sparse_matrix_from_LO(ham,fs)
    # compute its eigenvalues
    eig = eigen(Matrix(smat))
    # @test eig.values == [-6.681733497641263, -1.663545897706113, 0.8922390118623973, 1.000000000000007, 1.6458537005442135, 2.790321237291681, 3.000000000000001, 3.878480840626051, 7.266981109653349, 9.871403495369677]
    @test reduce(&, map(isapprox, eig.values, [-6.681733497641263, -1.663545897706113, 0.8922390118623973, 1.000000000000007, 1.6458537005442135, 2.790321237291681, 3.000000000000001, 3.878480840626051, 7.266981109653349, 9.871403495369677]))

    # for comparison check real-space Bose Hubbard chain - the eigenvalues should be the same
    hamr = BoseHubbardReal1D(fs,t=1.0)
    smatr, addsr = Hamiltonians.build_sparse_matrix_from_LO(hamr,fs)
    eigr = eigen(Matrix(smatr))
    @test eigr.values[1] ≈ eig.values[1] # check equality for ground state energy

    ham = Hamiltonians.HubbardMom1D(fs,t=1.0)
    m=momentum(ham) # define momentum operator
    mom_fs = diagonal_element(m, fs) # get momentum value as diagonal matrix element of operator
    @test isapprox(mom_fs, 0.0, atol = sqrt(eps())) # check whether momentum is zero
    @test reduce(&,[isapprox(mom_fs, diagonal_element(m,h[1]), atol = sqrt(eps())) for h in offdiagonals(ham, fs)]) # check that momentum does not change for offdiagonals
    # construct full matrix
    smat, adds = Hamiltonians.build_sparse_matrix_from_LO(ham,fs)
    # compute its eigenvalues
    eig = eigen(Matrix(smat))
    # @test eig.values == [-6.681733497641263, -1.663545897706113, 0.8922390118623973, 1.000000000000007, 1.6458537005442135, 2.790321237291681, 3.000000000000001, 3.878480840626051, 7.266981109653349, 9.871403495369677]
    @test reduce(&, map(isapprox, eig.values, [-6.681733497641263, -1.663545897706113, 0.8922390118623973, 1.000000000000007, 1.6458537005442135, 2.790321237291681, 3.000000000000001, 3.878480840626051, 7.266981109653349, 9.871403495369677]))
    @test eigr.values[1] ≈ eig.values[1] # check equality for ground state energy
end

@safetestset "lomc!" begin
    include("lomc.jl")
end

@testset "MemoryStrategy" begin
    # Define the initial Fock state with n particles and m modes
    n = m = 9
    add = nearUniform(BoseFS{n,m})
    H = BoseHubbardReal1D(add; u = 6.0, t = 1.0)
    dv = DVec(add => 1; style=IsStochasticWithThreshold(1.0))
    s_strat = DoubleLogUpdate(targetwalkers=100)

    @testset "NoMemory" begin
        seedCRNG!(12345)
        df = lomc!(
            H, copy(dv);
            laststep=100, s_strat, m_strat=NoMemory(), maxlength=2*dimension(H)
        ).df
        @test sum(df[:,:norm]) ≈ 2610 atol=1
    end

    @testset "DeltaMemory" begin
        seedCRNG!(12345)
        df = lomc!(
            H, copy(dv);
            laststep=100, s_strat, m_strat=DeltaMemory(1), maxlength=2*dimension(H)
        ).df
        @test sum(df[:,:norm]) ≈ 2610 atol=1

        seedCRNG!(12345)
        df = lomc!(
            H, copy(dv);
            laststep=100, s_strat, m_strat=DeltaMemory(10), maxlength=2*dimension(H)
        ).df
        @test sum(df[:,:norm]) ≈ 2236 atol=1
    end

    @testset "DeltaMemory2" begin
        seedCRNG!(12345)
        df = lomc!(
            H, copy(dv);
            laststep=100, s_strat, m_strat=Rimu.DeltaMemory2(1), maxlength=2*dimension(H)
        ).df
        @test sum(df[:,:norm]) ≈ 2610 atol=1

        seedCRNG!(12345)
        df = lomc!(
            H, copy(dv);
            laststep=100, s_strat, m_strat=Rimu.DeltaMemory2(10), maxlength=2*dimension(H)
        ).df
        @test sum(df[:,:norm]) ≈ 2038 atol=1
    end

    @testset "ShiftMemory" begin
        seedCRNG!(12345)
        df = lomc!(
            H, copy(dv);
            laststep=100, s_strat, m_strat=ShiftMemory(1), maxlength=2*dimension(H)
        ).df
        @test sum(df[:,:norm]) ≈ 2610 atol=1

        seedCRNG!(12345)
        df = lomc!(
            H, copy(dv);
            laststep=100, s_strat, m_strat=ShiftMemory(10), maxlength=2*dimension(H)
        ).df
        @test sum(df[:,:norm]) ≈ 3135 atol=1
    end
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
    v = [1,2,3]
    @test walkernumber(v) == norm(v,1)
    dvc= DVec(:a=>2-5im)
    @test StochasticStyle(dvc) isa DictVectors.IsStochastic2Pop
    @test walkernumber(dvc) == 2.0 + 5.0im
    Rimu.purge_negative_walkers!(dvc)
    @test walkernumber(dvc) == 2.0 + 0.0im
    dvi= DVec(:a=>Complex{Int32}(2-5im))
    @test StochasticStyle(dvi) isa DictVectors.IsStochastic2Pop
    dvr = DVec(i => cRandn() for i in 1:100; capacity = 100)
    @test walkernumber(dvr) ≈ norm(dvr,1)
end

using Rimu.Blocking
@testset "Blocking.jl" begin
    n=10
    a = rand(n)
    m = mean(a)
    @test m == sum(a)/n
    myvar(a,m) = sum((a .- m).^2)/n
    @test var(a) == sum((a .- m).^2)/(n-1)
    @test var(a, corrected=false) == sum((a .- m).^2)/n == myvar(a,m)
    @test var(a, corrected=false) == var(a, corrected=false, mean = m)
    # @benchmark myvar($a, $m)
    # @benchmark var($a, corrected=false, mean = $m)
    # evaluating the above shows that the library function is faster and avoids
    # memory allocations completely

    # test se
    a = collect(1:10)
    @test Rimu.Blocking.se(a) ≈ 0.9574271077563381
    @test Rimu.Blocking.se(a;corrected=false) ≈ 0.9082951062292475
    # test autocovariance
    @test autocovariance(a,1) ≈ 6.416666666666667
    @test autocovariance(a,1;corrected=false) ≈ 5.775
    # test covariance
    b = collect(2:11)
    @test covariance(a,b) ≈ 9.166666666666666
    @test covariance(a,b;corrected=false) ≈ 8.25
    c = collect(2:20) # should be truncated
    @test covariance(a,b) == covariance(a,c)
    @test covariance(a,b;corrected=false) == covariance(a,c;corrected=false)

    # Define the initial Fock state with n particles and m modes
    n = m = 9
    aIni = nearUniform(BoseFS{n,m})
    ham = BoseHubbardReal1D(aIni; u = 6.0, t = 1.0)
    pa = RunTillLastStep(laststep = 1000)

    # standard fciqmc
    s = DoubleLogUpdate(targetwalkers = 100)
    svec = DVec(Dict(aIni => 2))
    StochasticStyle(svec)
    vs = copy(svec)
    post_step = ProjectedEnergy(ham, svec)
    τ_strat = ConstantTimeStep()

    seedCRNG!(12345) # uses RandomNumbers.Xorshifts.Xoroshiro128Plus()
    # @time rdfs = fciqmc!(vs, pa, ham, s, r_strat, τ_strat, similar(vs))
    @time rdfs = lomc!(
        ham, vs; params = pa, s_strat = s, post_step, τ_strat, wm = similar(vs),
    ).df
    r = autoblock(rdfs, start=101)
    @test r.s̄ ≈ -5.14 atol=0.1
    @test r.σs ≈ 0.27 atol=0.1
    @test r.ē ≈ -5.52 atol=0.1
    @test r.σe ≈ 0.39 atol=0.1
    @test r.k == 6

    g = growthWitness(rdfs, b=50)
    # @test sum(g) ≈ -5725.3936298329545
    @test length(g) == nrow(rdfs)
    g = growthWitness(rdfs, b=50, pad = :false)
    @test length(g) == nrow(rdfs) - 50
    @test_throws AssertionError growthWitness(rdfs.norm, rdfs.shift[1:end-1],rdfs.dτ[1])
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

using Rimu.EmbarrassinglyDistributed
@testset "EmbarrassinglyDistributed" begin
    add = BoseFS((1,1,0,1))
    v = DVec(add => 2; capacity = 200)
    ham = BoseHubbardReal1D(add, u=4.0)
    @test setup_workers(4) == 4 # add workers and load code (Rimu and its modules)
    seedCRNGs_workers!(127)     # seed rgns on workers deterministically
    nt = d_lomc!(ham, v; eqsteps = 1_000, laststep = 21_000) # perform parallel lomc!
    @test [size(df)[1] for df in nt.dfs] == [6000, 6000, 6000, 6000]
    ntc = combine_dfs(nt) # combine results into one DataFrame
    @test size(ntc.df)[1] == 20997
    energies = autoblock(ntc) # perform `autoblock()` discarding `eqsteps` time steps
    # in a single line:
    # energies = d_lomc!(ham, v; eqsteps = 1_000, laststep = 21_000) |> combine_dfs |> autoblock
    @test ismissing(energies.ē) && ismissing(energies.σe)
    # golden master test on results because qmc evolution is deterministic
    @test energies.s̄ ≈ -4.1 atol=0.1
    @test energies.σs ≈ 0.006 atol=1e-3
end

@testset "BoseFS2C" begin
    bfs2c = BoseFS2C(BoseFS((1,2,0,4)),BoseFS((4,0,3,1)))
    @test typeof(bfs2c) <: BoseFS2C{7,8,4}
    @test Hamiltonians.numberoccupiedsites(bfs2c.bsa) == 3
    @test Hamiltonians.numberoccupiedsites(bfs2c.bsb) == 3
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

@testset "deprecated" begin
    @test @capture_err(EveryTimeStep()) ≠ ""
    @test @capture_err(EveryKthStep()) ≠ ""
    @suppress_err begin
        @test EveryTimeStep().k == 1
        @test EveryTimeStep().writeinfo == false

        @test EveryKthStep(k=1000).k == 1000
        @test EveryKthStep().k == 10
        @test EveryKthStep().writeinfo == false
    end
end

# Note: This last test is set up to work on Pipelines, within a Docker
# container, where everything runs as root. It should also work locally,
# where typically mpi is not (to be) run as root.
@testset "MPI" begin
    # read name of mpi executable from environment variable if defined
    # necessary for allow-run-as root workaround for Pipelines
    mpiexec = haskey(ENV, "JULIA_MPIEXEC") ? ENV["JULIA_MPIEXEC"] : "mpirun"
    is_local = !haskey(ENV, "CI")

    # use user installed julia executable if available
    if isfile(joinpath(homedir(),"bin/julia"))
        juliaexec = joinpath(homedir(),"bin/julia")
    else
        juliaexec = "julia"
    end
    run(`which $mpiexec`)

    if is_local
        flavours = ["os", "ptp", "ata"]
        rr = run(`$mpiexec -np 2 $juliaexec -t 1 mpi_runtests.jl`)
        @test rr.exitcode == 0
    else
        @info "not testing MPI on CI"
    end
end
