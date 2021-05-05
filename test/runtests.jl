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

using Rimu.FastBufs
@testset "FastBufs.jl" begin
    fb = FastBuf{Float64}(2000) # instantiate a FastBuf
    [push!(fb,rand()) for i=1:1000] # fill it with numbers
    @test length(fb)==1000
    @test FastBufs.capacity(fb)==2000
    fb2 = similar(fb)
    fb3 = FastBuf{Float64}(500)
    fb4 = FastBuf{Float64}(1500)
    @test_throws ErrorException copyto!(fb3,fb)
    copyto!(fb2,fb)
    copyto!(fb4,fb)
    @test fb == fb2 == fb4
    @test reverse(collect(fb)) == [pop!(fb) for i=1:length(fb)]
    @test isempty(fb)
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
    v = DVec(Dict(bfs => 10), 1000)
    @test rayleigh_quotient(m, v) ≈ -1.5707963267948966

    ham = Hamiltonians.HubbardMom1D(bfs)
    @test num_offdiagonals(ham,bfs) == 273
    @test get_offdiagonal(ham, bfs, 205) == (BoseFS((1,0,2,1,3,0,0,4)), 0.21650635094610965)
    @test diagonal_element(ham,bfs) ≈ 14.296572875253808
    m = momentum(ham)
    @test diagonal_element(m,bfs) ≈ -1.5707963267948966
    v = DVec(Dict(bfs => 10), 1000)
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


@testset "fciqmc.jl" begin
    ham = BoseHubbardReal1D(
        n = 9,
        m = 9,
        u = 6.0,
        t = 1.0,
        AT = BoseFS{9,9})
    aIni = nearUniform(ham)
    pa = RunTillLastStep(laststep = 100)

    # standard fciqmc
    s = LogUpdateAfterTargetWalkers(targetwalkers = 100)
    svec = DVec(Dict(aIni => 2), dimension(ham))
    StochasticStyle(svec)
    vs = copy(svec)
    seedCRNG!(12345) # uses RandomNumbers.Xorshifts.Xoroshiro128Plus()
    r_strat = EveryTimeStep()
    τ_strat = ConstantTimeStep()

    @time rdfs = fciqmc!(vs, pa, ham, s, r_strat, τ_strat, similar(vs))
    @test sum(rdfs[:,:spawns]) == 2932

    # fciqmc with delayed shift update
    pa = RunTillLastStep(laststep = 100)
    s = DelayedLogUpdateAfterTargetWalkers(targetwalkers = 100, a = 5)
    svec = DVec(Dict(aIni => 2), dimension(ham))
    StochasticStyle(svec)
    vs = copy(svec)
    seedCRNG!(12345) # uses RandomNumbers.Xorshifts.Xoroshiro128Plus()
    @time rdfs = fciqmc!(vs, pa, ham, s, r_strat, τ_strat, similar(vs))
    @test sum(rdfs[:,:spawns]) == 8230

    # replica fciqmc
    vv = [copy(svec),copy(svec)]
    s = LogUpdateAfterTargetWalkers(targetwalkers = 100)
    pb = RunTillLastStep(laststep = 100)
    seedCRNG!(12345) # uses RandomNumbers.Xorshifts.Xoroshiro128Plus()
    @time rr = fciqmc!(vv, pb, ham, s, r_strat, τ_strat, similar.(vv))
    @test sum(rr[1][:,:xHy]) ≈ -52734.63455801873
end

@testset "fciqmc with BoseFS" begin
    # Define the initial Fock state with n particles and m modes
    n = m = 9
    aIni = nearUniform(BoseFS{n,m})
    ham = BoseHubbardReal1D(aIni; u = 6.0, t = 1.0)

    pa = RunTillLastStep(laststep = 100)

    # standard fciqmc
    s = LogUpdateAfterTargetWalkers(targetwalkers = 100)
    svec = DVec(Dict(aIni => 2), dimension(ham))
    StochasticStyle(svec)
    vs = copy(svec)

    r_strat = EveryTimeStep(projector = UniformProjector())
    # r_strat = EveryTimeStep(projector = copy(svec))
    τ_strat = ConstantTimeStep()

    seedCRNG!(12345) # uses RandomNumbers.Xorshifts.Xoroshiro128Plus()
    @time rdfs = fciqmc!(vs, pa, ham, s, r_strat, τ_strat, similar(vs))
    @test sum(rdfs[:,:spawns]) == 2932

    # fciqmc with delayed shift update
    pa = RunTillLastStep(laststep = 100)
    s = DelayedLogUpdateAfterTargetWalkers(targetwalkers = 100, a = 5)
    svec = DVec(Dict(aIni => 2), dimension(ham))
    StochasticStyle(svec)
    vs = copy(svec)
    seedCRNG!(12345) # uses RandomNumbers.Xorshifts.Xoroshiro128Plus()
    @time rdfs = fciqmc!(vs, pa, ham, s, r_strat, τ_strat, similar(vs))
    @test sum(rdfs[:,:spawns]) == 8230

    # replica fciqmc
    vv = [copy(svec),copy(svec)]
    s = LogUpdateAfterTargetWalkers(targetwalkers = 1_000)
    pb = RunTillLastStep(laststep = 300)
    seedCRNG!(12345) # uses RandomNumbers.Xorshifts.Xoroshiro128Plus()
    @time rr = fciqmc!(vv, pb, ham, s, r_strat, τ_strat, similar.(vv))
    @test sum(rr[1][:,:xHy]) ≈ -9.998205101102287e6

    # replica fciqmc with multithreading
    tup1 = [copy(svec),copy(svec)]
    s = LogUpdateAfterTargetWalkers(targetwalkers = 1_000)
    pb = RunTillLastStep(laststep = 300)
    ws = Tuple(similar(svec) for i=1:Threads.nthreads())
    ww = [ws, copy.(ws)]
    seedCRNG!(12345) # uses RandomNumbers.Xorshifts.Xoroshiro128Plus()
    @time rr = fciqmc!(tup1, pb, ham, s, r_strat, ConstantTimeStep(), ww)

    # large bit string
    n = 200
    m = 200
    aIni = nearUniform(BoseFS{n,m})
    ham = BoseHubbardReal1D(aIni; u = 6.0, t = 1.0)
    iShift = diagonal_element(ham, aIni)

    # standard fciqmc
    tw = 1_000
    s = DoubleLogUpdate(targetwalkers = tw)
    svec = DVec(Dict(aIni => 20), 8*tw)
    StochasticStyle(svec)
    vs = copy(svec)
    r_strat = EveryTimeStep(projector = copy(svec))

    seedCRNG!(12345) # uses RandomNumbers.Xorshifts.Xoroshiro128Plus()
    pa = RunTillLastStep(laststep = 1, shift = iShift, dτ = 0.001)
    @time rdfs = fciqmc!(vs, pa, ham, s, r_strat, τ_strat, similar(vs))
    pa.laststep = 100
    @time rdfs = fciqmc!(vs, pa, rdfs, ham, s, r_strat, τ_strat, similar(vs))
    @test sum(rdfs[:,:spawns]) == 116169

    # single step
    ṽ, w̃, stats = Rimu.fciqmc_step!(ham, copy(vs), pa.shift, pa.dτ, 1.0, similar(vs))
    if Threads.nthreads() == 1 # I'm not sure why this is necessary, but there
        # seems to be a difference
        @test sum(stats) == 488
    elseif Threads.nthreads() == 4
        @test sum(stats) == 488
    end

    # single step multi threading
    cws = capacity(vs)÷Threads.nthreads()+1
    ws = Tuple(similar(vs,cws) for i=1:Threads.nthreads())
    ṽ, w̃, stats = Rimu.fciqmc_step!(ham, copy(vs), pa.shift, pa.dτ, 1.0, ws;
                    batchsize = length(vs)÷4+1)
    if Threads.nthreads() == 1
        @test sum(stats) == 461
    end

    # run 100 steps with multi
    pa.laststep = 200
    @time rdfs = fciqmc!(vs, pa, rdfs, ham, s, r_strat, τ_strat, ws)
    if Threads.nthreads() == 1
        @test sum(rdfs[:,:spawns]) == 135934
    end

    # threaded version of standard fciqmc!
    tw = 1_000
    s = DoubleLogUpdate(targetwalkers = tw)
    svec = DVec(Dict(aIni => 20), 8*tw)
    StochasticStyle(svec)
    vs = copy(svec)
    r_strat = EveryTimeStep(projector = copy(svec))

    seedCRNG!(12345) # uses RandomNumbers.Xorshifts.Xoroshiro128Plus()
    pa = RunTillLastStep(laststep = 1, shift = iShift, dτ = 0.001)
    @time rdfs = fciqmc!(vs, pa, ham, s, r_strat, τ_strat, ws)
    pa.laststep = 100
    @time rdfs = fciqmc!(vs, pa, rdfs, ham, s, r_strat, τ_strat, ws)
    if Threads.nthreads() == 1
        @test sum(rdfs[:,:spawns]) == 117277
    end
end

@testset "IsStochastic2PopWithThreshold" begin
    # Define the initial Fock state with n particles and m modes
    n = m = 9
    aIni = nearUniform(BoseFS{n,m})

    dvc = DVec(aIni => 2+3im; capacity = 10)
    @test_throws AssertionError setThreshold(dvc,1.0)
    dvcf = DVec(aIni => 2.0+3.0im; capacity = 10)
    setThreshold(dvcf, 1.3)
    @test StochasticStyle(dvcf) == DictVectors.IsStochastic2PopWithThreshold(1.3)
end

@testset "IsStochasticWithThreshold" begin
    # Define the initial Fock state with n particles and m modes
    n = m = 9
    aIni = nearUniform(BoseFS{n,m})
    ham = BoseHubbardReal1D(aIni; u = 6.0, t = 1.0)
    p = NoProjection() # ThresholdProject(1.0)

    # IsStochasticWithThreshold
    s = DoubleLogUpdate(targetwalkers = 100)
    svec = DVec(Dict(aIni => 2.0), dimension(ham))
    # Rimu.StochasticStyle(::Type{typeof(svec)}) = IsStochasticWithThreshold(1.0)
    @setThreshold svec 0.621
    @test StochasticStyle(svec) == IsStochasticWithThreshold(0.621)
    @setDeterministic svec
    @test StochasticStyle(svec) == IsDeterministic()
    setThreshold(svec, 1.0)
    @test StochasticStyle(svec) == IsStochasticWithThreshold(1.0)
    vs = copy(svec)
    pa = RunTillLastStep(laststep = 100)
    seedCRNG!(12345) # uses RandomNumbers.Xorshifts.Xoroshiro128Plus()
    @time rdfs = fciqmc!(vs, pa, ham, s, EveryTimeStep(), ConstantTimeStep(), copy(vs), p_strat = p)
    @test sum(rdfs[:,:norm]) ≈ 3578.975690432334

    # NoProjectionTwoNorm
    vs = copy(svec)
    pa = RunTillLastStep(laststep = 100)
    seedCRNG!(12345) # uses RandomNumbers.Xorshifts.Xoroshiro128Plus()
    @time rdfs = fciqmc!(vs, pa, ham, s, EveryTimeStep(), ConstantTimeStep(), copy(vs), p_strat = NoProjectionTwoNorm())
    @test sum(rdfs[:,:norm]) ≈ 3118.2470468282886

    # NoMemory
    vs = copy(svec)
    seedCRNG!(12345) # uses RandomNumbers.Xorshifts.Xoroshiro128Plus()
    pa = RunTillLastStep(laststep = 100)
    @time rdfs = fciqmc!(vs, pa, ham, s, EveryTimeStep(), ConstantTimeStep(), copy(vs), m_strat = NoMemory(), p_strat = p)
    @test sum(rdfs[:,:norm]) ≈ 3578.975690432334

    # DeltaMemory
    vs = copy(svec)
    seedCRNG!(12345) # uses RandomNumbers.Xorshifts.Xoroshiro128Plus()
    pa = RunTillLastStep(laststep = 100)
    @time rdfs = fciqmc!(vs, pa, ham, s, EveryTimeStep(), ConstantTimeStep(), copy(vs), m_strat = DeltaMemory(1), p_strat = p)
    @test sum(rdfs[:,:norm]) ≈ 3578.975690432334

    # DeltaMemory
    vs = copy(svec)
    seedCRNG!(12345) # uses RandomNumbers.Xorshifts.Xoroshiro128Plus()
    pa = RunTillLastStep(laststep = 100)
    @time rdfs = fciqmc!(vs, pa, ham, s, EveryTimeStep(), ConstantTimeStep(), copy(vs), m_strat = DeltaMemory(10), p_strat = p)
    @test sum(rdfs[:,:norm]) ≈ 2870.4131527075747
    @test sum(rdfs.shiftnoise) ≈ 0.6138924086486302
    # DeltaMemory2
    vs = copy(svec)
    seedCRNG!(12345) # uses RandomNumbers.Xorshifts.Xoroshiro128Plus()
    pa = RunTillLastStep(laststep = 100)
    @time rdfs = fciqmc!(vs, pa, ham, s, EveryTimeStep(), ConstantTimeStep(), copy(vs), m_strat = Rimu.DeltaMemory2(10), p_strat = p)
    @test sum(rdfs[:,:norm]) ≈ 3390.155742724108

    # ScaledThresholdProject
    vs = copy(svec)
    seedCRNG!(12345) # uses RandomNumbers.Xorshifts.Xoroshiro128Plus()
    pa = RunTillLastStep(laststep = 100)
    p_strat = ScaledThresholdProject(1.0)
    @time rdfs = fciqmc!(vs, pa, ham, s, EveryTimeStep(), ConstantTimeStep(), copy(vs), m_strat = DeltaMemory(10), p_strat = p_strat)
    @test sum(rdfs[:,:norm]) ≈ 3546.7449141934667

    # ProjectedMemory
    vs = copy(svec)
    seedCRNG!(12345) # uses RandomNumbers.Xorshifts.Xoroshiro128Plus()
    pa = RunTillLastStep(laststep = 100)
    p_strat = NoProjection() # ScaledThresholdProject(1.0)
    m_strat = Rimu.ProjectedMemory(5,UniformProjector(), vs)
    @time rdfs = fciqmc!(vs, pa, ham, s, EveryTimeStep(), ConstantTimeStep(), copy(vs), m_strat = m_strat, p_strat = p_strat)
    @test sum(rdfs[:,:norm]) ≈ 3365.538570019769

    # ShiftMemory
    vs = copy(svec)
    seedCRNG!(12345) # uses RandomNumbers.Xorshifts.Xoroshiro128Plus()
    pa = RunTillLastStep(laststep = 100)
    p_strat = NoProjection() #ScaledThresholdProject(1.0)
    @time rdfs = fciqmc!(vs, pa, ham, s, EveryTimeStep(), ConstantTimeStep(), copy(vs), m_strat = ShiftMemory(10), p_strat = p_strat)
    @test sum(rdfs[:,:norm]) ≈ 2474.7219619083903

    # applyMemoryNoise
    v2=DVec(Dict(aIni => 2))
    StochasticStyle(v2) # IsStochastic() is not suitable for DeltaMemory()
    @test_throws ErrorException Rimu.apply_memory_noise!(v2, v2, 0.0, 0.1, 20, DeltaMemory(3))
    @test 0 == Rimu.apply_memory_noise!(svec, copy(svec), 0.0, 0.1, 20, DeltaMemory(3))

    # momentum space - tests annihilation
    aIni = BoseFS((0,0,6,0,0,0))
    ham = BoseHubbardMom1D(aIni, u=6.0)
    s = DoubleLogUpdate(targetwalkers = 100)
    svec = DVec(Dict(aIni => 2.0), dimension(ham))
    Rimu.StochasticStyle(::Type{typeof(svec)}) = IsStochasticWithThreshold(1.0)
    StochasticStyle(svec)
    vs = copy(svec)
    pa = RunTillLastStep(laststep = 100)
    seedCRNG!(12345) # uses RandomNumbers.Xorshifts.Xoroshiro128Plus()
    @time rdfs = fciqmc!(vs, pa, ham, s, EveryTimeStep(), ConstantTimeStep(), copy(vs))
    @test sum(rdfs[:,:norm]) ≈ 3774.2132792843067
end

@testset "deterministic and multithreading" begin
    # set up parameters for simulations
    walkernumber = 20_000
    steps = 100
    dτ = 0.005

    # Define the initial Fock state with n particles and m modes
    n = m = 9
    aIni = nearUniform(BoseFS{n,m})
    ham = BoseHubbardReal1D(aIni; u = 6.0, t = 1.0)
    # ### Deterministic FCIQMC
    svec2 = DVec(Dict(aIni => 2.0), dimension(ham))
    Rimu.StochasticStyle(::Type{typeof(svec2)}) = IsDeterministic()
    StochasticStyle(svec2)

    pa = RunTillLastStep(laststep = steps,  dτ = dτ)
    τ_strat = ConstantTimeStep()
    s_strat = DoubleLogUpdate(targetwalkers = walkernumber)
    r_strat = EveryTimeStep()
    @time rdf = fciqmc!(svec2, pa, ham, s_strat, r_strat, τ_strat, similar(svec2))
    @test rdf.:shift[101] ≈ -1.5985012281209916
    # Multi-threading
    svec2 = DVec(Dict(aIni => 2.0), dimension(ham))
    pa = RunTillLastStep(laststep = steps,  dτ = dτ)
    cws = capacity(svec2)÷Threads.nthreads()+1
    ws = Tuple(similar(svec2,cws) for i=1:Threads.nthreads())
    @test Rimu.threadedWorkingMemory(svec2) == ws
    @time rdf = fciqmc!(svec2, pa, ham, s_strat, r_strat, τ_strat, ws)
    @test rdf.:shift[101] ≈ -1.5985012281209916
    mytdot2(x, ys) = sum(map(y->x⋅y,ys))
    mytdot(x, ys) = mapreduce(y->x⋅y,+,ys)
    @test dot(svec2, ws) == mytdot(svec2, ws) == mytdot2(svec2, ws)
    # @benchmark dot(svec2, ws) # 639.977 μs using Threads.@threads on 4 threads
    # @benchmark mytdot(svec2, ws) # 2.210 ms
    # @benchmark mytdot2(svec2, ws) # 2.154 ms
    # function myspawndot(x::AbstractDVec{K,T1}, ys::NTuple{N, AbstractDVec{K,T2}}) where {N, K, T1, T2}
    #     results = zeros(promote_type(T1,T2), N)
    #     @sync for i in 1:N
    #         Threads.@spawn results[i] = x⋅ys[i] # using dynamic scheduler
    #     end
    #     return sum(results)
    # end
    # @benchmark DictVectors.myspawndot(svec2, ws) # 651.476 μs
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
    v = DVec2(k=>1.0 for k in basis; capacity = dim+10) # corresponds to `ones(dim)`
    c = lomc!(ham, v).df
    @test a.shift ≈ c.shift
end

@testset "lomc!" begin
    # Define the initial Fock state with n particles and m modes
    n = m = 9
    aIni = nearUniform(BoseFS{n,m})
    ham = BoseHubbardReal1D(aIni; u = 6.0, t = 1.0)
    svec = DVec(Dict(aIni => 2), 2000)

    # fciqmc with default parameters
    pa = RunTillLastStep(shift = 0.0)
    nt = lomc!(ham, svec, params=pa, laststep = 100) # run for 100 time steps
    # continuation run
    nt = lomc!(nt, nt.params.laststep + 100) # run for another 100 steps
    @test size(nt.df)[1] == 201 # initial state + 200 steps

    # fciqmc with complex shift and norm
    svec = DVec(Dict(aIni => 2), 2000)
    pa = RunTillLastStep(shift = 0.0 + 0im) # makes shift and norm type ComplexF64
    nt = lomc!(ham, svec, params=pa, laststep = 100) # run for 100 time steps
    # continuation run
    nt = lomc!(nt, nt.params.laststep + 100) # run for another 100 steps
    @test size(nt.df)[1] == 201 # initial state + 200 steps

    # fciqmc with deterministic outcome by seeding the rng and turning off multithreading
    svec = DVec(Dict(aIni => 2), 2000)
    seedCRNG!(12345) # uses RandomNumbers.Xorshifts.Xoroshiro128Plus()
    nt = lomc!(ham, svec, laststep = 100, threading = false) # run for 100 time steps
    @test sum(nt.df.spawns) == 3279

    aIni2 = BoseFS((0,0,0,0,9,0,0,0,0))
    ham2 = BoseHubbardMom1D(aIni2; u = 1.0, t=1.0)
    sv2 = DVec(Dict(aIni2 => 2), 2000)
    seedCRNG!(12345) # uses RandomNumbers.Xorshifts.Xoroshiro128Plus()
    nt = lomc!(ham2, sv2, laststep = 30, threading = false,
                r_strat = EveryTimeStep(projector = copytight(sv2)),
                s_strat = DoubleLogUpdate(targetwalkers = 100))
    # need to analyse this - looks fishy
end

@testset "ReportingStrategy internals" begin
    aIni = BoseFS((2,4,0,0,1))
    ham = BoseHubbardMom1D(aIni)
    v = DVec(aIni => 2; capacity = 1)
    r = EveryTimeStep(projector = copytight(v))
    @test r.hproj == :auto
    @test_throws ErrorException Rimu.compute_proj_observables(v, ham, r)
    rr = Rimu.refine_r_strat(r, ham)
    @test rr.hproj⋅v == dot(v, ham, v)
    @test Rimu.compute_proj_observables(v, ham, rr) == (v⋅v, dot(v, ham, v))
end

@testset "ComplexNoiseCancellation" begin
    aIni = BoseFS((2,4,0,0,1))
    v = DVec(aIni => 2.0; capacity = 10)
    @setThreshold v 0.4
    @test_throws ErrorException Rimu.norm_project!(Rimu.ComplexNoiseCancellation(), v,4.0,5.0,0.6)
    seedCRNG!(12345) # uses RandomNumbers.Xorshifts.Xoroshiro128Plus()
    tnorm = Rimu.norm_project!(Rimu.ComplexNoiseCancellation(), v, 4.0+2im, 5.0+1im, 0.6)
    @test real(tnorm) ≈ norm(v)
    @test tnorm ≈ 0.004250179379897467 + 10.378749103100512im # TODO: is this ok

    ham = BoseHubbardReal1D(aIni; u = 6.0, t = 1.0)
    svec = DVec(aIni => 2; capacity = 200)
    p_strat = Rimu.ComplexNoiseCancellation(κ = 0.0)
    # fciqmc with default parameters
    pa = RunTillLastStep(shift = 0.0, dτ=0.001)
    s_strat = DoubleLogUpdate(targetwalkers = 100)
    # nt = lomc!(ham, svec, params=pa, s_strat= s_strat, laststep = 1000)
    @test_throws ErrorException nt = lomc!(ham, svec, params=pa, p_strat=p_strat, laststep = 1001) # run for 100 time steps
    svec = DVec(aIni => 2.0; capacity = 200)
    @setThreshold svec 0.4
    pa = RunTillLastStep(shift = 0.0+0im, dτ=0.001)
    p_strat = Rimu.ComplexNoiseCancellation(κ = 1.0)
    nt = lomc!(ham, svec, params=pa,s_strat= s_strat, p_strat=p_strat, laststep = 10) # run for 100 time steps
    @test Rimu.Blocking.gW(nt.df,4, pad= false) |> length == 7
end

@testset "helpers" begin
    v = [1,2,3]
    @test walkernumber(v) == norm(v,1)
    dvc= DVec(:a=>2-5im,capacity = 10)
    @test StochasticStyle(dvc) == DictVectors.IsStochastic2Pop()
    @test walkernumber(dvc) == 2.0 + 5.0im
    Rimu.purge_negative_walkers!(dvc)
    @test walkernumber(dvc) == 2.0 + 0.0im
    dvi= DVec(:a=>Complex{Int32}(2-5im),capacity = 10)
    @test StochasticStyle(dvi) == DictVectors.IsStochastic2Pop()
    dvr = DVec(i => cRandn() for i in 1:100; capacity = 100)
    @test walkernumber(dvr) == norm(dvr,1)
end

@testset "complex walkers" begin
    m = n = 6
    aIni = nearUniform(BoseFS{n,m})
    Ĥ = BoseHubbardReal1D(aIni; u = 6.0, t = 1.0)
    ζ = 0.08
    N=50
    s_strat = DoubleLogUpdate(ζ = ζ, ξ = ζ^2/4, targetwalkers = N + N*im)
    svec = DVec(aIni => 2+2im, capacity = (real(s_strat.targetwalkers)*2+100))
    r_strat = EveryTimeStep(projector = copytight(svec))

    # seed random number generator
    Rimu.ConsistentRNG.seedCRNG!(17+19)
    params = RunTillLastStep(dτ = 0.001, laststep = 200, shift = 0.0 + 0.0im)
    @time nt = lomc!(Ĥ, copy(svec); params, s_strat, r_strat)
    df = nt.df
    @test size(nt.df) == (201, 14)
    # TODO: Add sensible tests.

    N=50
    s_strat = DoubleLogUpdate(ζ = ζ, ξ = ζ^2/4, targetwalkers = N + N*im)
    svec = DVec(aIni => 2+2im, capacity = (real(s_strat.targetwalkers)*2+100))
    r_strat = EveryTimeStep(projector = copytight(svec))

    # seed random number generator
    Rimu.ConsistentRNG.seedCRNG!(17+19)
    params = RunTillLastStep(dτ = 0.001, laststep = 1000, shift = 0.0 + 0.0im)
    @time nt = lomc!(Ĥ, copy(svec); params, s_strat, r_strat)

end

using Rimu.Blocking # bring exported functions into name space
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
    svec = DVec(Dict(aIni => 2), dimension(ham))
    StochasticStyle(svec)
    vs = copy(svec)
    r_strat = EveryTimeStep(projector = copytight(svec))
    τ_strat = ConstantTimeStep()

    seedCRNG!(12345) # uses RandomNumbers.Xorshifts.Xoroshiro128Plus()
    # @time rdfs = fciqmc!(vs, pa, ham, s, r_strat, τ_strat, similar(vs))
    @time rdfs = lomc!(ham, vs; params = pa, s_strat = s, r_strat = r_strat,
        τ_strat = τ_strat, wm = similar(vs)
    ).df
    r = autoblock(rdfs, start=101)
    @test all(Tuple(r).≈ (-5.322585671036912, 0.23959423243049166, -6.284242115099837, 0.3694258182027044, 6))

    g = growthWitness(rdfs, b=50)
    # @test sum(g) ≈ -5725.3936298329545
    @test length(g) == nrow(rdfs)
    g = growthWitness(rdfs, b=50, pad = :false)
    @test length(g) == nrow(rdfs) - 50
    @test_throws AssertionError growthWitness(rdfs.norm, rdfs.shift[1:end-1],rdfs.dτ[1])
end

@testset "RimuIO" begin
    file = joinpath(@__DIR__, "tmp.arrow")
    df = DataFrame(a=[1, 2, 3], b=Complex{Float64}[1, 2, 3+im], d=rand(Complex{Int}, 3))
    RimuIO.save_df(file, df)
    df2 = RimuIO.load_df(file)
    @test df == df2
    rm(file)
end

using Rimu.EmbarrassinglyDistributed # bring relevant function into namespace
@testset "EmbarrassinglyDistributed" begin
    add = BoseFS((1,1,0,1))
    v = DVec(add => 2, capacity = 200)
    ham = BoseHubbardReal1D(add, u=4.0)
    @test setup_workers(4) == 4 # add workers and load code (Rimu and its modules)
    seedCRNGs_workers!(127)     # seed rgns on workers deterministically
    nt = d_lomc!(ham, v; eqsteps = 1_000, laststep = 21_000) # perform parallel lomc!
    @test [size(df)[1] for df in nt.dfs] == [6001, 6001, 6001, 6001]
    ntc = combine_dfs(nt) # combine results into one DataFrame
    @test size(ntc.df)[1] == 21_001
    energies = autoblock(ntc) # perform `autoblock()` discarding `eqsteps` time steps
    # in a single line:
    # energies = d_lomc!(ham, v; eqsteps = 1_000, laststep = 21_000) |> combine_dfs |> autoblock
    @test ismissing(energies.ē) && ismissing(energies.σe)
    # golden master test on results because qmc evolution is deterministic
    @test energies.s̄ ≈ -4.111231475595392
    @test energies.σs ≈ 0.005705218988651839
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

using Rimu.DictVectors: IsStochasticWithThresholdAndInitiator
@testset "IsDynamicSemistochastic" begin
    add = BoseFS((1, 1, 1))
    H = HubbardReal1D(add)
    dv1 = DVec2(add => 1; capacity=100)
    dv2 = DVec2(add => 1.0; capacity=100, style=IsDynamicSemistochastic())
    dv3 = DVec2(add => 1.0; capacity=100, style=IsStochasticWithThresholdAndInitiator())

    df1 = lomc!(H, dv1, laststep=10000).df
    df2 = lomc!(H, dv2, laststep=10000).df
    df3 = lomc!(H, dv3, laststep=10000).df
    σ1 = autoblock(df1, start=100).σs
    σ2 = autoblock(df2, start=100).σs
    σ3 = autoblock(df3, start=100).σs

    @test σ1 > σ2
    @test σ1 > σ3
end

@safetestset "KrylovKit" begin
    include("KrylovKit.jl")
end

using Rimu.RMPI
using Rimu.RMPI: sort_and_count!
@testset "RMPI" begin
    m = n = 6
    aIni = nearUniform(BoseFS{n,m})
    svec = DVec(aIni => 2, capacity = 10)
    dv = MPIData(svec)
    @test ConsistentRNG.check_crng_independence(dv) == mpi_size()*Threads.nthreads()*fieldcount(ConsistentRNG.CRNG)

    @testset "sort_and_count!" begin
        for l in (1, 2, 30, 1000)
            for k in (2, 10, 100)
                @testset "k=$k, l=$l" begin
                    ordfun(x) = hash(x) % k
                    vals = rand(Int, l)
                    counts = zeros(Int, k)
                    displs = zeros(Int, k)

                    sort_and_count!(counts, displs, vals, ordfun.(vals), (0, k-1))
                    @test issorted(vals, by=ordfun)
                    @test sum(counts) == l

                    for i in 0:(k - 1)
                        c = counts[i + 1]
                        d = displs[i + 1]
                        r = (1:c) .+ d
                        ords = ordfun.(vals)
                        @test all(ords[r] .== i)
                    end
                end
            end
        end
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
        flavours = ["os", "ptp"]
        for f in flavours
            savefile = joinpath(@__DIR__,"mpi_df_$f.arrow")

            rm(savefile, force = true) # make sure to remove any old file
            runfile = joinpath(@__DIR__,"script_mpi_minimum_$f.jl")
            rr = run(`$mpiexec -np 2 $juliaexec -t 1 $runfile`)
            @test rr.exitcode == 0
        end
        savefiles = [joinpath(@__DIR__,"mpi_df_$f.arrow") for f in flavours]
        dfs = [RimuIO.load_df(sf) for sf in savefiles]
        @test reduce(==, dfs) # require equal DataFrames from seeded qmc
        map(rm, savefiles)# clean up
    else
        @info "not testing MPI on CI"
    end
end
