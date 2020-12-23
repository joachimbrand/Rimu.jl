using Rimu
using Test
using LinearAlgebra
using Statistics, DataFrames

# the following is needed because random numbers of collections are computed
# differently after version 1.5, and thus the results of many tests change
# The following Bool is true if run on an old version of Julia
const OV = VERSION<v"1.5"

@testset "Rimu.jl" begin
    # Write your own tests here.
    @test 3==3
end

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
    svec = DVec(Dict(aIni => 2), ham(:dim))
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
    #@test reduce(&, Tuple(r).≈(-5.25223493152101, 0.19342756375228998, -6.527599639255635, 0.5197276766889821, 6))
    if OV
        @test reduce(&, Tuple(r).≈(-5.25223493152101, 0.19342756375229, -6.527599639255635, 0.47607219512729554, 6))
    else
        @test reduce(&, Tuple(r).≈(-5.714600548611788, 0.21631081209341332, -5.884807394477632, 0.3849918114544903, 6))
    end
    g = growthWitness(rdfs, b=50)
    # @test sum(g) ≈ -5725.3936298329545
    @test length(g) == nrow(rdfs)
    g = growthWitness(rdfs, b=50, pad = :false)
    @test length(g) == nrow(rdfs) - 50
    @test_throws AssertionError growthWitness(rdfs.norm, rdfs.shift[1:end-1],rdfs.dτ[1])
end

using Rimu.BitStringAddresses
import Rimu.BitStringAddresses: check_consistency, remove_ghost_bits
@testset "BitStringAddresses.jl" begin
    # BitAdd
    bs = BitAdd{40}(0xf342564fff)
    bs1 = BitAdd{40}(0xf342564ffd)
    bs2 = BitAdd{144}(big"0xf342564ffdf00dfdfdfdfdfdfdfdfdfdfdf")
    bs3 = BitAdd{44}(0xf342564fff)
    @test bs > bs1
    @test !(bs == bs1)
    @test !(bs < bs1)
    @test bs2 > bs1
    @test bs3 > bs
    @test bs & bs1 == bs1
    @test bs | bs1 == bs
    @test bs ⊻ bs1 == BitAdd{40}(2)
    @test count_ones(bs2) == 105
    @test count_zeros(bs2) == 39
    w = BitAdd{65}((UInt(31),UInt(15)))
    @test_throws ErrorException check_consistency(w)
    @test_throws ErrorException BitAdd((UInt(31),UInt(15)),65)
    wl = BitAdd((UInt(31),UInt(15)),85)
    @test bs2 == BitAdd(big"0xf342564ffdf00dfdfdfdfdfdfdfdfdfdfdf",144)
    fa = BitAdd{133}()
    @test trailing_zeros(bs<<3) == 3
    @test trailing_ones(fa) == 133
    @test trailing_ones(fa>>100) == 33
    @test trailing_zeros(fa<<100) == 100
    @test leading_zeros(fa>>130) == 130
    @test leading_ones(fa<<130) == 3
    @test bitstring(bs2) == "000011110011010000100101011001001111111111011111000000001101111111011111110111111101111111011111110111111101111111011111110111111101111111011111"
    @test repr(BoseFS(bs2)) == "BoseFS{BitAdd}((5,7,7,7,7,7,7,7,7,7,7,2,0,0,0,0,0,0,0,5,10,0,1,0,2,1,1,0,1,0,0,0,1,2,0,4,0,0,0,0))"
    @test onr(BoseFS(bs)) == [12,0,1,0,2,1,1,0,1,0,0,0,1,2,0,4]
    os = BoseFS{BitAdd}([12,0,1,0,2,1,1,0,1,0,0,0,1,2,0,4])
    @test os == BoseFS(bs)
    @test hash(os) == hash(BoseFS(bs))
    @test os.bs == bs
    bfs= BoseFS((1,0,2,1,2,1,1,3))
    onrep = onr(bfs)
    @test typeof(bfs)(onrep) == bfs
    ba=BoseFS{BStringAdd}((2,4,0,5,3))
    @test BitStringAddresses.i_onr(ba) == onr(ba) == onr(ba.bs, numModes(ba))
    @test BitStringAddresses.i_onr(os) == onr(os)
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

@testset "DictVectors.jl" begin
    @test FastDVec(i => i^2 for i in 1:10; capacity = 30)|> length == 10
    myfda = FastDVec("a" => 42; capacity = 40)
    myda2 = FastDVec{String,Int}(40)
    myda2["a"] = 42
    @test haskey(myda2,"a")
    @test !haskey(myda2,"b")
    @test myfda == myda2
    myda2["c"] = 422
    myda2["d"] = 45
    myda2["f"] = 412

    @test length(myda2) == 4

    for (key, val) in pairs(myda2)
        println("key ",key," val ",val)
    end
    show(myda2)

    @test norm(myda2)≈592.9730179358922
    @test norm(myda2,1)==921
    @test norm(myda2*2.0,1) == 2*norm(myda2,1)
    @test norm(myda2,Inf)==422
    @inferred norm(myda2,1)
    delete!(myda2,"d")
    delete!(myda2,"c")
    @test myda2.emptyslots == [3,2]
    myda3 = similar(myda2)
    copyto!(myda3,myda2)
    @test length(myda3)==2
    myda3["q"]= 3
    delete!(myda3,"q")
    @test myda2==myda3
    fdv = FastDVec([rand() for i=1:1000], 2000)
    ki = keys(fdv)
    @test sort(collect(ki))==collect(1:1000)
    cdv = FastDVec(fdv)
    @test cdv == fdv
    fdv[1600] = 10.0
    cdv[1600] = 10.0
    @test cdv == fdv
    dv = DVec(fdv)
    edv = empty(dv)
    copyto!(edv,dv)
    axpby!(0.1,dv,4.0,edv)
    dvc = copy(dv)
    @test dvc == dv
    y = empty(dv)
    axpy!(2.0, dv, y)
    @test norm(y, 1) ≈ norm(dv,1)*2
    ys = Tuple(empty(dv) for i in 1:Threads.nthreads())
    axpy!(2.0, dv, ys, batchsize=100)
    @test sum(norm.(ys, 1)) ≈ norm(dv,1)*2

    mdv = DVec(:a => 2; capacity = 10)
    @test mdv[:a] == 2
    @test mdv[:b] == 0
    @test length(mdv) == 1
    @test ismissing(missing*mdv)

    @test DFVec(:a => (2,3); capacity = 10) == DFVec(Dict(:a => (2,3)))
    dv = DVec(:a => 2; capacity = 100)
    cdv = copytight(dv)
    @test dv == cdv
    @test capacity(dv) > capacity(cdv)
end

using Rimu.ConsistentRNG
@testset "ConsistentRNG.jl" begin
    seedCRNG!(127) # uses `RandomNumbers.Xorshifts.Xoshiro256StarStar()`
    @test cRand(UInt128) == 0xad2acf8f66080104f395d0b7ed4713d9

    @test rand(ConsistentRNG.CRNGs[1],UInt128) == 0x0b0c30478c16f78daa91bcc785895269
    # Only looks at first element of the `NTuple`. This should be reproducible
    # regardless of `numthreads()`.
    @test rand(trng(),UInt16) == 0x4c52
    @test rand(newChildRNG(),UInt16) == 0xc4f1
end

@testset "Hamiltonians.jl" begin
    ham = BoseHubbardReal1D(
        n = 9,
        m = 9,
        u = 6.0,
        t = 1.0,
        AT = BSAdd64)
    @test ham(:dim) == 24310

    aIni = Rimu.Hamiltonians.nearUniform(ham)
    @test aIni == BSAdd64(0x15555)

    hp = Hops(ham,aIni)
    @test length(hp) == 18
    @test hp[18][1] == BSAdd64(0x000000000000d555)
    @test hp[18][2] ≈ -1.4142135623730951
    @test diagME(ham,aIni) == 0
    os = BoseFS([12,0,1,0,2,1,1,0,1,0,0,0,1,2,0,4])
    @test Rimu.Hamiltonians.bosehubbardinteraction(os) == 148
    @test Rimu.Hamiltonians.ebhm(os) == (53, 148)
    @test Rimu.Hamiltonians.numberoccupiedsites(os) == 9
    hnnn = Rimu.Hamiltonians.hopnextneighbour(0xf342564fff,3,16,25)
    bs = BitAdd{40}(0xf342564fff)
    hnnbs = Rimu.Hamiltonians.hopnextneighbour(bs,3,16,25)
    @test BitAdd{40}(hnnn[1]) == hnnbs[1]

    svec = DVec(Dict(aIni => 2.0), ham(:dim))
    v2 = ham(svec)
    v3 = ham*v2
    @test norm(v3,1) ≈ 1482.386824949077
    @test v2 == mul!(similar(svec), ham, svec)
    @test norm(v2) ≈ 12
    @test v2 == ham*svec
    @test dot(v2,ham,svec) == v2⋅(ham*svec) ≈ 144
    @test -⋅(UniformProjector(),ham,svec)≈⋅(NormProjector(),ham,svec)≈norm(v2,1)
    @test dot(Norm2Projector(),v2) ≈ norm(v2,2)
    @test Hamiltonians.LOStructure(ham) == Hamiltonians.HermitianLO()
    aIni2 = nearUniform(BoseFS{9,9})
    hamc = BoseHubbardReal1D(aIni2, u=6.0+0im, t=1.0+0im) # formally a complex operator
    @test Hamiltonians.LOStructure(hamc) == Hamiltonians.ComplexLO()
    @test dot(v3,ham,svec) ≈ dot(v3,hamc,svec) ≈ dot(svec,ham,v3) ≈ dot(svec,hamc,v3) ≈ 864
    hamcc = BoseHubbardReal1D(aIni2, u=6.0+0.1im, t=1.0+2im) # a complex operator
    vc2 = hamcc*svec
    @test isreal(dot(vc2,hamcc,svec))
    @test dot(vc2,hamc,svec) ≉ dot(svec,hamc,vc2)

    @test adjoint(ham) == ham' == ham
    @test Rimu.Hamiltonians.LOStructure(hamcc) == Rimu.Hamiltonians.ComplexLO()
    @test_throws ErrorException hamcc'
end

@testset "BoseHubbardMom1D" begin
    bfs= BoseFS((1,0,2,1,2,1,1,3))
    @test Hamiltonians.numberoccupiedsites(bfs) == 7
    @test Hamiltonians.numSandDoccupiedsites(bfs) == (7,3)
    @test Hamiltonians.numSandDoccupiedsites(onr(bfs)) == Hamiltonians.numSandDoccupiedsites(bfs)

    ham = Hamiltonians.BoseHubbardMom1D(bfs)
    @test numOfHops(ham,bfs) == 273
    @test hop(ham, bfs, 205) == (BoseFS{BSAdd64}((1,0,2,1,3,0,0,4)), 0.21650635094610965)
    @test diagME(ham,bfs) ≈ 14.296572875253808
    momentum = Momentum(ham)
    @test diagME(momentum,bfs) ≈ -1.5707963267948966
    v = DVec(Dict(bfs => 10), 1000)
    @test rayleigh_quotient(momentum, v) ≈ -1.5707963267948966

    ham = Hamiltonians.HubbardMom1D(bfs)
    @test numOfHops(ham,bfs) == 273
    @test hop(ham, bfs, 205) == (BoseFS{BSAdd64}((1,0,2,1,3,0,0,4)), 0.21650635094610965)
    @test diagME(ham,bfs) ≈ 14.296572875253808
    momentum = Momentum(ham)
    @test diagME(momentum,bfs) ≈ -1.5707963267948966
    v = DVec(Dict(bfs => 10), 1000)
    @test rayleigh_quotient(momentum, v) ≈ -1.5707963267948966

    fs = BoseFS((1,2,1,0)) # should be zero momentum
    ham = BoseHubbardMom1D(fs,t=1.0)
    m=Momentum(ham) # define momentum operator
    mom_fs = diagME(m, fs) # get momentum value as diagonal matrix element of operator
    @test isapprox(mom_fs, 0.0, atol = sqrt(eps())) # check whether momentum is zero
    @test reduce(&,[isapprox(mom_fs, diagME(m,h[1]), atol = sqrt(eps())) for h in Hops(ham, fs)]) # check that momentum does not change for hops
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
    m=Momentum(ham) # define momentum operator
    mom_fs = diagME(m, fs) # get momentum value as diagonal matrix element of operator
    @test isapprox(mom_fs, 0.0, atol = sqrt(eps())) # check whether momentum is zero
    @test reduce(&,[isapprox(mom_fs, diagME(m,h[1]), atol = sqrt(eps())) for h in Hops(ham, fs)]) # check that momentum does not change for hops
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
        AT = BSAdd64)
    aIni = nearUniform(ham)
    pa = RunTillLastStep(laststep = 100)

    # standard fciqmc
    s = LogUpdateAfterTargetWalkers(targetwalkers = 100)
    svec = DVec(Dict(aIni => 2), ham(:dim))
    StochasticStyle(svec)
    vs = copy(svec)
    seedCRNG!(12345) # uses RandomNumbers.Xorshifts.Xoroshiro128Plus()
    r_strat = EveryTimeStep()
    τ_strat = ConstantTimeStep()

    @time rdfs = fciqmc!(vs, pa, ham, s, r_strat, τ_strat, similar(vs))
    @test sum(rdfs[:,:spawns]) == (OV ? 2603 : 3263)

    # fciqmc with delayed shift update
    pa = RunTillLastStep(laststep = 100)
    s = DelayedLogUpdateAfterTargetWalkers(targetwalkers = 100, a = 5)
    svec = DVec(Dict(aIni => 2), ham(:dim))
    StochasticStyle(svec)
    vs = copy(svec)
    seedCRNG!(12345) # uses RandomNumbers.Xorshifts.Xoroshiro128Plus()
    @time rdfs = fciqmc!(vs, pa, ham, s, r_strat, τ_strat, similar(vs))
    @test sum(rdfs[:,:spawns]) == (OV ? 7233 : 9932)

    # replica fciqmc
    vv = [copy(svec),copy(svec)]
    s = LogUpdateAfterTargetWalkers(targetwalkers = 100)
    pb = RunTillLastStep(laststep = 100)
    seedCRNG!(12345) # uses RandomNumbers.Xorshifts.Xoroshiro128Plus()
    @time rr = fciqmc!(vv, pb, ham, s, r_strat, τ_strat, similar.(vv))
    @test sum(rr[1][:,:xHy]) ≈ (OV ? -40751.45601645894 : -45891.01102371609)
end

@testset "fciqmc with BoseFS" begin
    # Define the initial Fock state with n particles and m modes
    n = m = 9
    aIni = nearUniform(BoseFS{n,m})
    ham = BoseHubbardReal1D(aIni; u = 6.0, t = 1.0)

    pa = RunTillLastStep(laststep = 100)

    # standard fciqmc
    s = LogUpdateAfterTargetWalkers(targetwalkers = 100)
    svec = DVec(Dict(aIni => 2), ham(:dim))
    StochasticStyle(svec)
    vs = copy(svec)

    r_strat = EveryTimeStep(projector = UniformProjector())
    # r_strat = EveryTimeStep(projector = copy(svec))
    τ_strat = ConstantTimeStep()

    seedCRNG!(12345) # uses RandomNumbers.Xorshifts.Xoroshiro128Plus()
    @time rdfs = fciqmc!(vs, pa, ham, s, r_strat, τ_strat, similar(vs))
    @test sum(rdfs[:,:spawns]) == (OV ? 2603 : 3263)

    # fciqmc with delayed shift update
    pa = RunTillLastStep(laststep = 100)
    s = DelayedLogUpdateAfterTargetWalkers(targetwalkers = 100, a = 5)
    svec = DVec(Dict(aIni => 2), ham(:dim))
    StochasticStyle(svec)
    vs = copy(svec)
    seedCRNG!(12345) # uses RandomNumbers.Xorshifts.Xoroshiro128Plus()
    @time rdfs = fciqmc!(vs, pa, ham, s, r_strat, τ_strat, similar(vs))
    @test sum(rdfs[:,:spawns]) == (OV ? 7233 : 9932)

    # replica fciqmc
    vv = [copy(svec),copy(svec)]
    s = LogUpdateAfterTargetWalkers(targetwalkers = 1_000)
    pb = RunTillLastStep(laststep = 300)
    seedCRNG!(12345) # uses RandomNumbers.Xorshifts.Xoroshiro128Plus()
    @time rr = fciqmc!(vv, pb, ham, s, r_strat, τ_strat, similar.(vv))
    @test sum(rr[1][:,:xHy]) ≈ (OV ? -8.356031508027215e6 : -7.69252569645882e6)

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
    iShift = diagME(ham, aIni)

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
    @test sum(rdfs[:,:spawns]) == (OV ? 117364 : 116658)

    # single step
    ṽ, w̃, stats = Rimu.fciqmc_step!(ham, copy(vs), pa.shift, pa.dτ, 1.0, similar(vs))
    if Threads.nthreads() == 1 # I'm not sure why this is necessary, but there
        # seems to be a difference
        @test sum(stats) == (OV ? 436 : 479)
    elseif Threads.nthreads() == 4
        @test sum(stats) == (OV ? 643 : 479)
    end

    # single step multi threading
    cws = capacity(vs)÷Threads.nthreads()+1
    ws = Tuple(similar(vs,cws) for i=1:Threads.nthreads())
    ṽ, w̃, stats = Rimu.fciqmc_step!(ham, copy(vs), pa.shift, pa.dτ, 1.0, ws;
                    batchsize = length(vs)÷4+1)
    if Threads.nthreads() == 1
        @test sum(stats) == (OV ? 428 : 475) # test assuming nthreads() == 1
    end

    # run 100 steps with multi
    pa.laststep = 200
    @time rdfs = fciqmc!(vs, pa, rdfs, ham, s, r_strat, τ_strat, ws)
    if Threads.nthreads() == 1
        @test sum(rdfs[:,:spawns]) == (OV ? 136992 : 136905) # test assuming nthreads() == 1
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
        @test sum(rdfs[:,:spawns]) == (OV ? 118650 : 119854) # test assuming nthreads() == 1
    end
end

@testset "IsStochasticWithThreshold" begin
    # Define the initial Fock state with n particles and m modes
    n = m = 9
    aIni = nearUniform(BoseFS{n,m})
    ham = BoseHubbardReal1D(aIni; u = 6.0, t = 1.0)
    p = NoProjection() # ThresholdProject(1.0)

    # IsStochasticWithThreshold
    s = DoubleLogUpdate(targetwalkers = 100)
    svec = DVec(Dict(aIni => 2.0), ham(:dim))
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
    @test sum(rdfs[:,:norm]) ≈ (OV ? 3250.375173192328 : 3012.564012011806)

    # NoProjectionTwoNorm
    vs = copy(svec)
    pa = RunTillLastStep(laststep = 100)
    seedCRNG!(12345) # uses RandomNumbers.Xorshifts.Xoroshiro128Plus()
    @time rdfs = fciqmc!(vs, pa, ham, s, EveryTimeStep(), ConstantTimeStep(), copy(vs), p_strat = NoProjectionTwoNorm())
    @test sum(rdfs[:,:norm]) ≈ (OV ? 3518.9649297547053 : 3467.388948546654)

    # NoMemory
    vs = copy(svec)
    seedCRNG!(12345) # uses RandomNumbers.Xorshifts.Xoroshiro128Plus()
    pa = RunTillLastStep(laststep = 100)
    @time rdfs = fciqmc!(vs, pa, ham, s, EveryTimeStep(), ConstantTimeStep(), copy(vs), m_strat = NoMemory(), p_strat = p)
    @test sum(rdfs[:,:norm]) ≈ (OV ? 3250.3751731923294 : 3012.564012011806)

    # DeltaMemory
    vs = copy(svec)
    seedCRNG!(12345) # uses RandomNumbers.Xorshifts.Xoroshiro128Plus()
    pa = RunTillLastStep(laststep = 100)
    @time rdfs = fciqmc!(vs, pa, ham, s, EveryTimeStep(), ConstantTimeStep(), copy(vs), m_strat = DeltaMemory(1), p_strat = p)
    @test sum(rdfs[:,:norm]) ≈ (OV ? 3250.3751731923294 : 3012.564012011806)

    # DeltaMemory
    vs = copy(svec)
    seedCRNG!(12345) # uses RandomNumbers.Xorshifts.Xoroshiro128Plus()
    pa = RunTillLastStep(laststep = 100)
    @time rdfs = fciqmc!(vs, pa, ham, s, EveryTimeStep(), ConstantTimeStep(), copy(vs), m_strat = DeltaMemory(10), p_strat = p)
    @test sum(rdfs[:,:norm]) ≈ (OV ? 2841.683858917014 : 2683.334567725324)
    @test sum(rdfs.shiftnoise) ≈ (OV ? 0.456062023263646 : 0.282574064213998)
    # DeltaMemory2
    vs = copy(svec)
    seedCRNG!(12345) # uses RandomNumbers.Xorshifts.Xoroshiro128Plus()
    pa = RunTillLastStep(laststep = 100)
    @time rdfs = fciqmc!(vs, pa, ham, s, EveryTimeStep(), ConstantTimeStep(), copy(vs), m_strat = Rimu.DeltaMemory2(10), p_strat = p)
    @test sum(rdfs[:,:norm]) ≈ (OV ? 3407.75528796349 : 3114.518739252482)

    # ScaledThresholdProject
    vs = copy(svec)
    seedCRNG!(12345) # uses RandomNumbers.Xorshifts.Xoroshiro128Plus()
    pa = RunTillLastStep(laststep = 100)
    p_strat = ScaledThresholdProject(1.0)
    @time rdfs = fciqmc!(vs, pa, ham, s, EveryTimeStep(), ConstantTimeStep(), copy(vs), m_strat = DeltaMemory(10), p_strat = p_strat)
    @test sum(rdfs[:,:norm]) ≈ (OV ? 3174.5195839788425 : 3122.817384314045)

    # ProjectedMemory
    vs = copy(svec)
    seedCRNG!(12345) # uses RandomNumbers.Xorshifts.Xoroshiro128Plus()
    pa = RunTillLastStep(laststep = 100)
    p_strat = NoProjection() # ScaledThresholdProject(1.0)
    m_strat = Rimu.ProjectedMemory(5,UniformProjector(), vs)
    @time rdfs = fciqmc!(vs, pa, ham, s, EveryTimeStep(), ConstantTimeStep(), copy(vs), m_strat = m_strat, p_strat = p_strat)
    @test sum(rdfs[:,:norm]) ≈ (OV ? 3256.8110011126214 : 3054.5448859688427)

    # ShiftMemory
    vs = copy(svec)
    seedCRNG!(12345) # uses RandomNumbers.Xorshifts.Xoroshiro128Plus()
    pa = RunTillLastStep(laststep = 100)
    p_strat = NoProjection() #ScaledThresholdProject(1.0)
    @time rdfs = fciqmc!(vs, pa, ham, s, EveryTimeStep(), ConstantTimeStep(), copy(vs), m_strat = ShiftMemory(10), p_strat = p_strat)
    @test sum(rdfs[:,:norm]) ≈ (OV ? 3231.4229026099574 : 3298.7124102559173)

    # applyMemoryNoise
    v2=DVec(Dict(aIni => 2))
    StochasticStyle(v2) # IsStochastic() is not suitable for DeltaMemory()
    @test_throws ErrorException Rimu.applyMemoryNoise!(v2, v2, 0.0, 0.1, 20, DeltaMemory(3))
    @test 0 == Rimu.applyMemoryNoise!(svec, copy(svec), 0.0, 0.1, 20, DeltaMemory(3))

    # momentum space - tests annihilation
    aIni = BoseFS((0,0,6,0,0,0))
    ham = BoseHubbardMom1D(aIni, u=6.0)
    s = DoubleLogUpdate(targetwalkers = 100)
    svec = DVec(Dict(aIni => 2.0), ham(:dim))
    Rimu.StochasticStyle(::Type{typeof(svec)}) = IsStochasticWithThreshold(1.0)
    StochasticStyle(svec)
    vs = copy(svec)
    pa = RunTillLastStep(laststep = 100)
    seedCRNG!(12345) # uses RandomNumbers.Xorshifts.Xoroshiro128Plus()
    @time rdfs = fciqmc!(vs, pa, ham, s, EveryTimeStep(), ConstantTimeStep(), copy(vs))
    @test sum(rdfs[:,:norm]) ≈ (OV ? 3939.3835802371677 : 3687.674720780682)
end

@testset "dfvec.jl" begin
    df = DFVec(Dict(3=>(3.5,true)))

    df2 = DFVec(Dict(3=>(3.5,true)), 200)
    @test capacity(df2) == 341
    df3 = DFVec{Int,Float64,Int}(30)
    @test capacity(df3) == 42
    df4 = DFVec([1,2,3,4])

    dv = DVec([1,2,3,4])

    df5 = DFVec(dv)
    @test eltype(dv) == Int
    length(dv)
    @test df4 == df5
    @test df4 ≢ df5
    @test df == df2
    @test df ≠ df4
    @test df5 == dv # checking keys, values, but not flags

    dd = Dict("a"=>1,"b"=>2,"c"=>3)
    ddv = DVec(dd)
    values(ddv)
    fddv = FastDVec(dd)
    @test collect(values(fddv)) == collect(values(ddv))
    ddf = DFVec(ddv)
    @test collect(values(ddf)) == collect(values(ddv))
    dt = Dict("a"=>(1,false),"b"=>(2,true),"c"=>(3,true))
    dtv = DVec(dt)
    collect(values(dtv))
    collect(keys(dtv))
    eltype(dtv)
    valtype(dtv)
    dtf = DFVec(dt)
    @test collect(flags(dtf)) == [true, true, false]
    @test DFVec(dtv) == dtf
    ndfv = DFVec(dtf,500,UInt8)
    dt = Dict(i => (sqrt(i),UInt16(i)) for i in 1:1000)
    dtv = DFVec(dt)
    dv = DVec(dtv)
    @test dtv == dv
    @test dv ≠ DVec(dt)
    dvt = DVec(dt)
    fdvt = DFVec(dvt)
    @test fdvt == dtv
    dtv[218] = (14.7648230602334, 0x00ff) # change flag
    @test fdvt ≠ dtv
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
    svec2 = DVec(Dict(aIni => 2.0), ham(:dim))
    Rimu.StochasticStyle(::Type{typeof(svec2)}) = IsDeterministic()
    StochasticStyle(svec2)

    pa = RunTillLastStep(laststep = steps,  dτ = dτ)
    τ_strat = ConstantTimeStep()
    s_strat = DoubleLogUpdate(targetwalkers = walkernumber)
    r_strat = EveryTimeStep()
    @time rdf = fciqmc!(svec2, pa, ham, s_strat, r_strat, τ_strat, similar(svec2))
    @test rdf.:shift[101] ≈ -1.5985012281209916
    # Multi-threading
    svec2 = DVec(Dict(aIni => 2.0), ham(:dim))
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
    @test sum(nt.df.spawns) == (OV ? 3580 : 3638)

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
    @test tnorm ≈ 1.6216896894892896 + 2.2915515525535524im

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
    @test gW(nt.df,4, pad= false) |> length == 7
end

@testset "helpers" begin
    v = [1,2,3]
    @test walkernumber(v) == norm(v,1)
    dvc= DVec(:a=>2-5im,capacity = 10)
    @test StochasticStyle(dvc) == IsStochastic2Pop()
    @test walkernumber(dvc) == 2.0 + 5.0im
    dvi= DVec(:a=>Complex{Int32}(2-5im),capacity = 10)
    @test StochasticStyle(dvi) == IsStochastic2Pop()
    dvr = DVec(i => cRandn() for i in 1:100; capacity = 100)
    @test walkernumber(dvr) == norm(dvr,1)
end


# Note: This last test is set up to work on Pipelines, within a Docker
# container, where everything runs as root. It should also work locally,
# where typically mpi is not (to be) run as root.
@testset "MPI" begin
    wd = pwd() # move to test/ folder if running from Atom
    if wd[end-3:end] ≠ "test"
        cd("test")
    end
    # read name of mpi executable from environment variable if defined
    # necessary for allow-run-as root workaround for Pipelines
    mpiexec = haskey(ENV, "JULIA_MPIEXEC") ? ENV["JULIA_MPIEXEC"] : "mpirun"
    rr = run(`$mpiexec -np 2 julia mpiexample.jl`)
    @test rr.exitcode == 0
    cd(wd)
end
