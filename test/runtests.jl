using Rimu
using Test
using LinearAlgebra

@testset "Rimu.jl" begin
    # Write your own tests here.
    @test 3==3
end

using Statistics
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
r_strat = EveryTimeStep(projector = copy(svec))
τ_strat = ConstantTimeStep()

seedCRNG!(12345) # uses RandomNumbers.Xorshifts.Xoroshiro128Plus()
@time rdfs = fciqmc!(vs, pa, ham, s, r_strat, τ_strat, similar(vs))
r = autoblock(rdfs, start=101)
@test reduce(&, Tuple(r).≈(-5.956879167047395, 0.20131107806609871,  -5.156115604590991, 0.18436001934822677, 4))
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
    myda2 = FastDVec{String,Int}(40)
    myda2["a"] = 42
    @test haskey(myda2,"a")
    @test !haskey(myda2,"b")
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
    @test Hamiltonians.LOStructure(ham) == Hamiltonians.HermitianLO()
    aIni2 = nearUniform(BoseFS{9,9})
    hamc = BoseHubbardReal1D(aIni2, u=6.0+0im, t=1.0+0im) # formally a complex operator
    @test Hamiltonians.LOStructure(hamc) == Hamiltonians.ComplexLO()
    @test dot(v3,ham,svec) ≈ dot(v3,hamc,svec) ≈ dot(svec,ham,v3) ≈ dot(svec,hamc,v3) ≈ 864
    hamcc = BoseHubbardReal1D(aIni2, u=6.0+0.1im, t=1.0+2im) # a complex operator
    vc2 = hamcc*svec
    @test isreal(dot(vc2,hamcc,svec))
    @test dot(vc2,hamc,svec) ≉ dot(svec,hamc,vc2)
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
    @test sum(rdfs[:,:spawns]) == 1725

    # fciqmc with delayed shift update
    pa = RunTillLastStep(laststep = 100)
    s = DelayedLogUpdateAfterTargetWalkers(targetwalkers = 100, a = 5)
    svec = DVec(Dict(aIni => 2), ham(:dim))
    StochasticStyle(svec)
    vs = copy(svec)
    seedCRNG!(12345) # uses RandomNumbers.Xorshifts.Xoroshiro128Plus()
    @time rdfs = fciqmc!(vs, pa, ham, s, r_strat, τ_strat, similar(vs))
    @test sum(rdfs[:,:spawns]) == 2998

    # replica fciqmc
    vv = [copy(svec),copy(svec)]
    s = LogUpdateAfterTargetWalkers(targetwalkers = 100)
    pb = RunTillLastStep(laststep = 100)
    seedCRNG!(12345) # uses RandomNumbers.Xorshifts.Xoroshiro128Plus()
    @time rr = fciqmc!(vv, pb, ham, s, r_strat, τ_strat, similar.(vv))
    @test sum(rr[1][:,:xHy]) ≈ -12366.096729400324
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
    @test sum(rdfs[:,:spawns]) == 1725

    # fciqmc with delayed shift update
    pa = RunTillLastStep(laststep = 100)
    s = DelayedLogUpdateAfterTargetWalkers(targetwalkers = 100, a = 5)
    svec = DVec(Dict(aIni => 2), ham(:dim))
    StochasticStyle(svec)
    vs = copy(svec)
    seedCRNG!(12345) # uses RandomNumbers.Xorshifts.Xoroshiro128Plus()
    @time rdfs = fciqmc!(vs, pa, ham, s, r_strat, τ_strat, similar(vs))
    @test sum(rdfs[:,:spawns]) == 2998

    # replica fciqmc
    vv = [copy(svec),copy(svec)]
    s = LogUpdateAfterTargetWalkers(targetwalkers = 1_000)
    pb = RunTillLastStep(laststep = 300)
    seedCRNG!(12345) # uses RandomNumbers.Xorshifts.Xoroshiro128Plus()
    @time rr = fciqmc!(vv, pb, ham, s, r_strat, τ_strat, similar.(vv))
    @test sum(rr[1][:,:xHy]) ≈ -3.1790581344812755e6

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
    @test sum(rdfs[:,:spawns]) == 39288

    # single step
    ṽ, w̃, stats = Rimu.fciqmc_step!(ham, copy(vs), pa.shift, pa.dτ, 1.0, similar(vs))
    @test sum(stats) == 1319

    # single step multi threading
    cws = capacity(vs)÷Threads.nthreads()+1
    ws = Tuple(similar(vs,cws) for i=1:Threads.nthreads())
    ṽ, w̃, stats = Rimu.fciqmc_step!(ham, copy(vs), pa.shift, pa.dτ, 1.0, ws;
                    batchsize = length(vs)÷4+1)
    if Threads.nthreads() == 1
        @test sum(stats) == 1303 # test assuming nthreads() == 1
    end

    # run 100 steps with multi
    pa.laststep = 200
    @time rdfs = fciqmc!(vs, pa, rdfs, ham, s, r_strat, τ_strat, ws)
    if Threads.nthreads() == 1
        @test sum(rdfs[:,:spawns]) == 88269 # test assuming nthreads() == 1
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
        @test sum(rdfs[:,:spawns]) == 39106 # test assuming nthreads() == 1
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
Rimu.StochasticStyle(::Type{typeof(svec)}) = IsStochasticWithThreshold(1.0)
StochasticStyle(svec)
vs = copy(svec)
pa = RunTillLastStep(laststep = 100)
seedCRNG!(12345) # uses RandomNumbers.Xorshifts.Xoroshiro128Plus()
@time rdfs = fciqmc!(vs, pa, ham, s, EveryTimeStep(), ConstantTimeStep(), copy(vs), p_strat = p)
@test sum(rdfs[:,:norm]) ≈ 7658.142200557433

# NoMemory
vs = copy(svec)
seedCRNG!(12345) # uses RandomNumbers.Xorshifts.Xoroshiro128Plus()
pa = RunTillLastStep(laststep = 100)
@time rdfs = fciqmc!(vs, pa, ham, s, EveryTimeStep(), ConstantTimeStep(), copy(vs), m_strat = NoMemory(), p_strat = p)
@test sum(rdfs[:,:norm]) ≈ 7658.142200557433

# DeltaMemory
vs = copy(svec)
seedCRNG!(12345) # uses RandomNumbers.Xorshifts.Xoroshiro128Plus()
pa = RunTillLastStep(laststep = 100)
@time rdfs = fciqmc!(vs, pa, ham, s, EveryTimeStep(), ConstantTimeStep(), copy(vs), m_strat = DeltaMemory(1), p_strat = p)
@test sum(rdfs[:,:norm]) ≈ 7658.142200557433

# DeltaMemory
vs = copy(svec)
seedCRNG!(12345) # uses RandomNumbers.Xorshifts.Xoroshiro128Plus()
pa = RunTillLastStep(laststep = 100)
@time rdfs = fciqmc!(vs, pa, ham, s, EveryTimeStep(), ConstantTimeStep(), copy(vs), m_strat = DeltaMemory(10), p_strat = p)
@test sum(rdfs[:,:norm]) ≈ 7517.967201522948

# DeltaMemory2
vs = copy(svec)
seedCRNG!(12345) # uses RandomNumbers.Xorshifts.Xoroshiro128Plus()
pa = RunTillLastStep(laststep = 100)
@time rdfs = fciqmc!(vs, pa, ham, s, EveryTimeStep(), ConstantTimeStep(), copy(vs), m_strat = Rimu.DeltaMemory2(10), p_strat = p)
@test sum(rdfs[:,:norm]) ≈ 7741.941976381574

# ScaledThresholdProject
vs = copy(svec)
seedCRNG!(12345) # uses RandomNumbers.Xorshifts.Xoroshiro128Plus()
pa = RunTillLastStep(laststep = 100)
p_strat = ScaledThresholdProject(1.0)
@time rdfs = fciqmc!(vs, pa, ham, s, EveryTimeStep(), ConstantTimeStep(), copy(vs), m_strat = DeltaMemory(10), p_strat = p_strat)
@test sum(rdfs[:,:norm]) ≈ 7552.573951281426

# ProjectedMemory
vs = copy(svec)
seedCRNG!(12345) # uses RandomNumbers.Xorshifts.Xoroshiro128Plus()
pa = RunTillLastStep(laststep = 100)
p_strat = NoProjection() # ScaledThresholdProject(1.0)
m_strat = Rimu.ProjectedMemory(10,UniformProjector(), vs)
@time rdfs = fciqmc!(vs, pa, ham, s, EveryTimeStep(), ConstantTimeStep(), copy(vs), m_strat = m_strat, p_strat = p_strat)
@test sum(rdfs[:,:norm]) ≈ 7533.354766387553

# ShiftMemory
vs = copy(svec)
seedCRNG!(12345) # uses RandomNumbers.Xorshifts.Xoroshiro128Plus()
pa = RunTillLastStep(laststep = 100)
p_strat = NoProjection() #ScaledThresholdProject(1.0)
@time rdfs = fciqmc!(vs, pa, ham, s, EveryTimeStep(), ConstantTimeStep(), copy(vs), m_strat = ShiftMemory(10), p_strat = p_strat)
@test sum(rdfs[:,:norm]) ≈ 7868.300757047945

# applyMemoryNoise
v2=DVec(Dict(aIni => 2))
StochasticStyle(v2) # IsStochastic() is not suitable for DeltaMemory()
@test_throws ErrorException Rimu.applyMemoryNoise!(v2, v2, 0.0, 0.1, 20, DeltaMemory(3))
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
    @test rdf.:shift[101] ≈ -6.840080658204963
    # Multi-threading
    svec2 = DVec(Dict(aIni => 2.0), ham(:dim))
    pa = RunTillLastStep(laststep = steps,  dτ = dτ)
    cws = capacity(svec2)÷Threads.nthreads()+1
    ws = Tuple(similar(svec2,cws) for i=1:Threads.nthreads())
    @test Rimu.threadedWorkingMemory(svec2) == ws
    @time rdf = fciqmc!(svec2, pa, ham, s_strat, r_strat, τ_strat, ws)
    @test rdf.:shift[101] ≈ -6.840080658204963
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
