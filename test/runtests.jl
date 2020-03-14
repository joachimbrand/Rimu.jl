using Rimu
using Test
using LinearAlgebra

@testset "Rimu.jl" begin
    # Write your own tests here.
    @test 3==3
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
end

using Rimu.ConsistentRNG
@testset "ConsistentRNG.jl" begin
    seedCRNG!(12345) # uses RandomNumbers.Xorshifts.Xoroshiro128Plus()
    @test cRand(Int) == 6792904027504972346
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
    @time rdfs = fciqmc!(vs, pa, ham, s, EveryTimeStep())
    @test sum(rdfs[:,:spawns]) == 1751

    # fciqmc with delayed shift update
    pa = RunTillLastStep(laststep = 100)
    s = DelayedLogUpdateAfterTargetWalkers(targetwalkers = 100, a = 5)
    svec = DVec(Dict(aIni => 2), ham(:dim))
    StochasticStyle(svec)
    vs = copy(svec)
    seedCRNG!(12345) # uses RandomNumbers.Xorshifts.Xoroshiro128Plus()
    @time rdfs = fciqmc!(vs, pa, ham, s)
    @test sum(rdfs[:,:spawns]) == 2646

    # replica fciqmc
    tup1 = (copy(svec),copy(svec))
    s = LogUpdateAfterTargetWalkers(targetwalkers = 100)
    pb = RunTillLastStep(laststep = 100)
    seedCRNG!(12345) # uses RandomNumbers.Xorshifts.Xoroshiro128Plus()
    @time rr = fciqmc!(tup1, ham, pb, s)
    @test sum(rr[1][:,:xHy]) ≈ -10456.373910680508
end

@testset "fciqmc with BoseFS" begin
# Define the initial Fock state with n particles and m modes
n = m = 9
aIni = nearUniform(BoseFS{n,m})
ham = BoseHubbardReal1D(aIni; u = 6.0, t = 1.0)
# ham = BoseHubbardReal1D(
#     n = 9,
#     m = 9,
#     u = 6.0,
#     t = 1.0,
#     AT = typeof(aIni))
# ham, aIni = setupBoseHubbardReal1D(
#     n = 9,
#     m = 9,
#     u = 6.0,
#     t = 1.0
# )

pa = RunTillLastStep(laststep = 100)

# standard fciqmc
s = LogUpdateAfterTargetWalkers(targetwalkers = 100)
svec = DVec(Dict(aIni => 2), ham(:dim))
StochasticStyle(svec)
vs = copy(svec)
seedCRNG!(12345) # uses RandomNumbers.Xorshifts.Xoroshiro128Plus()
@time rdfs = fciqmc!(vs, pa, ham, s, EveryTimeStep())
@test sum(rdfs[:,:spawns]) == 1751

# fciqmc with delayed shift update
pa = RunTillLastStep(laststep = 100)
s = DelayedLogUpdateAfterTargetWalkers(targetwalkers = 100, a = 5)
svec = DVec(Dict(aIni => 2), ham(:dim))
StochasticStyle(svec)
vs = copy(svec)
seedCRNG!(12345) # uses RandomNumbers.Xorshifts.Xoroshiro128Plus()
@time rdfs = fciqmc!(vs, pa, ham, s)
@test sum(rdfs[:,:spawns]) == 2646

# replica fciqmc
tup1 = (copy(svec),copy(svec))
s = LogUpdateAfterTargetWalkers(targetwalkers = 100)
pb = RunTillLastStep(laststep = 100)
seedCRNG!(12345) # uses RandomNumbers.Xorshifts.Xoroshiro128Plus()
@time rr = fciqmc!(tup1, ham, pb, s)
@test sum(rr[1][:,:xHy]) ≈ -10456.373910680508

# large bit string
n = 200
m = 200
aIni = nearUniform(BoseFS{n,m})
ham = BoseHubbardReal1D(aIni; u = 6.0, t = 1.0)
# ham = BoseHubbardReal1D(
#     n = n,
#     m = m,
#     u = 6.0,
#     t = 1.0,
#     AT = typeof(aIni))
iShift = diagME(ham, aIni)

# standard fciqmc
tw = 1_000
s = DoubleLogUpdate(targetwalkers = tw)
svec = DVec(Dict(aIni => 20), 8*tw)
StochasticStyle(svec)
vs = copy(svec)
seedCRNG!(12345) # uses RandomNumbers.Xorshifts.Xoroshiro128Plus()
pa = RunTillLastStep(laststep = 1, shift = iShift, dτ = 0.001)
@time rdfs = fciqmc!(vs, pa, ham, s, EveryTimeStep())
pa.laststep = 100
@time rdfs = fciqmc!(vs, pa, rdfs, ham, s, EveryTimeStep())
@test sum(rdfs[:,:spawns]) == 39299

# single step
ṽ, w̃, stats = Rimu.fciqmc_step!(ham, copy(vs), pa.shift, pa.dτ, similar(vs))
@test sum(stats) == 1370

# single step multi threading
cws = capacity(vs)÷Threads.nthreads()+1
ws = Tuple(similar(vs,cws) for i=1:Threads.nthreads())
ṽ, w̃, stats = Rimu.fciqmc_step!(ham, copy(vs), pa.shift, pa.dτ, ws)
# @test sum(stats) == 1024
# TODO: need to look at threadsafty of RNG!!!

# run 100 steps with multi
pa.laststep = 200
@time rdfs = fciqmc!(vs, pa, rdfs, ham, s, EveryTimeStep(),ConstantTimeStep(), ws)
# @test sum(rdfs[:,:spawns]) == 76544
# TODO: need to look at threadsafty of RNG!!!

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
