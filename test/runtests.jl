using Rimu
using Test
using LinearAlgebra

@testset "Rimu.jl" begin
    # Write your own tests here.
    @test 3==3
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

    for (key, val) in myda2
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
end

@testset "fciqmc.jl" begin
    ham = BoseHubbardReal1D(
        n = 9,
        m = 9,
        u = 6.0,
        t = 1.0,
        AT = BSAdd64)
    aIni = nearUniform(ham)
    pa = FCIQMCParams(laststep = 100)
    s = LogUpdateAfterTargetWalkers(targetwalkers = 100)

    svec = DVec(Dict(aIni => 2), ham(:dim))
    StochasticStyle(svec)
    vs = copy(svec)
    seedCRNG!(12345) # uses RandomNumbers.Xorshifts.Xoroshiro128Plus()
    @time rdfs = fciqmc!(vs, pa, ham, s)
    @test sum(rdfs[:,:spawns]) == 1751
end
