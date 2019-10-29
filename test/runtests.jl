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
    if read(`whoami`,String) == "root\n" # relevant for Pipelines
        rr = run(`mpirun -np 2 --allow-run-as-root julia mpiexample.jl`)
    else
        rr = run(`mpirun -np 2 julia mpiexample.jl`)
    end
    @test rr.exitcode == 0
end
