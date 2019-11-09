using Rimu
using Test
using LinearAlgebra
using Rimu.ConsistentRNG
using BenchmarkTools

@testset "fciqmc.jl" begin
ham = BoseHubbardReal1D(
    n = 15,
    m = 9,
    u = 6.0,
    t = 1.0,
    AT = BSAdd64)
ham(:dim)
aIni = nearUniform(ham)
iShift = diagME(ham, aIni)

# standard fciqmc
s = LogUpdateAfterTargetWalkers(targetwalkers = 1_000)
svec = DVec(Dict(aIni => 20), ham(:dim))
StochasticStyle(svec)
vs = copy(svec)
seedCRNG!(12345) # uses RandomNumbers.Xorshifts.Xoroshiro128Plus()
pa = RunTillLastStep(laststep = 1, shift = iShift)
@time rdfs = fciqmc!(vs, pa, ham, s, EveryTimeStep())
pa.laststep = 1_000
@time rdfs = fciqmc!(vs, pa, rdfs, ham, s, EveryTimeStep())

@test sum(rdfs[:,:spawns]) == 580467

# # replica fciqmc
# tup1 = (copy(svec),copy(svec))
# s = LogUpdateAfterTargetWalkers(targetwalkers = 1_000)
# pb = RunTillLastStep(laststep = 1, shift = iShift)
# seedCRNG!(12345) # uses RandomNumbers.Xorshifts.Xoroshiro128Plus()
# @time rr = fciqmc!(tup1, ham, pb, s)
# pb.laststep = 10
# @time rr = fciqmc!(tup1, ham, pb, s)
#
# @test sum(rr[1][:,:xHy]) ≈ -10456.373910680508

sv = DVec(Dict(aIni => 20.0), 100)
hsv = ham(sv)
v2 = similar(sv,ham(:dim))
@benchmark ham(v2, hsv)

end

@testset "fciqmc with BoseBA" begin
n = 15
m = 9
aIni = BoseBA(n,m)
ham = BoseHubbardReal1D(
    n = n,
    m = m,
    u = 6.0,
    t = 1.0,
    AT = typeof(aIni))
iShift = diagME(ham, aIni)

# standard fciqmc
s = LogUpdateAfterTargetWalkers(targetwalkers = 1_000)
svec = DVec(Dict(aIni => 20), ham(:dim))
StochasticStyle(svec)
vs = copy(svec)
seedCRNG!(12345) # uses RandomNumbers.Xorshifts.Xoroshiro128Plus()
pa = RunTillLastStep(laststep = 1, shift = iShift)
@time rdfs = fciqmc!(vs, pa, ham, s, EveryTimeStep())
pa.laststep = 1_000
@time rdfs = fciqmc!(vs, pa, rdfs, ham, s, EveryTimeStep())

@test sum(rdfs[:,:spawns]) == 535061

sv = DVec(Dict(aIni => 20.0), 100)
hsv = ham(sv)
v2 = similar(sv,ham(:dim))
@benchmark ham(v2, hsv)

n = 200
m = 200
aIni = BoseBA(n,m)
ham = BoseHubbardReal1D(
    n = n,
    m = m,
    u = 6.0,
    t = 1.0,
    AT = typeof(aIni))
iShift = diagME(ham, aIni)

# standard fciqmc
tw = 1_000
s = LogUpdateAfterTargetWalkers(targetwalkers = tw)
svec = DVec(Dict(aIni => 20), 8*tw)
StochasticStyle(svec)
vs = copy(svec)
seedCRNG!(12345) # uses RandomNumbers.Xorshifts.Xoroshiro128Plus()
pa = RunTillLastStep(laststep = 1, shift = iShift, dτ = 0.001)
@time rdfs = fciqmc!(vs, pa, ham, s, EveryTimeStep())
pa.laststep = 1_000
@time rdfs = fciqmc!(vs, pa, rdfs, ham, s, EveryTimeStep())
@test sum(rdfs[:,:spawns]) == 518517

bIni = BoseBA(200,200)
svb = DVec(Dict(bIni => 20.0), 8*tw)
hsvb = ham(svb)
v2b = similar(svb,150_000)
@benchmark ham(v2b, hsvb)

end

@testset "fciqmc. wit BStringAdd" begin
ham = BoseHubbardReal1D(
    n = 15,
    m = 9,
    u = 6.0,
    t = 1.0,
    AT = BStringAdd)
ham(:dim)
aIni = nearUniform(ham)
iShift = diagME(ham, aIni)

# standard fciqmc
s = LogUpdateAfterTargetWalkers(targetwalkers = 1_000)
svec = DVec(Dict(aIni => 20), ham(:dim))
StochasticStyle(svec)
vs = copy(svec)
seedCRNG!(12345) # uses RandomNumbers.Xorshifts.Xoroshiro128Plus()
pa = RunTillLastStep(laststep = 1, shift = iShift)
@time rdfs = fciqmc!(vs, pa, ham, s, EveryTimeStep())
pa.laststep = 1_000
@time rdfs = fciqmc!(vs, pa, rdfs, ham, s, EveryTimeStep())

@test sum(rdfs[:,:spawns]) == 534068

ham = BoseHubbardReal1D(
    n = 200,
    m = 200,
    u = 6.0,
    t = 1.0,
    AT = BStringAdd)
ham(:dim)
aIni = nearUniform(ham)
iShift = diagME(ham, aIni)

# standard fciqmc
tw = 1_000
s = LogUpdateAfterTargetWalkers(targetwalkers = tw)
svec = DVec(Dict(aIni => 20), 8*tw)
StochasticStyle(svec)
vs = copy(svec)
seedCRNG!(12345) # uses RandomNumbers.Xorshifts.Xoroshiro128Plus()
pa = RunTillLastStep(laststep = 1, shift = iShift, dτ = 0.001)
@time rdfs = fciqmc!(vs, pa, ham, s, EveryTimeStep())
pa.laststep = 1_000
@time rdfs = fciqmc!(vs, pa, rdfs, ham, s, EveryTimeStep())
@test sum(rdfs[:,:spawns]) == 685156

sv = DVec(Dict(aIni => 20.0), 8*tw)
hsv = ham(sv)
v2 = similar(sv,150_000)
@benchmark ham(v2, hsv)
end

using Rimu.Hamiltonians

c1 = BSAdd64(0xf342564ffd)
c2 = BSAdd128(0xf342564ffdf00dfdfdfdfd037a3de)
bs1 = BitAdd{40}(0xf342564ffd)
bs2 = BitAdd{128}(0xf342564ffdf00dfdfdfdfd037a3de)
bs3 = BitAdd{144}(big"0xf342564ffdf00dfdfdfdfdfdfdfdfdfdfdf")
bb1 = BoseBA(bs1)
bb2 = BoseBA(bs2)
bb3 = BoseBA(bs3)
b1 = BoseBA(15,9)
b7 = BoseBA(200,200)
bs7 = b7.bs
b4 = BoseBA(128,80) # needs 4 chunks
bs4 = b4.bs
b5 = BoseBA(128,135) # needs 5 chunks
bs5 = b5.bs
b6 = BoseBA(128,250) # needs 6 chunks
bs6 = b6.bs
bfs1 = BoseFS(bs1)
bfs2 = BoseFS(bs2)
bfs3 = BoseFS(bs3)
bfs4 = BoseFS(bs4)
bfs5 = BoseFS(bs5)
bfs6 = BoseFS(bs6)
bfs7 = BoseFS(bs7)


ham = BoseHubbardReal1D(
    n = 15,
    m = 9,
    u = 6.0,
    t = 1.0,
    AT = BStringAdd)
a1 = nearUniform(ham)

ham = BoseHubbardReal1D(
    n = 200,
    m = 200,
    u = 6.0,
    t = 1.0,
    AT = BStringAdd)
a2 = nearUniform(ham)

# BSAdd64
@benchmark Hamiltonians.numberoccupiedsites($c1)
@benchmark Hamiltonians.bosehubbardinteraction(c1)
@benchmark Hamiltonians.hopnextneighbour(c1,4,17,24)

# BSAdd128
@benchmark Hamiltonians.numberoccupiedsites(c2)
@benchmark Hamiltonians.bosehubbardinteraction(c2)
@benchmark Hamiltonians.hopnextneighbour($c2,4,55,74)

# BitAdd
@benchmark Hamiltonians.numberoccupiedsites(bs1)
@benchmark Hamiltonians.bosehubbardinteraction(bs1)
@benchmark Hamiltonians.hopnextneighbour(bs1,4,17,24)

@benchmark Hamiltonians.numberoccupiedsites($bs2)
@benchmark Hamiltonians.bosehubbardinteraction($bs2)
@benchmark Hamiltonians.hopnextneighbour($bs2,4,55,74)

@benchmark Hamiltonians.numberoccupiedsites($bs3)
@benchmark Hamiltonians.bosehubbardinteraction($bs3)
@benchmark Hamiltonians.hopnextneighbour(bs3,4,40,105)

@benchmark Hamiltonians.numberoccupiedsites($bs4)
@benchmark Hamiltonians.bosehubbardinteraction($bs4)
@benchmark Hamiltonians.hopnextneighbour(bs4,4,80,128)

@benchmark $bs1 >>> 32
@benchmark $bs2 >>> 32
@benchmark $bs3 >>> 32
@benchmark $bs4 >>> 32

# BoseBA
@benchmark Hamiltonians.numberoccupiedsites($bb1)
@benchmark Hamiltonians.bosehubbardinteraction($bb1)
@benchmark Hamiltonians.hopnextneighbour(bb1,4,17,24)

@benchmark Hamiltonians.numberoccupiedsites($bb2)
@benchmark Hamiltonians.bosehubbardinteraction($bb2)
@benchmark Hamiltonians.hopnextneighbour(bb2,4,55,74)
@benchmark Hamiltonians.numberoccupiedsites($bb3)
@benchmark Hamiltonians.bosehubbardinteraction($bb3)
@benchmark Hamiltonians.hopnextneighbour(bb3,4,40,105)

@benchmark Hamiltonians.numberoccupiedsites(b1)
@benchmark Hamiltonians.bosehubbardinteraction(b1)
@benchmark Hamiltonians.hopnextneighbour(b1,3,9,15)
@benchmark Hamiltonians.hopnextneighbour(b1,4,9,15)
@benchmark Hamiltonians.numberoccupiedsites(b7)
@benchmark Hamiltonians.bosehubbardinteraction(b7)
@benchmark Hamiltonians.hopnextneighbour(b7,4,200,200)

@benchmark Hamiltonians.numberoccupiedsites($b4)
@benchmark Hamiltonians.bosehubbardinteraction($b2)
@benchmark Hamiltonians.hopnextneighbour($b2,4,80,128)

# BoseFS
@benchmark Hamiltonians.numberoccupiedsites($bfs1)
@benchmark Hamiltonians.bosehubbardinteraction($bfs1)
@benchmark Hamiltonians.hopnextneighbour($bfs1,4,17,24)

@benchmark Hamiltonians.numberoccupiedsites($bfs2)
@benchmark Hamiltonians.bosehubbardinteraction($bfs2)
@benchmark Hamiltonians.hopnextneighbour(bfs2,4,55,74)
@benchmark Hamiltonians.numberoccupiedsites($bfs3)
@benchmark Hamiltonians.bosehubbardinteraction($bfs3)
@benchmark Hamiltonians.hopnextneighbour($bfs3,4,40,105)
@benchmark Hamiltonians.numberoccupiedsites($bfs6)
@benchmark Hamiltonians.bosehubbardinteraction($bfs6)
@benchmark Hamiltonians.hopnextneighbour($bfs6,4,40,105)


# BStringAdd
@benchmark Hamiltonians.numberoccupiedsites(a1)
@benchmark Hamiltonians.bosehubbardinteraction(a1)
@benchmark Hamiltonians.hopnextneighbour(a1,3,9,15)
@benchmark Hamiltonians.numberoccupiedsites(a2)
@benchmark Hamiltonians.bosehubbardinteraction(a2)
@benchmark Hamiltonians.hopnextneighbour(a2,4,200,200)

# BoseBA with two chunks
aIni = bb2
ham = BoseHubbardReal1D(
    n = 74,
    m = 55,
    u = 6.0,
    t = 1.0,
    AT = typeof(aIni))

sv = DVec(Dict(aIni => 20.0), 200)
v2 = similar(sv,150_000)
# hsv = ham(sv)
@benchmark ham(v2, sv)
hsv = ham(sv)
@benchmark ham(v2, hsv)

# BSAdd128
aIni = c2
ham = BoseHubbardReal1D(
    n = 74,
    m = 55,
    u = 6.0,
    t = 1.0,
    AT = typeof(aIni))

sv = DVec(Dict(aIni => 20.0), 200)
v2 = similar(sv,150_000)
# hsv = ham(sv)
@benchmark ham(v2, sv)
hsv = ham(sv)
@benchmark ham(v2, hsv)

# BoseBA with seven chunks
aIni = b7
ham = BoseHubbardReal1D(
    n = 200,
    m = 200,
    u = 6.0,
    t = 1.0,
    AT = typeof(aIni))

sv = DVec(Dict(aIni => 20.0), 400)
v2 = similar(sv,150_000)
# hsv = ham(sv)
@benchmark ham(v2, sv)
hsv = ham(sv)
@benchmark ham(v2, hsv)

# BStringAdd with seven chunks
ham = BoseHubbardReal1D(
    n = 200,
    m = 200,
    u = 6.0,
    t = 1.0,
    AT = BStringAdd)
aIni = nearUniform(ham)
sv = DVec(Dict(aIni => 20.0), 400)
v2 = similar(sv,150_000)
# hsv = ham(sv)
@benchmark ham(v2, sv)
hsv = ham(sv)
@benchmark ham(v2, hsv)

# BoseFS with seven chunks
n = m = 200
aIni = BoseFS(n,m)
ham = BoseHubbardReal1D(
    n = n,
    m = m,
    u = 6.0,
    t = 1.0,
    AT = typeof(aIni))

sv = DVec(Dict(aIni => 20.0), 400)
v2 = similar(sv,150_000)
# hsv = ham(sv)
@benchmark ham(v2, sv)
hsv = ham(sv)
@benchmark ham(v2, hsv)
