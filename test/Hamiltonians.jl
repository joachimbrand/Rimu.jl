using LinearAlgebra
using Rimu
using Test

using Rimu.Hamiltonians: LOStructure

"""
    test_hamiltonian_interface(H, addr)

The main purpose of this test function is to check that all required methods are defined.
"""
function test_hamiltonian_interface(H, addr)
    @testset "$(nameof(typeof(H)))" begin
        @testset "*, mul!, and call" begin
            v = DVec2(addr => 2.0, capacity=100)
            v′ = H(v)
            v″ = H * v
            v‴ = similar(v)
            H(v‴, v)
            v⁗ = similar(v)
            mul!(v⁗, H, v)
            @test v′ == v″ == v‴ == v⁗
        end
        @testset "diagME" begin
            @test diagME(H, addr) ≥ 0
        end
        @testset "hopping" begin
            hops = Hops(H, addr)
            @test length(hops) == numOfHops(H, addr)
            for i in 1:length(hops)
                @test hops[i] == hop(H, addr, i)
            end
        end
        @testset "LOStructure" begin
            @test LOStructure(H) isa LOStructure
        end
        @testset "dimension" begin
            @test dimension(H) isa Int
            @test dimension(Float64, H) isa Float64
            @test dimension(Int, H) === dimension(H)
        end
    end
end

@testset "Interface tests" begin
    for (H, addr) in (
        (BoseHubbardReal1D(n=5, m=5, AT=BoseFS{5,5}), BoseFS((1, 1, 1, 2, 0))),
        (HubbardReal1D(BoseFS((1, 2, 3, 4))), BoseFS((1, 2, 3, 4))),

        (BoseHubbardMom1D(n=15, m=5, add=BoseFS((2, 2, 3, 3, 0))), BoseFS((2, 2, 3, 3, 0))),
        (HubbardMom1D(BoseFS((6, 0, 0, 4))), BoseFS((6, 0, 0, 4))),

        (BoseHubbardReal1D2C(BoseFS2C((1,2,3), (1,0,0))), BoseFS2C((1,2,3), (1,0,0))),
        (BoseHubbardMom1D2C(BoseFS2C((1,2,3), (1,0,0))), BoseFS2C((1,2,3), (1,0,0))),

        (ExtendedHubbardReal1D(BoseFS((1,0,0,0,1))), BoseFS((1,0,0,0,1))),
        (ExtendedBHReal1D(), BoseFS((1,0,0,0,1))),
    )
        test_hamiltonian_interface(H, addr)
    end
end

@testset "old tests" begin
    ham = BoseHubbardReal1D(
        n = 9,
        m = 9,
        u = 6.0,
        t = 1.0,
        AT = BoseFS{9,9})
    @test dimension(ham) == 24310

    aIni = Rimu.Hamiltonians.nearUniform(ham)
    @test aIni == BoseFS{9,9}((1,1,1,1,1,1,1,1,1))

    hp = Hops(ham,aIni)
    @test length(hp) == 18
    @test hp[18][1] == BoseFS{9,9}(BitString{17}(0x000000000000d555))
    @test hp[18][2] ≈ -√2
    @test diagME(ham,aIni) == 0
    os = BoseFS([12,0,1,0,2,1,1,0,1,0,0,0,1,2,0,4])
    @test Rimu.Hamiltonians.bosehubbardinteraction(os) == 148
    @test Rimu.Hamiltonians.extended_bose_hubbard_interaction(os) == (53, 148)
    @test Rimu.Hamiltonians.numberoccupiedsites(os) == 9
    hnnn = Rimu.Hamiltonians.hopnextneighbour(BoseFS{25,16}(BitString{40}(0xf342564fff)),3)
    bs = BoseFS(BitString{40}(0xf342564fff))
    hnnbs = Rimu.Hamiltonians.hopnextneighbour(bs,3)
    @test hnnn == hnnbs

    svec = DVec2(Dict(aIni => 2.0), dimension(ham))
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
